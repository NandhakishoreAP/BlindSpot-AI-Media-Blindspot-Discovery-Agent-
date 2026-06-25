import json
import re
import time
import string
import traceback
import httpx

from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError

from config import Config
from utils.logger import get_logger


class GeminiClient:
    """Client for Google Gemini API with session-level warmup caching and quota handling."""

    REQUEST_TIMEOUT_SECONDS = 60

    # Class-level warmup cache (persists across instances within a Python process)
    _session_warmed = False
    _session_warmed_model = ""
    _last_warmup_time = 0.0
    WARMUP_CACHE_TTL = 600  # seconds (10 min)

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("gemini_client")
        self.available = False
        self.model = config.GEMINI_MODEL
        self.last_eval_count = 0
        self.failures_count = 0

        # Quota state — separate from available so the pipeline stays live
        self.quota_exceeded = False
        self.quota_retry_after = 0.0  # epoch timestamp when retry is allowed

        api_key = config.GEMINI_API_KEY
        if not api_key:
            self.logger.error("GEMINI_API_KEY not configured.")
            return

        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            self.logger.error(f"Failed to create Gemini client: {e}")
            return

        # Use cached warmup if the same model was already warmed up
        if GeminiClient._session_warmed and GeminiClient._session_warmed_model == self.model:
            if time.time() - GeminiClient._last_warmup_time < GeminiClient.WARMUP_CACHE_TTL:
                self.available = True
                self.logger.info(f"Active model: {self.model} (warmup cached)")
                return

        warmup_ok = self.warmup_model()
        if warmup_ok:
            self.available = True
            GeminiClient._session_warmed = True
            GeminiClient._session_warmed_model = self.model
            GeminiClient._last_warmup_time = time.time()
            self.logger.info(f"Active model: {self.model}")
        elif self.quota_exceeded:
            # Temporary quota exhaustion — keep available so pipeline components
            # are instantiated; generate() will skip requests until retry window.
            self.available = True
            self.logger.warning(
                "Gemini quota temporarily exceeded during warmup. "
                "Pipeline will run with degraded quality until quota resets."
            )
        else:
            self.available = False
            self.logger.error("Gemini warmup failed. GeminiClient unavailable.")

    def warmup_model(self) -> bool:
        self.logger.info(f"Verifying Gemini and warming up model {self.model}...")
        if not self.health_check():
            self.logger.warning("Gemini health check failed during warm-up")
            return False
        warmup_start = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents="warmup",
                config={"max_output_tokens": 1}
            )
            warmup_duration = time.time() - warmup_start
            self.logger.info(f"Gemini model warmed in {warmup_duration:.2f} seconds")
            return True
        except ClientError as e:
            if e.code == 429:
                self.quota_exceeded = True
                self.quota_retry_after = time.time() + self._parse_retry_delay(e)
                self.logger.warning(
                    "Gemini quota exceeded during warmup. "
                    f"Retry after {self.quota_retry_after - time.time():.0f}s."
                )
                return False
            self.logger.error(f"Gemini warmup failed with client error ({e.code}): {e}")
            return False
        except Exception as e:
            self.logger.error(f"Gemini warmup request failed: {e}")
            return False

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except ClientError as e:
            if e.code == 429:
                self.quota_exceeded = True
                self.quota_retry_after = time.time() + self._parse_retry_delay(e)
                self.logger.warning(
                    "Gemini quota exceeded during health check. "
                    f"Retry after {self.quota_retry_after - time.time():.0f}s."
                )
            else:
                self.logger.warning(f"Gemini health check failed: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"Gemini health check failed: {e}")
            return False

    def pre_call_health_check(self, model: str = None) -> bool:
        return True

    @staticmethod
    def _parse_retry_delay(exc: ClientError) -> float:
        """Extract retry delay from a 429 ClientError's RetryInfo detail."""
        try:
            error_body = getattr(exc, "details", None) or {}
            details_list = error_body.get("error", {}).get("details", [])
            for detail in details_list:
                retry_delay = detail.get("retryDelay", "")
                if retry_delay:
                    # Format: "38.57s" or "38s"
                    import re as _re
                    match = _re.search(r"(\d+(?:\.\d+)?)", retry_delay)
                    if match:
                        return float(match.group(1))
            # Fallback: retry in 60 seconds
            return 60.0
        except Exception:
            return 60.0

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        num_predict: int = 128,
        num_ctx: int = 4096,
        format: any = None,
        preset_name: str = None,
        model: str = None,
        think: bool = False,
        timeout: float = None
    ) -> str:
        # Check quota before attempting — short-circuit if still in retry-wait
        if self.quota_exceeded:
            remaining = self.quota_retry_after - time.time()
            if remaining > 0:
                self.logger.warning(
                    f"Gemini quota exceeded. Skipping request. "
                    f"Retry in {remaining:.0f}s."
                )
                return ""
            self.logger.info("Gemini quota retry window reached, attempting request.")
            self.quota_exceeded = False
            self.quota_retry_after = 0.0

        # Match Ollama behavior: always prepend /no_think
        if system_prompt:
            system_prompt = "/no_think\n\n" + system_prompt
        else:
            system_prompt = "/no_think"

        target_model = model or self.model

        # Build generation config
        # Gemini 2.5 Flash uses thinking tokens that consume the output budget;
        # ensure max_output_tokens is large enough to leave room for real output.
        min_output = max(num_predict, 4096)
        gen_config = {
            "temperature": temperature,
            "max_output_tokens": min_output,
        }
        if system_prompt:
            gen_config["system_instruction"] = system_prompt
        if format:
            gen_config["response_mime_type"] = "application/json"

        active_timeout = timeout or getattr(self.config, 'REQUEST_TIMEOUT_SECONDS', 60.0)
        if isinstance(active_timeout, (int, float)) and active_timeout < 60.0:
            active_timeout = 60.0

        retry_delays = [0, 5, 10]
        last_exception = None

        for attempt_idx, delay in enumerate(retry_delays, 1):
            if delay > 0:
                self.logger.info(f"Retrying Gemini request, attempt {attempt_idx}/3 after waiting {delay} seconds...")
                time.sleep(delay)
            else:
                self.logger.debug(f"Gemini request started (Attempt 1/3)")

            start_time = time.time()
            try:
                response = self.client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=gen_config,
                )
                elapsed = time.time() - start_time
                self.logger.info("Gemini request completed")
                self.logger.info(f"Duration: {int(elapsed)} seconds")

                response_text = ""
                if response.candidates:
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if part.text:
                                    response_text += part.text

                self.logger.debug(f"[GEMINI RAW RESPONSE] {repr(response_text)}")
                if response.usage_metadata:
                    self.last_eval_count = response.usage_metadata.total_token_count or 0

                return response_text.strip()
            except ClientError as e:
                elapsed = time.time() - start_time
                if e.code == 429:
                    self.quota_exceeded = True
                    self.quota_retry_after = time.time() + self._parse_retry_delay(e)
                    self.logger.warning(
                        "Gemini quota exceeded (429). "
                        f"Retry after {self.quota_retry_after - time.time():.0f}s."
                    )
                    return ""
                self.logger.warning(
                    f"Gemini request failed on attempt {attempt_idx}/3. "
                    f"Duration: {int(elapsed)}s. Error: {e}"
                )
                last_exception = e
                self.failures_count += 1
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.warning(
                    f"Gemini request failed on attempt {attempt_idx}/3. "
                    f"Duration: {int(elapsed)}s. Error: {e}"
                )
                last_exception = e
                self.failures_count += 1

        self.logger.error(
            f"All 3 Gemini attempts failed. "
            f"Returning fallback empty string. Last error: {last_exception}"
        )
        return ""

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        num_predict: int = 128,
        num_ctx: int = 4096,
        preset_name: str = None,
        model: str = None,
        think: bool = False,
        timeout: float = None
    ) -> dict:
        response_text = ""
        try:
            response_text = self.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                num_predict=num_predict,
                num_ctx=num_ctx,
                format="json",
                preset_name=preset_name,
                model=model,
                think=think,
                timeout=timeout
            )
            self.logger.debug(f"Raw model output: {response_text}")
            if not response_text:
                return {}
            return self.extract_json(response_text)
        except Exception as e:
            self.logger.error(f"Gemini JSON generation failed: {e}")
            if response_text:
                self.failures_count += 1
            return {}

    def generate_json_with_retry(
        self,
        prompt: str,
        system_prompt: str = "",
        max_retries: int = 2,
        temperature: float = 0.2,
        num_predict: int = 128,
        num_ctx: int = 4096,
        preset_name: str = None,
        model: str = None,
        think: bool = False,
        timeout: float = None
    ) -> dict:
        current_prompt = prompt
        last_error = None

        for i in range(max_retries + 1):
            try:
                res = self.generate_json(
                    current_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    num_predict=num_predict,
                    num_ctx=num_ctx,
                    preset_name=preset_name,
                    model=model,
                    think=think,
                    timeout=timeout
                )
                if isinstance(res, dict):
                    if res == {}:
                        self.logger.info("JSON response is exactly empty dict {}, disabling retry loop.")
                        return {}
                    return res
                raise ValueError("Empty or invalid JSON dictionary returned")
            except (ValueError, RuntimeError) as e:
                last_error = e
                self.logger.warning(
                    f"JSON parsing failed attempt {i + 1}/{max_retries + 1}, retrying... Error: {e}"
                )
                if i < max_retries:
                    current_prompt = (
                        prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No explanations. No markdown."
                    )

        self.logger.error(f"Gemini JSON generation completely failed: {last_error}. Returning fallback empty dictionary.")
        return {}

    def repair_json_string(self, text: str) -> str:
        if not text:
            return "{}"
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        first_brace = text.find('{')
        first_bracket = text.find('[')
        start_idx = 0
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start_idx = first_brace
        elif first_bracket != -1:
            start_idx = first_bracket
        s = text[start_idx:]
        in_string = False
        escape = False
        stack = []
        repaired_chars = []
        i = 0
        while i < len(s):
            char = s[i]
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == '{':
                    stack.append('}')
                elif char == '[':
                    stack.append(']')
                elif char == '}':
                    if stack and stack[-1] == '}':
                        stack.pop()
                    else:
                        if '}' in stack:
                            while stack:
                                popped = stack.pop()
                                if popped == '}':
                                    break
                elif char == ']':
                    if stack and stack[-1] == ']':
                        stack.pop()
                    else:
                        if ']' in stack:
                            while stack:
                                popped = stack.pop()
                                if popped == ']':
                                    break
            repaired_chars.append(char)
            i += 1
        repaired = "".join(repaired_chars)
        if in_string:
            if repaired.endswith('\\'):
                repaired = repaired[:-1]
            repaired += '"'
        repaired = repaired.strip()
        if stack and stack[-1] == '}':
            last_comma_or_brace = max(repaired.rfind(','), repaired.rfind('{'))
            last_colon = repaired.rfind(':')
            if last_comma_or_brace > last_colon:
                repaired = repaired.rstrip()
                repaired += ': null'
        while True:
            prev_len = len(repaired)
            repaired = repaired.strip()
            if repaired.endswith(':'):
                repaired += ' null'
            elif repaired.endswith(','):
                repaired = repaired[:-1]
            else:
                colon_match = re.search(r':\s*$', repaired)
                if colon_match:
                    repaired = repaired[:colon_match.start()] + ': null'
                else:
                    break
            if len(repaired) == prev_len:
                break
        while stack:
            repaired += stack.pop()
        repaired = re.sub(r',\s*([\]}])', r'\1', repaired)
        return repaired

    def extract_json(self, text: str) -> dict:
        self.logger.debug(f"[JSON EXTRACTION INPUT] repr(text): {repr(text)}")
        self.logger.debug(f"Extracting JSON from text: {text}")
        if text:
            cleaned_text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        else:
            cleaned_text = ""
        self.logger.debug(f"Raw model output length: {len(cleaned_text)}")
        self.logger.debug(f"Raw output snippet: {cleaned_text[:1000]}")

        # 1. Direct json
        try:
            parsed = json.loads(cleaned_text)
            self.logger.debug("JSON extraction strategy used: direct_json")
            return self._clean_and_repair_json(parsed)
        except json.JSONDecodeError:
            pass

        # 2. Try JSON recovery engine
        try:
            repaired = self.repair_json_string(cleaned_text)
            parsed = json.loads(repaired)
            self.logger.info("JSON_RECOVERY_SUCCESS")
            self.logger.debug("JSON extraction strategy used: repaired_json_via_recovery_engine")
            return self._clean_and_repair_json(parsed)
        except Exception as e:
            self.logger.debug(f"JSON recovery engine failed: {e}")

        # 3. Markdown blocks
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_text)
        if markdown_match:
            try:
                parsed = json.loads(markdown_match.group(1).strip())
                self.logger.debug("JSON extraction strategy used: markdown_json")
                return self._clean_and_repair_json(parsed)
            except Exception as e:
                self.logger.debug(f"Markdown JSON repair failed: {traceback.format_exc()}")
            try:
                repaired = self.repair_json_string(markdown_match.group(1).strip())
                parsed = json.loads(repaired)
                self.logger.info("JSON_RECOVERY_SUCCESS")
                self.logger.debug("JSON extraction strategy used: markdown_repaired_json")
                return self._clean_and_repair_json(parsed)
            except Exception as e:
                self.logger.debug(f"Markdown JSON recovery failed: {traceback.format_exc()}")

        # 4. First-to-last brace extraction
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            candidate = cleaned_text[start_idx:end_idx + 1]
            try:
                parsed = json.loads(candidate)
                self.logger.debug("JSON extraction strategy used: brace_extraction")
                return self._clean_and_repair_json(parsed)
            except Exception as e:
                self.logger.debug(f"First-to-last brace extraction failed: {traceback.format_exc()}")
            try:
                repaired = self.repair_json_string(candidate)
                parsed = json.loads(repaired)
                self.logger.info("JSON_RECOVERY_SUCCESS")
                self.logger.debug("JSON extraction strategy used: brace_repaired_json")
                return self._clean_and_repair_json(parsed)
            except Exception as e:
                self.logger.debug(f"Repaired JSON parse failed: {traceback.format_exc()}")

        # 5. Repaired JSON
        try:
            repaired = cleaned_text
            repaired = repaired.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
            repaired = repaired.replace('`', '')
            repaired = "".join(char for char in repaired if char in string.printable)
            r_start = repaired.find('{')
            r_end = repaired.rfind('}')
            if r_start != -1 and r_end != -1 and r_start < r_end:
                repaired = repaired[r_start:r_end + 1]
            parsed = json.loads(repaired)
            self.logger.debug("JSON extraction strategy used: legacy_repaired_json")
            return self._clean_and_repair_json(parsed)
        except Exception as e:
            self.logger.debug(f"Legacy repaired JSON attempt failed: {e}")

        # 6. Fallback multi-candidate brace regex
        candidates = re.findall(r"\{.*?\}", cleaned_text, re.DOTALL)
        for idx, candidate in enumerate(candidates):
            try:
                parsed = json.loads(candidate)
                self.logger.debug(f"JSON extraction strategy used: brace_regex_candidate_{idx}")
                return self._clean_and_repair_json(parsed)
            except json.JSONDecodeError:
                pass

        # 7. Regex-based partial recovery
        try:
            matches = re.findall(r"\{[^{}]+?\}", cleaned_text, re.DOTALL)
            recovered = []
            for m in matches:
                if ("relevance" in m or "relevance_score" in m) and "quality" in m:
                    try:
                        parsed_obj = json.loads(m)
                        if isinstance(parsed_obj, dict):
                            recovered.append(self._clean_and_repair_json(parsed_obj))
                    except Exception as e:
                        self.logger.debug(f"Partial JSON regex failed: {traceback.format_exc()}")
                        try:
                            m_rep = m.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'").replace('`', '')
                            parsed_obj = json.loads(m_rep)
                            if isinstance(parsed_obj, dict):
                                recovered.append(self._clean_and_repair_json(parsed_obj))
                        except Exception as e:
                            self.logger.debug(f"Nested JSON extraction failed: {traceback.format_exc()}")
            if recovered:
                self.logger.debug(f"JSON extraction strategy used: regex_partial_recovery (recovered {len(recovered)} items)")
                return recovered
        except Exception as e:
            self.logger.debug(f"Regex partial recovery failed: {e}")

        self.logger.error("JSON parsing failure: All extraction strategies failed. Returning empty dict.")
        return {}

    def _clean_and_repair_json(self, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        recovered_text = None
        key_to_delete = None
        fs_val = data.get("framing_summary")
        if fs_val is None or not str(fs_val).strip() or str(fs_val).strip() == "Unknown":
            for k, v in data.items():
                k_str = str(k)
                v_str = str(v)
                if "framing_summary" in k_str or "framing_summary" in v_str:
                    combined = f"{k_str} : {v_str}"
                    match = re.search(r'framing_summary["\\]*\s*[:=]\s*["\\]*([\s\S]+?)(?:["\\]+\s*)?$', combined)
                    if match:
                        recovered_text = match.group(1).strip().rstrip('}').strip().strip('"').strip("'")
                        key_to_delete = k
                        break
                    elif ":" in combined:
                        parts = combined.split(":", 1)
                        recovered_text = parts[1].strip().strip('"').strip("'").strip().rstrip('}').strip()
                        key_to_delete = k
                        break
        cleaned = {}
        for k, v in data.items():
            if k == key_to_delete:
                continue
            k_str = str(k).strip()
            if not k_str:
                continue
            is_punct_or_quotes = all(char in string.punctuation or char in '"\' ' for char in k_str)
            if is_punct_or_quotes:
                continue
            cleaned[k] = v
        if recovered_text is not None:
            cleaned["framing_summary"] = recovered_text
        self.logger.debug(f"Final cleaned JSON: {cleaned}")
        return cleaned
