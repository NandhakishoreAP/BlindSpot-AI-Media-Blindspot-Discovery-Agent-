import json
import re
import time
import traceback
import string
# pyrefly: ignore [missing-import]
import ollama
import httpx

from config import Config
from utils.logger import get_logger

class ModelNotFoundError(Exception):
    pass

class ExtractionPreset:
    """
    Predefined configuration presets for extraction components.
    """
    CLAIM_ANALYZER_NAME = "CLAIM_ANALYZER"
    BLINDSPOT_DETECTOR_NAME = "BLINDSPOT_DETECTOR"
    EVIDENCE_EVALUATOR_NAME = "EVIDENCE_EVALUATOR"

    CLAIM_ANALYZER = {
        "temperature": 0.2,
        "num_predict": 192,
        "num_ctx": 4096
    }
    BLINDSPOT_DETECTOR = {
        "temperature": 0.2,
        "num_predict": 192,
        "num_ctx": 4096
    }
    EVIDENCE_EVALUATOR = {
        "temperature": 0.2,
        "num_predict": 192,
        "num_ctx": 4096
    }

FALLBACK_MODELS = ["qwen3:4b", "qwen3:1.7b", "phi4-mini"]

class OllamaClient:
    REQUEST_TIMEOUT_SECONDS = 60
    _model_verified = False
    _model_warmed = False

    def __init__(self, config: Config):
        """
        Stores config, initializes logger, creates Ollama Client, and stores the target model name.
        Attempts model warm-up with fallback models if primary fails.
        """
        self.config = config
        self.logger = get_logger("ollama_client")
        self.available = False
        timeout_val = getattr(config, 'REQUEST_TIMEOUT_SECONDS', self.REQUEST_TIMEOUT_SECONDS)
        # Standardize timeout to at least 60 seconds
        if isinstance(timeout_val, (int, float)) and timeout_val < 60:
            timeout_val = 60
        timeout_obj = httpx.Timeout(float(timeout_val), connect=10.0, read=float(timeout_val))
        self.client = ollama.Client(host=config.OLLAMA_BASE_URL, timeout=timeout_obj)
        self.model = config.OLLAMA_MODEL
        self.last_eval_count = 0
        self.failures_count = 0
        
        # Build fallback model list: primary first, then hardcoded fallbacks
        fallback_models = [config.OLLAMA_MODEL]
        for fb in FALLBACK_MODELS:
            if fb not in fallback_models:
                fallback_models.append(fb)

        # Try each model in sequence; use the first that warms up successfully
        for candidate in fallback_models:
            self.model = candidate
            if self.warmup_model():
                self.available = True
                self.logger.info(f"Active model: {self.model}")
                break
            self.logger.warning(f"Model {candidate} failed to warm up, trying fallback...")
        
        if not self.available:
            self.logger.error("All models failed to initialize. OllamaClient unavailable.")

    def warmup_model(self) -> bool:
        """
        Verifies Ollama availability, checks if model exists, and runs lightweight warmup.
        Returns True on success, False on any failure (does not raise).
        """
        self.logger.info(f"Verifying Ollama and warming up model {self.model}...")
        
        # 1. Verify Ollama availability
        if not self.health_check():
            self.logger.warning("Ollama unreachable during model warm-up")
            return False
        self.logger.info("Ollama reachable")

        # 2. Verify model exists
        try:
            models_list = self.client.list()
            models_raw = []
            if hasattr(models_list, 'models'):
                models_raw = models_list.models
            elif isinstance(models_list, dict):
                models_raw = models_list.get('models', [])
            elif hasattr(models_list, 'get'):
                models_raw = models_list.get('models', [])
            else:
                models_raw = list(models_list)

            available_models = []
            for m in models_raw:
                name = None
                if hasattr(m, 'model'):
                    name = m.model
                elif hasattr(m, 'name'):
                    name = m.name
                elif isinstance(m, dict):
                    name = m.get('model') or m.get('name')
                elif hasattr(m, 'get'):
                    name = m.get('model') or m.get('name')
                if name:
                    available_models.append(name)
            
            model_exists = False
            for m in available_models:
                if not m:
                    continue
                if m == self.model or m.split(':')[0] == self.model.split(':')[0]:
                    model_exists = True
                    break
            
            if not model_exists:
                try:
                    self.client.show(self.model)
                    model_exists = True
                except Exception as e:
                    self.logger.warning(f"Model warmup failed ({ollama_client.py}:119): {traceback.format_exc()}")

            if not model_exists:
                self.logger.warning(f"Model {self.model} not installed.")
                return False
            
            self.logger.info(f"Model found: {self.model}")
        except Exception as e:
            self.logger.error(f"Failed to verify model existence: {e}")
            return False

        # 3. Send lightweight warmup request
        warmup_start = time.time()
        try:
            self.client.generate(
                model=self.model,
                prompt="warmup",
                options={"num_predict": 1}
            )
            warmup_duration = time.time() - warmup_start
            self.logger.info("Model warmed")
            self.logger.info(f"Warmup duration: {warmup_duration:.2f} seconds")
            
            OllamaClient._model_verified = True
            OllamaClient._model_warmed = True
            return True
        except Exception as e:
            self.logger.error(f"Warmup request failed: {e}")
            return False

    def health_check(self) -> bool:
        """
        Calls self.client.list() to check if Ollama is running.
        Returns True if successful, logs warning and returns False otherwise.
        """
        try:
            self.client.list()
            return True
        except Exception as e:
            self.logger.warning(f"Ollama health check failed: {e}")
            return False

    def pre_call_health_check(self, model: str = None) -> bool:
        """
        Verifies Ollama server reachable and model exists.
        """
        target_model = model or self.model
        if not self.health_check():
            self.logger.warning("Ollama server unreachable during pre-call health check")
            return False
        try:
            models_list = self.client.list()
            models_raw = []
            if hasattr(models_list, 'models'):
                models_raw = models_list.models
            elif isinstance(models_list, dict):
                models_raw = models_list.get('models', [])
            elif hasattr(models_list, 'get'):
                models_raw = models_list.get('models', [])
            else:
                models_raw = list(models_list)

            available_models = []
            for m in models_raw:
                name = None
                if hasattr(m, 'model'):
                    name = m.model
                elif hasattr(m, 'name'):
                    name = m.name
                elif isinstance(m, dict):
                    name = m.get('model') or m.get('name')
                elif hasattr(m, 'get'):
                    name = m.get('model') or m.get('name')
                if name:
                    available_models.append(name)

            for m in available_models:
                if not m:
                    continue
                if m == target_model or m.split(':')[0] == target_model.split(':')[0]:
                    return True

            try:
                self.client.show(target_model)
                return True
            except Exception as e:
                self.logger.warning(f"Health check failed ({ollama_client.py}:206): {traceback.format_exc()}")

            self.logger.warning(f"Model {target_model} not found during pre-call health check")
            return False
        except Exception as e:
            self.logger.warning(f"Pre-call health check failed: {e}")
            return False

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
        """
        Calls Ollama generate endpoint and returns response text.
        Automatically prepends /no_think to the system prompt.
        Never allows exceptions to propagate.
        """
        # A. Disable Thinking Mode by prepending /no_think globally
        if system_prompt:
            system_prompt = "/no_think\n\n" + system_prompt
        else:
            system_prompt = "/no_think"

        target_model = model or self.model
        if not self.pre_call_health_check(target_model):
            self.logger.warning("Pre-call health check failed. Bypassing LLM generation and returning empty response.")
            self.failures_count += 1
            return ""

        generate_args = {
            "model": target_model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": num_ctx
            },
            "think": think,
            "keep_alive": "30m"
        }
        generate_args["system"] = system_prompt
        if format:
            generate_args["format"] = format
            generate_args["think"] = False

        self.logger.debug(f"Model: {generate_args['model']}")
        
        # Standardize timeout to at least 60 seconds
        active_timeout = timeout or getattr(self.config, 'REQUEST_TIMEOUT_SECONDS', 60.0)
        if isinstance(active_timeout, (int, float)) and active_timeout < 60.0:
            active_timeout = 60.0
            
        timeout_obj = httpx.Timeout(float(active_timeout), connect=10.0, read=float(active_timeout))
        
        # Retry with backoff: Attempt 1 -> immediate, Attempt 2 -> wait 5s, Attempt 3 -> wait 10s
        retry_delays = [0, 5, 10]
        last_exception = None
        
        for attempt_idx, delay in enumerate(retry_delays, 1):
            if delay > 0:
                self.logger.info(f"Retrying LLM request, attempt {attempt_idx}/3 after waiting {delay} seconds...")
                time.sleep(delay)
            else:
                self.logger.debug(f"LLM request started (Attempt 1/3)")
                
            start_time = time.time()
            try:
                temp_client = ollama.Client(host=self.config.OLLAMA_BASE_URL, timeout=timeout_obj)
                response = temp_client.generate(**generate_args)
                elapsed = time.time() - start_time
                
                self.logger.info("LLM request completed")
                self.logger.info(f"Duration: {int(elapsed)} seconds")

                # Extract metrics
                done_reason = getattr(response, "done_reason", "unknown")
                eval_count = getattr(response, "eval_count", 0)
                response_text = getattr(response, "response", "") or ""
                thinking_text = getattr(response, "thinking", "") or ""

                # Log raw response at debug level
                self.logger.debug(f"[OLLAMA RAW RESPONSE] repr(raw_response): {repr(response_text)}")
                self.logger.debug(f"Ollama raw response: {response_text}")
                self.logger.debug(f"Ollama raw thinking: {thinking_text}")

                actual_text = response_text
                if not actual_text.strip() and thinking_text.strip():
                    self.logger.warning("Main response content is empty but thinking trace exists. Using thinking trace.")
                    actual_text = thinking_text
                
                self.last_eval_count = eval_count
                return actual_text.strip()
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.warning(f"LLM request failed/timed out on attempt {attempt_idx}/3. Duration: {int(elapsed)} seconds. Error: {e}")
                last_exception = e
                self.failures_count += 1
                
        self.logger.error(f"All 3 LLM attempts failed. Returning fallback empty string. Last error: {last_exception}")
        return ""

    def repair_json_string(self, text: str) -> str:
        """
        Repairs truncated or malformed JSON by:
        - Closing unclosed quotes
        - Closing brackets and braces using a parser stack
        - Replacing dangling colons with null
        - Strips trailing commas
        - Repairing partially completed arrays/objects
        """
        if not text:
            return "{}"

        # Clean thinking blocks if not already done
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

        # Find starting brace or bracket to repair from
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

        # Close unclosed quote
        if in_string:
            if repaired.endswith('\\'):
                repaired = repaired[:-1]
            repaired += '"'

        repaired = repaired.strip()

        # Handle dangling key in dictionary contexts (e.g. {"a": 1, "b" -> {"a": 1, "b": null})
        if stack and stack[-1] == '}':
            last_comma_or_brace = max(repaired.rfind(','), repaired.rfind('{'))
            last_colon = repaired.rfind(':')
            if last_comma_or_brace > last_colon:
                repaired = repaired.rstrip()
                repaired += ': null'

        # Repeatedly fix trailing commas and colons
        while True:
            prev_len = len(repaired)
            repaired = repaired.strip()

            if repaired.endswith(':'):
                repaired += ' null'
            elif repaired.endswith(','):
                repaired = repaired[:-1]
            else:
                # Check for trailing colons with spaces
                colon_match = re.search(r':\s*$', repaired)
                if colon_match:
                    repaired = repaired[:colon_match.start()] + ': null'
                else:
                    break

            if len(repaired) == prev_len:
                break

        # Close remaining stack elements
        while stack:
            repaired += stack.pop()

        # Clean trailing commas inside list/dict structures
        repaired = re.sub(r',\s*([\]}])', r'\1', repaired)

        return repaired

    def extract_json(self, text: str) -> dict:
        """
        Extracts JSON from text using multi-stage fallback strategies:
        1. Direct json.loads(text) -> "direct_json"
        2. Markdown blocks regex matching -> "markdown_json"
        3. First-to-last brace matching -> "brace_extraction"
        4. Repair logic (quotes, spaces, control characters) -> "repaired_json"
        Raises ValueError if all attempts fail.
        """
        # Log raw text before extraction at debug level
        self.logger.debug(f"[JSON EXTRACTION INPUT] repr(text): {repr(text)}")
        self.logger.debug(f"Extracting JSON from text: {text}")

        # Clean thinking blocks from text first to avoid curly braces pollution
        if text:
            cleaned_text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        else:
            cleaned_text = ""

        # Log input length (limit log representation to 1000 characters)
        self.logger.debug(f"Raw model output length: {len(cleaned_text)}")
        self.logger.debug(f"Raw output snippet: {cleaned_text[:1000]}")

        # 1. Direct json
        try:
            parsed = json.loads(cleaned_text)
            self.logger.debug("JSON extraction strategy used: direct_json")
            return self._clean_and_repair_json(parsed)
        except json.JSONDecodeError:
            pass

        # 2. Try JSON recovery engine (repair_json_string)
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
                self.logger.debug(f"Markdown JSON repair failed ({ollama_client.py}:482): {traceback.format_exc()}")
            try:
                repaired = self.repair_json_string(markdown_match.group(1).strip())
                parsed = json.loads(repaired)
                self.logger.info("JSON_RECOVERY_SUCCESS")
                self.logger.debug("JSON extraction strategy used: markdown_repaired_json")
                return self._clean_and_repair_json(parsed)
            except Exception as e:
                self.logger.debug(f"Markdown JSON recovery failed ({ollama_client.py}:489): {traceback.format_exc()}")

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
                self.logger.debug(f"First-to-last brace extraction failed ({ollama_client.py}:502): {traceback.format_exc()}")
            try:
                repaired = self.repair_json_string(candidate)
                parsed = json.loads(repaired)
                self.logger.info("JSON_RECOVERY_SUCCESS")
                self.logger.debug("JSON extraction strategy used: brace_repaired_json")
                return self._clean_and_repair_json(parsed)
            except Exception as e:
                self.logger.debug(f"Repaired JSON parse failed ({ollama_client.py}:509): {traceback.format_exc()}")

        # 5. Repaired JSON (smart quotes, control characters, backticks, stray text)
        try:
            repaired = cleaned_text
            # Replace smart/curly quotes
            repaired = repaired.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
            # Remove stray backticks
            repaired = repaired.replace('`', '')
            # Clean control characters
            repaired = "".join(char for char in repaired if char in string.printable)
            
            # Find first { and last } again on repaired string
            r_start = repaired.find('{')
            r_end = repaired.rfind('}')
            if r_start != -1 and r_end != -1 and r_start < r_end:
                repaired = repaired[r_start:r_end + 1]
                
            parsed = json.loads(repaired)
            self.logger.debug("JSON extraction strategy used: legacy_repaired_json")
            return self._clean_and_repair_json(parsed)
        except Exception as e:
            self.logger.debug(f"Legacy repaired JSON attempt failed: {e}")

        # 6. Fallback multi-candidate brace regex matching
        candidates = re.findall(r"\{.*?\}", cleaned_text, re.DOTALL)
        for idx, candidate in enumerate(candidates):
            try:
                parsed = json.loads(candidate)
                self.logger.debug(f"JSON extraction strategy used: brace_regex_candidate_{idx}")
                return self._clean_and_repair_json(parsed)
            except json.JSONDecodeError:
                pass

        # 7. Regex-based partial recovery of dict objects
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
                        self.logger.debug(f"Partial JSON regex failed ({ollama_client.py}:554): {traceback.format_exc()}")
                        try:
                            m_rep = m.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'").replace('`', '')
                            parsed_obj = json.loads(m_rep)
                            if isinstance(parsed_obj, dict):
                                recovered.append(self._clean_and_repair_json(parsed_obj))
                        except Exception as e:
                            self.logger.debug(f"Nested JSON extraction failed ({ollama_client.py}:560): {traceback.format_exc()}")
            if recovered:
                self.logger.debug(f"JSON extraction strategy used: regex_partial_recovery (recovered {len(recovered)} items)")
                return recovered
        except Exception as e:
            self.logger.debug(f"Regex partial recovery failed: {e}")

        self.logger.error("JSON parsing failure: All extraction strategies failed. Returning empty dict.")
        return {}

    def _clean_and_repair_json(self, data: dict) -> dict:
        """
        Cleans up dictionary keys, removes punctuation-only keys,
        and repairs Qwen framing_summary nesting/escaping issues.
        """
        if not isinstance(data, dict):
            return data

        # D. Recover framing_summary if missing or Unknown in original data
        recovered_text = None
        key_to_delete = None
        
        fs_val = data.get("framing_summary")
        if fs_val is None or not str(fs_val).strip() or str(fs_val).strip() == "Unknown":
            # Search all keys and values to locate framing_summary
            for k, v in data.items():
                k_str = str(k)
                v_str = str(v)
                if "framing_summary" in k_str or "framing_summary" in v_str:
                    combined = f"{k_str} : {v_str}"
                    # Extract framing summary text
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

        # C. Remove malformed/punctuation keys (like ": ")
        cleaned = {}
        for k, v in data.items():
            if k == key_to_delete:
                continue
            k_str = str(k).strip()
            # Check if key is empty, punctuation-only, quote-only, or colon-only
            if not k_str:
                continue
            # punctuation-only / quote-only / colon-only
            is_punct_or_quotes = all(char in string.punctuation or char in '"\' ' for char in k_str)
            if is_punct_or_quotes:
                continue
            cleaned[k] = v

        if recovered_text is not None:
            cleaned["framing_summary"] = recovered_text

        # Log final cleaned JSON representation
        self.logger.debug(f"Final cleaned JSON: {cleaned}")
        return cleaned


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
        """
        Generates content from the LLM and parses the result into a dict.
        Never allows exceptions to propagate.
        """
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
            self.logger.error(f"JSON generation failed: {e}")
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
        """
        Calls generate_json() up to max_retries times.
        On failure, appends instructions and retries.
        """
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
                        # Disable retry loops when response is exactly {} to avoid wasting time
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

        self.logger.error(f"LLM json generation completely failed: {last_error}. Returning fallback empty dictionary.")
        return {}

if __name__ == "__main__":
    config = Config.from_env()
    client = OllamaClient(config)

    print("=== Ollama Health Check ===")
    is_healthy = client.health_check()
    print(f"Ollama healthy: {is_healthy}\n")

    print("=== Diagnostic JSON Extraction Tests ===")
    
    # Test 1: Valid JSON
    test_1 = '{"main_topic": "AI", "key_claims": ["claim 1"], "author_stance": "Neutral", "tone": "Balanced", "framing_summary": "Summary"}'
    # Test 2: Markdown wrapped JSON
    test_2 = 'Some text ```json\n{"main_topic": "Markdown", "key_claims": ["claim 2"], "author_stance": "Neutral", "tone": "Balanced", "framing_summary": "Summary"}\n``` other text'
    # Test 3: Broken framing_summary JSON (nested artifact key)
    test_3 = '{\n  "main_topic": "Broken Framing",\n  "key_claims": ["claim 3"],\n  "author_stance": "Neutral",\n  "tone": "Balanced",\n  ": ": "framing_summary:\\"Nested summary\\""\n}'
    # Test 4: Stray explanatory text before JSON
    test_4 = 'Here is the requested output:\n{"main_topic": "Stray Text", "key_claims": ["claim 4"], "author_stance": "Neutral", "tone": "Balanced", "framing_summary": "Summary"}\nEnd of response.'

    test_cases = [test_1, test_2, test_3, test_4]
    
    all_passed = True
    for idx, test_text in enumerate(test_cases, 1):
        try:
            parsed = client.extract_json(test_text)
            assert isinstance(parsed, dict), "parsed result must be dict"
            assert parsed.get("main_topic"), "main_topic check failed"
            assert len(parsed.get("key_claims", [])) > 0, "key_claims check failed"
            assert parsed.get("framing_summary"), "framing_summary check failed"
            print(f"Test {idx}: PASS")
        except Exception as e:
            print(f"Test {idx}: FAIL ({e})")
            all_passed = False

    print(f"\nAll JSON extraction tests passed: {all_passed}\n")

    if is_healthy:
        print("=== Sample JSON Generation ===")
        json_prompt = (
            "Return a JSON object with keys: name (string), score (integer 1-10), "
            "tags (list of 2 strings). Use any values."
        )
        try:
            json_response = client.generate_json(
                prompt=json_prompt,
                temperature=0.1
            )
            print(f"Sample JSON Generation Output:\n{json_response}\n")
        except Exception as err:
            print(f"Sample JSON Generation Failed: {err}")
