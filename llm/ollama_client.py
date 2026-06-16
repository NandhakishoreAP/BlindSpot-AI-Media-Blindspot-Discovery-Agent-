import json
import re
import time
import string
# pyrefly: ignore [missing-import]
import ollama

from config import Config
from utils.logger import get_logger

class ExtractionPreset:
    """
    Predefined configuration presets for extraction components.
    """
    CLAIM_ANALYZER_NAME = "CLAIM_ANALYZER"
    BLINDSPOT_DETECTOR_NAME = "BLINDSPOT_DETECTOR"
    EVIDENCE_EVALUATOR_NAME = "EVIDENCE_EVALUATOR"

    CLAIM_ANALYZER = {
        "temperature": 0.1,
        "num_predict": 256,
        "num_ctx": 4096
    }
    BLINDSPOT_DETECTOR = {
        "temperature": 0.2,
        "num_predict": 512,
        "num_ctx": 4096
    }
    EVIDENCE_EVALUATOR = {
        "temperature": 0.1,
        "num_predict": 256,
        "num_ctx": 4096
    }

class OllamaClient:
    def __init__(self, config: Config):
        """
        Stores config, initializes logger, creates Ollama Client, and stores the target model name.
        """
        self.config = config
        self.logger = get_logger("ollama_client")
        self.client = ollama.Client(host=config.OLLAMA_BASE_URL)
        self.model = config.OLLAMA_MODEL
        self.last_eval_count = 0

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

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        num_predict: int = 1024,
        num_ctx: int = 4096,
        format: any = None,
        preset_name: str = None
    ) -> str:
        """
        Calls Ollama generate endpoint and returns response text.
        Automatically prepends /no_think to the system prompt.
        Raises RuntimeError on failure.
        """
        try:
            # A. Disable Thinking Mode by prepending /no_think globally
            if system_prompt:
                system_prompt = "/no_think\n\n" + system_prompt
            else:
                system_prompt = "/no_think"

            generate_args = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "num_ctx": num_ctx
                }
            }
            generate_args["system"] = system_prompt
            if format:
                generate_args["format"] = format

            self.logger.debug(f"Model: {self.model}")
            
            start_time = time.time()
            response = self.client.generate(**generate_args)
            elapsed = time.time() - start_time
            
            # Extract metrics
            done_reason = getattr(response, "done_reason", "unknown")
            eval_count = getattr(response, "eval_count", 0)
            prompt_eval_count = getattr(response, "prompt_eval_count", 0)
            response_text = getattr(response, "response", "") or ""
            thinking_text = getattr(response, "thinking", "") or ""

            # Fallback to thinking trace if main response is empty
            actual_text = response_text
            if not actual_text.strip() and thinking_text.strip():
                self.logger.warning("Main response content is empty but thinking trace exists. Using thinking trace for extraction.")
                actual_text = thinking_text
            
            # Store metrics
            self.last_eval_count = eval_count
            
            # Diagnostics logging
            self.logger.info(f"Ollama generation completed in {elapsed:.2f} seconds. Reason: {done_reason}")
            self.logger.debug(
                f"Metrics - Duration: {elapsed:.2f}s | "
                f"Prompt Tokens: {prompt_eval_count} | "
                f"Generated Tokens: {eval_count} | "
                f"Response Length: {len(actual_text)} chars"
            )
            if preset_name:
                self.logger.debug(f"Preset used: {preset_name}")
            self.logger.debug(f"num_predict: {num_predict}")
            self.logger.debug(f"temperature: {temperature}")
            self.logger.debug(f"total execution time: {elapsed:.2f} seconds")
            
            return actual_text.strip()
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def extract_json(self, text: str) -> dict:
        """
        Extracts JSON from text using multi-stage fallback strategies:
        1. Direct json.loads(text) -> "direct_json"
        2. Markdown blocks regex matching -> "markdown_json"
        3. First-to-last brace matching -> "brace_extraction"
        4. Repair logic (quotes, spaces, control characters) -> "repaired_json"
        Raises ValueError if all attempts fail.
        """
        # Log input length (limit log representation to 1000 characters)
        self.logger.debug(f"Raw model output length: {len(text)}")
        self.logger.debug(f"Raw output snippet: {text[:1000]}")

        # 1. Direct json
        try:
            parsed = json.loads(text)
            self.logger.debug("JSON extraction strategy used: direct_json")
            return self._clean_and_repair_json(parsed)
        except json.JSONDecodeError:
            pass

        # 2. Markdown blocks
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if markdown_match:
            try:
                parsed = json.loads(markdown_match.group(1).strip())
                self.logger.debug("JSON extraction strategy used: markdown_json")
                return self._clean_and_repair_json(parsed)
            except json.JSONDecodeError:
                pass

        # 3. First-to-last brace extraction
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            candidate = text[start_idx:end_idx + 1]
            try:
                parsed = json.loads(candidate)
                self.logger.debug("JSON extraction strategy used: brace_extraction")
                return self._clean_and_repair_json(parsed)
            except json.JSONDecodeError:
                pass

        # 4. Repaired JSON (smart quotes, control characters, backticks, stray text)
        try:
            repaired = text
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
            self.logger.debug("JSON extraction strategy used: repaired_json")
            return self._clean_and_repair_json(parsed)
        except Exception as e:
            self.logger.debug(f"JSON repair attempt failed: {e}")

        # 5. Fallback multi-candidate brace regex matching
        candidates = re.findall(r"\{.*?\}", text, re.DOTALL)
        for idx, candidate in enumerate(candidates):
            try:
                parsed = json.loads(candidate)
                self.logger.debug(f"JSON extraction strategy used: brace_regex_candidate_{idx}")
                return self._clean_and_repair_json(parsed)
            except json.JSONDecodeError:
                pass

        self.logger.error("JSON parsing failure: All extraction strategies failed.")
        raise ValueError("Could not extract JSON from LLM response")

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


    def validate_analysis_response(self, data: dict) -> dict:
        """
        Guarantees that analysis responses contain all required fields with fallback defaults.
        """
        if not isinstance(data, dict):
            data = {}

        required_fields = {
            "main_topic": "Unknown",
            "key_claims": [],
            "author_stance": "Unknown",
            "tone": "Unknown",
            "framing_summary": "Unknown"
        }

        cleaned = {}
        for field, default in required_fields.items():
            val = data.get(field)
            if val is None:
                cleaned[field] = default
            elif field == "key_claims":
                if not isinstance(val, list):
                    cleaned[field] = default
                else:
                    cleaned[field] = [str(c).strip() for c in val if c is not None and str(c).strip() != ""]
            else:
                val_str = str(val).strip()
                cleaned[field] = val_str if val_str else default

        return cleaned

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        num_predict: int = 1024,
        num_ctx: int = 4096,
        preset_name: str = None
    ) -> dict:
        """
        Generates content from the LLM and parses the result into a dict.
        """
        response_text = self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
            format="json",
            preset_name=preset_name
        )
        self.logger.debug(f"Raw model output: {response_text}")
        return self.extract_json(response_text)

    def generate_json_with_retry(
        self,
        prompt: str,
        system_prompt: str = "",
        max_retries: int = 2,
        temperature: float = 0.1,
        num_predict: int = 1024,
        num_ctx: int = 4096,
        preset_name: str = None
    ) -> dict:
        """
        Calls generate_json() up to max_retries times.
        On failure, appends instructions and retries.
        """
        current_prompt = prompt
        last_error = None

        for i in range(max_retries + 1):
            try:
                return self.generate_json(
                    current_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    num_predict=num_predict,
                    num_ctx=num_ctx,
                    preset_name=preset_name
                )
            except (ValueError, RuntimeError) as e:
                last_error = e
                self.logger.warning(
                    f"JSON parsing failed attempt {i + 1}/{max_retries + 1}, retrying... Error: {e}"
                )
                if i < max_retries:
                    current_prompt = (
                        prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No explanations. No markdown."
                    )

        raise last_error

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
            validated = client.validate_analysis_response(parsed)
            # Assert crucial properties exist and values are cleaned
            assert validated["main_topic"] != "Unknown", "main_topic check failed"
            assert len(validated["key_claims"]) > 0, "key_claims check failed"
            if idx == 3:
                assert validated["framing_summary"] == "Nested summary", "framing_summary repair check failed"
            else:
                assert validated["framing_summary"] == "Summary", "framing_summary check failed"
            
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
