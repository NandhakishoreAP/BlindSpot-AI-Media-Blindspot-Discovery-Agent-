import json
import re
# pyrefly: ignore [missing-import]
import ollama

from config import Config
from utils.logger import get_logger

class OllamaClient:
    def __init__(self, config: Config):
        """
        Stores config, initializes logger, creates Ollama Client, and stores the target model name.
        """
        self.config = config
        self.logger = get_logger("ollama_client")
        self.client = ollama.Client(host=config.OLLAMA_BASE_URL)
        self.model = config.OLLAMA_MODEL

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

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.7, format: str = None) -> str:
        """
        Builds messages list, calls Ollama chat endpoint, and returns content of the response.
        Raises RuntimeError on failure.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            chat_args = {
                "model": self.model,
                "messages": messages,
                "options": {"temperature": temperature}
            }
            if format:
                chat_args["format"] = format

            response = self.client.chat(**chat_args)

            if hasattr(response, "message") and hasattr(response.message, "content"):
                return response.message.content.strip()
            elif isinstance(response, dict) and "message" in response:
                msg = response["message"]
                if isinstance(msg, dict):
                    return msg.get("content", "").strip()
                elif hasattr(msg, "content"):
                    return msg.content.strip()
            
            return str(response).strip()
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def extract_json(self, text: str) -> dict:
        """
        Centralizes JSON extraction logic:
        1. Tries direct json.loads()
        2. Extracts JSON from markdown blocks: ```json ... ``` or ``` ... ```
        3. Extracts JSON using curly brace regex matching
        Raises ValueError if all attempts fail.
        """
        # 1. Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Extract from markdown code blocks
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if markdown_match:
            try:
                return json.loads(markdown_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3. Extract using brace matching
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0).strip())
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract JSON from LLM response")

    def generate_json(self, prompt: str, system_prompt: str = "", temperature: float = 0.3) -> dict:
        """
        Generates content from the LLM in JSON mode and parses the result into a dict.
        """
        response = self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            format="json"
        )
        return self.extract_json(response)

    def generate_json_with_retry(
        self,
        prompt: str,
        system_prompt: str = "",
        max_retries: int = 3,
        temperature: float = 0.3
    ) -> dict:
        """
        Calls generate_json() up to max_retries times.
        On failure, appends strict instructions to prompt and retries.
        """
        current_prompt = prompt
        last_error = None

        for i in range(max_retries):
            try:
                return self.generate_json(
                    current_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature
                )
            except (ValueError, RuntimeError) as e:
                last_error = e
                self.logger.warning(
                    f"JSON parsing failed attempt {i + 1}/{max_retries}, retrying... Error: {e}"
                )
                if i < max_retries - 1:
                    current_prompt = (
                        prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No explanations. No markdown."
                    )

        raise last_error

if __name__ == "__main__":
    try:
        config = Config.from_env()
        client = OllamaClient(config)

        print("=== Test 1: Health Check ===")
        is_healthy = client.health_check()
        print(f"Ollama healthy: {is_healthy}\n")

        print("=== Test 2: JSON Extraction/Parsing Mock Test ===")
        mocked_responses = [
            # Direct json
            '{"name": "direct", "score": 9, "tags": ["a", "b"]}',
            # Markdown block
            'Here is the JSON:\n```json\n{"name": "markdown", "score": 8, "tags": ["c", "d"]}\n```\nHope you like it!',
            # Brace matched
            'Sure, the data is: {"name": "brace", "score": 7, "tags": ["e", "f"]} and that is all.'
        ]
        for idx, mock_text in enumerate(mocked_responses, 1):
            try:
                parsed = client.extract_json(mock_text)
                print(f"Mock Response {idx} parsed successfully: {parsed}")
            except Exception as e:
                print(f"Mock Response {idx} failed to parse: {e}")
        print()

        # If Ollama is running and healthy, run integration tests
        if is_healthy:
            print("=== Test 3: Simple Text Generation ===")
            text_response = client.generate(
                prompt="In one sentence, what is media bias?",
                temperature=0.7
            )
            print(f"Response:\n{text_response}\n")

            print("=== Test 4: JSON Generation ===")
            json_prompt = (
                "Return a JSON object with keys: name (string), score (integer 1-10), "
                "tags (list of 2 strings). Use any values."
            )
            json_response = client.generate_json(
                prompt=json_prompt,
                temperature=0.3
            )
            print(f"Response:\n{json_response}\n")
        else:
            print("Skipping Live Ollama generation tests because health check failed (Ollama not running).")

    except Exception as err:
        print(f"\nError occurred during testing: {err}")
