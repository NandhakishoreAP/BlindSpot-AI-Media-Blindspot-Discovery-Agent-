from config import Config
from llm.ollama_client import OllamaClient
from llm.gemini_client import GeminiClient


def create_llm_client(config: Config):
    backend = config.LLM_BACKEND or "ollama"
    if backend == "gemini":
        return GeminiClient(config)
    return OllamaClient(config)
