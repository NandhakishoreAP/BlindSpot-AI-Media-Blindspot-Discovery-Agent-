# quick_test.py
from llm.ollama_client import OllamaClient
from config import Config

client = OllamaClient(Config.from_env())

response = client.generate(
    prompt="What is climate change?",
    temperature=0.1
)

print(response)