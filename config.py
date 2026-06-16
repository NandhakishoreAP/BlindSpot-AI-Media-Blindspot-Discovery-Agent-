import os
from pathlib import Path
from dotenv import load_dotenv

# Load all values from .env using python-dotenv
load_dotenv()

class Config:
    # Typed class attributes
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str
    MAX_SEARCH_ATTEMPTS: int = 5
    CONFIDENCE_THRESHOLD: int = 70
    MAX_SEARCH_RESULTS: int = 5
    REPORTS_DIR: str = "reports"

    @classmethod
    def from_env(cls):
        # Reads from environment variables using os.getenv() with fallback defaults
        cls.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        cls.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        
        try:
            cls.MAX_SEARCH_ATTEMPTS = int(os.getenv("MAX_SEARCH_ATTEMPTS", "5"))
        except ValueError:
            cls.MAX_SEARCH_ATTEMPTS = 5
            
        try:
            cls.CONFIDENCE_THRESHOLD = int(os.getenv("CONFIDENCE_THRESHOLD", "70"))
        except ValueError:
            cls.CONFIDENCE_THRESHOLD = 70
            
        try:
            cls.MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
        except ValueError:
            cls.MAX_SEARCH_RESULTS = 5
            
        cls.REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
        
        # Creates the reports/ directory automatically if it doesn't exist
        Path(cls.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
        
        return cls()

    def __repr__(self) -> str:
        # Prints all config values cleanly
        return (
            f"Config(\n"
            f"  OLLAMA_BASE_URL={getattr(self, 'OLLAMA_BASE_URL', None)!r},\n"
            f"  OLLAMA_MODEL={getattr(self, 'OLLAMA_MODEL', None)!r},\n"
            f"  MAX_SEARCH_ATTEMPTS={getattr(self, 'MAX_SEARCH_ATTEMPTS', 5)},\n"
            f"  CONFIDENCE_THRESHOLD={getattr(self, 'CONFIDENCE_THRESHOLD', 70)},\n"
            f"  MAX_SEARCH_RESULTS={getattr(self, 'MAX_SEARCH_RESULTS', 5)},\n"
            f"  REPORTS_DIR={getattr(self, 'REPORTS_DIR', 'reports')!r}\n"
            f")"
        )
