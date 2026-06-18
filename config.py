import os
from pathlib import Path
from dotenv import load_dotenv

# Load all values from .env using python-dotenv
load_dotenv()

class Config:
    # Typed class attributes
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str
    MODEL_CLAIM_ANALYZER: str = "qwen3:1.7b"
    MODEL_BLINDSPOT_DETECTOR: str = "qwen3:1.7b"
    MODEL_EVIDENCE_EVALUATOR: str = "qwen3:1.7b"
    MODEL_REPORT_GENERATOR: str = "qwen3:1.7b"
    MAX_SEARCH_ATTEMPTS: int = 1
    CONFIDENCE_THRESHOLD: int = 70
    MAX_SEARCH_RESULTS: int = 15
    MAX_RESEARCH_SECONDS: int = 90
    REQUEST_TIMEOUT_SECONDS: int = 60
    REPORTS_DIR: str = "reports"

    @classmethod
    def from_env(cls):
        # Reads from environment variables using os.getenv() with fallback defaults
        cls.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        cls.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
        cls.MODEL_CLAIM_ANALYZER = os.getenv("MODEL_CLAIM_ANALYZER", "qwen3:1.7b")
        cls.MODEL_BLINDSPOT_DETECTOR = os.getenv("MODEL_BLINDSPOT_DETECTOR", "qwen3:1.7b")
        cls.MODEL_EVIDENCE_EVALUATOR = os.getenv("MODEL_EVIDENCE_EVALUATOR", "qwen3:1.7b")
        cls.MODEL_REPORT_GENERATOR = os.getenv("MODEL_REPORT_GENERATOR", "qwen3:1.7b")
        
        try:
            cls.MAX_SEARCH_ATTEMPTS = int(os.getenv("MAX_SEARCH_ATTEMPTS", "1"))
        except ValueError:
            cls.MAX_SEARCH_ATTEMPTS = 1
            
        try:
            cls.CONFIDENCE_THRESHOLD = int(os.getenv("CONFIDENCE_THRESHOLD", "70"))
        except ValueError:
            cls.CONFIDENCE_THRESHOLD = 70
            
        try:
            cls.MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "15"))
        except ValueError:
            cls.MAX_SEARCH_RESULTS = 15

        try:
            cls.MAX_RESEARCH_SECONDS = int(os.getenv("MAX_RESEARCH_SECONDS", "90"))
        except ValueError:
            cls.MAX_RESEARCH_SECONDS = 90

        try:
            cls.REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
        except ValueError:
            cls.REQUEST_TIMEOUT_SECONDS = 60
            
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
            f"  MAX_SEARCH_ATTEMPTS={getattr(self, 'MAX_SEARCH_ATTEMPTS', 1)},\n"
            f"  CONFIDENCE_THRESHOLD={getattr(self, 'CONFIDENCE_THRESHOLD', 70)},\n"
            f"  MAX_SEARCH_RESULTS={getattr(self, 'MAX_SEARCH_RESULTS', 5)},\n"
            f"  MAX_RESEARCH_SECONDS={getattr(self, 'MAX_RESEARCH_SECONDS', 90)},\n"
            f"  REQUEST_TIMEOUT_SECONDS={getattr(self, 'REQUEST_TIMEOUT_SECONDS', 20)},\n"
            f"  REPORTS_DIR={getattr(self, 'REPORTS_DIR', 'reports')!r}\n"
            f")"
        )

