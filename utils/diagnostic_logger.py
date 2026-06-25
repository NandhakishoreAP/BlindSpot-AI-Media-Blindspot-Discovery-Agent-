import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from .logger import get_logger


class DiagnosticLogger:
    """
    Writes structured JSON snapshots at each pipeline stage for debugging and analysis.
    Each snapshot captures the stage name, timestamp, duration, key inputs, and outputs.
    """

    def __init__(self, reports_dir: str = "reports") -> None:
        self.logger = get_logger("diagnostic_logger")
        self.entries: list[dict] = []
        self._reports_dir = reports_dir

    def _safe(self, val: Any, max_len: int = 200) -> Any:
        if isinstance(val, str):
            return val[:max_len]
        if isinstance(val, dict):
            return {k: self._safe(v, max_len) for k, v in val.items()}
        if isinstance(val, list):
            return [self._safe(v, max_len) for v in val[:10]]
        return val

    def log(
        self,
        stage: str,
        inputs: Optional[Dict] = None,
        outputs: Optional[Dict] = None,
        duration: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "duration_seconds": duration,
            "inputs": self._safe(inputs or {}),
            "outputs": self._safe(outputs or {}),
            "error": error,
            "metadata": self._safe(metadata or {}),
        }
        self.entries.append(entry)
        self.logger.info(f"DIAG [{stage}] duration={duration}s error={error}")

    def flush(self, article_url: str = "") -> None:
        if not self.entries:
            return
        safe_url = "".join(c for c in article_url if c.isalnum() or c in ("/", ":", ".", "_")).strip().replace(
            "/", "_"
        )[:60]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostics_{safe_url}_{timestamp}.json"
        try:
            diag_dir = Path(self._reports_dir)
            diag_dir.mkdir(parents=True, exist_ok=True)
            filepath = diag_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({"entries": self.entries}, f, indent=2, default=str)
            self.logger.info(f"Diagnostics saved to {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to save diagnostics: {e}")
