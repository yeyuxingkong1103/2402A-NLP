"""本地反馈存储，为自适应重排提供受限分数修正。"""
import json
from pathlib import Path


class FeedbackStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> dict:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def record(self, chunk_ids: list[str | int], helpful: bool) -> None:
        data = self._load()
        for chunk_id in dict.fromkeys(str(item) for item in chunk_ids):
            votes = data.setdefault(chunk_id, {"helpful": 0, "not_helpful": 0})
            key = "helpful" if helpful else "not_helpful"
            votes[key] = int(votes.get(key, 0)) + 1
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.path)
        except OSError:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            return

    def adjustment(self, chunk_id: str | int) -> float:
        votes = self._load().get(str(chunk_id), {})
        try:
            score = 0.03 * (int(votes.get("helpful", 0)) - int(votes.get("not_helpful", 0)))
        except (AttributeError, TypeError, ValueError):
            return 0.0
        return round(max(-0.15, min(0.15, score)), 4)
