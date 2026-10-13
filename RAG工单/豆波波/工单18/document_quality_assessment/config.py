"""Configuration loading for the document quality skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AssessmentConfig:
    """All classification thresholds are configurable and serializable."""

    scan_text_threshold: int = 32
    scan_ratio_threshold: float = 0.70
    simhash_max_distance: int = 8
    simhash_max_pairs: int = 500
    context_radius: int = 48
    sensitive_patterns: dict[str, str] = field(
        default_factory=lambda: {
            "phone": r"(?<!\d)(?:1[3-9]\d{9})(?!\d)",
            "email": r"(?i)(?<![\w.+-])([\w.+-]+@[\w-]+(?:\.[\w-]+)+)(?![\w.-])",
            "id_card": r"(?<!\d)(?:\d{17}[\dXx])(?!\d)",
        }
    )
    enable_bank_card: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "AssessmentConfig":
        if not value:
            return cls()
        known = {k: v for k, v in value.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_file(cls, path: str | Path) -> "AssessmentConfig":
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        try:
            import yaml

            loaded = yaml.safe_load(raw) or {}
        except ImportError:
            try:
                from ruamel.yaml import YAML

                loaded = YAML(typ="safe").load(raw) or {}
            except ImportError:
                import json

                loaded = json.loads(raw)
        return cls.from_mapping(loaded.get("assessment", loaded))
