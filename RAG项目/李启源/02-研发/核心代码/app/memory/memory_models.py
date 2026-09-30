"""Shared data contract for the memory orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryContext:
    """Combined short-term and long-term context for prompt construction."""

    short_term_messages: list[dict[str, Any]] = field(default_factory=list)
    long_term_summaries: list[dict[str, Any]] = field(default_factory=list)
    user_preferences: list[dict[str, Any]] = field(default_factory=list)
    important_facts: list[dict[str, Any]] = field(default_factory=list)
    role_config: dict[str, Any] | None = None
