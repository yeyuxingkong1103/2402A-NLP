# -*- coding: utf-8 -*-
"""Small configuration helper shared by API, scripts and data adapters."""

import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def load_env(path=None):
    """Load KEY=VALUE pairs without overriding values supplied by the shell."""
    selected = path or os.environ.get("RAG_ENV_FILE")
    env_file = Path(selected) if selected else BASE / (".env.local" if (BASE / ".env.local").exists() else ".env")
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def setting(name, default=""):
    load_env()
    return os.environ.get(name, default)


def enabled(name, default=False):
    fallback = "true" if default else "false"
    return setting(name, fallback).strip().lower() in {"1", "true", "yes", "on"}
