"""Skill registry and invocation contract for document quality assessment."""

from __future__ import annotations

import importlib.util
from pathlib import Path


class SkillKitManager:
    def __init__(self, root: str):
        self.root = Path(root)
        self.skills: dict[str, dict] = {}

    def discover(self) -> list[str]:
        for path in self.root.glob("*/skill.md"):
            self.skills[path.parent.name] = {"path": str(path), "description": path.read_text(encoding="utf-8")[:500]}
        return sorted(self.skills)

    def list_skills(self) -> list[str]:
        return sorted(self.skills)

    def invoke_skill(self, name: str, **kwargs):
        if name not in self.skills:
            raise KeyError(name)
        module_path = self.root / name / "skill_impl.py"
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run(**kwargs)
