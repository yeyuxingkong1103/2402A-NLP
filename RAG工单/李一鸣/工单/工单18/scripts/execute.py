from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Skill import SkillKitManager


if __name__ == "__main__":
    manager = SkillKitManager(str(ROOT))
    print(json.dumps({"skills": manager.discover()}, ensure_ascii=False, indent=2))
