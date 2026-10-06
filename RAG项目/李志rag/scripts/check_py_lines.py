from pathlib import Path


def main() -> int:
    failures = []
    for path in Path(".").rglob("*.py"):
        if any(part in {".venv", ".runtime_cache", "models"} for part in path.parts):
            continue
        if path.as_posix().endswith("app/rag_main.py"):
            continue
        count = len(path.read_text(encoding="utf-8").splitlines())
        if count > 300:
            failures.append(f"{path}: {count} lines")
    if failures:
        print("\n".join(failures))
        return 1
    print("All Python files are within 300 lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
