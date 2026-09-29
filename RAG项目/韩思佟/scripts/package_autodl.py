"""Create the small AutoDL model-launch update; excludes data, secrets and models."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from app.config import BASE

FILES = [
    "deploy/autodl/start_llm.sh",
    "deploy/autodl/env.example",
    "deploy/autodl/preflight.py",
    "deploy/autodl/stop.sh",
]


def package():
    output = BASE / "outputs" / "rag-autodl-model-only.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name in FILES:
            path = BASE / name
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, name)
    with ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP 损坏: {bad}")
        if set(archive.namelist()) != set(FILES):
            raise RuntimeError("ZIP 文件清单不正确")
    print(f"PACKAGE_OK: {output} ({output.stat().st_size} bytes)")
    return output


if __name__ == "__main__":
    package()
