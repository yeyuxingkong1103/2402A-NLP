"""Fast deployment checks using only Python's standard library; never install/download."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")


def safetensors_size(path):
    """Check the header and expected file length, without reading model tensors."""
    require_file(path)
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors: {path}")
        header_size = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= header_size <= 100_000_000:
            raise ValueError(f"Invalid safetensors header: {path}")
        header = json.loads(stream.read(header_size))
    offsets = [v["data_offsets"] for k, v in header.items() if k != "__metadata__"]
    if not offsets or any(a < 0 or b < a for a, b in offsets):
        raise ValueError(f"Invalid tensor offsets: {path}")
    data_size = max(b for _, b in offsets)
    expected = 8 + header_size + data_size
    if path.stat().st_size != expected:
        raise ValueError(f"Incomplete model file: {path} (expected {expected}, got {path.stat().st_size} bytes)")
    return data_size


def check_model(model):
    require_file(model / "config.json")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    require_file(model / "tokenizer_config.json")
    require_file(model / "tokenizer.json")
    index_path = model / "model.safetensors.index.json"
    require_file(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = set(index.get("weight_map", {}).values())
    if not shards:
        raise ValueError(f"No model shards listed in {index_path}")
    total = 0
    for shard in sorted(shards):
        path = (model / shard).resolve()
        if not path.is_relative_to(model.resolve()):
            raise ValueError("Model index contains a path outside its directory")
        total += safetensors_size(path)
    expected = index.get("metadata", {}).get("total_size")
    if expected and abs(total - expected) > max(1_048_576, expected * 0.01):
        raise ValueError(f"Model tensor size mismatch: {total} vs index {expected}")
    print(f"Qwen files OK: {len(shards)} shards, {total / 1024**3:.1f} GiB, architecture={config.get('architectures')}")


def check_project(project):
    for name in ("app/main.py", "app/rag.py", "app/db.py", "app/static/index.html",
                 "app/static/app.js", "app/static/app.css", "scripts/check_demo.py"):
        require_file(project / name)
    db = Path(os.environ.get("RAG_MILVUS_URI", str(project / "db/milvus.db")))
    if not db.is_absolute():
        db = project / db
    if not db.exists() or (db.is_dir() and not any(p.is_file() and p.stat().st_size for p in db.rglob("*") if p.name != "LOCK")):
        raise ValueError(f"Knowledge database is missing/empty: {db}")
    if db.is_file():
        require_file(db)
    embed = project / "models/bge-small-zh-v1.5"
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "modules.json", "1_Pooling/config.json"):
        require_file(embed / name)
    safetensors_size(embed / "model.safetensors")
    print("Web assets, knowledge database and local BGE model OK")


def check_environment(python, packages):
    if not Path(python).is_file():
        raise ValueError(f"Python environment not found: {python}")
    code = "from importlib.metadata import version; import json,sys; print(json.dumps({p:version(p) for p in sys.argv[1:]}))"
    result = subprocess.run([python, "-c", code, *packages], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError(f"Dependency check failed in {python}:\n{result.stderr.strip()}")
    print(f"Environment OK ({python}): {result.stdout.strip()}")


def check_gpu(python):
    code = """
import torch
if not torch.cuda.is_available():
    raise RuntimeError('No accessible GPU. Start this instance in GPU mode.')
p = torch.cuda.get_device_properties(0)
if p.total_memory < 70 * 1024**3:
    raise RuntimeError('This full-precision demo requires at least 70 GiB GPU VRAM.')
cuda = tuple(int(x) for x in torch.version.cuda.split('.')[:2])
if p.major >= 10 and cuda < (12, 8):
    raise RuntimeError('This GPU requires newer CUDA support in the vllm environment (PyTorch CUDA >=12.8). Do not reinstall the base RAG environment.')
print(f'GPU OK: {p.name}, {p.total_memory / 1024**3:.1f} GiB; vllm torch={torch.__version__}, CUDA={torch.version.cuda}')
"""
    result = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "GPU check failed")
    print(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--api-python", required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    try:
        check_project(args.project)
        check_model(args.model_path)
        check_environment(args.api_python, ["torch", "fastapi", "uvicorn", "openai", "pymilvus", "milvus-lite", "sentence-transformers", "fakeredis", "rank-bm25", "jieba"])
        check_environment(args.vllm_python, ["torch", "vllm", "transformers"])
        if args.gpu:
            check_gpu(args.vllm_python)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1
    print("Preflight passed (no downloads)." if args.gpu else "CPU preflight passed. GPU execution and actual chat still need the GPU launch check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
