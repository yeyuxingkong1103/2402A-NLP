"""环境自检：Python / 依赖 / GPU / 模型 / Milvus / Redis / 配置 / 知识库。

用法：
    D:\\an\\envs\\rags_\\python.exe tools/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def line(level: str, name: str, detail: str = "") -> None:
    print(f"{level} {name:<34} {detail}")


def check_python() -> bool:
    version = platform.python_version()
    good = sys.version_info[:2] >= (3, 10)
    line(OK if good else FAIL, "Python 版本", f"{version}（{sys.executable}）")
    return good


def check_packages() -> bool:
    required = {
        "torch": "模型推理",
        "transformers": "模型加载",
        "numpy": "数值计算",
        "yaml": "配置解析",
        "fastapi": "Web 服务",
        "uvicorn": "ASGI 服务器",
        "pydantic": "模型校验",
        "pymilvus": "向量库客户端",
        "redis": "缓存客户端",
        "jieba": "中文分词",
        "rank_bm25": "BM25 检索",
        "httpx": "接口冒烟测试",
    }
    optional = {"sentence_transformers": "稠密向量一致性比对", "pdfplumber": "PDF 知识文档解析"}
    all_ok = True
    for name, purpose in required.items():
        try:
            module = importlib.import_module(name)
            line(OK, name, f"{getattr(module, '__version__', '')} · {purpose}")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            line(FAIL, name, f"{type(exc).__name__} · {purpose}")
    for name, purpose in optional.items():
        # 可选依赖只用 find_spec 探测，避免导入副作用的无关噪声
        if importlib.util.find_spec(name) is not None:
            line(OK, name, f"已安装 · 可选 · {purpose}")
        else:
            line(WARN, name, f"未安装 · 可选 · {purpose}")
    return all_ok


def check_gpu() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            line(OK, "GPU", f"{name} · {total:.1f} GiB · CUDA {torch.version.cuda}")
            return True
        line(WARN, "GPU", "CUDA 不可用，将回退 CPU（速度显著下降）")
        return False
    except Exception as exc:  # noqa: BLE001
        line(FAIL, "GPU", str(exc))
        return False


def check_models() -> bool:
    from role_rag.config import get_config

    config = get_config()
    embedder = config.embedder_path()
    llm = config.llm_path()
    good = True
    for label, path in (("嵌入模型 BGE-M3", embedder), ("生成模型 Qwen3", llm)):
        if path.is_dir():
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1024**3
            line(OK, label, f"{path} · {size:.2f} GiB")
        else:
            good = False
            line(FAIL, label, f"目录不存在：{path}")
    sparse_head = embedder / "sparse_linear.pt"
    if sparse_head.is_file():
        line(OK, "BGE-M3 稀疏头", str(sparse_head))
    else:
        good = False
        line(FAIL, "BGE-M3 稀疏头", f"缺少 {sparse_head}")
    return good


def _tcp(host: str, port: int, timeout: float = 1.5) -> bool:
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def check_services() -> bool:
    from role_rag.config import get_config

    config = get_config()
    ok = True
    milvus_uri = str(config.get("milvus.uri"))
    host = milvus_uri.split("//")[-1].split(":")[0]
    port = int(milvus_uri.rsplit(":", 1)[-1].strip("/"))
    if _tcp(host, port):
        line(OK, "Milvus 端口", f"{host}:{port}")
        try:
            from role_rag.store.milvus_store import get_milvus

            info = get_milvus(config).ping()
            line(OK, "Milvus 连接", f"v{info['version']} · collections={info['collections']}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            line(FAIL, "Milvus 连接", str(exc))
    else:
        ok = False
        line(FAIL, "Milvus 端口", f"{host}:{port} 未监听（WSL 中执行：cd ~ && bash standalone_embed.sh start）")

    redis_host = str(config.get("redis.host"))
    redis_port = int(config.get("redis.port"))
    if _tcp(redis_host, redis_port):
        line(OK, "Redis 端口", f"{redis_host}:{redis_port}")
        try:
            from role_rag.store.redis_store import get_redis

            info = get_redis(config).ping()
            line(OK, "Redis 连接", f"v{info['version']} · keys={info['keys']}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            line(FAIL, "Redis 连接", str(exc))
    else:
        ok = False
        line(FAIL, "Redis 端口", f"{redis_host}:{redis_port} 未监听（WSL 中执行：sudo service redis-server start）")
    return ok


def check_project() -> bool:
    from role_rag.config import get_config
    from role_rag.roles import RoleRegistry

    config = get_config()
    ok = True
    line(OK, "项目根目录", str(config.root))
    registry = RoleRegistry.from_config(config)
    line(OK, "角色注册表", f"共 {len(registry)} 个，启用 {registry.enabled_ids()}")
    kb_root = config.kb_dir
    for scope in registry.enabled_ids():
        directory = kb_root / scope
        files = sorted(directory.glob("*.md")) if directory.is_dir() else []
        if files:
            line(OK, f"知识库 {scope}", f"{len(files)} 个文件")
        else:
            ok = False
            line(FAIL, f"知识库 {scope}", f"目录为空或不存在：{directory}")
    shared = kb_root / "shared"
    shared_files = sorted(shared.glob("*.md")) if shared.is_dir() else []
    line(OK if shared_files else WARN, "公共知识库 shared", f"{len(shared_files)} 个文件")
    return ok


def main() -> int:
    print("=" * 92)
    print("Role RAG_try 环境自检")
    print("=" * 92)
    results = [
        ("Python", check_python()),
        ("依赖包", check_packages()),
        ("GPU", check_gpu()),
    ]
    print("-" * 92)
    results.append(("模型文件", check_models()))
    results.append(("外部服务", check_services()))
    results.append(("项目配置", check_project()))
    print("-" * 92)
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"{FAIL} 未通过：{', '.join(failed)}")
        return 1
    print(f"{OK} 全部检查通过，可以执行：python tools/ingest_cli.py --all --recreate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
