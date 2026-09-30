# -*- coding: utf-8 -*-
"""服务健康自检：一条命令看清 Docker/Redis/MySQL/Milvus/外部 API 是否可用。

用法：
    python scripts/check_services.py               # 本地服务检查（不调外部 API）
    python scripts/check_services.py --with-api    # 额外对 Embedding/Reranker/LLM 各做一次最小调用

设计约定：
- 每项检查输出 ✅/❌ + 失败原因 + 可直接照抄的修复命令；任一失败退出码非 0。
- 不打印任何密钥、令牌、完整连接串，只显示 host:port 与库名。
- 检查顺序：Docker 容器 → Redis → MySQL → Milvus（含向量一致性）→ 外部 API（可选）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GREEN = "✅"
RED = "❌"

# 汇总失败项，决定退出码
failures: list[str] = []


def report(name: str, ok: bool, detail: str = "", fix: str = "", key: str = "") -> None:
    """打印一项检查结果；失败时把 key（默认用 name）登记进 failures 决定退出码。

    key 用于失败键与展示名不同的场合（如展示名"容器 my-redis"、失败键"my-redis"）。
    所有失败必须经本函数登记 failures——此前 docstring 承诺"失败时登记"但实现没有
    append，只有显式 append 的检查才影响退出码，出现"打印 ❌ 但退出码 0"的假绿灯
    （批次 32 修复）；直接 failures.append 的旧写法已全部收敛删除，避免重复计数。
    """
    mark = GREEN if ok else RED
    print(f"{mark} {name}")
    if detail:
        print(f"   {detail}")
    if not ok:
        failures.append(key or name)
        if fix:
            print(f"   修复：{fix}")


def load_env() -> None:
    """加载项目根 .env（不覆盖已有环境变量），并剔除代理变量防止本地请求被劫持。"""
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)  # 沙箱/系统代理会劫持 127.0.0.1 请求，先清掉
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def check_docker() -> bool:
    """检查 1：my-redis 与 milvus-standalone 容器是否 Up。返回两容器是否都健康。"""
    print("\n[1] Docker 容器")
    docker = shutil.which("docker")
    if docker is None:
        report(
            "docker 命令可用",
            False,
            "PATH 中找不到 docker",
            "先启动 Docker Desktop（本机在 D:\\Docker）",
            key="docker",
        )
        return False

    try:
        out = subprocess.run(
            [docker, "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report("docker daemon 可用", False, f"docker 命令执行失败：{exc}", "先启动 Docker Desktop（本机在 D:\\Docker）", key="docker")
        return False

    if out.returncode != 0:
        # daemon 没起来时 docker ps 会报错，视为 Docker Desktop 未启动
        report("docker daemon 可用", False, (out.stderr or "未知错误").strip()[:120], "先启动 Docker Desktop（本机在 D:\\Docker）", key="docker")
        return False

    statuses: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if "\t" in line:
            name, status = line.split("\t", 1)
            statuses[name.strip()] = status.strip()

    all_ok = True
    for container in ("my-redis", "milvus-standalone"):
        status = statuses.get(container)
        if status and status.startswith("Up"):
            report(f"容器 {container}", True, status)
        elif status and "Exited" in status:
            # Exited(134) 是本项目已知问题（见开发路线图已知限制第 5 条）
            report(
                f"容器 {container}",
                False,
                f"状态：{status}",
                f"docker start {container}；若反复 Exited(134)，见《开发路线图》已知限制第 5 条",
                key=container,
            )
            all_ok = False
        elif status:
            report(f"容器 {container}", False, f"状态：{status}", f"docker start {container}", key=container)
            all_ok = False
        else:
            report(
                f"容器 {container}",
                False,
                "容器不存在",
                f"确认 docker-compose 配置后重建该容器（docker ps -a 查看）",
                key=container,
            )
            all_ok = False
    return all_ok


def check_redis() -> bool:
    """检查 2：Redis ping 是否返回 PONG。"""
    print("\n[2] Redis")
    try:
        import redis  # noqa: PLC0415
    except ImportError:
        report("redis 库可用", False, "缺少 redis 包", "pip install redis（进 backend 依赖环境安装）", key="redis")
        return False

    url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    # 只展示 host:port/db，不展示可能带密码的完整 URL
    display = url.split("@")[-1] if "@" in url else url
    try:
        client = redis.Redis.from_url(url, socket_connect_timeout=3, socket_timeout=3)
        pong = client.ping()
    except Exception as exc:  # noqa: BLE001 —— 自检脚本要兜住一切连接错误
        report(f"Redis ping（{display}）", False, f"连接失败：{type(exc).__name__}: {exc}", "docker start my-redis", key="redis")
        return False
    report(
        f"Redis ping（{display}）",
        ok=pong is True,
        detail="PONG" if pong else "异常返回",
        key="redis",
    )
    return pong is True


def check_mysql() -> dict[str, int] | None:
    """检查 3：MySQL 连接 + 关键表行数。成功时返回表行数字典（供检查 4 复用）。"""
    print("\n[3] MySQL")
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = os.getenv("MYSQL_PORT", "3306")
    database = os.getenv("MYSQL_DATABASE", "legal_rag")
    try:
        import pymysql  # noqa: PLC0415
    except ImportError:
        report("pymysql 库可用", False, "缺少 pymysql 包", "pip install pymysql（进 backend 依赖环境安装）", key="mysql")
        return None

    try:
        conn = pymysql.connect(
            host=host,
            port=int(port),
            user=os.getenv("MYSQL_USER", "legal_rag"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=database,
            connect_timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        report(
            f"MySQL 连接（{host}:{port}/{database}）",
            False,
            f"{type(exc).__name__}: {exc}",
            "确认 MySQL 服务已启动（本机服务，非 Docker）：net start mysql 或启动对应服务",
            key="mysql",
        )
        return None

    counts: dict[str, int] = {}
    try:
        with conn.cursor() as cur:
            for table in ("documents", "document_chunks", "law_versions", "document_versions"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 —— 表名是写死白名单
                counts[table] = cur.fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        report(f"MySQL 关键表查询（{host}:{port}/{database}）", False, f"{type(exc).__name__}: {exc}", "确认库结构完整：SHOW TABLES", key="mysql")
        conn.close()
        return None
    conn.close()
    report(
        f"MySQL 连接（{host}:{port}/{database}）",
        True,
        f"documents={counts['documents']}  document_chunks={counts['document_chunks']}  "
        f"law_versions={counts['law_versions']}  document_versions={counts['document_versions']}",
    )
    return counts


def check_milvus(chunk_count: int | None) -> bool:
    """检查 4：Milvus 连接 + 集合存在 + 向量数与 document_chunks 一致 + 版本状态统计。"""
    print("\n[4] Milvus")
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")
    collection = os.getenv("MILVUS_COLLECTION_NAME", "legal_documents")
    ok = True
    try:
        from pymilvus import MilvusClient  # noqa: PLC0415
    except ImportError:
        report("pymilvus 库可用", False, "缺少 pymilvus 包", "pip install pymilvus（进 backend 依赖环境安装）", key="milvus")
        return False

    client = MilvusClient(uri=f"http://{host}:{port}")
    try:
        client.list_collections(timeout=5)
    except Exception as exc:  # noqa: BLE001
        report(
            f"Milvus 连接（{host}:{port}）",
            False,
            f"{type(exc).__name__}: {exc}",
            "docker start milvus-standalone",
            key="milvus",
        )
        return False
    report(f"Milvus 连接（{host}:{port}）", True)

    if not client.has_collection(collection):
        report(f"集合 {collection} 存在", False, "集合缺失", "重跑索引构建：python import_mysql.py 或对应索引脚本", key="milvus_collection")
        return False
    report(f"集合 {collection} 存在", True)

    # 向量数一致性：这是最关键的检查——Milvus 里的实体数必须等于 MySQL 的 document_chunks 数
    # 注意：get_collection_stats 的 row_count 在部分 Milvus 版本上返回 0（统计缓存不可靠），
    # 因此用 load + query count(*) 作为权威口径
    client.load_collection(collection)
    count_rows = client.query(collection, filter='chunk_key != ""', output_fields=["count(*)"])
    vector_count = int(str(count_rows).split("'count(*)': ")[1].split("}")[0])
    if chunk_count is None:
        report(
            "向量数一致性",
            False,
            f"Milvus 实体数={vector_count}，但 MySQL 不可用无法比对",
            "先修复 MySQL 再跑一次自检",
            key="milvus_consistency",
        )
        return False
    if vector_count == chunk_count:
        report("向量数一致性", True, f"Milvus={vector_count} == MySQL.document_chunks={chunk_count}")
    else:
        report(
            "向量数一致性",
            False,
            f"Milvus 实体数={vector_count} != MySQL.document_chunks={chunk_count}（差 {vector_count - chunk_count:+d}）",
            "差异常见于审核后向量清理/重建中断：重跑向量重建脚本，或对 approved 版本重建索引",
            key="milvus_consistency",
        )
        ok = False

    # 版本状态统计：从 MySQL 侧取（version_status 的真相源在 MySQL）
    try:
        import pymysql  # noqa: PLC0415

        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "legal_rag"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "legal_rag"),
            connect_timeout=5,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT version_status, COUNT(*) FROM document_versions GROUP BY version_status")
            rows = cur.fetchall()
        conn.close()
        stat_text = "  ".join(f"{status}={count}" for status, count in rows) or "（无数据）"
        approved = dict(rows).get("approved", 0)
        others = sum(count for status, count in rows if status != "approved")
        report(
            "版本状态统计",
            True,
            f"{stat_text}   （approved={approved}，其它={others}）"
            + ("  ⚠ 有非 approved 版本，发布前需处理" if others else ""),
        )
    except Exception as exc:  # noqa: BLE001
        report("版本状态统计", False, f"{type(exc).__name__}: {exc}", "确认 MySQL 可用后重跑", key="milvus_stats")
        ok = False
    return ok


def check_external_api() -> None:
    """检查 5（--with-api 才跑）：Embedding / Reranker / LLM 各一次最小调用。"""
    print("\n[5] 外部 API（最小调用）")

    def masked(url: str) -> str:
        """只显示 API 域名，不带路径参数与密钥。"""
        return url.split("//")[-1].split("/")[0] if "//" in url else url

    # —— Embedding：走项目自己的客户端（自带批次 15 的重试）——
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "backend"))
        from app.models.embedding import SiliconFlowEmbeddingClient  # noqa: PLC0415

        client = SiliconFlowEmbeddingClient(
            api_url=os.getenv("EMBEDDING_API_BASE_URL", ""),
            api_key=os.getenv("EMBEDDING_API_KEY", ""),
            model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
        )
        vectors = client.embed(["健康检查"])
        report(
            f"Embedding（{masked(os.getenv('EMBEDDING_API_BASE_URL', ''))}）",
            len(vectors) == 1 and len(vectors[0]) > 0,
            f"模型 {os.getenv('EMBEDDING_MODEL', '')} 返回 {len(vectors[0])} 维",
            key="embedding",
        )
    except Exception as exc:  # noqa: BLE001
        report(
            "Embedding",
            False,
            f"{type(exc).__name__}: {str(exc)[:160]}",
            "确认 EMBEDDING_API_KEY 有效、网络可达 api.siliconflow.cn",
            key="embedding",
        )

    # —— Reranker：一次最小的 rerank 调用 ——
    try:
        import json  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        url = os.getenv("RERANKER_API_BASE_URL", "")
        payload = json.dumps(
            {"model": os.getenv("RERANKER_MODEL", ""), "query": "经济补偿", "documents": ["劳动合同法第四十七条"], "top_n": 1}
        ).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('RERANKER_API_KEY', '')}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        results = body.get("results", [])
        report(
            f"Reranker（{masked(url)}）",
            bool(results),
            f"模型 {os.getenv('RERANKER_MODEL', '')} 返回 {len(results)} 条",
            key="reranker",
        )
    except Exception as exc:  # noqa: BLE001
        report(
            "Reranker",
            False,
            f"{type(exc).__name__}: {str(exc)[:160]}",
            "确认 RERANKER_API_KEY 有效、网络可达 api.siliconflow.cn",
            key="reranker",
        )

    # —— LLM：一次 chat.completions 最小调用 ——
    try:
        import json  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        base = os.getenv("LLM_API_BASE_URL", "").rstrip("/")
        payload = json.dumps(
            {
                "model": os.getenv("LLM_MODEL", ""),
                "messages": [{"role": "user", "content": "回复两个字：正常"}],
                "max_tokens": 64,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('LLM_API_KEY', '')}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        content = body["choices"][0]["message"]["content"]
        llm_ok = bool(content)
        # 空内容多半是 max_tokens 被截断或模型异常，写明原因便于排查（曾是假绿灯路径：
        # 打印 ❌ 但不登记 failures，退出码照样 0）
        llm_detail = (
            f"模型 {os.getenv('LLM_MODEL', '')} 返回 {len(content)} 字"
            if llm_ok
            else "返回空内容（可能 max_tokens 过小或模型异常）"
        )
        report(f"LLM（{masked(base)}）", llm_ok, llm_detail, key="llm")
    except Exception as exc:  # noqa: BLE001
        report(
            "LLM",
            False,
            f"{type(exc).__name__}: {str(exc)[:160]}",
            "确认 LLM_API_KEY 有效、网络可达 api.deepseek.com",
            key="llm",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="法律 RAG 项目服务健康自检")
    parser.add_argument("--with-api", action="store_true", help="额外对 Embedding/Reranker/LLM 各做一次最小调用（花几分钱）")
    args = parser.parse_args()

    load_env()
    print("=== 服务健康自检 ===")

    docker_ok = check_docker()
    check_redis()
    counts = check_mysql()
    check_milvus(counts["document_chunks"] if counts else None)
    if args.with_api:
        check_external_api()
    else:
        print("\n[5] 外部 API —— 跳过（加 --with-api 才会真实调用，避免浪费额度）")

    print("\n=== 结果 ===")
    if failures:
        print(f"{RED} {len(failures)} 项失败：{', '.join(sorted(set(failures)))}（退出码 1）")
        return 1
    print(f"{GREEN} 全部通过（退出码 0）" + ("" if docker_ok else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
