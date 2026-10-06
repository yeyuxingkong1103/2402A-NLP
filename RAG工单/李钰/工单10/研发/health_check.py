# -*- coding: utf-8 -*-
"""
健康检查 + 数据持久化管理
工单编号: 人工智能 NLP-RAG-金融问答系统部署

提供:
  1. /api/health       健康检查 (Docker HEALTHCHECK 使用)
  2. /api/data/export  数据导出
  3. /api/data/import  数据导入
  4. /api/status       系统状态
"""
import os, sys, json, shutil, logging, time
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v10

logger = logging.getLogger(__name__)


def check_system() -> Dict:
    """完整健康检查"""
    checks = {}

    # 1. Flask 进程
    checks["process"] = "running"

    # 2. 磁盘空间
    try:
        total, used, free = shutil.disk_usage("/")
        checks["disk"] = {
            "total_mb": round(total / 1024 / 1024, 1),
            "used_mb": round(used / 1024 / 1024, 1),
            "free_mb": round(free / 1024 / 1024, 1),
            "ok": free > 100 * 1024 * 1024,  # 至少 100MB
        }
    except Exception as e:
        checks["disk"] = {"error": str(e), "ok": True}

    # 3. 数据目录
    for d in [config_v10.DATA_DIR, config_v10.CACHE_DIR, config_v10.LOG_DIR]:
        checks[f"dir_{os.path.basename(d)}"] = {
            "exists": os.path.isdir(d),
            "ok": os.path.isdir(d),
        }

    # 4. LLM 连接 (只检查配置, 不测试 API)
    checks["llm"] = {
        "configured": bool(config_v10.LLM_API_KEY),
        "model": config_v10.LLM_MODEL,
    }

    # 5. Neo4j (可选)
    if config_v10.NEO4J_URI:
        checks["neo4j"] = {"configured": True, "uri": config_v10.NEO4J_URI}
    else:
        checks["neo4j"] = {"configured": False}

    # 6. V9 引擎
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", "..", "工单9", "研发"))
        import qa_engine_v9
        checks["v9_engine"] = {"ok": True}
    except Exception as e:
        checks["v9_engine"] = {"ok": False, "error": str(e)}

    all_ok = all(c.get("ok", True) for c in checks.values())
    return {
        "status": "healthy" if all_ok else "unhealthy",
        "checks": checks,
        "uptime": round(time.time() - _start_time, 1) if '_start_time' in dir() else 0,
        "version": "v10.0",
    }


_start_time = time.time()


def export_data() -> Dict:
    """导出数据 (持久化目录打包)"""
    import subprocess
    export_path = os.path.join(config_v10.DATA_DIR, f"export_{int(time.time())}.tar.gz")
    try:
        subprocess.run(
            ["tar", "-czf", export_path, "-C", config_v10.DATA_DIR, "."],
            check=True, timeout=60
        )
        return {"ok": True, "export_path": export_path,
                "size_kb": round(os.path.getsize(export_path) / 1024, 1)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_system_dirs() -> Dict:
    """获取系统目录信息"""
    result = {}
    for name, path in [("data", config_v10.DATA_DIR), ("cache", config_v10.CACHE_DIR),
                        ("logs", config_v10.LOG_DIR), ("shared", config_v10.SHARED_DIR)]:
        total_size = 0
        file_count = 0
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                file_count += len(files)
                for f in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        result[name] = {
            "path": path,
            "exists": os.path.isdir(path),
            "files": file_count,
            "size_kb": round(total_size / 1024, 1),
        }
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(check_system(), indent=2, ensure_ascii=False))
