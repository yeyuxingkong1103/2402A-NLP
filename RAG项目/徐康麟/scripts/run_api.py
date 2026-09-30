# -*- coding: utf-8 -*-
"""启动 FastAPI 服务。

用法：
    python scripts/run_api.py --port 8000
    python scripts/run_api.py --offline --port 8000     # 离线模式（无 key / 无 GPU）
    python scripts/run_api.py --reload                  # 开发热重载

会自动读取项目根目录的 ``.env``（已存在的环境变量优先，不覆盖）——
以前只有 ``run.sh`` 会读它，Windows 下直接跑本脚本时 ``.env`` 里的
``DEEPSEEK_API_KEY`` 之类会被**静默忽略**，排查起来很费时间。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.api.app import create_app  # noqa: E402
from legal_rag.config import RagConfig  # noqa: E402
from legal_rag.logging_setup import setup_logging  # noqa: E402
from legal_rag.roles import DEFAULT_ROLE_ID, get_role  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> int:
    """把 ``.env`` 读进 ``os.environ``（**已存在的变量优先**，不覆盖）。返回生效条数。"""
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return 0
    count = 0
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            count += 1
    return count


def _judge_summary() -> str:
    """一行说清"入库前判定"用谁当裁判、能不能用（起服务时就该看见）。"""
    from legal_rag.api.fitcheck import RoleFitJudge
    try:
        status = RoleFitJudge().provider_status()
    except Exception as exc:  # noqa: BLE001 - 打印信息不该拦住服务
        return f"状态未知（{type(exc).__name__}: {exc}）"
    if status["available"]:
        return f"{status['judge']} / {status['model']}（可用）"
    return ("不可用 —— 未配置 DEEPSEEK_API_KEY，且 Ollama 不可用；"
            "此时网页会如实显示「无法判定」，不会假装通过")


def _auth_summary(config) -> str:
    """一行说清强制鉴权开没开 —— **这是"要不要对外发布"的判据，必须醒目**。"""
    if config.auth.require_auth:
        return ("已开启（AUTH_REQUIRED/REQUIRE_AUTH=true）：/chat、/sessions、/documents* "
                "会校验登录态，且用登录态挡住伪造的 user_id")
    return ("**未开启（默认）**：接口接受任意 user_id，任何人可冒充任何账号 —— "
            "本机调试可以，**对外发布前必须置 AUTH_REQUIRED=true**")


def _start_warmup(app) -> None:
    """后台预热（**不阻塞服务启动**）：嵌入冷加载 + BM25 建索引/载缓存。

    真机实测这两项分别是「首查 7.6–9.3s」与「懒触发时约 160s / 11.9 万块」，本该在启动
    后台付掉。**注意 engine 是 lifespan 里的 ``build()`` 才建的**，比 ``create_app()``
    返回晚 —— 具体等待与预热策略见 ``legal_rag/api/warmup.py``（v3 第 1 次跑就因为这个
    拿到 None、预热静默失效、前 10 题落在降级窗口里）。
    """
    from legal_rag.api.warmup import start_warmup

    start_warmup(app, role_id=DEFAULT_ROLE_ID)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动法律 RAG HTTP 服务")
    parser.add_argument("--host", default=None, help="监听地址（默认取配置 API_HOST）")
    parser.add_argument("--port", type=int, default=None, help="监听端口（默认取配置 API_PORT）")
    parser.add_argument("--offline", action="store_true",
                        help="离线模式：内存向量库 + 零依赖向量化 + 余弦重排 + Mock 大模型")
    parser.add_argument("--reload", action="store_true", help="开发热重载")
    parser.add_argument("--log-level", default="info", help="uvicorn 日志级别")
    parser.add_argument("--no-sys-metrics", action="store_true",
                        help="不启动系统指标周期采样器（/metrics 里就没有 process_*/host_*/gpu_*）")
    parser.add_argument("--no-warmup", action="store_true",
                        help="不在启动后台预热（嵌入冷加载 + BM25 索引）；排查启动问题时用")
    return parser


def main(argv=None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)

    env_loaded = load_dotenv()
    print(f"配置文件   : {ROOT / '.env'}"
          + (f"（已加载 {env_loaded} 条；已存在的环境变量优先）" if env_loaded else "（不存在或为空）"))

    if args.offline:
        os.environ["RAG_OFFLINE"] = "1"
    if args.no_sys_metrics:
        os.environ["SYS_METRICS_ENABLED"] = "0"

    config = RagConfig.from_env()
    host = args.host or config.api_host
    port = args.port or config.api_port

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "未安装 uvicorn，无法启动服务：\n"
            "    .venv\\Scripts\\python.exe -m pip install -r requirements-full.txt"
        ) from exc

    print(f"启动服务：http://{host}:{port}   （离线模式：{args.offline}）")
    print(f"  ★ 网页控制台：http://{host}:{port}/ui   ← 用户交互都在这里")
    print(f"  向量库={config.vector_store}  向量化={config.embedding_provider}  "
          f"重排={config.rerank_provider}  大模型={config.llm_provider}")
    print(f"  入库前判定：{_judge_summary()}")
    print(f"  强制鉴权：{_auth_summary(config)}")
    print(f"  Ollama={config.ollama.base_url}  嵌入模型={config.ollama.embedding_model}  "
          f"生成模型={config.ollama.llm_model}")
    print(f"  并发上限={config.concurrency.max_concurrent_requests}  "
          f"线程池={config.concurrency.thread_pool_size}  "
          f"系统指标每 {config.logging.sys_metrics_interval}s/"
          f"{'开' if config.logging.sys_metrics_enabled and not args.no_sys_metrics else '关'}")
    print(f"  上传目录={config.upload_path}  单文件上限={config.upload.max_upload_mb}MB  "
          f"单次最多={config.upload.max_upload_files} 个")
    print("  接口：")
    for line in (
        "GET    /ui                   ★ 网页控制台（角色选择 / 聊天 / 知识库 / 入库前判定）",
        "GET    /health               组件健康（聚合状态 + 后端可达性/延迟 + GPU + 并发 + 活条数）",
        "GET    /metrics              Prometheus 指标（业务 + 系统层）",
        "GET    /roles                角色库",
        "POST   /auth/register        注册（只收 username+password；成功 201 + HttpOnly Cookie）",
        "POST   /auth/login           登录（失败统一 401 invalid_credentials，防用户名枚举）",
        "GET    /auth/me              当前登录用户（Cookie；脚本可用 Authorization: Bearer）",
        "POST   /auth/logout          登出（立刻失效该令牌并清 Cookie，返回 204）",
        "GET    /auth/username-available  用户名查重（别名 /auth/check-username）",
        "POST   /sessions             创建/登记会话",
        "GET    /sessions?user_id=    会话列表",
        "POST   /chat                 问答（stream=true 时流式；stream_mode=sse 带引用帧）",
        "POST   /ingest               重建/增量更新知识库",
        "GET    /documents/upload     上传能力自描述（允许的扩展名/上限）",
        "POST   /documents/upload     上传 PDF/txt/md/json 并立即入库（multipart，字段名 files）",
        "GET    /documents/fit-check  判定器状态（谁是裁判、能不能用）",
        "POST   /documents/fit-check  ★ 入库前判定：这份资料是否符合该角色设定（只判定不入库）",
        "GET    /documents            已入库文档列表",
        "GET    /documents/jobs/{id}  上传入库 job 进度",
        "DELETE /documents/{doc_id}   删除该文档的分块与文件",
    ):
        print(f"    {line}")

    app = create_app(config)
    if args.no_warmup:
        print("  预热：**已关闭**（--no-warmup）—— 首查要等 bge-m3 冷加载，"
              "BM25 会在第一次检索时才开始建")
    else:
        _start_warmup(app)
        print(f"  预热：已在后台开始（角色={DEFAULT_ROLE_ID}：嵌入冷加载 + BM25 建索引/载缓存）")

    if args.reload:
        print("  [注意] --reload 模式下预热只作用于父进程，热重载出的 worker 不会预热")
        uvicorn.run("legal_rag.api.app:create_app", factory=True,
                    host=host, port=port, reload=True, log_level=args.log_level)
    else:
        uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
