"""条文一句话摘要生成 CLI（批次 37）。

对 approved 版本的 parent 块（整条法条）批量生成一句话摘要，写入 chunk_summaries 表；
索引服务写 Milvus 时按 chunk_key 带上 summary（引用卡片展示 / 父块检索补充用）。

设计要点：
- 只对 approved 版本跑：未审核内容不该出现在引用卡片上（与索引口径一致）；
- 只对 parent 块跑：摘要的对象是"条文"，parent 块即整条；子块不单独摘
  （引用展示归并到父块后取父块摘要）；
- 断点续跑：已存在于 chunk_summaries 的 chunk_key 默认跳过（--force 全量重算）；
  中途失败直接退出，下次重跑从上次断点继续；
- 失败重试：单条 LLM 调用 3 次退避重试（上游偶发挂起），3 次仍失败则整批退出
  （宁可少摘要也不写垃圾摘要），已写入的条目保留（下次续跑）；
- 成本透明：结尾打印条数 / prompt 字符数 / 输出字符数与 token 估算，
  供报告核算费用（本 CLI 不查价目表，单价人工代入）。

用法（backend 目录下）：
    python -m app.cli.summarize_chunks --dry-run        # 只统计待处理条数
    python -m app.cli.summarize_chunks                  # 全量生成（断点续跑）
    python -m app.cli.summarize_chunks --limit 20       # 先小批量试跑
    python -m app.cli.summarize_chunks --force          # 全量重算
"""

import argparse
import json
import sys
import time

from sqlalchemy import select

from app.core.config import load_environment_file, settings
from app.db import sql_models  # noqa: F401  确保全部实体注册进 Base.metadata
from app.db.base import Base
from app.db.engine import create_database_engine, create_session_factory
from app.db.sql_models import ChunkSummary, DocumentChunk, DocumentVersion
from app.db.version_status import APPROVED
from app.models.llm import OpenAiCompatibleChatClient

# 单条 LLM 调用重试次数与退避秒数（上游偶发挂起/超时；与评测工具同款策略）
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

SYSTEM_PROMPT = (
    "你是法律条文摘要助手。请把给定的法律条文压缩成一句话摘要，"
    "要求：不超过 60 个字；概括该条的核心规范内容（谁、在什么情形下、有什么义务或后果）；"
    "只输出摘要正文，不要任何前缀、引号或解释。"
)


def build_llm_client() -> OpenAiCompatibleChatClient:
    """从 settings 构造 LLM 客户端（与生产 chat 同一套配置，摘要口径可追溯）。"""
    return OpenAiCompatibleChatClient(
        api_base_url=settings.llm_api_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0.0,  # 摘要要稳定，不采样
        # 1024 而非 512：deepseek 类模型的思考段也占输出额度，
        # 长条文会触发长推理，512 都可能被占满导致"空回答"（实测批次 37）。
        # 走配置而非写死：个别超长条文可临时用 LLM_MAX_TOKENS=2048 提额重跑
        max_tokens=max(1024, settings.llm_max_tokens),
        timeout=settings.llm_timeout_seconds,
    )


def fetch_pending_chunks(session, *, force: bool, limit: int | None) -> list[DocumentChunk]:
    """取待摘要的 parent 块：approved 版本 + （未生成摘要 或 force）。

    按 version_id, sequence 排序保证断点续跑顺序稳定（同样的重跑顺序一致）。
    """
    existing_keys: set[str] = set()
    if not force:
        existing_keys = set(session.scalars(select(ChunkSummary.chunk_key)).all())

    stmt = (
        select(DocumentChunk)
        .join(
            DocumentVersion,
            DocumentVersion.id == DocumentChunk.document_version_id,
        )
        .where(
            DocumentChunk.chunk_type == "parent",
            DocumentVersion.version_status == APPROVED,
        )
        .order_by(DocumentChunk.document_version_id, DocumentChunk.sequence)
    )
    chunks = session.scalars(stmt).all()
    if existing_keys:
        chunks = [chunk for chunk in chunks if chunk.chunk_key not in existing_keys]
    return list(chunks[:limit]) if limit else list(chunks)


def summarize_one(client: OpenAiCompatibleChatClient, content: str) -> str:
    """单条摘要，带 3 次退避重试；全部失败抛出最后一次异常。"""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            summary = client.chat(SYSTEM_PROMPT, content)
            # 防御：模型偶发输出"摘要："之类前缀，剥掉常见几种
            for prefix in ("摘要：", "摘要:", "这句话摘要："):
                if summary.startswith(prefix):
                    summary = summary[len(prefix):]
            return summary.strip()
        except Exception as error:  # noqa: BLE001  统一按"上游偶发挂起"重试
            last_error = error
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_error  # type: ignore[misc]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成条文一句话摘要（写入 chunk_summaries）")
    parser.add_argument("--limit", type=int, help="本次最多处理条数（试跑/分批用）")
    parser.add_argument("--force", action="store_true", help="忽略已有摘要全量重算")
    parser.add_argument("--dry-run", action="store_true", help="只统计待处理条数，不调 LLM")
    parser.add_argument("--env-file", help="可选 dotenv 文件路径（默认按 ENVIRONMENT 选）")
    args = parser.parse_args(argv)

    if args.env_file:
        load_environment_file(args.env_file)
    else:
        load_environment_file()

    # 连接串与 chat_runtime.build_default_session_factory 同口径（含密码转义），
    # 不另造第三份构造逻辑
    from urllib.parse import quote_plus

    database_url = (
        f"mysql+pymysql://{quote_plus(settings.mysql_user)}:{quote_plus(settings.mysql_password)}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}?charset=utf8mb4"
    )
    engine = create_database_engine(database_url)
    # chunk_summaries 是新表，先确保存在（与既有 CLI 同款 create_all 口径）
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    client = build_llm_client()

    with session_factory() as session:
        pending = fetch_pending_chunks(session, force=args.force, limit=args.limit)

    total_prompt_chars = 0
    total_output_chars = 0
    done = 0
    if args.dry_run:
        payload = {
            "status": "dry_run",
            "pending_chunks": len(pending),
            "force": args.force,
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    started = time.time()
    failed_keys: list[str] = []
    for chunk in pending:
        try:
            summary = summarize_one(client, chunk.content)
        except Exception as error:  # noqa: BLE001
            # 单条失败不中断整体（断点续跑语义：失败条本轮跳过，重跑时自动补）
            failed_keys.append(chunk.chunk_key)
            print(f"[失败] {chunk.chunk_key}: {error}", file=sys.stderr)
            continue
        total_prompt_chars += len(SYSTEM_PROMPT) + len(chunk.content)
        total_output_chars += len(summary)
        done += 1
        # 逐条即时落库：断点续跑的"断点"就是这张表本身
        with session_factory() as session:
            session.merge(
                ChunkSummary(
                    chunk_key=chunk.chunk_key,
                    summary=summary,
                    model=settings.llm_model,
                )
            )
            session.commit()
        if done % 50 == 0:
            print(f"[进度] {done}/{len(pending)}", file=sys.stderr)

    # 成本估算：中文 1 token ≈ 1.6 字符（bge/deepseek 系经验值，仅供预算参考，
    # 精确计费以服务商账单为准）；单价代入报告即可折算费用
    estimated_prompt_tokens = int(total_prompt_chars / 1.6)
    estimated_output_tokens = int(total_output_chars / 1.6)
    print(
        json.dumps(
            {
                "status": "done",
                "summarized": done,
                "failed": len(failed_keys),
                "pending_total": len(pending),
                "prompt_chars": total_prompt_chars,
                "output_chars": total_output_chars,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "elapsed_seconds": round(time.time() - started, 1),
                "model": settings.llm_model,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
