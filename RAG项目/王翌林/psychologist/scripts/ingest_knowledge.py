"""离线知识库构建：解析 → 分块 → BGE-M3 向量化 → Milvus 入库 → MySQL 元数据。

用法：
    # 按三个角色的知识库目录批量入库（默认）
    python scripts/ingest_knowledge.py --all

    # 指定角色
    python scripts/ingest_knowledge.py --persona code=cbt_chen

    # 入库单个文件
    python scripts/ingest_knowledge.py --file /path/to/book.pdf --persona-id 2

    # 重建索引（先删除该角色已有向量与元数据）
    python scripts/ingest_knowledge.py --all --drop-existing

分块策略：fixed / sentence / paragraph / heading / semantic（默认 paragraph）
"""
import argparse
import os
import sys
import time

# 把项目根目录插入 sys.path 首位，保证脚本从任意目录运行时都能 import 到 src.* 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings  # noqa: E402
from src.core.logging import get_logger, setup_logging  # noqa: E402
from src.db.milvus import ensure_collections  # noqa: E402
from src.db.mysql import init_db, session_scope  # noqa: E402
from src.rag.pipeline import offline_ingest_directory, offline_ingest_file, offline_rebuild_persona  # noqa: E402
from src.services import knowledge_service, persona_service  # noqa: E402
from src.services.persona_seed import KNOWLEDGE_DIRS  # noqa: E402

logger = get_logger("scripts.ingest")


def _resolve_persona_id(db, spec: str) -> int:
    """支持 --persona-id 1 或 --persona code=cbt_chen 两种写法。"""
    if spec.startswith("code="):
        # 按角色编码查库得到真实 ID；查不到直接退出，避免后续用错误 ID 入库
        persona = persona_service.get_persona_by_code(db, spec.split("=", 1)[1])
        if not persona:
            raise SystemExit(f"角色编码不存在：{spec}")
        return persona.id
    # 否则按纯数字 ID 解析
    return int(spec)


def main() -> None:
    # argparse 定义多种入库模式：--all 批量 / --persona 指定角色 / --file 单文件 / --dir 目录
    parser = argparse.ArgumentParser(description="离线知识库构建")
    parser.add_argument("--all", action="store_true", help="按角色目录批量入库（三个角色）")
    parser.add_argument("--persona", default=None, help="角色：code=cbt_chen 或 数字 ID")
    parser.add_argument("--persona-id", type=int, default=None, help="角色 ID")
    parser.add_argument("--file", default=None, help="单个文件路径")
    parser.add_argument("--dir", default=None, help="目录路径（配合 --persona-id）")
    parser.add_argument("--strategy", default="paragraph",
                        choices=["fixed", "sentence", "paragraph", "heading", "semantic"])
    parser.add_argument("--drop-existing", action="store_true", help="重建：先清空该角色向量与元数据")
    args = parser.parse_args()

    # 初始化日志、MySQL 表结构与 Milvus 集合，保证后续入库有可用的存储
    setup_logging()
    init_db()
    ensure_collections()

    start = time.time()
    with session_scope() as db:
        if args.file:
            # 单文件入库：必须显式给出 persona_id，否则无法确定归属角色
            if not args.persona_id:
                raise SystemExit("--file 需要同时指定 --persona-id")
            result = offline_ingest_file(db, args.file, args.persona_id, args.strategy)
            logger.info("单文件入库结果：%s", result)
        elif args.dir:
            # 目录批量入库：同样需要 persona_id 指定知识归属角色
            if not args.persona_id:
                raise SystemExit("--dir 需要同时指定 --persona-id")
            results = offline_ingest_directory(db, args.dir, args.persona_id, args.strategy)
            logger.info("目录入库完成，共 %d 个文件", len(results))
        elif args.persona or args.persona_id:
            # 指定角色入库：--persona-id 优先，否则用 --persona 编码反查 ID
            persona_id = args.persona_id or _resolve_persona_id(db, args.persona)
            persona = persona_service.get_persona(db, persona_id)
            # 从预配置的 KNOWLEDGE_DIRS 取该角色的知识库目录列表（未配置则为空）
            dirs = KNOWLEDGE_DIRS.get(persona.persona_code, [])
            result = offline_rebuild_persona(
                db, persona_id, dirs, drop_existing=args.drop_existing, strategy=args.strategy
            )
            logger.info("角色 %s 入库完成：文档 %d，成功 %d，分块 %d",
                        persona.name, result["documents"], result["success"], result["chunks"])
        elif args.all:
            # 批量模式：遍历所有角色（含停用），逐个按目录入库
            for persona in persona_service.list_personas(db, only_active=False):
                dirs = KNOWLEDGE_DIRS.get(persona.persona_code, [])
                if not dirs:
                    logger.warning("角色 %s 未配置知识库目录，跳过", persona.persona_code)
                    continue
                logger.info("==== 开始入库角色：%s（%s）====", persona.name, persona.persona_code)
                result = offline_rebuild_persona(
                    db, persona.id, dirs, drop_existing=args.drop_existing, strategy=args.strategy
                )
                logger.info("角色 %s 完成：成功 %d/%d，分块 %d",
                            persona.name, result["success"], result["documents"], result["chunks"])
        else:
            # 没有任何模式参数时，打印帮助而不是静默退出，便于用户了解用法
            parser.print_help()
            return

        # 入库结束后汇总打印知识库统计，便于确认入库效果
        stats = knowledge_service.knowledge_stats(db)
        logger.info("知识库统计：%s", stats)
    logger.info("总耗时 %.1fs", time.time() - start)


if __name__ == "__main__":
    main()