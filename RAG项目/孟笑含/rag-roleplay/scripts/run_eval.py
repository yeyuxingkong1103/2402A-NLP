# -*- coding: utf-8 -*-
"""RAGAS 基线评测一键脚本：准备数据 → 入库 → 逐题跑 RAG 管线 → 打分 → 出报告。

用法：
    python scripts/run_eval.py              # 用 data/eval/eval_dataset.json
    python scripts/run_eval.py 题库.json     # 指定题库
"""
import asyncio
# 解析：异步模块（评测主流程）
import json
# 解析：JSON（保存报告）
import logging
# 解析：日志
import sys
# 解析：命令行参数
from datetime import datetime
# 解析：时间戳（报告文件名）
from pathlib import Path
# 解析：路径

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# 解析：日志配置
logger = logging.getLogger("rag-eval")
# 解析：评测 logger

ROOT = Path(__file__).resolve().parent.parent
# 解析：项目根目录
sys.path.insert(0, str(ROOT))
# 解析：把项目根加入导入路径（脚本从任意目录运行都能 import app）

from app.config import settings  # noqa: E402
# 解析：配置（noqa 抑制"导入不在顶部"告警）
from app.core.llm import OpenAILLM  # noqa: E402
# 解析：大模型客户端
from app.eval.dataset import load_eval_dataset  # noqa: E402
# 解析：题库加载
from app.eval.ragas_metrics import LocalEmbeddings, build_judge_llm, run_ragas  # noqa: E402
# 解析：RAGAS 封装
from app.eval.report import format_report  # noqa: E402
# 解析：报告格式化
from app.eval.runner import EvalPipeline  # noqa: E402
# 解析：评测管线
from app.models.db import SessionLocal  # noqa: E402
# 解析：数据库会话
from app.models.tables import Role  # noqa: E402
# 解析：角色表
from app.rag.milvus_store import MilvusStore  # noqa: E402
# 解析：Milvus
from app.rag.models import BGEM3Embedder, BGEReranker  # noqa: E402
# 解析：本地 BGE 模型
from app.services.knowledge_service import KnowledgeService  # noqa: E402
# 解析：知识库服务

DATASET_PATH = ROOT / "data" / "eval" / "eval_dataset.json"
# 解析：默认题库路径
KNOWLEDGE_PDF = ROOT / "data" / "knowledge" / "hypertension_guide.pdf"
# 解析：评测知识文档
REPORT_DIR = ROOT / "eval_report"
# 解析：报告目录


def ensure_knowledge_uploaded(knowledge_service, role_id: int, pdf_path: Path) -> None:
    """知识库没有该文档时上传（幂等）。"""
    existing = {item["source"] for item in knowledge_service.list_sources(role_id)}
    # 解析：库内已有文档名
    if pdf_path.name in existing:
        # 解析：已入库
        logger.info("知识库已有 %s，跳过上传", pdf_path.name)
        # 解析：跳过
        return
        # 解析：返回
    logger.info("上传知识文档 %s ...", pdf_path.name)
    # 解析：日志
    result = knowledge_service.ingest_pdf(
        # 解析：上传入库
        role_id=role_id, pdf_bytes=pdf_path.read_bytes(), source=pdf_path.name
        # 解析：角色、字节、文件名
    )
    logger.info("入库完成：%s 块", result["chunks"])
    # 解析：日志


# RAGAS 评测主流程：题库加载 → 入库 → 逐题跑管线 → 打分 → 报告
async def main() -> None:
    # 解析：评测主流程
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET_PATH
    # 解析：命令行参数指定题库，否则用默认
    dataset = load_eval_dataset(dataset_path)
    # 解析：加载题库（含字段校验）
    logger.info("题库：%s（%d 题）", dataset.name, len(dataset.samples))
    # 解析：日志

    # 找角色
    with SessionLocal() as db:
        # 解析：开数据库会话
        role = db.query(Role).filter(Role.name == dataset.role).first()
        # 解析：按题库指定的角色名查角色
    if role is None:
        # 解析：角色不存在
        raise SystemExit(f"角色不存在：{dataset.role}")
        # 解析：退出
    logger.info("评测角色：%s（id=%s）", role.name, role.id)
    # 解析：日志

    # 真实组件：本地 BGE-m3 + Milvus + BGE 重排 + DeepSeek 生成
    logger.info("加载本地模型（首次约 30-60 秒）...")
    # 解析：日志
    embedder = BGEM3Embedder(settings.embedding_model_path)
    # 解析：加载 BGE-m3
    reranker = BGEReranker(settings.rerank_model_path)
    # 解析：加载重排模型
    knowledge_service = KnowledgeService(
        # 解析：构造知识库服务（真实组件）
        embedder=embedder,
        # 解析：向量化
        reranker=reranker,
        # 解析：重排
        milvus=MilvusStore(settings.milvus_host, settings.milvus_port),
        # 解析：Milvus
        chunk_size=settings.chunk_size,
        # 解析：分块大小
        overlap=settings.chunk_overlap,
        # 解析：重叠
        chunking_mode=settings.chunking_mode,
        # 解析：分块模式（对比评测时用环境变量切换）
        semantic_threshold=settings.semantic_threshold,
        # 解析：语义阈值
        parent_chunk_size=settings.parent_chunk_size,
        # 解析：父块大小
    )
    ensure_knowledge_uploaded(knowledge_service, role.id, KNOWLEDGE_PDF)
    # 解析：知识入库（幂等）

    llm = OpenAILLM(
        # 解析：构造生成大模型
        base_url=settings.llm_base_url,
        # 解析：地址
        api_key=settings.llm_api_key,
        # 解析：密钥
        model=settings.llm_model,
        # 解析：模型
        timeout=settings.llm_timeout,
        # 解析：超时
    )
    pipeline = EvalPipeline(
        # 解析：构造评测管线
        knowledge_service=knowledge_service,
        # 解析：知识库服务
        llm=llm,
        # 解析：生成模型
        role=role,
        # 解析：角色
        top_k=settings.rag_top_k,
        # 解析：注入条数
        recall_k=settings.rag_recall_k,
        # 解析：召回条数
    )

    logger.info("逐题跑 RAG 管线（检索→重排→生成）...")
    # 解析：日志
    samples = await pipeline.run_all(dataset.samples)
    # 解析：逐题真实 RAG 流程生成回答

    logger.info("RAGAS 打分（LLM-as-judge: %s）...", settings.llm_model)
    # 解析：日志
    judge = build_judge_llm(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    # 解析：构造 judge（默认与生成同模型，可换）
    scores, rows = run_ragas(samples, judge, LocalEmbeddings(embedder))
    # 解析：RAGAS 五指标打分

    # 报告
    REPORT_DIR.mkdir(exist_ok=True)
    # 解析：确保报告目录
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 解析：时间戳
    report_md = format_report(
        # 解析：格式化报告
        dataset.name, role.name, scores, rows, model=settings.llm_model
        # 解析：名称、角色、分数、明细、模型
    )
    (REPORT_DIR / f"baseline_{stamp}.md").write_text(report_md, encoding="utf-8")
    # 解析：保存 markdown 报告
    (REPORT_DIR / f"baseline_{stamp}.json").write_text(
        # 解析：保存 JSON（含生成样本，供重判与对比）
        json.dumps(
            # 解析：序列化
            {"scores": scores, "rows": rows,
             "samples": samples},
            # 解析：分数、明细、样本
            ensure_ascii=False, indent=2,
            # 解析：中文不转义 + 缩进
        ),
        encoding="utf-8",
        # 解析：UTF-8
    )
    logger.info("报告已保存到 %s", REPORT_DIR / f"baseline_{stamp}.md")
    # 解析：日志
    print("\n" + report_md)
    # 解析：终端输出报告


if __name__ == "__main__":
    # 解析：脚本入口
    asyncio.run(main())
    # 解析：异步执行主流程
