# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-LightRAG优化
"""
LightRAG 知识图谱构建模块（验收标准 1：优化实体/关系类型抽取）。

核心设计：
    1. 领域定制实体类型：针对招股说明书（力源信息 / 兴图新科）设计 13 类实体
       （公司、人员、股东、产品与服务、行业、财务指标、募投项目、技术标准、
       工程奖项、技术与资质、机构、地点、事件），通过 addon_params[
       "entity_types_guidance"] 注入抽取 Prompt，替换默认通用类型；
    2. 领域定制关系类型：控股/任职/供应/客户/募投/参与制定/荣获/资质等
       13 类关系，并在 guidance 中给出抽取要求；
    3. LLM：DeepSeek（deepseek-chat），异步并发 16 路；
    4. Embedding：本地 bge-m3（1024 维），批量编码；
    5. 增量更新演示：先插入招股说明书1，再增量插入招股说明书2，
       记录各自耗时与图谱规模变化，验证增量更新无需重建整个知识库。

用法：
    python lightrag_build.py --trial          # 小试验证（仅插入样本片段）
    python lightrag_build.py --doc all        # 全量构建（默认，先1后2增量插入）
    python lightrag_build.py --stats          # 输出图谱统计
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    EMBED_MODEL_PATH, EMBED_DIM, EMBED_BATCH_SIZE,
    WORK_DIR, TXT_DIR, CHUNK_TOKEN_SIZE, CHUNK_OVERLAP_TOKEN,
    LLM_MAX_ASYNC, EMBED_MAX_ASYNC, EMBED_BATCH_NUM, RESULT_DIR, EMBED_TIMEOUT,
)
from logger import get_logger

logger = get_logger(__name__)

# ============================================================
# 一、领域定制的实体类型与关系类型（验收标准 1 核心）
# ============================================================
ENTITY_TYPES_GUIDANCE = """请基于招股说明书（金融证券领域）文本进行实体与关系抽取，必须使用以下为本领域定制的实体类型，不要使用通用类型。

## 实体类型（从中选择最贴切的一类，格式为"中文名(English)"）：
- 公司(Company)：法人主体，包括发行人、子公司、分公司、参股公司、关联方企业、供应商、客户、竞争对手、同行业公司、保荐机构（券商）、会计师事务所、律师事务所等
- 人员(Person)：自然人，包括法定代表人、实际控制人、董事、监事、高级管理人员（总经理/副总经理/财务总监/董事会秘书）、核心技术人员、控股股东代表
- 股东(Shareholder)：持股主体，包括控股股东、发起人股东、法人股东、机构投资者、合伙企业等
- 产品与服务(Product)：公司生产销售的产品、提供的服务与技术解决方案（如电子元器件分销、专用通信设备等）
- 行业(Industry)：公司所处行业、细分行业、上下游行业（如电子信息行业、军用电子行业）
- 财务指标(FinancialMetric)：营业收入、净利润、毛利率、主营业务收入构成等财务数据项（抽取时在描述中注明报告期与金额）
- 募投项目(Project)：本次发行募集资金投资项目、在建工程、技改项目、补充流动资金安排
- 技术标准(TechnicalStandard)：公司参与起草/制定的国家标准、行业标准、军用标准
- 工程奖项(EngineeringAward)：公司参与建设并荣获国家/省部级奖项的工程项目（如国家科技进步一等奖）
- 技术与资质(TechnologyQualification)：专利、软件著作权、专有技术、军工资质、认证证书、高新技术企业认定
- 机构(Institution)：政府部门、监管机构（证监会/发改委）、军队单位、行业协会、高校科研院所
- 地点(Location)：注册地、生产基地、经营场所、募投项目实施地点
- 事件(Event)：重大事件，包括首次公开发行、重大合同签订、股权变动、对外投资等

## 关系类型（relationship_keywords 优先使用以下关键词）：
- 控股/持股（controlled_by、holds_shares_in）：股东→公司，描述中注明持股比例
- 任职（serves_as）：人员→公司/股东，描述中注明具体职务
- 供应关系（supplies_to）：供应商→公司
- 客户关系（is_customer_of）：客户→公司
- 子公司关系（is_subsidiary_of）：子公司→母公司
- 所属行业（belongs_to_industry）：公司/产品→行业，注明上/下游
- 募投投资（invests_in）：公司→募投项目，描述中注明拟投入金额
- 参与制定（participates_in_drafting）：公司→技术标准
- 荣获奖项（won_award_for）：公司→工程奖项，注明奖项等级与年份
- 拥有资质（owns_qualification）：公司→技术与资质
- 位于（located_in）：公司/项目→地点
- 竞争关系（competes_with）：公司→公司
- 中介服务（sponsors/audits/legal_advises）：保荐机构/会计师/律师事务所→发行人

## 抽取要求：
0. entity_type 字段必须严格按"中文名(English)"格式书写（如"公司(Company)"），不得只写英文；
1. 实体名称使用文档中的全称（如"武汉力源信息科技股份有限公司"），首次出现后可用简称，但同一实体保持名称一致；
2. 持股比例、金额、占比等数字信息必须写入实体描述或关系描述；
3. 财务数据必须注明所属报告期（如 2008 年、2009 年 1-6 月）；
4. 关系描述要体现业务实质（如"力源信息是德州仪器在中国的一级授权分销商"）。"""

# ============================================================
# 二、DeepSeek 异步 LLM 函数
# ============================================================


async def deepseek_llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list = [],
    **kwargs,
) -> str:
    """LightRAG 使用的 LLM 函数：调用 DeepSeek API（带重试）。"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    f"{LLM_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {LLM_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": LLM_MODEL,
                        "messages": messages,
                        "temperature": LLM_TEMPERATURE,
                        "max_tokens": kwargs.get("max_tokens", LLM_MAX_TOKENS),
                    },
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            wait = min(2 ** attempt * 2, 20)
            logger.warning(f"LLM 调用失败（第{attempt}次）：{e}，{wait}s 后重试")
            await asyncio.sleep(wait)
    raise RuntimeError("DeepSeek API 连续 3 次调用失败")


# ============================================================
# 三、bge-m3 异步 Embedding 函数
# ============================================================

_embed_model = None


def _get_embed_model():
    """懒加载本地 bge-m3 模型。"""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cpu")
        logger.info(f"bge-m3 模型已加载：{EMBED_MODEL_PATH}")
    return _embed_model


async def bge_m3_embed(texts: list[str]) -> np.ndarray:
    """批量文本向量化（线程池中执行，避免阻塞事件循环）。"""
    model = _get_embed_model()
    vecs = await asyncio.to_thread(
        model.encode, texts, batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=False,
    )
    return np.asarray(vecs)


# ============================================================
# 四、LightRAG 实例构建
# ============================================================


def build_rag(working_dir: Path = WORK_DIR):
    """创建配置完成的 LightRAG 实例。"""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    rag = LightRAG(
        working_dir=str(working_dir),
        # LLM：DeepSeek
        llm_model_func=deepseek_llm_func,
        llm_model_name=LLM_MODEL,
        llm_model_max_async=LLM_MAX_ASYNC,
        # Embedding：本地 bge-m3
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            func=bge_m3_embed,
            model_name="bge-m3",
        ),
        embedding_batch_num=EMBED_BATCH_NUM,
        embedding_func_max_async=EMBED_MAX_ASYNC,
        default_embedding_timeout=EMBED_TIMEOUT,
        # 切块
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN,
        # 实体抽取：关闭二次补捞（gleaning），控制调用量
        entity_extract_max_gleaning=0,
        # ★ 验收标准1：注入招股书领域定制的实体类型与关系类型
        addon_params={
            "language": "Simplified Chinese",
            "entity_types_guidance": ENTITY_TYPES_GUIDANCE,
        },
    )
    return rag


# ============================================================
# 五、图谱构建（增量插入演示）
# ============================================================


async def _insert_one(rag, txt_path: Path, label: str) -> dict:
    """插入单个文档并记录耗时。"""
    text = txt_path.read_text(encoding="utf-8")
    t0 = time.time()
    await rag.ainsert(text, file_paths=[txt_path.name])
    elapsed = time.time() - t0
    logger.info(f"[{label}] 插入完成：{len(text)} 字符，耗时 {elapsed:.1f}s")
    return {"file": txt_path.name, "chars": len(text), "seconds": round(elapsed, 1)}


def graph_stats(rag) -> dict:
    """读取当前图谱规模统计（从 graphml 持久化文件）。"""
    import networkx as nx

    graphml = Path(rag.working_dir) / "graph_chunk_entity_relation.graphml"
    g = nx.read_graphml(graphml)
    types = {}
    for _, data in g.nodes(data=True):
        t = data.get("entity_type", "未知") or "未知"
        types[t] = types.get(t, 0) + 1
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "entity_type_dist": dict(sorted(types.items(), key=lambda x: -x[1])),
    }


async def _delete_existing_docs(rag) -> int:
    """删除已存在的文档（failed 状态重插时 LightRAG 会按文件名判重跳过，需先删）。"""
    status_file = Path(rag.working_dir) / "kv_store_doc_status.json"
    if not status_file.exists():
        return 0
    doc_ids = list(json.loads(status_file.read_text(encoding="utf-8")).keys())
    for doc_id in doc_ids:
        try:
            await rag.adelete_by_doc_id(doc_id)
            logger.info(f"已删除旧文档记录：{doc_id[:24]}...")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"删除旧文档 {doc_id[:24]} 失败：{e}")
    return len(doc_ids)


async def run_build(mode: str, rebuild: bool = False) -> None:
    """执行构建流程。"""
    if mode == "trial":
        # 小试：仅插入 6 个片段，验证 LLM/Embedding/图谱全链路
        trial_dir = WORK_DIR.parent / "lightrag_trial"
        rag = build_rag(trial_dir)
        await rag.initialize_storages()
        try:
            text = (TXT_DIR / "招股说明书1.txt").read_text(encoding="utf-8")
            samples = [text[i:i + 1500] for i in range(0, 9000, 1500)]
            t0 = time.time()
            await rag.ainsert("\n\n".join(samples), file_paths=["trial_sample.txt"])
            print(f"小试完成：5 个片段，耗时 {time.time() - t0:.1f}s")
            print(json.dumps(graph_stats(rag), ensure_ascii=False, indent=2))
        finally:
            await rag.finalize_storages()
        return

    # 全量构建：先招股书1，再增量插入招股书2
    steps = []
    rag = build_rag()
    await rag.initialize_storages()
    try:
        if rebuild:
            n = await _delete_existing_docs(rag)
            print(f"rebuild 模式：已删除 {n} 条旧文档记录，将重新处理")

        for label, fname in [("招股说明书1(全量索引)", "招股说明书1.txt"),
                             ("招股说明书2(增量更新)", "招股说明书2.txt")]:
            step = await _insert_one(rag, TXT_DIR / fname, label)
            step["graph_after"] = graph_stats(rag)
            steps.append(step)
            print(f"[{label}] {step['seconds']}s | 节点 {step['graph_after']['nodes']} "
                  f"| 边 {step['graph_after']['edges']}")
            # 每份文档插入后立即导出图谱快照（json，防止意外丢失）
            snapshot = RESULT_DIR / f"graph_stats_after_{len(steps)}.json"
            snapshot.write_text(
                json.dumps(step["graph_after"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    finally:
        await rag.finalize_storages()

    (RESULT_DIR / "build_report.json").write_text(
        json.dumps({"steps": steps}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("构建完成，报告已保存 build_report.json")


def main():
    parser = argparse.ArgumentParser(description="LightRAG 知识图谱构建")
    parser.add_argument("--trial", action="store_true", help="小试验证")
    parser.add_argument("--stats", action="store_true", help="输出图谱统计")
    parser.add_argument("--rebuild", action="store_true",
                        help="删除旧文档记录后重建（用于修复 failed 状态）")
    args = parser.parse_args()

    if args.stats:
        rag = build_rag()
        print(json.dumps(graph_stats(rag), ensure_ascii=False, indent=2))
        return

    asyncio.run(run_build("trial" if args.trial else "all", rebuild=args.rebuild))


if __name__ == "__main__":
    main()
