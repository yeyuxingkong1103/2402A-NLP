# -*- coding: utf-8 -*-
"""
RAGAS 评测脚本（M3 基线 / M5 迭代对比共用）

评测三类样本，输出四类指标：

    检索集  -> recall@k、MRR          （自算，确定性指标）
    生成集  -> faithfulness           （RAGAS，LLM 作为裁判）
    安全集  -> refusal_accuracy       （自算，拒答判定）

★ 为什么检索指标不用 RAGAS 计算（见技术决策记录 ADR-007）★
    recall@k 与 MRR 本质是集合运算，给定标注结果与检索返回列表，
    数值完全可精确计算。若交给 LLM 判定，会引入随机性 ——
    同一份检索结果两次评测可能得出不同数值，直接污染 V1/V2/V3 的对比数据，
    产生虚假的提升或下降。因此这两项一律自算，保证版本对比可信。

用法：
    python -m scripts.ragas_eval --pipeline v1
    python -m scripts.ragas_eval --pipeline v1 --tag baseline
    python -m scripts.ragas_eval --pipeline v1 --skip-ragas   # 跳过 LLM 评测（省费用）
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 允许以脚本方式直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# 金标准页码标识：(源文件名关键字, 页码)
PageKey = Tuple[str, int]
# 检索命中标识：(源文件名, 该块覆盖的**完整页码列表**)
# 注意是完整列表而非单一主页码：chunk 可以跨页，答案正文可能位于后半部分，
# 只记 page_no 会让跨页块的次页命中被漏判（详见 ADR-026）。
HitPages = Tuple[str, Tuple[int, ...]]


# ===========================================================================
# 数据集
# ===========================================================================

DATASET_FILE = "测试数据集.json"


def load_dataset(path: Optional[Path] = None) -> Dict[str, Any]:
    """加载评测数据集"""
    path = path or (settings.eval_path / DATASET_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"评测数据集不存在：{path}\n"
            f"请先准备 {DATASET_FILE}（含 retrieval / generation / safety 三部分）"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _page_key(item: Dict[str, Any]) -> PageKey:
    """把标注中的来源统一转成 (文件名关键字, 页码) 元组"""
    return (str(item.get("file", "")).strip(), int(item.get("page", 0)))


def _pages_match(retrieved: HitPages, gold: PageKey) -> bool:
    """
    判断一个检索命中是否命中标注页。

    命中条件：该 chunk **覆盖的任一页**等于标注页，且文件名匹配。

    为什么不是只看主页码：
        page_no 的语义是「chunk 覆盖的首个内容块页码」，而 chunk 可跨页。
        答案正文完全可能位于跨页块的后半部分（例如某块 page_no=6、
        page_nums=[6,7]，答案文字实际在第 7 页）。只比对 page_no 会把
        这种命中判为未命中 —— 全库 127 个块中有 29 个（23%）是跨页块，
        因此这是一个系统性偏差，而非个例。

    文件名采用「包含」匹配：标注中可只写关键字（如 "42581"），
    避免因 PDF 文件名过长或含空格导致匹配失败。
    """
    r_file, r_pages = retrieved
    g_file, g_page = gold
    if not r_pages:
        return False
    if not (g_file in r_file or r_file in g_file):
        return False
    return g_page in r_pages


# ===========================================================================
# 检索指标（自算）
# ===========================================================================

def compute_retrieval_metrics(
    retrieved_pages: List[HitPages],
    gold_pages: List[PageKey],
    top_k_list: List[int],
) -> Dict[str, float]:
    """
    计算单条样本的检索指标。

    recall@k : 前 k 个结果中命中任一标注页，记 1，否则 0
    rr       : 首个命中结果的排名倒数（命中不了记 0）
    """
    metrics: Dict[str, float] = {}

    for k in top_k_list:
        head = retrieved_pages[:k]
        hit = any(_pages_match(r, g) for r in head for g in gold_pages)
        metrics[f"recall@{k}"] = 1.0 if hit else 0.0

    rr = 0.0
    for rank, page in enumerate(retrieved_pages, start=1):
        if any(_pages_match(page, g) for g in gold_pages):
            rr = 1.0 / rank
            break
    metrics["rr"] = rr
    return metrics


def eval_retrieval(pipeline_name: str, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    检索集评测。

    直接用检索到的页码与标注页码比对，不经过答案生成环节，
    保证指标只反映「检索能力」，不受生成质量干扰。
    """
    from backend.rag_pipeline import get_pipeline_by_name

    pipeline = get_pipeline_by_name(pipeline_name)
    top_k_list = settings.top_k_list
    max_k = max(top_k_list)

    per_sample: List[Dict[str, Any]] = []

    for sample in samples:
        question = sample["question"]
        gold_pages = [_page_key(g) for g in sample.get("gold_pages", [])]

        try:
            # 只做检索，不做生成，避免 LLM 开销与干扰
            hits = pipeline.retrieve(question, top_k=max_k)
            retrieved_pages = [
                (
                    h.get("file_name", ""),
                    # 优先用完整页码列表；缺失时退回主页码（兼容异常数据）
                    tuple(h.get("page_nums") or [int(h.get("page_no", 0))]),
                )
                for h in hits
            ]
            metrics = compute_retrieval_metrics(retrieved_pages, gold_pages, top_k_list)
        except Exception as exc:
            logger.exception("检索评测失败：%s | %s", sample.get("id"), exc)
            metrics = {f"recall@{k}": 0.0 for k in top_k_list}
            metrics["rr"] = 0.0
            retrieved_pages = []

        per_sample.append({
            "id": sample.get("id"),
            "question": question,
            "gold_pages": [f"{f}:{p}" for f, p in gold_pages],
            "retrieved_pages": [
                f"{f}:{','.join(str(p) for p in pages)}" for f, pages in retrieved_pages
            ],
            **metrics,
        })
        logger.info(
            "[检索] %s | recall@%d=%.0f | rr=%.3f | 标注=%s | 命中=%s",
            sample.get("id"), max_k, metrics.get(f"recall@{max_k}", 0.0),
            metrics["rr"], per_sample[-1]["gold_pages"],
            per_sample[-1]["retrieved_pages"][:5],
        )

    # 汇总：各指标取平均
    summary: Dict[str, float] = {}
    for key in per_sample[0].keys() if per_sample else []:
        if isinstance(per_sample[0].get(key), float):
            values = [s[key] for s in per_sample]
            summary[key] = round(statistics.mean(values), 4) if values else 0.0

    summary["mrr"] = summary.pop("rr", 0.0)
    return {"summary": summary, "samples": per_sample, "count": len(per_sample)}


# ===========================================================================
# 生成忠实度（RAGAS）
# ===========================================================================

def _patch_ragas_imports() -> None:
    """
    兼容垫片：绕过 ragas 与 langchain-community 的版本脱节。

    问题现象：
        ragas 0.4.x 内部仍然执行
            from langchain_community.chat_models.vertexai import ChatVertexAI
        但 langchain-community 0.4.x 已移除 VertexAI 集成，
        导致 `import ragas` 直接抛 ModuleNotFoundError。

    处理方式：
        本项目完全不使用 Google VertexAI，因此注入一个占位模块满足其导入需求，
        而不去改动任何第三方包的实际文件（保证环境可复现、可重新部署）。

    说明：
        这是临时兼容措施。待 ragas 与 langchain-community 版本对齐后可移除本函数。
    """
    import sys
    import types

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        import importlib
        importlib.import_module(module_name)
        return  # 该模块确实存在，无需垫片
    except ImportError:
        pass

    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # noqa: D401
        """占位类：本项目不使用 VertexAI，仅用于满足 ragas 的导入需求。"""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "本项目未接入 Google VertexAI，该占位实现不应被实例化。"
            )

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim
    logger.debug("已注入 langchain_community.chat_models.vertexai 兼容垫片")

def eval_generation(
    pipeline_name: str,
    samples: List[Dict[str, Any]],
    *,
    use_ragas: bool = True,
) -> Dict[str, Any]:
    """
    生成集评测：计算答案忠实度 faithfulness。

    faithfulness 衡量「答案是否完全由检索到的原文支撑」，
    是本项目防幻觉要求（N2）的量化验证手段。

    该指标需要语义级判断，无法用规则精确计算，因此使用 RAGAS，
    以 deepseek-v4-flash 作为裁判模型。
    """
    from backend.rag_pipeline import get_pipeline_by_name

    pipeline = get_pipeline_by_name(pipeline_name)
    records: List[Dict[str, Any]] = []

    # 先跑一遍链路，收集 (问题, 答案, 检索上下文)
    for sample in samples:
        question = sample["question"]
        try:
            result = pipeline.answer(question, use_cache=False)
            contexts = [s.get("summary", "") for s in result.get("sources", [])]
            # 摘要不足以支撑忠实度判断，取回完整正文
            contexts = _fetch_full_contexts(result.get("sources", []))
            records.append({
                "id": sample.get("id"),
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "ground_truth": sample.get("ground_truth", ""),
                "sources": [
                    {"file_name": s.get("file_name"), "page_no": s.get("page_no")}
                    for s in result.get("sources", [])
                ],
            })
            logger.info("[生成] %s | 答案长度=%d | 上下文数=%d",
                        sample.get("id"), len(records[-1]["answer"]), len(contexts))
        except Exception as exc:
            logger.exception("生成评测失败：%s | %s", sample.get("id"), exc)

    summary: Dict[str, Any] = {"count": len(records)}

    if not use_ragas or not records:
        summary["faithfulness"] = None
        summary["note"] = "已跳过 RAGAS 评测" if not use_ragas else "无有效样本"
        return {"summary": summary, "samples": records}

    try:
        _patch_ragas_imports()
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness

        dataset = Dataset.from_dict({
            "question": [r["question"] for r in records],
            "answer": [r["answer"] for r in records],
            "contexts": [r["contexts"] for r in records],
            "ground_truth": [r["ground_truth"] for r in records],
        })

        logger.info("开始 RAGAS 评测（裁判模型=%s），共 %d 条样本…",
                    settings.deepseek_model, len(records))

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            llm=_build_ragas_llm(),
            embeddings=_build_ragas_embeddings(),
            raise_exceptions=False,
            show_progress=True,
        )

        scores = result.to_pandas() if hasattr(result, "to_pandas") else None
        if scores is not None:
            for i, record in enumerate(records):
                try:
                    record["faithfulness"] = float(scores.iloc[i]["faithfulness"])
                except Exception:
                    record["faithfulness"] = None
            # 注意：RAGAS 对个别样本可能返回 nan（裁判模型未给出可用判定）。
            # 统计均值时必须显式排除 nan —— statistics.mean 一旦遇到 nan，
            # 整个结果会变成 nan，导致所有样本的有效评分全部丢失。
            valid = [
                r["faithfulness"] for r in records
                if r.get("faithfulness") is not None
                and not math.isnan(r["faithfulness"])
            ]
            summary["valid_count"] = len(valid)
            summary["faithfulness"] = round(statistics.mean(valid), 4) if valid else None
        else:
            summary["faithfulness"] = None

    except ImportError as exc:
        logger.error("RAGAS 未安装，跳过忠实度评测：%s", exc)
        summary["faithfulness"] = None
        summary["note"] = "RAGAS 未安装（pip install ragas）"
    except Exception as exc:
        logger.exception("RAGAS 评测失败：%s", exc)
        summary["faithfulness"] = None
        summary["note"] = f"RAGAS 评测异常：{exc}"

    return {"summary": summary, "samples": records}


def _fetch_full_contexts(sources: List[Dict[str, Any]]) -> List[str]:
    """按来源的 chunk_id 取回完整正文，作为 RAGAS 的上下文"""
    from backend.db import mysql

    chunk_ids = [s.get("chunk_id") for s in sources if s.get("chunk_id")]
    if not chunk_ids:
        return []
    try:
        rows = mysql.fetch_chunks_by_ids(chunk_ids)
        order = {cid: i for i, cid in enumerate(chunk_ids)}
        rows.sort(key=lambda r: order.get(r["chunk_id"], 999))
        return [r.get("content", "") for r in rows]
    except Exception as exc:
        logger.warning("取回完整上下文失败：%s", exc)
        return []


def _build_ragas_llm():
    """构造 RAGAS 使用的裁判模型（复用生成模型的 OpenAI 兼容接口）"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        timeout=settings.llm_timeout,
    )


def _build_ragas_embeddings():
    """构造 RAGAS 使用的向量模型（复用本地 BGE-M3，避免额外的接口开销）"""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.bge_m3_path,
        model_kwargs={"device": settings.embed_device},
        encode_kwargs={"normalize_embeddings": True},
    )


# ===========================================================================
# 安全集：拒答准确率
# ===========================================================================

REFUSAL_MARKERS = [
    "未找到相关内容",
    "没有找到相关内容",
    "知识库中未收录",
    "无法回答",
    "没有相关信息",
    "未收录",
]


def _is_refusal(answer: str) -> bool:
    """判断答案是否为拒答"""
    if not answer:
        return True
    return any(marker in answer for marker in REFUSAL_MARKERS)


def eval_safety(pipeline_name: str, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    安全集评测：拒答准确率。

    样本分两类：
        知识库中确实不存在答案的问题（如询问其它标准的内容、编造的标准号）
            -> 期望拒答
        知识库中存在答案的正常问题（对照样本）
            -> 期望正常作答

    拒答准确率 = 两类样本判定正确的比例。
    这一指标直接验证 N2（防幻觉）是否落到实处 ——
    在标准合规场景中，一个页码正确但内容编造的答案，比明确的「未找到」危害大得多。
    """
    from backend.rag_pipeline import get_pipeline_by_name

    pipeline = get_pipeline_by_name(pipeline_name)
    per_sample: List[Dict[str, Any]] = []

    for sample in samples:
        question = sample["question"]
        should_refuse = bool(sample.get("should_refuse", True))

        try:
            result = pipeline.answer(question, use_cache=False)
            answer = result.get("answer", "")
            refused = _is_refusal(answer)
            correct = (refused == should_refuse)
        except Exception as exc:
            logger.exception("安全评测失败：%s | %s", sample.get("id"), exc)
            refused, correct, answer = False, False, f"<异常：{exc}>"

        per_sample.append({
            "id": sample.get("id"),
            "question": question,
            "should_refuse": should_refuse,
            "actual_refuse": refused,
            "correct": correct,
            "answer": answer[:150],
            "category": sample.get("category", ""),
        })
        logger.info(
            "[安全] %s | 期望拒答=%s | 实际拒答=%s | %s",
            sample.get("id"), should_refuse, refused,
            "✓" if correct else "✗",
        )

    correct_count = sum(1 for s in per_sample if s["correct"])
    accuracy = correct_count / len(per_sample) if per_sample else 0.0

    return {
        "summary": {
            "refusal_accuracy": round(accuracy, 4),
            "count": len(per_sample),
            "correct": correct_count,
        },
        "samples": per_sample,
    }


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 评测：检索 / 生成 / 安全三类样本")
    parser.add_argument("--pipeline", default="v1", choices=["v1", "v2", "v3"],
                        help="要评测的链路版本")
    parser.add_argument("--tag", default=None,
                        help="结果文件标签，默认使用链路版本名")
    parser.add_argument("--dataset", default=None, help="自定义评测集路径")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="跳过 RAGAS 忠实度评测（不调用裁判模型，节省费用）")
    parser.add_argument("--only", default=None,
                        choices=["retrieval", "generation", "safety"],
                        help="只评测其中一类样本")
    args = parser.parse_args()

    setup_logging()
    settings.ensure_directories()

    dataset = load_dataset(Path(args.dataset) if args.dataset else None)
    tag = args.tag or args.pipeline
    started = time.time()

    logger.info("=" * 68)
    logger.info("开始评测 | 链路=%s | 标签=%s", args.pipeline, tag)
    logger.info("=" * 68)

    report: Dict[str, Any] = {
        "meta": {
            "pipeline": args.pipeline,
            "tag": tag,
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": DATASET_FILE,
            "top_k_list": settings.top_k_list,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }
    }

    run_all = args.only is None

    if run_all or args.only == "retrieval":
        samples = dataset.get("retrieval", [])
        logger.info("检索集评测开始，共 %d 条样本", len(samples))
        report["retrieval"] = eval_retrieval(args.pipeline, samples)

    if run_all or args.only == "generation":
        samples = dataset.get("generation", [])
        logger.info("生成集评测开始，共 %d 条样本", len(samples))
        report["generation"] = eval_generation(
            args.pipeline, samples, use_ragas=not args.skip_ragas)

    if run_all or args.only == "safety":
        samples = dataset.get("safety", [])
        logger.info("安全集评测开始，共 %d 条样本", len(samples))
        report["safety"] = eval_safety(args.pipeline, samples)

    elapsed = round(time.time() - started, 1)
    report["meta"]["elapsed_seconds"] = elapsed

    out_file = settings.eval_result_path / f"eval_{tag}.json"
    out_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 控制台汇总 ----
    logger.info("=" * 68)
    logger.info("评测完成 | 耗时 %.1f 秒 | 结果已写入 %s", elapsed, out_file.name)
    r = report.get("retrieval", {}).get("summary", {})
    g = report.get("generation", {}).get("summary", {})
    s = report.get("safety", {}).get("summary", {})
    logger.info("  检索指标  : %s", {k: v for k, v in r.items()})
    logger.info("  生成指标  : faithfulness=%s", g.get("faithfulness"))
    logger.info("  安全指标  : refusal_accuracy=%s", s.get("refusal_accuracy"))
    logger.info("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
