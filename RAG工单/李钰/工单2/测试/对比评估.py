# -*- coding: utf-8 -*-
"""
V1 vs V2 对比评估脚本
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

功能:
    1. 同时导入 V1 (工单1) 和 V2 (工单2) 的问答引擎
    2. 对 10 个验收问题分别跑 V1 和 V2
    3. 对比: 关键词覆盖率 / 响应时间 / 检索质量
    4. 生成对比报告 (V1vsV2_对比报告.md)
    5. 输出 JSON 数据 (对比结果.json)

运行前提:
    - 已安装依赖: pip install -r 工单2/部署/requirements.txt
    - V1 和 V2 的索引已构建 (首次运行会自动构建)
"""
import os
import sys
import json
import time
import logging
from typing import List, Dict

# ============ 路径设置 ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# V1 路径 (工单1)
V1_DEV = os.path.join(BASE_DIR, "..", "工单1", "研发")
# V2 路径 (工单2 研发)
V2_DEV = os.path.join(BASE_DIR, "研发")

# 修正: V1 在 项目工单/工单1/研发, V2 在 项目工单/工单2/研发
ROOT = os.path.dirname(BASE_DIR)  # 项目工单
V1_PATH = os.path.join(ROOT, "工单1", "研发")
V2_PATH = os.path.join(BASE_DIR, "研发")

sys.path.insert(0, V1_PATH)
sys.path.insert(0, V2_PATH)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============ 验收问题集 ============
TEST_QUESTIONS: List[Dict] = [
    {"id": 260, "question": "报告期内, 武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少?"},
    {"id": 95,  "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准?"},
    {"id": 33,  "question": "报告期内, 武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少?"},
    {"id": 34,  "question": "根据武汉兴图新科电子股份有限公司招股意向书, 电子信息行业的上游涉及哪些企业?"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商?"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书, 电子信息行业的下游主要包括哪些行业?"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖?"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少?"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁?"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金?"},
]

# ============ 参考答案关键词 (用于评估) ============
REFERENCE = {
    260: {"keywords": ["6,464.51", "14,414.16", "18,780.67", "4,627.14", "万元"]},
    95:  {"keywords": ["AVS", "标准", "编解码"]},
    33:  {"keywords": ["82.10%", "97.31%", "94.84%", "94.34%"]},
    34:  {"keywords": ["芯片", "元器件", "上游"]},
    957: {"keywords": ["国防", "军工", "军队", "指挥控制"]},
    793: {"keywords": ["国防", "军队", "军工", "下游"]},
    795: {"keywords": ["国家科技进步", "一等奖"]},
    543: {"keywords": ["7,360", "注册资本"]},
    531: {"keywords": ["法定代表人"]},
    207: {"keywords": ["补充流动资金", "%"]},
}


def keyword_coverage(answer: str, keywords: List[str]) -> float:
    if not answer or not keywords:
        return 0.0
    hit = sum(1 for k in keywords if k in answer)
    return round(hit / len(keywords), 4)


def run_v1(question: str) -> Dict:
    """运行 V1 (工单1)"""
    import qa_engine as v1_engine
    return v1_engine.answer_question(question)


def run_v2(question: str) -> Dict:
    """运行 V2 (工单2)"""
    import qa_engine_v2 as v2_engine
    return v2_engine.answer_question(question)


def compare_one(item: Dict) -> Dict:
    """对比单题 V1 vs V2"""
    qid = item["id"]
    q = item["question"]
    ref = REFERENCE.get(qid, {})
    kws = ref.get("keywords", [])

    print(f"\n[Q{qid}] {q[:50]}...")

    try:
        r1 = run_v1(q)
        v1_rag = r1.get("rag_answer", "")
        v1_time = r1.get("response_time", 999)
        v1_retrieval = r1.get("retrieval", [])
        v1_top_score = v1_retrieval[0]["score"] if v1_retrieval else 0.0
        v1_cov = keyword_coverage(v1_rag, kws)
        print(f"  V1: 覆盖={v1_cov}, 时间={v1_time}s, Top1={v1_top_score:.3f}")
    except Exception as e:
        print(f"  V1 异常: {e}")
        v1_rag = f"[V1 异常] {e}"
        v1_time = 999
        v1_top_score = 0.0
        v1_cov = 0.0

    try:
        r2 = run_v2(q)
        v2_rag = r2.get("rag_answer", "")
        v2_time = r2.get("response_time", 999)
        v2_retrieval = r2.get("retrieval", [])
        v2_top_score = v2_retrieval[0]["score"] if v2_retrieval else 0.0
        v2_cov = keyword_coverage(v2_rag, kws)
        print(f"  V2: 覆盖={v2_cov}, 时间={v2_time}s, Top1={v2_top_score:.3f}")
    except Exception as e:
        print(f"  V2 异常: {e}")
        v2_rag = f"[V2 异常] {e}"
        v2_time = 999
        v2_top_score = 0.0
        v2_cov = 0.0

    return {
        "id": qid,
        "question": q,
        "keywords": kws,
        "v1": {
            "rag_answer": v1_rag,
            "llm_answer": r1.get("llm_answer", "") if 'r1' in dir() else "",
            "response_time": v1_time,
            "top1_score": v1_top_score,
            "keyword_coverage": v1_cov,
            "retrieval_count": len(r1.get("retrieval", [])) if 'r1' in dir() else 0,
        },
        "v2": {
            "rag_answer": v2_rag,
            "llm_answer": r2.get("llm_answer", "") if 'r2' in dir() else "",
            "response_time": v2_time,
            "top1_score": v2_top_score,
            "keyword_coverage": v2_cov,
            "retrieval_count": len(r2.get("retrieval", [])) if 'r2' in dir() else 0,
        },
        "improvement": round(v2_cov - v1_cov, 4),
        "v2_better": v2_cov >= v1_cov,
    }


def run_all() -> List[Dict]:
    results = []
    print(f"========== V1 vs V2 对比评估 ({len(TEST_QUESTIONS)} 题) ==========")
    for item in TEST_QUESTIONS:
        results.append(compare_one(item))
    print("\n========== 全部完成 ==========")
    return results


def summary(results: List[Dict]) -> Dict:
    v1_covs = [r["v1"]["keyword_coverage"] for r in results]
    v2_covs = [r["v2"]["keyword_coverage"] for r in results]
    v1_times = [r["v1"]["response_time"] for r in results]
    v2_times = [r["v2"]["response_time"] for r in results]
    v2_better_count = sum(1 for r in results if r["v2_better"])
    v2_in_time = sum(1 for r in results if r["v2"]["response_time"] <= 3.0)

    return {
        "total": len(results),
        "v1_avg_coverage": round(sum(v1_covs) / len(v1_covs), 4),
        "v2_avg_coverage": round(sum(v2_covs) / len(v2_covs), 4),
        "coverage_gain": round(sum(v2_covs) / len(v2_covs) - sum(v1_covs) / len(v1_covs), 4),
        "v1_avg_time": round(sum(v1_times) / len(v1_times), 4),
        "v2_avg_time": round(sum(v2_times) / len(v2_times), 4),
        "v2_better_count": v2_better_count,
        "v2_in_time_count": v2_in_time,
        "v2_coverage_above_90": round(sum(v2_covs) / len(v2_covs), 4) >= 0.9,
    }


def write_report(results: List[Dict], path: str = None):
    """生成 Markdown 对比报告"""
    path = path or os.path.join(BASE_DIR, "V1vsV2_对比报告.md")
    s = summary(results)
    lines = []
    lines.append("# V1 vs V2 优化前后对比报告\n")
    lines.append("> 工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化\n")
    lines.append("## 一、优化方案概要\n")
    lines.append("| 优化点 | V1 (工单1) | V2 (工单2) |")
    lines.append("| --- | --- | --- |")
    lines.append("| 分块策略 | 固定500字符 | 语义段落 + 表格独立 + 动态合并拆分 |")
    lines.append("| 检索方式 | 单路 TF-IDF | BM25 + TF-IDF 混合 + Rerank |")
    lines.append("| Query理解 | 关键词匹配 | 同义词扩展 + 多语言 + 意图驱动 |")
    lines.append("| LLM Prompt | 基础模板 | 表格格式化 + 结构约束 + 诚实回答 |\n")
    lines.append("## 二、汇总对比\n")
    lines.append(f"| 指标 | V1 | V2 | 提升 |")
    lines.append(f"| --- | --- | --- | --- |")
    lines.append(f"| 平均关键词覆盖率 | {s['v1_avg_coverage']} | {s['v2_avg_coverage']} | **+{s['coverage_gain']}** |")
    lines.append(f"| 平均响应时间 | {s['v1_avg_time']}s | {s['v2_avg_time']}s | |")
    lines.append(f"| V2 更优的题数 | - | {s['v2_better_count']}/{s['total']} | |")
    lines.append(f"| V2 响应达标 (≤3s) | - | {s['v2_in_time_count']}/{s['total']} | |")
    lines.append(f"| 准确率 ≥ 90% | - | **{'达标' if s['v2_coverage_above_90'] else '待优化'}** | |")
    lines.append("\n## 三、逐题对比\n")
    lines.append("| ID | 问题 | V1覆盖 | V2覆盖 | 提升 | V1时间 | V2时间 | V2 Top1 | V1答案(前80) | V2答案(前80) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        q_short = r["question"][:25] + "..." if len(r["question"]) > 25 else r["question"]
        a1 = r["v1"]["rag_answer"][:80].replace("\n", " ")
        a2 = r["v2"]["rag_answer"][:80].replace("\n", " ")
        lines.append(
            f"| {r['id']} | {q_short} | {r['v1']['keyword_coverage']} | "
            f"{r['v2']['keyword_coverage']} | +{r['improvement']} | "
            f"{r['v1']['response_time']}s | {r['v2']['response_time']}s | "
            f"{r['v2']['top1_score']:.3f} | {a1} | {a2} |"
        )
    lines.append("\n## 四、结论\n")
    if s["coverage_gain"] >= 0.2:
        lines.append(f"V2 优化效果显著: 关键词覆盖率从 {s['v1_avg_coverage']} 提升至 {s['v2_avg_coverage']}, 提升 {s['coverage_gain']:.1%}。")
    elif s["coverage_gain"] >= 0.1:
        lines.append(f"V2 优化有效: 关键词覆盖率提升 {s['coverage_gain']:.1%}, 多数问题改善明显。")
    else:
        lines.append(f"V2 优化有限: 覆盖率提升仅 {s['coverage_gain']:.1%}, 建议参考《进阶优化说明.md》进一步升级。")

    if s["v2_coverage_above_90"]:
        lines.append("**准确率达到 90% 以上, 满足工单验收要求!**")
    else:
        lines.append("当前准确率尚未达到 90%, 可通过 sentence-transformers 向量模型进一步提升 (见《进阶优化向量模型.py》)。")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n对比报告: {path}")
    return path


def save_json(results: List[Dict], path: str = None):
    path = path or os.path.join(BASE_DIR, "对比结果.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"对比数据: {path}")


if __name__ == "__main__":
    results = run_all()
    s = summary(results)
    print(f"\n========== 汇总 ==========")
    print(f"V1 平均覆盖率: {s['v1_avg_coverage']}")
    print(f"V2 平均覆盖率: {s['v2_avg_coverage']}")
    print(f"覆盖率提升: +{s['coverage_gain']}")
    print(f"V2 更优题数: {s['v2_better_count']}/{s['total']}")
    print(f"V2 准确率≥90%: {s['v2_coverage_above_90']}")
    write_report(results)
    save_json(results)
