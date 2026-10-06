# -*- coding: utf-8 -*-
"""
工单05 公共工具模块（供 build_index / multi_turn_chat / ablation / evaluation 复用）
工单编号：人工智能NLP-RAG-Query理解优化任务

职责：
  1. 统一路径（项目根目录、结果目录）与结果落盘（JSON / Markdown）
  2. 封装 Pipeline 的构造（工单05 预设 wo05_multiturn，可一键关闭 Query 理解做消融）
  3. 定义 5 轮对话的「判定标准」（评测口径），供消融实验与 RAG 评估共用，
     避免把标准散落在多个脚本里导致口径不一致
  4. 提供带容错的多轮执行器 run_script()：单轮失败不中断整个脚本
     （工单验收项「系统稳定性：高可用性、容错机制」）

说明：
  · 本模块不修改 rag_core 任何代码，只做调用与结果整理；
  · 判定标准是「最小必要证据」口径（关键信息点是否出现），
    若需要更精确的参考答案，可在 results/ground_truth.json 中覆盖
    （格式见 load_ground_truth 的注释）。
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

# --- 路径处理：把项目根目录（工单作业/）加入 sys.path，保证能 import rag_core ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core import config                      # noqa: E402
from rag_core.pipeline import PRESETS, Pipeline  # noqa: E402
from rag_core.query_understand import Turn       # noqa: E402

# --- 工单目录与结果目录 ---
WO_DIR = Path(__file__).resolve().parents[1]          # 工单05-Query理解优化-多轮对话/
SRC_DIR = WO_DIR / "src"
RESULT_DIR = WO_DIR / "results"
DOCS_DIR = WO_DIR / "docs"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# --- 数据源（两份招股说明书：多轮对话第 4/5 轮需要跨文档） ---
PDF_PATHS = [config.PDF_PROSPECTUS_1, config.PDF_PROSPECTUS_2]
DOC_NAMES = ["招股说明书1", "招股说明书2"]
COLLECTION = "prospectus"                 # 与 rag_core.api 服务共用的集合名

# 答案中表示「没找到」的标记（与 rag_core.generator 的拒答话术一致）
REFUSAL_MARKERS = ("未能找到", "was not found")

# 检索/答案判定用的时间格式（招股书报告期均为 20xx 年）
_YEAR_PAT = re.compile(r"20\d{2}")


# ---------------------------------------------------------------------------
# 5 轮对话的判定标准（评测口径）
# ---------------------------------------------------------------------------
# 每轮定义：
#   expected_doc      该轮应该命中的文档（跨文档检索的核心证据）
#   entity            该轮真正要问的主体（指代消解的目标）
#   aspect_keywords   检索片段必须覆盖的考察点（缺了说明检索跑偏）
#   answer_keywords   答案必须包含的关键信息点（最小充分证据）
#   min_years         答案中至少出现的年份个数（数值类问题的防漏检）
#   exam              该轮考察的 Query 理解能力（写入文档/报告）
TURN_SPECS: list[dict] = [
    {
        "turn": 1, "qid": 1,
        "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "expected_doc": "招股说明书1",
        "entity": "武汉兴图新科电子股份有限公司",
        "aspect_keywords": ["军用领域"],
        "answer_keywords": ["军用"],
        "min_years": 2,
        "exam": "多轮起点：自包含问题（无指代），用于建立对话主体",
    },
    {
        "turn": 2, "qid": 2,
        "question": "他参与的哪个工程荣获了国家科技进步一等奖？",
        "expected_doc": "招股说明书1",
        "entity": "武汉兴图新科电子股份有限公司",
        "aspect_keywords": ["科技进步一等奖"],
        "answer_keywords": ["科技进步一等奖"],
        "min_years": 0,
        "exam": "指代消解：人称代词「他」→ 上一轮主体（跨句消解）",
    },
    {
        "turn": 3, "qid": 3,
        "question": "这个公司的法定代表人是谁？",
        "expected_doc": "招股说明书1",
        "entity": "武汉兴图新科电子股份有限公司",
        "aspect_keywords": ["法定代表人"],
        "answer_keywords": ["法定代表人"],
        "min_years": 0,
        "exam": "指代消解 + 话题继承：「这个公司」延续主体，问点换成「法定代表人」",
    },
    {
        "turn": 4, "qid": 4,
        "question": "那武汉力源信息技术股份有限公司呢？",
        "expected_doc": "招股说明书2",
        "entity": "武汉力源信息技术股份有限公司",
        "aspect_keywords": ["法定代表人"],
        "answer_keywords": ["法定代表人"],
        "min_years": 0,
        "exam": "省略句补全 + 主体切换：换成新公司，同时沿用上一轮的问点（最难一轮）",
    },
    {
        "turn": 5, "qid": 5,
        "question": "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
        "expected_doc": "招股说明书2",
        "entity": "武汉力源信息技术股份有限公司",
        "aspect_keywords": ["销售处"],
        "answer_keywords": ["销售处"],
        "min_years": 0,
        "exam": "主体延续 + 跨文档检索：跳转到《招股说明书2》的组织结构图（图像块）",
    },
]

# 问题 → 判定标准 的索引，供各脚本按轮次取用
SPEC_BY_TURN = {s["turn"]: s for s in TURN_SPECS}


def load_ground_truth() -> dict:
    """
    载入人工参考答案（可选）。

    若 results/ground_truth.json 存在，则用它覆盖默认的「最小必要证据」口径，
    文件格式（键为轮次）：
        {
          "1": {"answer_keywords": ["2018年", "2019年"], "min_years": 3,
                "reference_answer": "报告期内军用领域收入分别为 …（全文）"},
          ...
        }
    reference_answer 会作为 evaluate.EvalRecord.ground_truth 用于 RAGAS 风格评估。
    文件不存在时返回空字典（使用内置判定标准）。
    """
    path = RESULT_DIR / "ground_truth.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:                                   # 容错：坏文件不影响主流程
        print(f"  [warn] 参考答案文件解析失败，改用内置判定标准：{e}")
        return {}


def merged_spec(turn: int, ground_truth: dict | None = None) -> dict:
    """把人工参考答案合并进内置判定标准（人工优先）。"""
    spec = dict(SPEC_BY_TURN[turn])
    gt = (ground_truth or {}).get(str(turn)) or (ground_truth or {}).get(turn)
    if isinstance(gt, dict):
        for k, v in gt.items():
            if k == "reference_answer":
                spec["reference_answer"] = v
            elif v not in (None, "", []):
                spec[k] = v
    return spec


# ---------------------------------------------------------------------------
# Pipeline 构造
# ---------------------------------------------------------------------------
def build_pipeline(
    use_query_understanding: bool = True,
    reranker: str | None = None,
    collection: str = COLLECTION,
    top_k: int | None = None,
) -> Pipeline:
    """
    构造工单05 流水线。

    Args:
        use_query_understanding: False 时关闭 Query 理解（消融实验的对照组），
                                 此时 pipeline 直接拿原始问题去检索，不做指代消解。
        reranker: 覆盖预设重排器（none/tfidf/llm/adaptive/cascade）。
                  默认用 wo05_multiturn 预设的 cascade（精度优先）；
                  若需要严格满足 3 秒约束，可传 tfidf（省一次 LLM 调用）。
        top_k: 送入生成的片段数。
    """
    cfg = PRESETS["wo05_multiturn"]
    cfg = replace(cfg, use_query_understanding=use_query_understanding)
    if reranker:
        cfg = replace(cfg, reranker=reranker)
    if top_k:
        cfg = replace(cfg, top_k=top_k)
    p = Pipeline(cfg, collection=collection)
    p.load_index()                       # 装载 BM25 索引；向量库检索时按需读取
    return p


def index_ready(collection: str = COLLECTION) -> bool:
    """检查索引是否已建立（BM25 落盘文件存在即视为就绪）。"""
    return (config.INDEX_DIR / "bm25.pkl").exists()


# ---------------------------------------------------------------------------
# 判定函数（检索层 / 答案层）
# ---------------------------------------------------------------------------
def is_refusal(answer: str) -> bool:
    """答案是否为「未找到」的拒答话术。"""
    return (not answer) or any(m in answer for m in REFUSAL_MARKERS)


def count_years(text: str) -> int:
    """统计文本中出现的 20xx 年份个数（数值类问题用来防「只答一年」）。"""
    return len(set(_YEAR_PAT.findall(text or "")))


def judge_retrieval(docs: list[dict], spec: dict) -> dict:
    """
    检索层判定（不依赖 LLM，客观可复现）：
      · 期望文档命中：top-k 片段里是否有片段来自该轮应有的文档（跨文档证据）
      · 考察点覆盖：期望文档的片段中是否覆盖了本轮考察点关键词
      · 实体出现：片段中是否出现了本轮真正要问的公司实体
    """
    exp = spec["expected_doc"]
    kws = spec.get("aspect_keywords", [])
    hit_docs = [d for d in (docs or []) if exp in str(d.get("doc", ""))]
    aspect_hit = any(
        all(k in (d.get("text") or "") for k in kws) for d in hit_docs
    ) if kws else bool(hit_docs)
    entity_hit = any(spec["entity"] in (d.get("text") or "") for d in (docs or []))
    return {
        "期望文档命中": bool(hit_docs),
        "考察点覆盖": bool(aspect_hit),
        "实体出现": bool(entity_hit),
        "通过": bool(hit_docs) and bool(aspect_hit),
    }


def judge_answer(answer: str, spec: dict) -> dict:
    """
    答案层判定（「最小必要证据」口径）：
      · 非拒答：答案不是「未能找到…」
      · 关键信息点：answer_keywords 全部出现在答案中
      · 年份数量：数值类问题要求出现足够多的年份
    全部满足记为「正确」，用于统计多轮问答准确率。
    """
    text = answer or ""
    missing = [k for k in spec.get("answer_keywords", []) if k not in text]
    years = count_years(text)
    years_ok = years >= int(spec.get("min_years", 0))
    ok = (not is_refusal(text)) and (not missing) and years_ok
    return {
        "非拒答": not is_refusal(text),
        "关键信息点": not missing,
        "漏答信息点": missing,
        "年份数量": years,
        "年份达标": years_ok,
        "正确": bool(ok),
    }


def retrieval_stats(docs: list[dict]) -> dict:
    """统计检索结果中的文档分布，用于发现「跨文档串味」问题。"""
    dist: dict[str, int] = {}
    for d in docs or []:
        name = str(d.get("doc", "未知"))
        dist[name] = dist.get(name, 0) + 1
    return dist


# ---------------------------------------------------------------------------
# 多轮执行器（带容错）
# ---------------------------------------------------------------------------
def run_script(
    pipeline: Pipeline,
    questions: list[str],
    top_k_show: int = 3,
    verbose: bool = True,
) -> list[dict]:
    """
    按顺序跑完多轮问题，自动维护对话历史（rag_core.query_understand.Turn）。

    单轮异常不会中断整个脚本：记录 error 字段后继续下一轮，
    下一轮的历史仍然保留（答案为空），满足「容错机制」验收项。

    Returns:
        每轮的完整记录（问题 / 理解结果 / 检索片段 / 答案 / 耗时 / 异常）
    """
    history: list[Turn] = []
    records: list[dict] = []

    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        rec: dict = {"turn": i, "question": q, "error": None}
        try:
            trace = pipeline.ask(q, history=history, return_trace=True)
            rec["understanding"] = trace.get("understanding", {})
            rec["rewritten"] = rec["understanding"].get("改写后", q)
            rec["intent"] = rec["understanding"].get("意图", "")
            rec["entities"] = rec["understanding"].get("实体", [])
            rec["sub_questions"] = rec["understanding"].get("子问题", [])
            rec["answer"] = trace.get("answer", "")
            rec["docs"] = [
                {
                    "doc": d.get("doc"), "page": d.get("page"),
                    "type": d.get("type"), "section": d.get("section"),
                    "score": round(float(d.get("final_score", d.get("score", 0.0))), 4),
                    "text": (d.get("text") or "")[:400],
                }
                for d in (trace.get("docs") or [])[:top_k_show]
            ]
            rec["timings"] = {k: round(v, 3) for k, v in (trace.get("timings") or {}).items()}
            history.append(Turn(question=q, answer=rec["answer"],
                                docs=trace.get("docs") or []))
        except Exception as e:                                # 容错：记录并继续
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["answer"] = ""
            rec["docs"] = []
            history.append(Turn(question=q, answer=""))

        rec["latency"] = round(time.perf_counter() - t0, 3)
        records.append(rec)

        if verbose:
            _print_turn(rec, top_k_show)
    return records


def _print_turn(rec: dict, top_k_show: int) -> None:
    """按工单要求打印一轮的完整链路信息。"""
    print(f"\n{'=' * 78}")
    print(f"第 {rec['turn']} 轮")
    print(f"Q（原始问题）：{rec['question']}")
    if rec.get("error"):
        print(f"  [错误] 本轮处理失败（已容错跳过）：{rec['error']}")
        return
    print(f"  · 识别意图：{rec.get('intent', '')}")
    if rec.get("rewritten") and rec["rewritten"] != rec["question"]:
        print(f"  · 指代消解 → 检索式：{rec['rewritten']}")
    else:
        print(f"  · 检索式：{rec['question']}（无需改写）")
    if rec.get("entities"):
        print(f"  · 识别实体：{'、'.join(rec['entities'])}")
    if len(rec.get("sub_questions") or []) > 1:
        print(f"  · 子问题分解：{rec['sub_questions']}")

    print(f"  · 检索片段 Top{top_k_show}：")
    for j, d in enumerate(rec.get("docs", [])[:top_k_show], 1):
        snippet = (d.get("text") or "").replace("\n", " ")[:70]
        print(f"      [{j}] 《{d.get('doc')}》第{d.get('page')}页 "
              f"({d.get('type')}, 分数 {d.get('score')})：{snippet}…")

    ans = (rec.get("answer") or "").replace("\n", " ")
    print(f"  · 生成答案：{ans[:300]}{'…' if len(ans) > 300 else ''}")
    print(f"  · 本轮耗时：{rec['latency']:.3f}s（3 秒约束："
          f"{'满足' if rec['latency'] <= 3 else '超出'}）")


# ---------------------------------------------------------------------------
# 结果落盘与格式化
# ---------------------------------------------------------------------------
def write_json(path: Path, obj) -> Path:
    """写 JSON（UTF-8，中文不转义），自动建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_md(path: Path, text: str) -> Path:
    """写 Markdown（UTF-8）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def render_turn_md(rec: dict, spec: dict | None = None) -> str:
    """把一轮记录渲染成 Markdown 片段（多轮对话记录 / 消融实验共用）。"""
    lines = [f"### 第 {rec['turn']} 轮", ""]
    lines.append(f"- **原始问题**：{rec['question']}")
    if spec:
        lines.append(f"- **考察能力**：{spec.get('exam', '')}")
    if rec.get("error"):
        lines.append(f"- **异常**：`{rec['error']}`（已容错，脚本继续执行）")
        return "\n".join(lines) + "\n"

    lines.append(f"- **识别意图**：{rec.get('intent', '')}")
    if rec.get("rewritten") and rec["rewritten"] != rec["question"]:
        lines.append(f"- **指代消解后的检索式**：`{rec['rewritten']}`")
    else:
        lines.append(f"- **检索式**：`{rec['question']}`（无需改写）")
    lines.append(f"- **识别实体**：{'、'.join(rec.get('entities') or []) or '（无）'}")
    if len(rec.get("sub_questions") or []) > 1:
        lines.append(f"- **子问题分解**：{'；'.join(rec['sub_questions'])}")

    lines.append("- **检索片段（前 3）**：")
    for j, d in enumerate((rec.get("docs") or [])[:3], 1):
        snippet = (d.get("text") or "").replace("\n", " ")[:90]
        lines.append(f"  {j}. 《{d.get('doc')}》第 {d.get('page')} 页 "
                     f"（{d.get('type')}，分数 {d.get('score')}）：{snippet}…")
    lines.append(f"- **生成答案**：{(rec.get('answer') or '').strip()}")
    lines.append(f"- **本轮耗时**：{rec.get('latency', 0):.3f}s"
                 f"（3 秒约束：{'满足' if rec.get('latency', 9) <= 3 else '超出'}）")
    lines.append("")
    return "\n".join(lines)


def latency_summary(records: list[dict]) -> dict:
    """统计多轮耗时（平均/最大/满足 3 秒约束的轮次比例）。"""
    lats = [r.get("latency", 0.0) for r in records]
    if not lats:
        return {"轮数": 0}
    ok = sum(1 for x in lats if x <= 3.0)
    return {
        "轮数": len(lats),
        "平均耗时(s)": round(sum(lats) / len(lats), 3),
        "最大耗时(s)": round(max(lats), 3),
        "满足3秒约束轮次": ok,
        "3秒达标率": round(ok / len(lats), 4),
    }


def print_llm_usage(prefix: str = "") -> None:
    """打印 LLM 用量（调用次数 / 缓存命中 / token / 预估费用）。"""
    try:
        from rag_core import llm
        u = llm.get_usage()
        print(f"\n{prefix}LLM 调用 {u['calls']} 次（缓存命中 {u['cached']} 次）| "
              f"输入 {u['prompt_tokens']} tokens / 输出 {u['completion_tokens']} tokens | "
              f"预估费用 ￥{u['cost_estimate_cny']}")
        return u
    except Exception as e:                                    # 统计失败不影响主流程
        print(f"{prefix}[warn] LLM 用量统计失败：{e}")
        return {}


def check_data_files() -> None:
    """启动前检查 PDF 素材是否存在，给出明确的中文提示。"""
    missing = [str(p) for p in PDF_PATHS if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "未找到招股说明书 PDF，请确认工单附件路径：\n  " + "\n  ".join(missing)
        )
