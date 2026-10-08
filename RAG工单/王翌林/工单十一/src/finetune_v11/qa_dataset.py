# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Embedding模型微调任务
src/finetune_v11/qa_dataset.py —— 工单十一 微调问答对数据集生成（纯函数，可单测）

流程（对应工单"数据集生成"要求）：
  1) 从 data/ccf_reports/chunks/*_chunks.json 读取金融年报语料 chunk；
  2) 按文档分层抽样（保证 9 份年报均匀覆盖）；
  3) 为每个 chunk 构造 LLM prompt，生成"可由该 chunk 回答"的金融问题；
  4) 解析 LLM 输出为问题列表，组成 (query, positive_chunk) 正例对；
  5) 固定随机种子切分 train/dev，保证可复现。
数据集格式：正例对（query, positive）——适配对比损失/多负例排序损失。
"""
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

WORK_ORDER = "人工智能NLP-RAG-Embedding模型微调任务"

QA_GEN_SYSTEM = (
    "你是金融领域的问答数据构造专家。给定一段上市公司年报/招股说明书文本，"
    "生成能够仅依据该文本回答的中文金融问题。只输出 JSON 数组字符串，"
    "不要输出任何其他内容。"
)

QA_GEN_TEMPLATE = """请基于以下金融文本生成 {n} 个问题：
要求：
1. 问题必须是金融领域的专业问题（营收/净利润/股息/保费/资产/风险指标/业务结构等）；
2. 每个问题都能仅依据该文本回答；
3. 问题之间不重复，表述自然多样（可含数字追问、同比变化、构成明细等角度）；
4. 只输出 JSON 数组，例如 ["问题1","问题2"]。

文本：
\"\"\"
{text}
\"\"\""""


def build_qa_prompt(chunk_text: str, n_questions: int = 2,
                    max_chars: int = 1200) -> List[Dict[str, str]]:
    """工单十一：构造问答对生成 prompt（超长截断，控制 token 成本）"""
    text = chunk_text[:max_chars]
    return [{"role": "system", "content": QA_GEN_SYSTEM},
            {"role": "user", "content": QA_GEN_TEMPLATE.format(n=n_questions, text=text)}]


def parse_questions(content: str) -> List[str]:
    """工单十一：解析 LLM 输出为问题列表（容忍 ```json 围栏与编号列表兜底）"""
    if not content:
        return []
    text = content.strip()
    # 去掉 markdown 代码围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    # 优先按 JSON 数组解析
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(q).strip() for q in arr
                    if isinstance(q, str) and len(q.strip()) >= 6]
        except json.JSONDecodeError:
            pass
    # 兜底：按行解析（"1. xxx" / "- xxx"）
    questions = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:\d+[.、)]|[-*])\s*", "", line).strip().strip('"')
        if len(line) >= 6 and ("?" in line or "？" in line or line.endswith("。")):
            questions.append(line)
    return questions


def iter_chunks(chunks_dir: Path,
                min_chars: int = 200) -> Iterable[Dict[str, Any]]:
    """工单十一：遍历 chunks 目录，产出过滤后的 chunk（doc_id/page/chunk_id/text）"""
    for fp in sorted(Path(chunks_dir).glob("*_chunks.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        for c in data.get("chunks", []):
            text = (c.get("text") or "").strip()
            if len(text) < min_chars:
                continue
            yield {"chunk_id": c.get("chunk_id") or f"{data.get('doc_id')}_{c.get('global_index')}",
                   "doc_id": c.get("doc_id") or data.get("doc_id"),
                   "page": c.get("page"),
                   "text": text}


def sample_chunks(chunks: List[Dict[str, Any]], per_doc: int = 24,
                  seed: int = 42) -> List[Dict[str, Any]]:
    """工单十一：按 doc_id 分层抽样（每份年报至多 per_doc 个，保证领域覆盖均匀）"""
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    rng = random.Random(seed)
    sampled: List[Dict[str, Any]] = []
    for doc_id in sorted(by_doc):
        pool = by_doc[doc_id]
        rng.shuffle(pool)
        sampled.extend(pool[:per_doc])
    rng.shuffle(sampled)
    return sampled


def make_pair(question: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
    """工单十一：(query, positive) 正例对记录"""
    return {"query": question, "positive": chunk["text"],
            "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
            "page": chunk["page"], "work_order": WORK_ORDER}


def split_train_dev(pairs: List[Dict[str, Any]], dev_ratio: float = 0.1,
                    seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """工单十一：按 chunk_id 去重切分（同一 chunk 的问题不跨 train/dev，防泄漏）"""
    by_chunk: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        by_chunk.setdefault(p["chunk_id"], []).append(p)
    chunk_ids = sorted(by_chunk)
    rng = random.Random(seed)
    rng.shuffle(chunk_ids)
    n_dev = max(1, round(len(chunk_ids) * dev_ratio))
    dev_ids = set(chunk_ids[:n_dev])
    train = [p for cid in chunk_ids[n_dev:] for p in by_chunk[cid]]
    dev = [p for cid in chunk_ids[:n_dev] for p in by_chunk[cid]]
    return train, dev


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in
            Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
