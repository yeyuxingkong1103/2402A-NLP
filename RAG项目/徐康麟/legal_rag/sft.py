"""SFT 数据构造与校验（纯函数，便于单测；不依赖 torch/GPU）。

三条硬规矩，都来自本项目实测踩过的坑：

1. **训练集绝不能与评测集重叠** —— 否则 v4 的分数是假的。做法：问题归一去重 + 与
   ``eval/qa_set.jsonl`` 做精确与近似（difflib 相似度）双重比对，命中就丢。
   ⚠️ **已知局限**：这是**词面**比对，纯改写（「商标侵权如何认定」vs「什么行为构成商标侵权」）
   拦不住；对策是再加一道**语义闸门**（用 bge-m3 向量近似，云端能做），见 ``docs/FINETUNE.md``。
2. **引用必须能在给定证据里找到** —— 答案里出现的每个法名、每个 ``第X条`` 都必须出现在
   证据（检索到的条文）里，否则就是**教模型编条文**，这是法律 RAG 最不能犯的错。
3. **证据只用语料** —— 生成时把"检索到的条文"当唯一依据，答案必须能被证据支撑（同上校验）。

数据格式（OpenAI messages，与 vLLM/transformers 的 chat template 对齐）::

    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}],
     "evidence": ["中华人民共和国商标法 第五十七条 ..."], "source": "...", "meta": {...}}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

#: 归一化时剥掉的空白与常见标点（全角/半角都算）
_PUNCT_RE = re.compile(r"[\s，。、；：？！,.;:?!~`'\"“”‘’（）()《》〈〉\[\]【】—\-…]+")

#: 法名引用：``《中华人民共和国商标法》`` / ``《民法典》``
CITATION_RE = re.compile(r"《([^》]{2,40})》")
#: 条号：``第五十七条`` / ``第57条``
ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千零〇两0-9]+条")

#: 默认相似度门槛（超过就认为"与评测集太像"，丢）
DEFAULT_OVERLAP_THRESHOLD = 0.85


@dataclass
class SftSample:
    """一条 SFT 样本：``question`` 问、``answer`` 答、``evidence`` 是**唯一**依据。"""

    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)
    source: str = ""
    meta: dict = field(default_factory=dict)

    def to_chat_record(self, system_prompt: str = "") -> dict:
        return {
            "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                        + [{"role": "user", "content": self.question},
                           {"role": "assistant", "content": self.answer}],
            "evidence": list(self.evidence),
            "source": self.source,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SftSample":
        return cls(
            question=str(payload.get("question") or ""),
            answer=str(payload.get("answer") or ""),
            evidence=[str(x) for x in (payload.get("evidence") or [])],
            source=str(payload.get("source") or ""),
            meta=dict(payload.get("meta") or {}),
        )


def normalize_question(text: str) -> str:
    """问题归一化：去空白与标点、全半角统一（用于查重与重叠检测）。"""
    return _PUNCT_RE.sub("", str(text or "")).replace("２", "2").lower()


def similarity(left: str, right: str) -> float:
    """归一化后的相似度（0~1）。短问题用精确比对，长问题用序列相似度。"""
    a, b = normalize_question(left), normalize_question(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def overlap_with_eval(questions: list[str], eval_questions: list[str],
                      threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> list[dict]:
    """返回与评测集重叠的问题（``{"question", "eval_question", "ratio"}``）。

    **这是防"自欺欺人"的闸门**：训练集一旦混进评测题，v4 涨分就毫无意义。
    """
    hits: list[dict] = []
    for question in questions:
        for eval_question in eval_questions:
            ratio = similarity(question, eval_question)
            if ratio >= threshold:
                hits.append({"question": question, "eval_question": eval_question,
                             "ratio": round(ratio, 4)})
                break
    return hits


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = sum(a * a for a in left) ** 0.5
    norm_r = sum(b * b for b in right) ** 0.5
    return dot / (norm_l * norm_r) if norm_l and norm_r else 0.0


def semantic_overlap(questions: list[str], eval_questions: list[str], embed,
                     threshold: float = 0.92) -> list[dict]:
    """**语义**闸门：用向量近邻抓"纯改写"型泄漏（词面闸门抓不到的那种）。

    为什么需要（词面闸门的已知局限）：「商标侵权如何认定」与「什么行为构成商标侵权」
    词面相似度只有 ~0.6，但语义几乎相同 —— 这种泄漏会让 v4 看起来"变聪明了"。

    ``embed`` 是注入的嵌入函数（``list[str] -> list[list[float]]``），因此本函数**纯逻辑、可单测**，
    真实调用时才接 bge-m3。门槛 0.92 偏保守：宁可漏报（人工抽检兜底），不误杀有效样本。
    """
    if not questions or not eval_questions:
        return []
    vectors = embed(list(questions) + list(eval_questions))
    if len(vectors) != len(questions) + len(eval_questions):
        raise ValueError("embed 返回的向量条数与输入不符")
    train_vectors = vectors[: len(questions)]
    eval_vectors = vectors[len(questions):]
    hits: list[dict] = []
    for question, vector in zip(questions, train_vectors):
        best_ratio, best_match = 0.0, ""
        for eval_question, eval_vector in zip(eval_questions, eval_vectors):
            ratio = _cosine(vector, eval_vector)
            if ratio > best_ratio:
                best_ratio, best_match = ratio, eval_question
        if best_ratio >= threshold:
            hits.append({"question": question, "eval_question": best_match,
                         "ratio": round(best_ratio, 4)})
    return hits


def citation_problems(answer: str, evidence: list[str]) -> list[str]:
    """答案里的引用是否都能在证据里找到；返回问题列表（空 = 全部有据）。

    两条都查：
    * 法名（``《…》``）必须出现在证据文本里（防止"引一部没给的法规"）；
    * 条号（``第X条``）必须出现在证据文本里（防止"引一条没给的条文"—— 这是最要命的）。
    """
    blob = "\n".join(evidence or [])
    problems: list[str] = []
    for name in {m.group(1).strip() for m in CITATION_RE.finditer(answer or "")}:
        if name and name not in blob:
            problems.append(f"法名不在证据里：{name}")
    for article in {m.group(0) for m in ARTICLE_RE.finditer(answer or "")}:
        if article and article not in blob:
            problems.append(f"条号不在证据里：{article}")
    return problems


def dedupe(samples: list[SftSample],
           threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> tuple[list[SftSample], int]:
    """按问题去重（含近似）；返回 ``(保留, 丢掉条数)``，保持原顺序。"""
    kept: list[SftSample] = []
    dropped = 0
    for sample in samples:
        if any(similarity(sample.question, item.question) >= threshold for item in kept):
            dropped += 1
            continue
        kept.append(sample)
    return kept, dropped


def split_train_val(samples: list[SftSample], *, val_ratio: float = 0.05,
                    seed: int = 20260923) -> tuple[list[SftSample], list[SftSample]]:
    """确定性划分（同 seed 同结果），val 至少 1 条（样本足够时）。"""
    import random

    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    if len(ordered) < 2 or val_ratio <= 0:
        return ordered, []
    val_size = max(1, int(len(ordered) * val_ratio))
    return ordered[val_size:], ordered[:val_size]


def mask_prompt_labels(labels: list[int], prompt_len: int,
                       ignore_index: int = -100) -> list[int]:
    """把前 ``prompt_len`` 个 token 的 label 置为 ``ignore_index``（只在答案上学）。

    为什么必须掩：不掩就是在训练模型"复述用户问题 + 系统提示"，白学还伤效果。
    """
    if prompt_len <= 0:
        return list(labels)
    return [ignore_index] * min(prompt_len, len(labels)) + list(labels)[prompt_len:]


#: 提示词里各分区的标题（与 `legal_rag/generate/prompt.py` 保持一致；改那边要同步改这里）
_BLOCK_HEADERS = ("【检索知识】", "【用户长期记忆】", "【历史对话】", "【用户问题】")


def trim_evidence(messages: list[dict], keep_chars: int) -> tuple[list[dict], int]:
    """只裁**证据块**，保留记忆/历史/问题与答案。返回 (新 messages, 丢掉的字数)。

    为什么必须这么裁：`encode_chat` 原来是**右截断**（``full_ids[:max_len]``），而提示
    长、答案在尾 —— 提示一超长，答案就被切光。2026-09-28 实测：`max_len=2048` 而提示
    平均 3032 token，**912 条里有 514 条只剩 ≤1 个监督 token**，等于没训。
    证据是"可裁"的部分（按相关性排序，留头部最相关的），问题和答案不可裁。
    """
    if not messages or not messages[-1].get("content"):
        return messages, 0
    # 取**最后一条** user：few-shot 示例也是 user，取错了就会裁示例而不是裁证据
    index = next((i for i in range(len(messages) - 1, -1, -1)
                  if messages[i].get("role") == "user"), None)
    if index is None:
        return messages, 0
    content = str(messages[index].get("content") or "")
    if "【检索知识】" not in content:
        return messages, 0
    start = content.find("【检索知识】")
    # 证据块的结束 = 它之后第一个其他分区标题（记忆/历史/问题）；没有就到最后
    end = len(content)
    for header in _BLOCK_HEADERS[1:]:
        position = content.find(header, start + len("【检索知识】"))
        if position != -1:
            end = min(end, position)
    context = content[start:end]
    if keep_chars >= len(context):
        return messages, 0
    trimmed = messages[:]
    trimmed[index] = {**messages[index],
                      "content": content[:start] + context[:keep_chars] + content[end:]}
    return trimmed, len(context) - keep_chars


def fit_evidence_to_budget(messages: list[dict], tokenizer, *, max_len: int,
                           answer_tokens: int, render=None) -> tuple[list[dict], int]:
    """二分找出**最大可留的证据字数**，使 提示+答案 ≤ max_len。返回 (messages, 丢掉字数)。

    二分而不是精确解：token 数对字符数不是线性的（中英混排），逐字符试代价太大。
    20 次左右的 tokenize 换一次准确落位，构建期完全可以接受。
    """
    if render is None:
        def render(msgs: list[dict]) -> str:
            return tokenizer.apply_chat_template(msgs[:-1], tokenize=False,
                                                add_generation_prompt=True)

    budget = max_len - answer_tokens
    if budget <= 0:
        return messages, 0

    def _prompt_len(msgs: list[dict]) -> int:
        return len(tokenizer(render(msgs), add_special_tokens=False)["input_ids"])

    if _prompt_len(messages) <= budget:
        return messages, 0
    low, high = 0, 1 << 20
    best: tuple[list[dict], int] = (messages, 0)
    for _ in range(20):
        if low > high:
            break
        middle = (low + high) // 2
        candidate, dropped = trim_evidence(messages, middle)
        if _prompt_len(candidate) <= budget:
            best = (candidate, dropped)
            low = middle + 1
        else:
            high = middle - 1
    return best


def encode_chat(messages: list[dict], tokenizer, *, max_len: int = 2048,
                ignore_index: int = -100) -> dict:
    """把一条对话编成 ``{"input_ids", "labels", "supervision"}``：只对最后一条 assistant 算 loss。

    截断策略（2026-09-28 改，之前的右截断把答案切光了）：

    1. 先按原样渲染；装得下就直接用。
    2. 装不下 → **只裁证据块**（`trim_evidence` + 二分），问题和答案原样保留。
    3. 还装不下（答案本身太长/没有证据块可裁）→ 才退回左截断提示，并如实在
       ``supervision["left_truncated_prompt"]`` 里标出来。
    """
    if not messages:
        raise ValueError("messages 为空")
    last = messages[-1]
    if last.get("role") != "assistant":
        raise ValueError("最后一条必须是 assistant（否则没有可学的目标）")
    eos = getattr(tokenizer, "eos_token", "") or ""
    answer_ids = list(tokenizer(f"{last.get('content') or ''}{eos}",
                                add_special_tokens=False)["input_ids"])

    def _render_prompt_of(msgs: list[dict]) -> str:
        return tokenizer.apply_chat_template(msgs[:-1], tokenize=False,
                                            add_generation_prompt=True)

    prompt_text = _render_prompt_of(messages)
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    dropped_context_chars = 0
    if len(prompt_ids) + len(answer_ids) > max_len:
        messages, dropped_context_chars = fit_evidence_to_budget(
            messages, tokenizer, max_len=max_len, answer_tokens=len(answer_ids))
        prompt_text = _render_prompt_of(messages)
        prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    # 兜底：答案优先。提示左截断（丢掉最前面的系统/证据），答案尽量完整。
    answer_kept = answer_ids
    left_truncated = False
    if len(prompt_ids) + len(answer_ids) > max_len:
        left_truncated = True
        keep_prompt = max(1, max_len - len(answer_ids))
        if len(answer_ids) > max_len - 1:
            answer_kept = answer_ids[:max_len - 1]
            keep_prompt = 1
        prompt_ids = prompt_ids[-keep_prompt:]
    full_ids = prompt_ids + answer_kept
    prompt_len = len(prompt_ids)
    labels = mask_prompt_labels(full_ids, prompt_len, ignore_index)
    supervised = sum(1 for label in labels if label != ignore_index)
    return {"input_ids": full_ids,
            "labels": labels,
            "supervision": {"prompt_tokens": prompt_len,
                            "answer_tokens": len(answer_ids),
                            "supervised_tokens": supervised,
                            "dropped_context_chars": dropped_context_chars,
                            "left_truncated_prompt": left_truncated}}


def supervision_summary(rows: Iterable[dict]) -> dict:
    """汇总监督信号是否被截断吃掉（**训练前必须看**，不然就是白烧 GPU）。"""
    items = [row.get("supervision") or {} for row in rows]
    if not items:
        return {"rows": 0}
    supervised = [int(item.get("supervised_tokens") or 0) for item in items]
    dropped = sum(int(item.get("dropped_context_chars") or 0) for item in items)
    starved = sum(1 for value in supervised if value <= 1)
    trimmed = sum(1 for item in items if int(item.get("dropped_context_chars") or 0) > 0)
    left_cut = sum(1 for item in items if item.get("left_truncated_prompt"))
    ordered = sorted(supervised)

    def _at(q: float) -> int:
        position = (len(ordered) - 1) * q
        return ordered[int(position)]

    return {
        "rows": len(items),
        "supervised_mean": round(sum(supervised) / len(supervised), 1),
        "supervised_median": _at(0.5),
        "supervised_p05": _at(0.05),
        "starved_rows": starved,
        "starved_share": round(starved / len(items), 4),
        "prompt_trimmed_rows": trimmed,
        "prompt_trimmed_share": round(trimmed / len(items), 4),
        "left_truncated_rows": left_cut,
        "dropped_context_chars_total": dropped,
    }


def dump_jsonl(records: list[dict], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n"：Windows 上默认会把 \n 翻成 \r\n，而仓库/云端都按 LF 处理
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def load_jsonl(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def eval_questions(path: str | Path) -> list[str]:
    """读评测题集里的问题（含多轮的每一轮，多轮问题同样不能进训练集）。"""
    questions: list[str] = []
    for payload in load_jsonl(path):
        turns = payload.get("turns") or []
        if turns:
            questions.extend(str(turn) for turn in turns)
        elif payload.get("question"):
            questions.append(str(payload["question"]))
    return questions


# ---------------------------------------------------------------- 证据抽取与教师回复解析

def to_inference_messages(sample: "SftSample", *, role_id: str = "",
                          history: list["Message"] | None = None) -> list[dict]:
    """把一条样本渲染成**与线上推理同形**的 messages（system + 检索知识 + 用户问题 → 答案）。

    ⚠️ 为什么必须这样（2026-09-28 真机查出的事故）：原来的 `to_chat_record()` 只写
    ``[user: 问题, assistant: 答案]``，而 **`evidence` 是独立字段、不进 messages**
    （`train_lora.py` 只渲染 messages）⇒ 那 796 条训练样本里**一条都没有把法条放进提示**。
    结果模型学的是"**凭记忆答法律题**"，与"只依据给定条文作答"的目标**正好相反** ——
    这一次事故同时解释了两件事：① 上一轮 LoRA 通过率 ±0（要学的技能根本没在输入里）；
    ② `must_refuse` 反而退化（没有证据也照样学着作答）。

    做法上**直接复用线上那套构造函数**（`build_messages` / `build_context_block` /
    `build_question_block`），把证据包成 `Chunk`/`SearchHit` 喂进去 —— 这样提示格式
    **不可能**与推理漂移（自己另写一份格式，早晚要漂）。
    """
    from .generate.prompt import build_messages
    from .roles import DEFAULT_ROLE_ID, get_role
    from .schemas import Chunk, SearchHit

    role = None
    role_key = role_id or DEFAULT_ROLE_ID
    try:
        role = get_role(role_key)
    except Exception:                                        # noqa: BLE001 - 角色表拿不到也不该炸
        role = None

    hits = [SearchHit(chunk=Chunk(id=f"sft-{index}", text=str(text), source=sample.source or ""),
                      score=1.0)
            for index, text in enumerate(sample.evidence)]
    return build_messages(role, hits, history, sample.question)


def evidence_blocks(text: str, *, min_chars: int = 120, limit: int = 0) -> list[str]:
    """把一份语料切成"证据块"（给教师模型当唯一依据）。

    跳过 crawler 写的元数据头（``# 标题`` 与 ``- 字段：值``），其余按空行分段，
    只留长度 >= ``min_chars`` 的段（太短的没有可问的信息），可选 ``limit`` 截断。
    """
    blocks: list[str] = []
    current: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        if not current and (line.startswith("#") or line.startswith("- ")):
            continue                      # 元数据头
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    kept = [block for block in blocks if len(block) >= min_chars]
    return kept[:limit] if limit > 0 else kept


def parse_teacher_reply(text: str) -> tuple[str, str]:
    """从教师回复里抠出 ``(question, answer)``。

    优先按 JSON 解析（``{"question":…, "answer":…}``，允许被 ```json 包裹）；
    退一步认"问题：/回答："分栏。都拿不到就抛 ``ValueError``（**不许静默用原文当答案**）。
    """
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("教师回复为空")
    candidate = raw
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if 0 <= start < end:
        try:
            payload = json.loads(candidate[start:end + 1])
            question = str(payload.get("question") or "").strip()
            answer = str(payload.get("answer") or "").strip()
            if question and answer:
                return question, answer
        except (json.JSONDecodeError, TypeError):
            pass
    question_match = re.search(r"(?:问题|问)\s*[：:]\s*(.+)", raw)
    answer_match = re.search(r"(?:回答|答)\s*[：:]\s*(.+)", raw, re.DOTALL)
    if question_match and answer_match:
        question = question_match.group(1).strip()
        answer = answer_match.group(1).strip()
        if question and answer:
            return question, answer
    raise ValueError("教师回复里找不到 question/answer")
