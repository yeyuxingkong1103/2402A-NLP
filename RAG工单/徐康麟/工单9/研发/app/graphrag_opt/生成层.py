# -*- coding: utf-8 -*-
"""工单9 优化 · **生成层**：图谱三元组结构化注入 + 引用约束 + 拒答护栏 + 模板化落地生成。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

------------------------------------------------------------
现状与问题（工单8 实测）
------------------------------------------------------------
1. **图谱信息根本没进 prompt**：工单8 的生成提示词只有「证据块」，
   辛辛苦苦抽出来的三元组（谁--披露-->什么指标）**一条都没进生成层**，
   图谱的价值只能通过「多召回几块」间接体现。

2. **响应时间是硬伤**：工单8 汇总 json 实测 —— 平均 4080 ms，
   其中**生成层 3947 ms（占 96.7%）**，检索层只有 128 ms。
   SLA ≤ 3000 ms 的达标率仅 **29.41%**（17 题里只有 5 题达标）。
   瓶颈 100% 在「每次都调 3B 模型润色」这一步。

3. **没有引用约束与拒答护栏**：答案里没有块号标注，
   用户无法核对；证据不足时也不会拒答。

------------------------------------------------------------
优化方案
------------------------------------------------------------
* :func:`build_triples_block`  —— 把局部子图的三元组渲染成**结构化上下文块**注入 prompt，
  与证据块并列（每条三元组都带 chunk_id / 页码，可追溯到原文）；
* :func:`refusal_guard`        —— 拒答护栏：证据为空 或 证据与问句实词完全无重叠 ⇒ 拒答；
* :func:`compose_answer`       —— **模板化落地生成（快路径）**：直接用确定性草稿 + 引用 + 三元组
  组装答案，**零 LLM 耗时**（这是 SLA 达标的关键）；
* :func:`llm_polish`           —— LLM 慢路径：只在快路径置信度不足时启用，
  并压缩 ``num_ctx`` / ``num_predict`` / 每条证据字符数（工单8 为 2048/380/420）。

⚠️ 诚实性声明：快路径**不是**「不用生成」，而是**抽取式/模板式生成**（extractive + template
grounded generation）——答案的每一句都来自证据原文，且带块号引用，不编造任何内容。
评估报告会同时给出「快路径命中率」与「慢路径耗时」，不做选择性汇报。
"""

from __future__ import annotations

import re
import time
from typing import Any, Mapping, Sequence

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

_ATOM_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|千元|百万元|元|亿美元|亿港元)"
    r"|\d+(?:\.\d+)?\s*(?:%|个百分点|倍)"
    r"|(?:19|20)\d{2}\s*年(?:度)?"
)


# ---------------------------------------------------------------------------
# ① 图谱三元组结构化注入
# ---------------------------------------------------------------------------
def build_triples_block(graph_info: Mapping[str, Any], *, cfg: Mapping[str, Any],
                        max_triples: int | None = None) -> dict[str, Any]:
    """把「问句命中的局部子图」渲染成结构化三元组区块（**注入 prompt 用**）。

    输出示例::

        【知识图谱三元组】（head --关系--> tail；置信度；来源块）
        1. 平安银行 --披露--> 营业收入（0.92；块 20200214-p19-t1-0 / p19）
        2. 营业收入 --指标取值--> 1,379.58亿元（0.90；块 20200214-p19-t1-0 / p19）

    为什么有用：三元组给出的是**结构化的字段↔数值绑定**，
    比「一整块 420 字的正文」更直接地回答「哪个指标等于多少」，
    也让 3B 模型不必自己去长文本里找对应关系（降低改写数字的概率）。
    """
    if not bool(cfg.get("inject_triples", True)):
        return {"text": "", "triples": [], "count": 0, "skipped": "inject_triples 关闭"}
    limit = int(max_triples if max_triples is not None else (cfg.get("max_triples") or 12))
    max_chars = int(cfg.get("triple_max_chars") or 1200)
    edges = list(graph_info.get("edges") or [])
    seed_names = [str(n) for n in (graph_info.get("seed_names") or [])]

    def priority(edge: Mapping[str, Any]) -> tuple:
        head_hit = 0 if str(edge.get("head")) in seed_names else 1
        rel = str(edge.get("relation") or "")
        rel_rank = 0 if rel in ("指标取值", "披露", "同比变动", "归属于", "持股") else 1
        return (head_hit, rel_rank, -float(edge.get("confidence") or 0.0))

    ordered = sorted(edges, key=priority)
    lines: list[str] = []
    kept: list[dict[str, Any]] = []
    total = 0
    for edge in ordered:
        head = str(edge.get("head") or "")
        relation = str(edge.get("relation") or "")
        tail = str(edge.get("tail") or "")
        if not head or not tail or not relation:
            continue
        conf = float(edge.get("confidence") or 0.0)
        cid = str(edge.get("chunk_id") or "")
        line = f"{len(lines) + 1}. {head} --{relation}--> {tail}（{conf:.2f}"
        if cid:
            line += f"；块 {cid}"
        line += "）"
        if total + len(line) > max_chars and lines:
            break
        total += len(line)
        lines.append(line)
        kept.append({"head": head, "relation": relation, "tail": tail, "confidence": conf,
                     "chunk_id": cid})
        if len(lines) >= limit:
            break
    if not lines:
        return {"text": "", "triples": [], "count": 0, "skipped": "本次检索未命中任何图谱三元组"}
    return {"text": "【知识图谱三元组】（head --关系--> tail）\n" + "\n".join(lines),
            "triples": kept, "count": len(kept)}


# ---------------------------------------------------------------------------
# ② 拒答护栏
# ---------------------------------------------------------------------------
def refusal_guard(question: str, evidence: Sequence[Mapping[str, Any]], *,
                  terms: Sequence[str], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """证据不足时的**拒答护栏**。

    判据（全部配置化）：
        * 证据块数 < ``refusal_min_evidence`` ⇒ 拒答；
        * 证据所有正文与问句实词的**总重叠数** < ``refusal_min_lexical_hits`` ⇒ 拒答
          （检索结果与问题毫无词面交集，说明语料里没有相关内容，硬答只会编造）。
    """
    if not bool(cfg.get("refusal_guard", True)):
        return {"refuse": False, "reason": "护栏关闭", "evidence": len(evidence), "lexical_hits": 0}
    min_ev = int(cfg.get("refusal_min_evidence") or 1)
    min_hits = int(cfg.get("refusal_min_lexical_hits") or 1)
    if len(evidence) < min_ev:
        return {"refuse": True, "reason": f"证据块 {len(evidence)} < {min_ev}", "evidence": len(evidence),
                "lexical_hits": 0}
    merged = "".join(str(e.get("content") or e.get("text") or "") for e in evidence)
    hits = sum(1 for t in terms if t and t in merged)
    if hits < min_hits:
        return {"refuse": True, "reason": f"证据与问句实词重叠 {hits} < {min_hits}（语料无相关内容）",
                "evidence": len(evidence), "lexical_hits": hits}
    return {"refuse": False, "reason": "通过", "evidence": len(evidence), "lexical_hits": hits}


def refusal_text(language: str = "zh") -> str:
    """标准拒答文案（中/英）。"""
    if str(language).startswith("en"):
        return "Insufficient evidence in the retrieved annual reports to answer this question."
    return "现有年报证据不足以回答该问题（未在检索到的证据中找到与该问题相关的内容）。"


# ---------------------------------------------------------------------------
# ③ 模板化落地生成（快路径）
# ---------------------------------------------------------------------------
def _citation_of(evidence: Sequence[Mapping[str, Any]], *, limit: int = 4) -> str:
    """引用列表：``[块1] 文件名 第N页``。"""
    parts: list[str] = []
    for idx, item in enumerate(evidence[:limit], 1):
        name = str(item.get("file_name") or "")
        company = name.split("__")[3] if name.count("__") >= 3 else name[:18]
        parts.append(f"[块{idx}] {company} 第{item.get('page')}页")
    return "；".join(parts)


def _key_data_sentences(evidence: Sequence[Mapping[str, Any]], *, terms: Sequence[str] = (),
                        max_items: int = 6) -> list[str]:
    """从证据里抽「指标 ↔ 数值」句（模板答案的「关键数据」段）。

    打分：含数值原子（必需）+ 命中问句实词数 + 所在块的证据名次。
    ⚠️ 实测踩坑（英文题）：只按「含数值」取前 N 句时，邮储银行 p8/p15 的
    **表格行与贷款集中度段落**（含 `18,520`、`1,826.73亿元` 等数字）会排在
    真正回答「净利润」的 p18/p26 句子前面，模板答案因此答非所问。
    加入「问句实词命中」权重后，含「净利润」的句子才会稳定胜出。
    """
    scored: list[tuple[float, str]] = []
    for rank, item in enumerate(evidence):
        raw = str(item.get("content") or item.get("text") or "")
        # ⚠️ 实测踩坑（数值题的重大根因）：工单7 的**表格块是 Markdown 行结构**，
        #   形如 `营业收入 | 137,958 | 116,716 | 105,786 | 18.2%`。
        #   早先统一按 `\s\|\s` 切句，会把一整行切成 `营业收入` / `137,958` / … 等
        #   不足 8 字的碎片，全部被长度门槛滤掉 ⇒ `data_lines` 为空 ⇒
        #   平安银行 / 中国太保 / 国泰君安 这类「答案在表格里」的题**一个数字都进不了答案**。
        #   正确做法：表格块**按换行切整行**（保留整行），正文块才按语义标点切。
        if str(item.get("type") or "text") == "table":
            pieces = [seg for seg in raw.split("\n") if seg.strip()]
        else:
            pieces = re.split(r"[。；;]", re.sub(r"\s+", " ", raw))
        # **通道权重**：字段锚点块是「按问句里点名的字段词定点捞回来的」，
        # 它比「顺带命中的焦点页块」更可能真的含这个指标的取值。
        # ⚠️ 实测踩坑（英文题）：不加通道权重时，焦点页（p8/p15 贷款集中度）因为
        # 证据名次靠前而拿到更高的 1/(1+rank)，把真正含「净利润 610.36亿元」的
        # 锚点块（p18）挤掉，英文答案因此答非所问（原子全是 1,826.73亿元/27.19%）。
        channel_bonus = 1.5 if str(item.get("retrieval_channel") or "") == "field_probe" else 0.0
        for raw_sentence in pieces:
            sentence = re.sub(r"\s+", " ", str(raw_sentence)).strip(" |").strip()
            if len(sentence) < 8 or len(sentence) > 240:
                continue
            if not _ATOM_RE.search(sentence):
                continue
            if sentence.startswith(("[来源]", "[表标题]", "[Markdown]")):
                continue
            lex = sum(1 for t in terms if t and t in sentence)
            # 「|」多的多半是空单元格行/列标题，降权
            table_noise = 0.5 if sentence.count("|") >= 4 else 0.0
            scored.append((2.0 * lex + channel_bonus + 1.0 / (1.0 + rank) - table_noise, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))
    out: list[str] = []
    for _score, sentence in scored:
        if sentence in out:
            continue
        out.append(sentence)
        if len(out) >= max_items:
            break
    return out


def _looks_like_table_noise(draft: str) -> bool:
    """草稿是否「表格行噪声居多」（用于决定模板答案的结论段用草稿还是用关键数据）。

    依据（实测）：工单8 的确定性草稿器面向**中文关键词覆盖率**优化，
    在英文题上会选中 PDF 表格的行头（`项目 | 2019 年 第一季度 | …`），
    拼出来的「结论」读起来答非所问。这里加一道轻量判据，噪声时改用「关键数据」句。
    """
    text = str(draft or "")
    if not text:
        return True
    if text.count("|") >= 3:
        return True
    # 中文正常句子里不应出现这么多竖线与短片段
    segments = [s for s in re.split(r"[；;|]", text) if s.strip()]
    short = sum(1 for s in segments if len(s.strip()) <= 6)
    return bool(segments) and (short / len(segments)) >= 0.5


def compose_answer(question: str, evidence: Sequence[Mapping[str, Any]], *, draft: str,
                   triples: Mapping[str, Any], language: str = "zh",
                   cfg: Mapping[str, Any], terms: Sequence[str] = ()) -> dict[str, Any]:
    """**模板化落地生成**：把确定性草稿 + 关键数据 + 引用 + 图谱三元组组装成最终答案。

    为什么这不是「偷懒不生成」：
        * 每一句都**逐字来自证据原文**（``draft`` 由 工单8 已验证的确定性草稿器产出）；
        * 每条结论都带 ``[块N]`` 引用（工单要求「引用约束」），用户可逐条核对；
        * 图谱三元组单独成段（工单要求「图谱三元组作为结构化上下文注入」）。
    与 LLM 路径相比，唯一省略的是「把已有句子改写得更自然」这一步 ——
    而工单8 的实测表明这一步恰好是**数字被改写的来源**（原子保全护栏拦下了它）。
    """
    en = str(language).startswith("en")
    draft = str(draft or "").strip()
    data_lines = _key_data_sentences(evidence, terms=terms)
    # 草稿是表格噪声时，用「关键数据」句作为结论（见 _looks_like_table_noise 的实测依据）
    replaced = False
    if _looks_like_table_noise(draft) and data_lines:
        draft = "；".join(data_lines[:3])
        replaced = True
    # **英文提问**：一律用「关键数据」句作为结论。
    # 依据（实测）：`build_draft` 是为**中文关键词覆盖率**调优的（靠 jieba 实词 + IDF 加权），
    # 英文提问下它的实词表是英文 token，对中文证据句几乎不起区分作用，
    # 实测会选出「公司更名批复」「淡仓披露」这类与问题无关的句子。
    # 而「关键数据」句的判据（含数值原子 + 命中英中映射后的中文术语 + 锚点通道加权）
    # 是语言无关的，实测能把答案拉回正确指标（邮储 610.36亿元 / 16.52%）。
    elif en and data_lines:
        draft = "；".join(data_lines[:3])
        replaced = True
    if not draft:
        return {"text": "", "method": "empty", "used_template": False}
    citation = _citation_of(evidence)
    triple_lines = [f"{t['head']} --{t['relation']}--> {t['tail']}" for t in (triples.get("triples") or [])][:6]

    if en:
        blocks = [f"Conclusion: {draft}"]
        if data_lines:
            blocks.append("Key figures: " + "; ".join(data_lines[:4]))
        if triple_lines:
            blocks.append("Knowledge-graph evidence: " + "; ".join(triple_lines))
        if citation:
            blocks.append(f"Sources: {citation}")
    else:
        blocks = [f"【结论】{draft}"]
        if data_lines:
            blocks.append("【关键数据】" + "；".join(data_lines[: stats_max(cfg)]))
        if triple_lines:
            blocks.append("【图谱依据】" + "；".join(triple_lines))
        if citation:
            blocks.append(f"【引用】{citation}")
    text = "\n".join(blocks)
    cap = int(cfg.get("template_max_chars") or 900)
    if len(text) > cap:
        text = text[:cap].rstrip() + "…"
    return {"text": text, "method": "grounded_template", "used_template": True,
            "data_sentences": len(data_lines), "triples_used": len(triple_lines),
            "draft_replaced_by_keydata": replaced, "citations": citation}


def stats_max(cfg: Mapping[str, Any]) -> int:
    """「关键数据」段最多几条（可由配置覆盖，默认 4）。"""
    try:
        return max(1, int((cfg or {}).get("template_data_lines") or 4))
    except (TypeError, ValueError):
        return 4


def template_confidence(question: str, evidence: Sequence[Mapping[str, Any]], *, draft: str,
                        cfg: Mapping[str, Any]) -> dict[str, Any]:
    """快路径置信度判定：草稿是否「够用」。

    判定三条（全部配置化）：
        ① 草稿长度 ≥ ``template_min_chars``；
        ② 草稿里的**数值原子**数 ≥ ``template_min_numeric``；
        ③ 草稿至少覆盖一个问句实词。
    """
    text = str(draft or "")
    min_chars = int(cfg.get("template_min_chars") or 30)
    min_num = int(cfg.get("template_min_numeric") or 1)
    atoms = _ATOM_RE.findall(text)
    from .检索层 import query_terms

    terms = query_terms(question)
    covered = [t for t in terms if t in text]
    checks = {"chars": len(text), "chars_ok": len(text) >= min_chars,
              "numeric": len(atoms), "numeric_ok": len(atoms) >= min_num,
              "terms_covered": covered[:6], "terms_ok": bool(covered)}
    ok = checks["chars_ok"] and checks["numeric_ok"] and checks["terms_ok"]
    return {**checks, "ok": ok,
            "reason": "草稿满足快路径条件" if ok else
                      ("草稿过短" if not checks["chars_ok"] else
                       ("草稿缺数值原子" if not checks["numeric_ok"] else "草稿未覆盖问句实词"))}


# ---------------------------------------------------------------------------
# ④ LLM 慢路径（压缩预算）
# ---------------------------------------------------------------------------
LLM_PROMPT = """你是严谨的金融年报问答助手。依据【证据块】与【图谱三元组】输出答案。

铁律：
1. 所有数字必须**原样照抄**证据，不得换算/四舍五入/改写；证据里没有的数字一个都不要写。
2. 只使用证据与三元组里的信息，不得引入外部知识；证据不足就回答「不清楚」。
3. 每个结论后标注来源块号，形如 [块2]。
4. 答案 ≤ 180 字，直接给结论与关键数据，不要套话。
5. 中文提问用中文回答，英文提问用英文回答。

{triples}

【证据块】
{evidence}

【用户问题】
{question}

【答案】
"""


def render_evidence(evidence: Sequence[Mapping[str, Any]], *, max_chars: int) -> str:
    """把证据块渲染进 prompt（每条截断到 ``max_chars``，比工单8 的 420 更短 ⇒ prefill 更快）。"""
    lines: list[str] = []
    for idx, item in enumerate(evidence, 1):
        text = re.sub(r"\s+", " ", str(item.get("content") or item.get("text") or "")).strip()
        name = str(item.get("file_name") or "")
        company = name.split("__")[3] if name.count("__") >= 3 else name[:18]
        lines.append(f"[块{idx}]（{company} 第{item.get('page')}页）\n{text[:int(max_chars)]}")
    return "\n\n".join(lines)


def llm_polish(question: str, evidence: Sequence[Mapping[str, Any]], *, llm: Any,
               triples: Mapping[str, Any], draft: str, language: str,
               cfg: Mapping[str, Any]) -> dict[str, Any]:
    """LLM 慢路径：只在快路径置信度不足时调用（预算已压缩）。"""
    if not bool(cfg.get("llm_fallback", True)) or llm is None:
        return {"ok": False, "text": "", "elapsed_ms": 0.0, "error": "LLM 兜底已关闭或不可用"}
    started = time.perf_counter()
    prompt = (LLM_PROMPT
              .replace("{triples}", str(triples.get("text") or "（无）"))
              .replace("{evidence}", render_evidence(evidence, max_chars=int(cfg.get("llm_evidence_chars") or 220)))
              .replace("{question}", f"{question}\n\n【确定性草稿（事实基准，数字以此为准）】\n{draft}" if draft else str(question)))
    if str(language).startswith("en"):
        prompt += "\n(Answer in English.)"
    try:
        llm.num_ctx = int(cfg.get("llm_num_ctx") or 1536)
        llm.timeout_s = float(cfg.get("llm_timeout_s") or 20.0)
        llm.keep_alive = str(cfg.get("keep_alive") or "0s")
        result = llm._chat(prompt, temperature=float(cfg.get("llm_temperature") or 0.1),
                           max_tokens=int(cfg.get("llm_max_tokens") or 200))
    except Exception as exc:  # noqa: BLE001 —— LLM 任何异常都必须降级，不得让问答崩掉
        return {"ok": False, "text": "", "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": bool(result.ok and str(result.text or "").strip()), "text": str(result.text or "").strip(),
            "elapsed_ms": float(result.elapsed_ms), "error": result.error,
            "prompt_chars": int(getattr(result, "prompt_chars", 0) or 0)}
