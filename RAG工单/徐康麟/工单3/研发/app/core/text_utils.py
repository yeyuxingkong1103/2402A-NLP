# -*- coding: utf-8 -*-
"""工单3 文本/数字/分词/命中判定工具（设计/接口设计.md §3.4 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

要点：
    * ``normalize_cell`` 落实 R1（去 ``\\n``、折叠空白、中文分段拼接）与「``[ ◆ ]`` → 未披露」占位符规则；
    * ``strip_printed_page`` 同时清洗 **两种** 印刷页码形态：PDF1 页脚 ``1-1-N``、
      PDF2 页眉第 2 行裸数字（实测 350/350 页满足 ``值 == 物理页 - 1``），
      但**只用「位置 + 值」双判据**，绝不无脑删正文里的独立数字行；
    * ``evidence_contains`` 是**命中判定的唯一实现**（T5/T8/T9/T11 必须调用它）。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .errors import RagError  # noqa: F401 —— 便于上层统一 except RagError

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 未披露占位符：合并单元格值可能是符号而非 None（实测 PDF2 物理 2/22/24/27/306 共 5 页 22 处）
UNDISCLOSED = "未披露"
_SYMBOL_CELL = re.compile(r"^\[\s*[◆◇○●○※*＊×xX/—\-–]+\s*\]$|^[◆◇※]+$")
# 占位符可能**夹在单位文本里**（实测：`[ ◆ ]元`、`[ ◆ ]倍(…)`、`[ ◆ ]万元`、`[ ◆ ]年[ ◆ ]月[ ◆ ]日`）
_PLACEHOLDER = re.compile(r"\[\s*[◆◇※]+\s*\]|[◆◇※]")
# 连续「占位符+单位」组合（年月日时分）：折叠为 未披露（年/月/日）
_CONSEC_PLACEHOLDER = re.compile(r"(?:未披露(?:年|月|日|时|分)){2,}")

# 停用词（中文为主，够用即可；不做无谓的巨型词表）
STOPWORDS: frozenset[str] = frozenset(
    """
    的 了 和 与 及 或 在 是 为 对 从 到 有 无 不 也 都 而 并 且 就 被 把 将 由 于 以 之 其 该 此 这 那 我们 公司
    本公司 本公司 上述 其中 因此 由于 根据 按照 相关 有关 进行 具有 通过 以及 方面 情况 时候 可以 需要 应当
    应该 已经 尚未 目前 本次 报告 期间 单位 万元 元 万股 股 年 月 日 末 期初 期末 如下 所示 详见 见 表 图
    项目 合计 小计 序号 名称 内容 金额 占比 比例 数 量 上述 其他 其中 一 二 三 四 五 六 七 八 九 十
    the a an of to in for on and or is are was were be been with by as at from that this these those it its
    """.split()
)

_NUM_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_NUM_PERCENT = re.compile(r"\d+(?:\.\d+)?%")
_NUM_DECIMAL = re.compile(r"(?<![\d.,])\d+\.\d+(?![\d,])")
_NUM_PLAIN = re.compile(r"(?<![\d.,])\d{2,}(?![\d.,])")  # 2 位以上纯整数（避开「1、」「2、」这类序号）
_FULLWIDTH = {chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}
_FULLWIDTH["\u3000"] = " "
_CJK = re.compile(r"[\u4e00-\u9fff]")
_PUNCT = re.compile(r"[\s\u3000!-/:-@\[-`{-~，。、；：？！“”‘’（）《》〈〉【】…—～·]+")

_JIEBA: Any = None
_JIEBA_LOCK: Any = None


def _jieba() -> Any:
    """惰性导入并缓存 jieba（同时压低其 INFO 日志，避免污染 stdout）。"""
    global _JIEBA, _JIEBA_LOCK
    if _JIEBA is None:
        import logging as _logging
        import threading

        if _JIEBA_LOCK is None:
            _JIEBA_LOCK = threading.Lock()
        with _JIEBA_LOCK:
            if _JIEBA is None:
                import jieba

                jieba.setLogLevel(_logging.ERROR)  # 默认会打印「Building prefix dict」到 stderr
                _JIEBA = jieba
    return _JIEBA


def collapse_whitespace(text: str) -> str:
    """折叠连续空白为单个空格并去首尾空白。"""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _to_halfwidth(text: str) -> str:
    """全角 ASCII 与全角空格转半角（保留中文标点语义由调用方决定）。"""
    return str(text or "").translate(_FULLWIDTH)


def normalize_text(text: str) -> str:
    """通用文本归一：全角转半角 + 折叠空白 + 去首尾空白。"""
    return collapse_whitespace(_to_halfwidth(str(text or "")))


def normalize_cell(text: str | None) -> str:
    """单元格归一（R1 + 占位符规则）。

    * ``None`` → ``""``；
    * 行尾连字符去除（英文断词）→ ``\\n`` 变空格 → 折叠空白；
    * 中文之间的空格去掉（``'有限公司成立日\\n期'`` → ``'有限公司成立日期'``）；
    * ``[ ◆ ]`` / ``[◆]`` / ``◆`` → ``未披露``（实测 PDF2 有 22 处，**不得丢弃或置空**），
      **含夹在单位中的形态**：``'[ ◆ ]元'`` → ``'未披露元'``、``'[ ◆ ]年[ ◆ ]月[ ◆ ]日'`` → ``'未披露（年/月/日）'``。
    """
    if text is None:
        return ""
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if _SYMBOL_CELL.match(raw.strip()):
        return UNDISCLOSED
    raw = _PLACEHOLDER.sub(UNDISCLOSED, raw)      # 占位符替换必须在空白折叠之前（保住单位）
    raw = re.sub(r"-\s*\n\s*", "", raw)           # 行尾连字符：直接拼接
    raw = raw.replace("\n", " ")
    raw = collapse_whitespace(_to_halfwidth(raw))
    raw = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", raw)  # 中文之间不留空格
    match = _CONSEC_PLACEHOLDER.search(raw)
    if match:
        units = "/".join(re.findall(r"年|月|日|时|分", match.group(0)))
        raw = raw.replace(match.group(0), f"{UNDISCLOSED}（{units}）")
    if _SYMBOL_CELL.match(raw):
        return UNDISCLOSED
    return raw


def strip_printed_page(text: str, *, page: int | None = None, head_lines: int = 2) -> str:
    """清洗页眉页脚噪声（印刷页码/运行标题），**只处理文本，不改 page 字段**。

    参数扩展说明（设计 §3.4 允许新增带默认值的参数）：
        page       当前页的 **1-based 物理页码**；给了才能做「值 == 物理页 - 1」双判据。
        head_lines 参与裸数字判据的页首行数（默认 2）。

    判据（双条件，避免误删正文）：
        ① 整行形如 ``1-1-128``（PDF1 页脚印刷页码）→ 删；
        ② 位于页首 ``head_lines`` 行内、且整行是纯数字、且 ``int(值) == page - 1`` → 删（PDF2 页眉）；
        ③ 位于页首 2 行、长度 <= 45、含「招股意向书/招股说明书」且不含句号 → 判为运行标题 → 删。
    """
    lines = str(text or "").split("\n")
    kept: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if re.fullmatch(r"\d+-\d+-\d+", stripped):           # ① PDF1 页脚 1-1-N
            continue
        if idx < head_lines:
            bare = re.fullmatch(r"\d+", _to_halfwidth(stripped))
            if bare and page is not None and int(bare.group(0)) == int(page) - 1:
                continue                                      # ② PDF2 页眉裸数字
            if idx < 2 and len(stripped) <= 45 and ("招股意向书" in stripped or "招股说明书" in stripped) and "。" not in stripped:
                continue                                      # ③ 运行标题
        kept.append(line)
    return "\n".join(kept).strip()


_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*|\n+")
_ABBR = re.compile(r"[A-Za-z]\.$")


def split_sentences(text: str, *, lang: str = "auto") -> list[str]:
    """按中英文句末标点切句（保留标点）；``lang`` 仅作提示，规则对中英通用。"""
    raw = str(text or "")
    pieces: list[str] = []
    for part in _SENT_SPLIT.split(raw):
        if not part:
            continue
        if pieces and _ABBR.search(pieces[-1]) and len(pieces[-1]) <= 4:
            pieces[-1] = pieces[-1] + part                            # 英文缩写（No. 之类）不切
        else:
            pieces.append(part)
    return [p.strip() for p in pieces if p.strip()]


def extract_numbers(text: str) -> list[str]:
    """抽取千分位金额、百分比、小数、2 位以上整数（保序去重）。"""
    raw = _to_halfwidth(str(text or ""))
    found: list[str] = []
    for pattern in (_NUM_THOUSANDS, _NUM_PERCENT, _NUM_DECIMAL, _NUM_PLAIN):
        found.extend(pattern.findall(raw))
    seen: set[str] = set()
    ordered: list[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def extract_percentages(text: str) -> list[str]:
    """抽取百分比（保序去重）。"""
    raw = _to_halfwidth(str(text or ""))
    seen: set[str] = set()
    ordered: list[str] = []
    for item in _NUM_PERCENT.findall(raw):
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


_NUMERIC_HINTS = ("多少", "几", "占比", "比例", "金额", "比重", "合计", "总额", "数量", "几年", "几个", "rate", "how much", "how many")


def has_numeric_signal(question: str) -> bool:
    """问题是否在问数字（用于加权与可答性判断）。"""
    text = str(question or "")
    if extract_numbers(text):
        return True
    lowered = text.lower()
    return any(hint in lowered for hint in _NUMERIC_HINTS)


_KEYWORD_NOISE = re.compile(r"^[^0-9a-zA-Z\u4e00-\u9fff]+$|^\d{1,2}$|^(markdown|来源|表标题|关键数字)$")


def extract_keywords(text: str, *, topk: int = 8, weight_dict: Mapping[str, float] | None = None) -> list[str]:
    """关键词：jieba 词频 + 加权词典（词典命中额外加成，长词优先）。

    噪声过滤（实测教训）：Markdown 分隔行 ``---``、单元格里的 1~2 位纯数字、
    以及 content 模板标签（``[Markdown]`` 等）都会污染关键词，必须剔除。
    """
    weights = dict(weight_dict or {})
    body = normalize_text(text)
    counts: Counter[str] = Counter(t for t in tokenize(body) if not _KEYWORD_NOISE.match(t))
    for word in weights:
        hit = body.count(word)
        if hit:
            counts[word] += hit
    scored = sorted(
        counts.items(),
        key=lambda kv: (-(kv[1] * float(weights.get(kv[0], 1.0)) + (2.0 if kv[0] in weights else 0.0)), -len(kv[0])),
    )
    return [word for word, _ in scored[: max(int(topk), 1)]]


def tokenize(text: str, *, use_jieba: bool = True, min_len: int = 2) -> list[str]:
    """分词：中文 jieba + 英文/数字整体保留 + 停用词过滤 + 补入金额/百分比 token。

    金额（``6,464.51``）会被 jieba 拆散，故显式补入 ``extract_numbers`` 的结果，
    保证 golden 数值能被 BM25 精确命中。
    """
    norm = _to_halfwidth(str(text or "")).lower()
    tokens: list[str] = []
    if use_jieba:
        tokens = [t for t in _jieba().cut(norm) if t.strip()]
    else:
        tokens = re.findall(r"[0-9a-z]+|[\u4e00-\u9fff]+", norm)
    out: list[str] = []
    for token in tokens:
        item = token.strip()
        if not item:
            continue
        if re.fullmatch(r"[0-9a-z][0-9a-z\.%,]*", item):
            out.append(item)                      # 数字/英文整体保留（含小数点与百分号）
            continue
        if item in STOPWORDS or _CJK.fullmatch(item) is None and len(item) < min_len:
            continue
        if len(item) < min_len:
            continue
        out.append(item)
    out.extend(extract_numbers(norm))
    return out


def warmup_tokenizer(*, logger: Any = None,
                     sample: str = "预热：报告期内公司来自军用领域的收入占主营业务收入的比重") -> dict[str, Any]:
    """预热 jieba 分词器并返回耗时（**在线服务启动时必须调用**）。

    实测（本机）：jieba 首次 ``cut`` 需构建前缀词典，耗时 **约 1.06 s**；后续调用仅 0.07 ms。
    若不预热，第一个问答请求会白吃约 1 s 的「首字 ≤ 3 s」预算（与 torch/transformers 的 eager import 同理）。
    """
    log = logger
    if log is None:
        from .logging_conf import get_logger

        log = get_logger("text_utils")
    started = time.perf_counter()
    tokens = tokenize(sample)
    first_ms = round((time.perf_counter() - started) * 1000, 2)
    started = time.perf_counter()
    tokenize(sample)
    warm_ms = round((time.perf_counter() - started) * 1000, 2)
    info = {"cold_ms": first_ms, "warm_ms": warm_ms, "tokens": len(tokens), "backend": "jieba"}
    log.log_event("utils.tokenizer_warmup", **info)
    return info


def keyword_score(text: str, keywords: Sequence[str]) -> float:
    """加权词典命中得分：命中次数 × 词长权重（长词更具体，权重更高）。"""
    body = normalize_text(text)
    if not body or not keywords:
        return 0.0
    score = 0.0
    for word in keywords:
        if not word:
            continue
        count = body.count(word)
        if count:
            score += count * (1.0 + 0.1 * (len(word) - 2))
    return round(score, 4)


def keyword_coverage(query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
    """查询词在文档中的覆盖率（0~1，空查询返回 0.0）。"""
    q = {t for t in query_tokens if t}
    if not q:
        return 0.0
    d = {t for t in doc_tokens if t}
    return round(len(q & d) / len(q), 4)


def text_digest(text: str, *, limit: int = 120) -> dict[str, Any]:
    """文本摘要：字符数 + 开头片段 + sha1 前 8 位。"""
    body = str(text or "")
    return {
        "chars": len(body),
        "head": body[:limit],
        "sha1": hashlib.sha1(body.encode("utf-8", errors="replace")).hexdigest()[:8],
    }


def squash_text(text: str) -> str:
    """压缩为「无空白无标点」形式，用于包含判定（保留中文与字母数字）。"""
    return _PUNCT.sub("", _to_halfwidth(str(text or "")).lower())


def chunk_field(chunk: Any, name: str, default: Any = None) -> Any:
    """从「块」里取字段，**同时兼容 dict/Mapping 与对象**两种形态（t14 修复，§17）。

    实测缺陷（t14）：命定判定等实现里写 ``getattr(chunk, "content", None)`` → 传 dict（JSONL 回读、
    HTTP 响应、评估产物回读都是 dict）时取到 ``None``，循环走完**静默返回未命中**，
    把「形状不匹配」伪装成「真实未命中」。所有从块取字段的地方都应改用本函数。

    * ``Mapping``（含 dict）→ ``chunk.get(name, default)``；
    * 对象 → ``getattr(chunk, name, default)``；
    * 纯字符串且 ``name == "content"`` → 字符串自身（把「纯文本片段」也视为块内容）。
    """
    if isinstance(chunk, Mapping):
        return chunk.get(name, default)
    if isinstance(chunk, str) and name == "content":
        return chunk
    return getattr(chunk, name, default)


def evidence_contains(chunk_text: str, evidence: str, *, threshold: float = 0.90) -> bool:
    """**命中判定的唯一实现**（设计 §3.4）。

    ① 归一化（去空白/标点/全半角）后证据原文被 chunk 文本包含 → True；
    ② 否则按 token 覆盖率 >= threshold → True；
    ③ 证据为空 → False（空证据不算命中，避免假命中）。
    """
    if not str(evidence or "").strip():
        return False
    if squash_text(evidence) and squash_text(evidence) in squash_text(chunk_text):
        return True
    ev_tokens = tokenize(evidence)
    if not ev_tokens:
        return False
    return keyword_coverage(ev_tokens, tokenize(chunk_text)) >= float(threshold)


# ---------------------------------------------------------------------------
# JSONL 读写
# ---------------------------------------------------------------------------
def to_jsonl_line(obj: Mapping[str, Any]) -> str:
    """对象 → 单行 JSON（``ensure_ascii=False``，非 ASCII 原样保留）。"""
    return json.dumps(dict(obj), ensure_ascii=False, default=str)


def read_jsonl(path: Path | str) -> Iterator[dict]:
    """逐行读取 JSONL（空行跳过；坏行抛 ValueError 由调用方处理，不静默丢弃）。"""
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSONL 解析失败：{exc}") from exc


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = False) -> int:
    """写 JSONL（默认追加），返回写入行数；父目录自动创建。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    count = 0
    with target.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(to_jsonl_line(row) + "\n")
            count += 1
    return count
