"""分块判定：合并、超长处理、以及「不确定就交人裁决」（D1/D3）。

Q2 裁决为 A（规则式）：切分边界由**段落对齐 + 章节边界**决定，不加载任何模型。
理由见 specs/003 的 D2 —— 嵌入式分块会让 chunk_id 随模型版本漂移，而
docs/04 §8 把向量空间漂移列为整条管线唯一会静默失效的环节。
"""

from __future__ import annotations

from .core import (
    ENDS_CJK, HIGH, LOW, MIN_RATIO, SENT_BOUNDARY, STANDALONE_TYPES,
    TERMINAL, UNFINISHED_MIN_LEN, char_len, size_of,
)


class Decisions:
    """待裁决项。它是本步骤的一等产出物，不是错误日志——
    它的存在就是「脚本没有私自决策」的证据。

    答复支持四档优先级的通配（见 resolve），否则几十条待裁决项没法批量处理。
    """

    def __init__(self, doc_id: str, answers: dict) -> None:
        self.doc_id, self.answers, self.items = doc_id, answers, []

    def resolve(self, decision_id: str, kind: str, parent: str = "") -> str | None:
        """从最具体到最宽泛：精确 id > 按父章节 > 按类别 > 全通配。

        父章节比类别更具体，所以优先——按父章节批量授权，`section` 元数据的
        损失最小（同父之下都是兄弟小节），不会跨到别的章节去。
        """
        for key in (decision_id, "parent:" + parent, "kind:" + kind, "*"):
            if key and key in self.answers:
                return self.answers[key]
        return None

    def record(self, kind: str, blocks: list[dict], why: str, choices: list[str],
               excerpt: str = "", parent: str = "") -> str:
        did = "%s:D%03d" % (self.doc_id, len(self.items) + 1)
        self.items.append({
            "decision_id": did, "kind": kind, "doc_id": self.doc_id,
            "parent": parent or "文档根",
            "block_ids": [b["block_id"] for b in blocks],
            "page": blocks[0]["page"] if blocks else None,
            "excerpt": excerpt or (blocks[0]["text"][:90] if blocks else ""),
            "why": why, "choices": choices,
        })
        return did

    def answer_for(self, blocks: list[dict]) -> str | None:
        """按首块 block_id 反查该段所属待裁决项的答复。"""
        if not blocks:
            return None
        anchor = blocks[0]["block_id"]
        for item in self.items:
            if item["block_ids"] and item["block_ids"][0] == anchor:
                return self.resolve(item["decision_id"], item["kind"], item["parent"])
        return None

    def pending(self) -> list[dict]:
        return [i for i in self.items
                if self.resolve(i["decision_id"], i["kind"], i["parent"]) is None]


# ---- 合并 -----------------------------------------------------------------

def note_small_sections(units: list[dict], decisions: Decisions) -> None:
    """D3 第 1 类。必须在合并**之前**记录，否则 answer_for 查不到答复。"""
    for unit in units:
        for seg in unit["segments"]:
            if seg[0]["block_type"] in STANDALONE_TYPES:
                continue                       # 自成一块的，不参与合并，无需裁决
            size = size_of(seg)
            if size >= LOW * MIN_RATIO:
                continue
            parent = " > ".join(unit["key"][:-1]) or "文档根"
            decisions.record(
                "section_too_small", seg,
                "该节仅 %d 字，不足下限 %d 的 %.0f%%；补足必须跨越父章节（%s）"
                % (size, LOW, MIN_RATIO * 100, parent),
                ["keep", "merge_up"], parent=parent)


def merge(units: list[dict], opts, decisions: Decisions) -> tuple[list[dict], int]:
    """按文档顺序合并相邻段。

    并入条件：前段当前长度 < 下限、合并后不超上限、且二者都是可合并类型
    （表格独占，永不参与合并）。合并本身还须满足 D1 —— 两段同属一个父章节；
    人工答复 merge_up 的段可越过父章节边界（那是明确授权）。
    """
    chunks: list[dict] = []
    crossed = 0
    for unit in units:
        key = unit["key"]
        for seg in unit["segments"]:
            mergable = seg[0]["block_type"] not in STANDALONE_TYPES
            size = size_of(seg)
            if chunks and mergable:
                prev = chunks[-1]
                same_parent = prev["key"][:-1] == key[:-1]
                authorized = decisions.answer_for(seg) == "merge_up"
                fits = (opts.cross_section and prev["size"] < LOW
                        and prev["size"] + size <= HIGH and prev["mergable"])
                if fits and (same_parent or authorized):
                    if not same_parent:
                        crossed += 1
                    prev["blocks"] += seg
                    prev["size"] += size
                    prev["key"] = key
                    continue
            chunks.append({"key": key, "blocks": list(seg), "size": size,
                           "mergable": mergable})
    return chunks, crossed


# ---- 超长处理 -------------------------------------------------------------

def sentence_pieces(text: str, limit: int) -> list[str]:
    """按句边界切成长度不超 limit 的片段；单句自身超限时只能整句保留。"""
    pieces, cur = [], ""
    for part in SENT_BOUNDARY.split(text):
        if not part:
            continue
        if cur and char_len(cur) + char_len(part) > limit:
            pieces.append(cur)
            cur = ""
        cur += part
    if cur:
        pieces.append(cur)
    return pieces or [text]


def _split_text(out: list[list[dict]], block: dict, limit: int) -> None:
    for i, piece in enumerate(sentence_pieces(block["text"], limit)):
        out.append([{**block, "text": piece, "_sub": i}])


def apply_oversized(chunks: list[dict], out: list[list[dict]],
                    decisions: Decisions, opts) -> None:
    """超长正文块按句边界二次切分；自成一块的类型原样保留（docs/04 §7）。"""
    for ch in chunks:
        first = ch["blocks"][0]
        if first["block_type"] in STANDALONE_TYPES:
            if first["block_type"] == "table" and char_len(first["text"]) > HIGH:
                did = decisions.record(
                    "oversized_table", [first],
                    "表格 %d 字，超过上限 %d。docs/04 §7 规定整表成一块——"
                    "切了表头，后半张表就失去列含义" % (char_len(first["text"]), HIGH),
                    ["keep", "split"])
                if decisions.resolve(did, "oversized_table") == "split":
                    _split_text(out, first, opts.high)
                    continue
            out.append(list(ch["blocks"]))
            continue

        buf: list[dict] = []
        used = 0
        for block in ch["blocks"]:
            size = char_len(block["text"])
            if size > HIGH:
                if buf:
                    out.append(buf)
                    buf, used = [], 0
                _split_text(out, block, opts.high)
                continue
            if buf and used + size > HIGH:
                out.append(buf)
                buf, used = [], 0
            buf.append(block)
            used += size
        if buf:
            out.append(buf)


def audit_unfinished(chunks: list[list[dict]], decisions: Decisions) -> None:
    """D3 第 2 类。判据刻意收紧：长度≥60、以汉字结尾、无终止标点。

    实测教训——只判「结尾无标点」在本语料上命中 24 个块，其中 `指南与共识`、
    机构名列表、`关键词 …`、图题**全是误报**（它们本来就是不带句号的独立条目）。
    收紧后实测命中 0，即只在真正异常时才打扰人。
    """
    for seg in chunks:
        last = seg[-1]
        tail = last["text"].rstrip()
        if (last["block_type"] == "text" and char_len(tail) >= UNFINISHED_MIN_LEN
                and tail[-1] not in TERMINAL and ENDS_CJK.search(tail)):
            decisions.record(
                "unfinished_block", [last],
                "该块 %d 字、以汉字结尾且无终止标点，句子可能被截断，但找不到可续接的相邻块"
                % char_len(tail), ["keep", "merge_next"], excerpt=tail[-90:])
