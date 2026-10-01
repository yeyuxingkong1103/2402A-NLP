"""文档分块：固定长度 / 句子 / 段落 / 标题 / 语义。

- paragraph：按空行切段后贪心合并到 chunk_size（默认），最稳健
- fixed     ：固定字符窗口 + 重叠
- sentence  ：按中英文句末标点分句后合并
- title     ：按 markdown 标题切节，标题拼到该节每个 chunk 前（保留章节上下文）
- semantic  ：句子 embedding 相邻相似度找话题断点，再合并到 chunk_size
             （需要外部传入 embed 函数，否则退化为段落分块）
"""
from __future__ import annotations

import re

STRATEGIES = {"fixed", "sentence", "paragraph", "semantic", "title"}

# 对外可调分块参数的边界，接口和离线 CLI 都从这里取，避免两边各写一份后漂移。
#
# 上界不是洁癖：检索会把 TOP_K 条 chunk 整段塞进系统消息，而系统消息**不受**
# HISTORY_MAX_CHARS 约束（那只管历史），chunk_size 直接决定 prompt 会不会顶穿上下文。
# 实测（qwen3:8b / n_ctx=16384）：chunk_size=2000 时 prompt 15355 tokens 还装得下，
# 3000 时 18850 tokens 就超了，服务端静默丢最老轮次。
# 1000 是留给「整体预算」兜底的安全起点，别随手放宽。
#
# 背景：这里的上界此前只加在 /knowledge/upload 的查询参数上，而 scripts/ingest.py
# 是直接写 Milvus 的离线入口，绕过接口就能塞进超大 chunk。
CHUNK_SIZE_MIN, CHUNK_SIZE_MAX = 50, 1000
CHUNK_OVERLAP_MIN, CHUNK_OVERLAP_MAX = 0, 500


class Chunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50, strategy: str = "paragraph"):
        if strategy not in STRATEGIES:
            raise ValueError(f"未知分块策略: {strategy}（可选 {sorted(STRATEGIES)}）")
        # 接口层有 ge/le 校验，但 scripts/ingest.py 与 seed 是直接构造 Chunker 的：
        # chunk_size<=0 时切出来的片全是空串，固定窗口那条 while 更会原地打转。这里兜一层。
        self.chunk_size = max(chunk_size, 1)
        # overlap 必须小于 chunk_size：_chunk_by_units 会把上一个 chunk 的尾部整段
        # 接到下一个前面，overlap >= chunk_size 时每个 chunk 都被撑大，越滚越长
        # （实测 chunk_size=50 / overlap=500 时 chunk 长到 536 字、数量也暴涨）。
        self.overlap = min(max(overlap, 0), max(self.chunk_size - 1, 0))
        self.strategy = strategy

    def chunk(self, text: str, embed=None) -> list[str]:
        """embed：语义分块用的向量化函数（``texts -> list[vector]``），其它策略忽略。"""
        if not text or not text.strip():
            return []

        if self.strategy == "fixed":
            return self._chunk_fixed(text)
        if self.strategy == "sentence":
            return self._chunk_by_units(self._split_sentences(text))
        if self.strategy == "title":
            return self._chunk_title(text)
        if self.strategy == "semantic":
            return self._chunk_semantic(text, embed)
        # paragraph（默认）
        return self._chunk_by_units(self._split_paragraphs(text))

    # ---- 切分单元 ----
    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        # 只把**空行**当段落边界，单个换行不算：PDF 抽出来的正文几乎每行都换行，
        # 若把单换行也当边界，paragraph 策略会退化成近乎按行切，chunk 又碎又多。
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # 用 lookbehind：句末标点留在前一句里（先切掉标点、之后再拼回去会连成一片）。
        # 只认中英文句末与分号，**不含英文句点**——正文里的小数、缩写、版本号会被切碎。
        parts = re.split(r"(?<=[。！？!?；;])\s*", text)
        return [p.strip() for p in parts if p.strip()]

    # ---- 标题分块 ----
    _HEADING_RE = re.compile(r"(?m)^(#{1,6}\s+.+?)\s*$")

    def _split_headings(self, text: str) -> list[tuple[str, str]]:
        """按 markdown 标题切节，返回 [(标题, 正文), ...]，无标题则一整节标题为空。"""
        if not any(self._HEADING_RE.match(ln) for ln in text.splitlines()):
            return [("", text)]
        parts = self._HEADING_RE.split(text)
        sections: list[tuple[str, str]] = []
        pre = parts[0].strip()
        if pre:
            sections.append(("", pre))
        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            body = (parts[i + 1] if i + 1 < len(parts) else "").strip()
            sections.append((heading, body))
        return sections

    def _chunk_title(self, text: str) -> list[str]:
        out: list[str] = []
        for heading, body in self._split_headings(text):
            if not body:
                continue
            # 标题拼到该节每个 chunk 前：检索命中的 chunk 自带上文章节，回答不丢语境。
            # 但拼是在切完之后做的，所以**必须先把标题的位置留出来**：chunk_size 是
            # "上界受 prompt 预算约束"（见文件头），不留的话 chunk 会涨到
            # chunk_size + len(heading)（实测 chunk_size=1000 + 300 字标题 → 1303 字符），
            # 长标题（PDF 转 md 常见）能把整个预算撑破。
            prefix = self._fit_heading(heading)
            budget = self.chunk_size - (len(prefix) + 1 if prefix else 0)
            chunks = self._chunk_by_units(self._split_paragraphs(body), budget)
            if prefix:
                chunks = [f"{prefix}\n{c}" for c in chunks]
            out.extend(chunks)
        return out

    def _fit_heading(self, heading: str) -> str:
        """标题最多占 chunk_size 的三分之一；更长的截断加省略号。

        标题行长度没有天然上界（`^#{1,6}\\s+.+$`），长到能独自吃光预算时，
        正文就没地方放了——那比截断标题更糟。
        """
        if not heading:
            return ""
        cap = max(1, self.chunk_size // 3)
        return heading if len(heading) <= cap else heading[:cap] + "…"

    # ---- 语义分块 ----
    def _chunk_semantic(self, text: str, embed) -> list[str]:
        sents = self._split_sentences(text)
        if embed is None or len(sents) <= 1:
            # 语义分块依赖 embedding，没给就退化为段落分块
            return self._chunk_by_units(self._split_paragraphs(text))

        import numpy as np

        vecs = np.asarray(embed(sents), dtype=float)
        sims: list[float] = []
        for i in range(len(vecs) - 1):
            a, b = vecs[i], vecs[i + 1]
            # 除范数是防御性的：本仓库的 embed_texts 会归一化，但 chunker 与外部约定的
            # 最小契约只是「texts -> list[vector]」，不能假定实现方一定归一化过。
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            sims.append(float(np.dot(a, b) / (na * nb)) if na and nb else 0.0)

        # 断点阈值取相似度的 30% 分位：相邻句相似度跌破它即判定话题切换。
        # 用当篇的分位数而不是绝对阈值：不同文档的相似度基线差很多，只有相对高低可比。
        # 这里 sims 为空只是防御——只有一句的情况上面已经提前返回了。
        thr = float(np.percentile(sims, 30)) if sims else 0.0
        segments: list[str] = []
        cur = sents[0]
        for i, s in enumerate(sents[1:], 1):
            if sims[i - 1] < thr:
                segments.append(cur)
                cur = s
            else:
                cur += s
        if cur:
            segments.append(cur)
        return self._chunk_by_units(segments)

    # ---- 分块 ----
    def _chunk_fixed(self, text: str) -> list[str]:
        # 重叠切分：窗口按 step 前进，窗口右端一旦到达或越过末尾就收工——不这样判的话
        # 末尾会再产出一个完全落在上一片重叠区里的短 chunk（内容几乎重复）。
        step = max(self.chunk_size - self.overlap, 1)
        chunks = []
        i, n = 0, len(text)
        while i < n:
            piece = text[i : i + self.chunk_size].strip()
            if piece:
                chunks.append(piece)
            if i + self.chunk_size >= n:
                break
            i += step
        return chunks

    def _split_oversized(self, units: list[str], limit: int | None = None) -> list[str]:
        """把单独就超过 limit（默认 chunk_size）的单元硬切。

        否则贪心合并里那句 `if not cur: cur = unit` 会把它整段放行：PDF 解析出来
        的长文常常整节没有空行，一段就是几千字，直接成为一个 chunk。检索会把这个
        巨块整段塞进系统消息，把 prompt 顶穿上下文——chunk_size 就成了摆设。
        """
        cap = self.chunk_size if limit is None else max(1, limit)
        out: list[str] = []
        for unit in units:
            if len(unit) <= cap:
                out.append(unit)
                continue
            for i in range(0, len(unit), cap):
                piece = unit[i : i + cap].strip()
                if piece:
                    out.append(piece)
        return out

    def _chunk_by_units(self, units: list[str], budget: int | None = None) -> list[str]:
        """把单元拼成不超过 budget（默认 chunk_size）的 chunk。

        budget 可调是为了给「切完还要在前面拼标题」的 title 策略留位置。
        """
        limit = self.chunk_size if budget is None else max(1, budget)
        chunks: list[str] = []
        cur = ""
        for unit in self._split_oversized(units, limit):
            if not unit:
                continue
            if not cur:
                cur = unit
            elif len(cur) + len(unit) + 1 <= limit:  # +1 是拼接时那个换行符的位置
                cur = f"{cur}\n{unit}"
            else:
                chunks.append(cur)
                # 新 chunk 带上上一 chunk 的尾部作为重叠。重叠要按剩下多少空间倒着让位，
                # 不能整段照搬：overlap 只是「想要多少」，limit 是硬上界。
                room = limit - len(unit) - 1
                tail = chunks[-1][-self.overlap :] if self.overlap and room > 0 else ""
                cur = f"{tail[-room:]}\n{unit}".strip() if tail else unit
        if cur:
            chunks.append(cur)
        return chunks
