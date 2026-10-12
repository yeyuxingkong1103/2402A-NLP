#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re

from PIL import Image

from common.constants import LLMType
from api.db.services.llm_service import LLMBundle
from common.connection_utils import timeout
from rag.app.picture import vision_llm_chunk as picture_vision_llm_chunk
from rag.prompts.generator import vision_llm_figure_describe_prompt


# ---------------------------------------------------------------------------
# 工单14 · 三元组后处理（确定性去歧义）
#
# VLM 提示词要求输出 (部件A, 方位关系, 部件B) 形式的三元组。实测表明，模型无法稳定
# 遵守「同一部件对只输出一条关系」这类跨行集合约束（C/D/E 三轮迭代分别仍有 9/8/6 次
# 违反）。因此把该约束下沉到代码，保证 100% 生效。
# ---------------------------------------------------------------------------

# 匹配 (A, 位于xxx/重合, B)
TRIPLE_RE = re.compile(r"\(\s*([^,()]+?)\s*,\s*(位于[^,()]+?|重合)\s*,\s*([^,()]+?)\s*\)")

# 尺寸/符号标注，不是部件：D、h₁、x₂ 等
DIM_SYMBOL_RE = re.compile(r"^[hHxXdD][₀-₉¹²³]*[0-9]*$")

# 主轴优先级：左右(2) > 上下(1) > 内外/重合(0)
REL_PRIORITY = {
    "位于左侧": 2,
    "位于右侧": 2,
    "位于上方": 1,
    "位于下方": 1,
}
DEFAULT_PRIORITY = 0


def _rel_priority(rel):
    return REL_PRIORITY.get(rel, DEFAULT_PRIORITY)


def dedup_component_triples(text, tag=""):
    """对 VLM 输出中 SECTION 3 的三元组做确定性去歧义。

    规则：
      1. 以 frozenset({A, B}) 分组，同一部件对只保留 1 条关系；
      2. 冲突时按主轴优先级择一：左右 > 上下 > 内外/重合；
      3. 过滤含尺寸符号（h/x/D）的三元组 —— 它们不是部件；
      4. 记录全部被丢弃的关系到日志。

    非 SECTION 3 的内容原样保留。
    """
    if not text:
        return text

    head = re.search(r"(?:^|\n)(#+\s*SECTION\s*3[^\n]*\n)", text)
    if not head:
        logging.info(f"[TriplePostProcess]{tag} no SECTION 3 found, text left unchanged")
        return text

    start = head.end()
    nxt = re.search(r"\n#+\s*SECTION\s*4", text[start:])
    end = start + nxt.start() if nxt else len(text)
    block = text[start:end]

    found = TRIPLE_RE.findall(block)
    if not found:
        logging.info(f"[TriplePostProcess]{tag} SECTION 3 has no triples, left unchanged")
        return text

    kept = {}          # frozenset({A,B}) -> (A, rel, B)
    order = []         # 保持首次出现顺序
    dropped = []       # (A, rel, B, reason)

    for raw_a, raw_rel, raw_b in found:
        a, rel, b = raw_a.strip(), raw_rel.strip(), raw_b.strip()

        if a == b:
            dropped.append((a, rel, b, "self-pair"))
            continue
        if DIM_SYMBOL_RE.match(a) or DIM_SYMBOL_RE.match(b):
            dropped.append((a, rel, b, "dimension-symbol"))
            continue

        key = frozenset((a, b))
        if key not in kept:
            kept[key] = (a, rel, b)
            order.append(key)
            continue

        cur = kept[key]
        if _rel_priority(rel) > _rel_priority(cur[1]):
            # 新高优先级，替换旧的，旧记录为被丢弃
            dropped.append((cur[0], cur[1], cur[2], "lower-priority"))
            kept[key] = (a, rel, b)
        else:
            dropped.append((a, rel, b, "lower-priority"))

    lines = [f"({kept[k][0]}, {kept[k][1]}, {kept[k][2]})" for k in order]
    new_block = "\n" + "\n".join(lines) + "\n"

    logging.info(
        f"[TriplePostProcess]{tag} found={len(found)} kept={len(lines)} "
        f"dropped={len(dropped)} pairs={len(kept)}"
    )
    for a, rel, b, reason in dropped:
        logging.info(
            f"[TriplePostProcess]{tag} DROP ({a}, {rel}, {b}) reason={reason}"
        )

    return text[:start] + new_block + text[end:]


def vision_figure_parser_figure_data_wrapper(figures_data_without_positions):
    if not figures_data_without_positions:
        return []
    return [
        (
            (figure_data[1], [figure_data[0]]),
            [(0, 0, 0, 0, 0)],
        )
        for figure_data in figures_data_without_positions
        if isinstance(figure_data[1], Image.Image)
    ]


def vision_figure_parser_docx_wrapper(sections, tbls, callback=None,**kwargs):
    if not sections:
        return tbls
    try:
        vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
        callback(0.7, "Visual model detected. Attempting to enhance figure extraction...")
    except Exception:
        vision_model = None
    if vision_model:
        figures_data = vision_figure_parser_figure_data_wrapper(sections)
        try:
            docx_vision_parser = VisionFigureParser(vision_model=vision_model, figures_data=figures_data, **kwargs)
            boosted_figures = docx_vision_parser(callback=callback)
            tbls.extend(boosted_figures)
        except Exception as e:
            callback(0.8, f"Visual model error: {e}. Skipping figure parsing enhancement.")
    return tbls

def vision_figure_parser_figure_xlsx_wrapper(images,callback=None, **kwargs):
    tbls = []
    if not images:
        return []
    try:
        vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
        callback(0.2, "Visual model detected. Attempting to enhance Excel image extraction...")
    except Exception:
        vision_model = None
    if vision_model:
        figures_data = [((
                        img["image"],   # Image.Image
                        [img["image_description"]]     # description list (must be list)
                    ),
                    [
                        (0, 0, 0, 0, 0)   # dummy position
                    ]) for img in images]
        try:
            parser = VisionFigureParser(vision_model=vision_model, figures_data=figures_data, **kwargs)
            callback(0.22, "Parsing images...")
            boosted_figures = parser(callback=callback)
            tbls.extend(boosted_figures)
        except Exception as e:
            callback(0.25, f"Excel visual model error: {e}. Skipping vision enhancement.")
    return tbls

def vision_figure_parser_pdf_wrapper(tbls, callback=None, **kwargs):
    if not tbls:
        return []
    try:
        vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
        callback(0.7, "Visual model detected. Attempting to enhance figure extraction...")
    except Exception:
        vision_model = None
    if vision_model:
        def is_figure_item(item):
            return (
                isinstance(item[0][0], Image.Image) and
                isinstance(item[0][1], list)
            )
        figures_data = [item for item in tbls if is_figure_item(item)]
        try:
            docx_vision_parser = VisionFigureParser(vision_model=vision_model, figures_data=figures_data, **kwargs)
            boosted_figures = docx_vision_parser(callback=callback)
            tbls = [item for item in tbls if not is_figure_item(item)]
            tbls.extend(boosted_figures)
        except Exception as e:
            callback(0.8, f"Visual model error: {e}. Skipping figure parsing enhancement.")
    return tbls


shared_executor = ThreadPoolExecutor(max_workers=10)


class VisionFigureParser:
    def __init__(self, vision_model, figures_data, *args, **kwargs):
        self.vision_model = vision_model
        self._extract_figures_info(figures_data)
        assert len(self.figures) == len(self.descriptions)
        assert not self.positions or (len(self.figures) == len(self.positions))

    def _extract_figures_info(self, figures_data):
        self.figures = []
        self.descriptions = []
        self.positions = []

        for item in figures_data:
            # position
            if len(item) == 2 and isinstance(item[0], tuple) and len(item[0]) == 2 and isinstance(item[1], list) and isinstance(item[1][0], tuple) and len(item[1][0]) == 5:
                img_desc = item[0]
                assert len(img_desc) == 2 and isinstance(img_desc[0], Image.Image) and isinstance(img_desc[1], list), "Should be (figure, [description])"
                self.figures.append(img_desc[0])
                self.descriptions.append(img_desc[1])
                self.positions.append(item[1])
            else:
                assert len(item) == 2 and isinstance(item[0], Image.Image) and isinstance(item[1], list), f"Unexpected form of figure data: get {len(item)=}, {item=}"
                self.figures.append(item[0])
                self.descriptions.append(item[1])

    def _assemble(self):
        self.assembled = []
        self.has_positions = len(self.positions) != 0
        for i in range(len(self.figures)):
            figure = self.figures[i]
            desc = self.descriptions[i]
            pos = self.positions[i] if self.has_positions else None

            figure_desc = (figure, desc)

            if pos is not None:
                self.assembled.append((figure_desc, pos))
            else:
                self.assembled.append((figure_desc,))

        return self.assembled

    def __call__(self, **kwargs):
        callback = kwargs.get("callback", lambda prog, msg: None)

        @timeout(30, 3)
        def process(figure_idx, figure_binary):
            description_text = picture_vision_llm_chunk(
                binary=figure_binary,
                vision_model=self.vision_model,
                prompt=vision_llm_figure_describe_prompt(),
                callback=callback,
            )
            return figure_idx, description_text

        futures = []
        for idx, img_binary in enumerate(self.figures or []):
            futures.append(shared_executor.submit(process, idx, img_binary))

        for future in as_completed(futures):
            figure_num, txt = future.result()
            if txt:
                # 工单14：三元组确定性去歧义（同对仅存 1 条，主轴优先，滤尺寸符号）
                txt = dedup_component_triples(txt, tag=f" figure={figure_num}")
                self.descriptions[figure_num] = txt + "\n".join(self.descriptions[figure_num])

        self._assemble()

        return self.assembled
