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
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from common.constants import LLMType
from api.db.services.llm_service import LLMBundle
from common.connection_utils import timeout
from rag.app.picture import vision_llm_chunk as picture_vision_llm_chunk
from rag.prompts.generator import vision_llm_figure_describe_prompt


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

    # ---- 工单15 新增（原文件没有这个方法）--------------------------------
    # 给每个图纸块打上「图号 + 页码」标记，形如：`【技术图纸】图3｜第11页`
    #
    # 为什么需要：检索层要能**确定地**认出「这一块讲的是哪张图」，才能做到
    # 「问题提到图3 → 优先召回图3 的描述块」。原实现里图号只以 OCR 文字的形式
    # 零散混在正文结尾（本工单实测是 "图3 图 2 1" 这种糊成一团的），没法可靠匹配；
    # 页码则完全没进 chunk —— 而问题恰恰是拿「第11页图3」来指代图纸的。
    #
    # 图号从 OCR 出来的图注里取（取第一个匹配）；页码从位置戳里取
    # （deepdoc/parser/pdf_parser.py 的 cropout() 里 poss.append 的是
    #  (页码, left, right, top, bottom)，页码是 page_from 起的 0 基，所以 +1）。
    def _mark_figures(self, captions):
        """captions: 标记前的原始图注（每个元素是 OCR 出来的图注列表）。

        图号**只从图注里取**，不能从整段（含 VLM 正文）里取 —— 实测 VLM 会把
        第 11 页的「图 3」读成「图 2」，从整段里取第一个 "图N" 就会取错。
        """
        for i in range(len(self.descriptions)):
            positions = self.positions[i] if i < len(self.positions) else []
            pages = sorted({p[0] for p in positions})

            cap = "\n".join(captions[i]) if i < len(captions) and isinstance(captions[i], list) \
                else str(captions[i]) if i < len(captions) else ""
            figs = []
            for m in re.finditer(r"图\s*([0-9]+)", cap):
                if m.group(1) not in figs:
                    figs.append(m.group(1))

            logging.info(f"[VisualRAG] 图纸#{i} 位置戳页码={pages} 图注={cap[:100]!r} 取到图号={figs}")

            if not figs and not pages:
                continue
            parts = []
            if figs:
                # 一个图块常常跨页、或把相邻几张图的图注一起吸进来（实测
                # 「图3\n图 2\n1」），所以图号全列出来，别只取第一个。
                parts.append("图" + "/图".join(figs))
            for p in pages:
                # 每一页单独写成一个「第N页」，这样问题里写「第11页」能直接命中
                parts.append(f"第{p + 1}页")
            self.descriptions[i] = f"【技术图纸】{'｜'.join(parts)}\n" + str(self.descriptions[i])
    # ----------------------------------------------------------------------

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

        # 先把原始图注留一份：下面 descriptions[i] 会被 VLM 正文覆盖掉
        captions = [list(d) if isinstance(d, list) else [str(d)] for d in self.descriptions]

        for future in as_completed(futures):
            figure_num, txt = future.result()
            if txt:
                self.descriptions[figure_num] = txt + "\n".join(self.descriptions[figure_num])

        self._mark_figures(captions)

        self._assemble()

        return self.assembled
