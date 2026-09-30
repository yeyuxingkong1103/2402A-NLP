"""
ocr_engine.py — OCR 引擎封装

给 pdf_parser 用的识别后端，按能力从高到低探测，任一环节缺失自动降级：

    PaddleOCR 3.x VL   版面预处理（方向分类 + 文档矫正 + 文本行方向）+ 文本识别
    PaddleOCR 普通     只做检测识别，不做版面处理
    None               都装不上，调用方按"没有 OCR"处理

对外只暴露 recognize()，返回文本行与从版面框几何重建出的表格
（表格统一转成 Markdown，直接复用 pdf_parser._table_to_text）。
"""

from __future__ import annotations

# 少于这个规模的框集合不当作表格，避免把普通段落排成网格
_TABLE_MIN_ROWS = 2
_TABLE_MIN_COLS = 2
_TABLE_MAX_COLS = 8
# 单元格文本超过这个长度基本是正文，不是表项
_TABLE_CELL_MAX_LEN = 40


def _cluster(values: list[float], tol: float) -> list[list[float]]:
    """把一维坐标按容差聚成若干组，返回分组（每组是原始值的列表）。"""
    groups: list[list[float]] = []
    for value in sorted(values):
        if groups and value - groups[-1][-1] <= tol:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _row_groups(boxes: list[tuple], tol: float) -> list[list[int]]:
    """按纵向中心把框聚成文本行，返回每行包含的框下标。"""
    centers = [(box[1] + box[3]) / 2 for box in boxes]
    groups = _cluster(centers, tol)
    rows: list[list[int]] = []
    for group in groups:
        limit = max(group)
        floor = min(group)
        rows.append([i for i, c in enumerate(centers) if floor <= c <= limit])
    return rows


def rebuild_table(texts: list[str], boxes: list[tuple]) -> list[list[str]]:
    """用版面框的几何关系把文字还原成二维表，不像表格就返回空。

    图片里的表格没有矢量线可抓，只能靠"同一行的框纵向对齐、不同行横向成列"
    这个特征来还原：先把框按纵坐标聚成行，再按横坐标聚成列，
    最后把文字填回行列交叉处。
    """
    if len(texts) != len(boxes) or len(boxes) < _TABLE_MIN_ROWS * _TABLE_MIN_COLS:
        return []

    heights = sorted(box[3] - box[1] for box in boxes)
    tol_y = max(4.0, heights[len(heights) // 2] * 0.6)
    rows = _row_groups(boxes, tol_y)

    # 表格的每一行都该有同样多的单元格。流程图虽然也能聚成行列，
    # 但行内框数忽多忽少，靠这条把它挡在外面。
    counts = [len(row) for row in rows]
    common = max(set(counts), key=counts.count)
    if len(rows) < _TABLE_MIN_ROWS or common < _TABLE_MIN_COLS:
        return []
    if counts.count(common) < len(rows) * 0.8:
        return []

    widths = sorted(box[2] - box[0] for box in boxes)
    tol_x = max(12.0, widths[len(widths) // 2] * 1.2)
    col_centers = [
        sum(c) / len(c)
        for c in _cluster([(box[0] + box[2]) / 2 for box in boxes], tol_x)
    ]
    # 列数要正好对上每行的单元格数，多出来说明聚类把列并了或裂了，不可信
    if not (_TABLE_MIN_COLS <= len(col_centers) <= _TABLE_MAX_COLS):
        return []
    if len(col_centers) != common:
        return []

    grid: list[list[str]] = []
    for row in rows:
        cells = [""] * len(col_centers)
        for index in row:
            box = boxes[index]
            center = (box[0] + box[2]) / 2
            cell = min(range(len(col_centers)), key=lambda c: abs(col_centers[c] - center))
            cells[cell] = (cells[cell] + " " + texts[index]).strip()
        grid.append(cells)

    filled = [cell for row in grid for cell in row if cell]
    # 表格里每格都短、且确实填满了大半格子，才认定还原成功
    if not filled or max(len(c) for c in filled) > _TABLE_CELL_MAX_LEN:
        return []
    if len(filled) < len(grid) * len(col_centers) * 0.5:
        return []
    return grid


class OCREngine:
    """OCR 引擎。初始化失败不抛异常，用 available 表示是否可用。"""

    def __init__(self):
        self._engine = None
        self._is_vl = False
        self._init()

    def _init(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            return

        # 3.x 支持版面预处理参数，走这条即 VL 路径；2.x 会因未知参数报错，
        # 落到下面的普通初始化。mkldnn 在这台机器上会触发 onednn 报错，关掉。
        try:
            self._engine = PaddleOCR(
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_textline_orientation=True,
                lang="ch",
                enable_mkldnn=False,
            )
            self._is_vl = True
            return
        except Exception:
            self._engine = None

        try:
            self._engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        except Exception:
            self._engine = None

    @property
    def available(self) -> bool:
        return self._engine is not None

    @property
    def is_vl(self) -> bool:
        """是否启用版面分析（VL）路径，报告与排查用。"""
        return self._is_vl

    def _recognize_3x(self, image):
        """3.x 的 predict 接口，返回 (文本行, 版面框)。"""
        results = self._engine.predict(input=image)
        page = results[0] if results else {}
        texts = list(page.get("rec_texts") or [])
        # rec_boxes 是 numpy 数组，不能用 or 兜底，那样会触发数组真值歧义
        raw = page.get("rec_boxes")
        boxes = [] if raw is None else [tuple(float(v) for v in box) for box in raw]
        if len(boxes) != len(texts):  # 框和文字对不上时就放弃几何信息
            boxes = []
        return texts, boxes

    def _recognize_2x(self, image):
        """2.x 的 ocr 接口，返回 (文本行, 版面框)。"""
        results = self._engine.ocr(image, cls=True)
        texts: list[str] = []
        boxes: list[tuple] = []
        for page in results or []:
            for item in page or []:
                if not item or not item[1] or not item[1][0]:
                    continue
                texts.append(item[1][0])
                points = item[0] or []
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    boxes.append((min(xs), min(ys), max(xs), max(ys)))
                else:
                    boxes.append(())
        return texts, boxes

    @staticmethod
    def _table_markdown(texts: list[str], boxes: list[tuple]) -> str:
        """从版面框重建表格并转 Markdown，重建不出表格时返回空串。"""
        if not boxes or len(boxes) != len(texts):
            return ""
        pairs = [(t.strip(), b) for t, b in zip(texts, boxes) if t and t.strip()]
        rows = rebuild_table([t for t, _ in pairs], [b for _, b in pairs])
        if not rows:
            return ""
        from pdf_parser import _table_to_text

        return _table_to_text(rows)

    def recognize(self, image) -> dict:
        """识别单张图片，返回 {"text": 纯文本, "table": Markdown 表格}。

        表格是从版面框几何重建出来的，重建失败时 table 为空串，
        文字仍然完整保留在 text 里，不会丢内容。
        """
        if self._engine is None:
            return {"text": "", "table": ""}

        try:
            if self._is_vl:
                texts, boxes = self._recognize_3x(image)
            else:
                texts, boxes = self._recognize_2x(image)
        except Exception:
            # 单张图识别失败只丢这张，不影响其它图片
            return {"text": "", "table": ""}

        clean = [t.strip() for t in texts if t and t.strip()]
        try:
            table = self._table_markdown(texts, boxes)
        except Exception:
            # 表格重建只是增强，失败时保留纯文本即可
            table = ""
        return {"text": "\n".join(clean), "table": table}


def get_engine() -> OCREngine | None:
    """构造引擎，装不上返回 None。"""
    engine = OCREngine()
    return engine if engine.available else None
