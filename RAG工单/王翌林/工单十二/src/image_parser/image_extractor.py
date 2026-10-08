# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/image_parser/image_extractor.py —— 工单四 PDF 图像提取模块

职责（见 docs/01_图像解析方案.md §三）：
  1. 使用 PyMuPDF 逐页遍历 PDF，提取全部位图图像（L1 位图提取）；
  2. 保存图像到 data/images/{doc_name}/page_{page}_img_{index}.png；
  3. 记录元数据：doc_id / page / image_index / bbox / width / height / format；
  4. 过滤小图标（宽或高 < 50px）与 md5 重复图（水印/页眉图标去重）；
  5. 输出 data/images/{doc_name}_images.json 结构化清单。

说明：
  - 招股说明书中的组织结构图/流程图若为矢量绘图，将在后续 Step 以
    L2（get_drawings 聚类渲染）补充，本模块先交付 L1 位图提取；
  - 统一转为 PNG 落盘（PIL 归一化色彩空间），避免 jpeg/cmyk 混杂。
"""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf  # 工单四：PyMuPDF 1.28 新 API（弃用 fitz 别名）
from loguru import logger
from PIL import Image

# 工单四：常量配置
WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"
MIN_SIDE_PX = 50          # 工单四：宽或高小于 50px 的小图标过滤阈值
DEFAULT_OUT_DIR = "data/images"   # 工单四：图像与清单默认输出目录

# 工单四：L2 矢量渲染触发关键词（页面文字命中才做矢量聚类渲染，
# 避免把全文档的表格网格误渲染为图像；招股说明书2 组织结构图即矢量绘制）
DIAGRAM_KEYWORDS = (
    "组织结构", "组织架构", "架构图", "流程图", "示意图",
    "结构如下图", "如下图所示", "结构如图",
)
L2_MIN_REGION_W = 150     # 工单四：L2 聚类区域最小宽度（pt）
L2_MIN_REGION_H = 120     # 工单四：L2 聚类区域最小高度（pt）
L2_MIN_PATHS = 4          # 工单四：聚类内矢量图元数下限
L2_MERGE_GAP = 20.0       # 工单四：矩形合并间距（pt）
L2_DPI = 200              # 工单四：矢量区域渲染分辨率
L2_RASTER_OVERLAP = 0.5   # 工单四：与已有位图 bbox 重叠率超此值则跳过（防重复）


class ImageExtractor:
    """工单四：PDF 图像提取器（L1 位图提取 + 过滤 + 去重 + 元数据落盘）"""

    def __init__(self, out_dir: str = DEFAULT_OUT_DIR, min_side_px: int = MIN_SIDE_PX,
                 enable_l2: bool = True):
        # 工单四：输出根目录（图像文件与 {doc_name}_images.json 均在其下）
        self.out_dir = Path(out_dir)
        self.min_side_px = min_side_px
        self.enable_l2 = enable_l2   # 工单四：是否启用 L2 矢量区域渲染

    # ------------------------------------------------------------------
    def extract_pdf(self, pdf_path: str, doc_id: Optional[str] = None,
                    out_json: Optional[str] = None) -> Dict[str, Any]:
        """工单四：提取单个 PDF 的全部图像并输出清单 JSON"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
        # 工单四：doc_name 取 PDF 文件名（去扩展名），与文本/表格库 doc_id 规范一致
        doc_id = doc_id or pdf_path.stem
        img_dir = self.out_dir / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        images: List[Dict[str, Any]] = []
        md5_seen: Dict[str, int] = {}     # 工单四：md5 -> 首次出现的全局序号（去重）
        stats = {
            "pages": 0, "raw_hits": 0, "extracted": 0,
            "filtered_small": 0, "duplicates": 0, "failed": 0,
            "l2_renders": 0,
        }

        def _admit(meta: Dict[str, Any]) -> bool:
            """工单四：小图标过滤 + md5 去重 + 孤儿文件清理，返回是否保留"""
            nonlocal global_seq
            if meta["width"] < self.min_side_px or meta["height"] < self.min_side_px:
                stats["filtered_small"] += 1
                Path(meta.pop("path")).unlink(missing_ok=True)
                return False
            if meta["md5"] in md5_seen:
                stats["duplicates"] += 1
                Path(meta.pop("path")).unlink(missing_ok=True)
                return False
            md5_seen[meta["md5"]] = len(images)
            global_seq += 1
            meta["image_id"] = f"img_{global_seq:03d}"   # 工单四：全局编号，供检索库引用
            images.append(meta)
            stats["extracted"] += 1
            return True

        with pymupdf.open(pdf_path) as doc:
            stats["pages"] = doc.page_count
            global_seq = 0
            # 工单四：预计算 L2 触发页集合——关键词命中页 + 其下一页
            #（"组织结构如下图"引文常在图的前一页，图在下一页，如招股说明书2 p38引文→p39图）
            trigger_pages = set()
            if self.enable_l2:
                for pno in range(doc.page_count):
                    if any(kw in doc.load_page(pno).get_text()
                           for kw in DIAGRAM_KEYWORDS):
                        trigger_pages.add(pno)
                        trigger_pages.add(min(pno + 1, doc.page_count - 1))

            for page_no in range(doc.page_count):          # 工单四：逐页遍历
                page = doc.load_page(page_no)
                page_raster_bboxes: List[Any] = []         # 工单四：本页已提取位图 bbox（L2 防重）
                # full=True 返回 smask 等完整信息；xref 为图像对象引用
                for raw_idx, img_info in enumerate(page.get_images(full=True), start=1):
                    stats["raw_hits"] += 1
                    xref = img_info[0]
                    meta = self._extract_one(
                        doc, page, xref, raw_idx,
                        doc_id=doc_id, img_dir=img_dir, seq=global_seq + 1,
                    )
                    if meta is None:
                        stats["failed"] += 1
                        continue
                    if _admit(meta) and meta.get("bbox"):
                        page_raster_bboxes.append(meta["bbox"])

                # 工单四：L2 矢量区域渲染——仅触发页执行（关键词命中页及其下一页）
                if self.enable_l2 and page.number in trigger_pages:
                    for k, clip in enumerate(
                        self._vector_regions(page, page_raster_bboxes), start=1
                    ):
                        meta = self._render_clip(
                            page, clip, k, doc_id=doc_id, img_dir=img_dir,
                            seq=global_seq + 1,
                        )
                        if meta is None:
                            stats["failed"] += 1
                            continue
                        if _admit(meta):
                            stats["l2_renders"] += 1

        result = {
            "doc_id": doc_id,
            "source_pdf": str(pdf_path),
            "work_order": WORK_ORDER,                       # 工单四：工单编号溯源
            "extract_source": ("L1_raster+L2_vector" if stats["l2_renders"]
                               else "L1_raster"),           # 工单四：提取层级标识
            "min_side_px": self.min_side_px,
            "images": images,
            "stats": stats,
            "elapsed_sec": round(time.time() - t0, 2),
        }

        # 工单四：清单输出（默认 data/images/{doc_name}_images.json，可由 --out 覆盖）
        out_path = Path(out_json) if out_json else self.out_dir / f"{doc_id}_images.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["out_json"] = str(out_path)
        logger.info(
            f"[image_extractor] {doc_id}: pages={stats['pages']} "
            f"raw={stats['raw_hits']} kept={stats['extracted']} "
            f"small={stats['filtered_small']} dup={stats['duplicates']} "
            f"failed={stats['failed']} -> {out_path}"
        )
        return result

    # ------------------------------------------------------------------
    def _extract_one(self, doc: "pymupdf.Document", page: "pymupdf.Page",
                     xref: int, page_index: int, doc_id: str,
                     img_dir: Path, seq: int) -> Optional[Dict[str, Any]]:
        """工单四：提取单个 xref 图像 → 统一转 PNG 落盘并组装元数据"""
        try:
            info = doc.extract_image(xref)                 # 工单四：原始字节与格式
            raw = info["image"]
            width, height = int(info.get("width", 0)), int(info.get("height", 0))
            src_ext = info.get("ext", "png")

            # 工单四：bbox——图像在页面上的摆放位置（get_image_rects 可返回多个，取首个）
            bbox = None
            try:
                rects = page.get_image_rects(xref)
                if rects:
                    r = rects[0]
                    bbox = [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)]
            except Exception:                           # 工单四：未摆放的孤立图像对象
                bbox = None

            # 工单四：统一转 PNG（CMYK/灰度归一为 RGB；非法字节判 failed）
            buf = io.BytesIO()
            with Image.open(io.BytesIO(raw)) as im:
                if im.mode in ("CMYK", "P", "RGBA", "LA"):
                    im = im.convert("RGB")
                elif im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            md5 = hashlib.md5(png_bytes).hexdigest()
            # 工单四：文件名 page_{page}_img_{index}.png（page 从 1 计）
            fname = f"page_{page.number + 1:03d}_img_{page_index}.png"
            fpath = img_dir / fname
            fpath.write_bytes(png_bytes)

            return {
                "doc_id": doc_id,
                "page": page.number + 1,                # 工单四：1-based 页码
                "image_index": page_index,              # 工单四：页内序号（与文件名一致）
                "bbox": bbox,                           # 工单四：页面位置 [x0,y0,x1,y1]
                "width": width,                         # 工单四：像素宽（原始）
                "height": height,                       # 工单四：像素高（原始）
                "format": src_ext,                      # 工单四：源格式（统一转 PNG 落盘）
                "extract_source": "L1_raster",          # 工单四：L1 位图提取
                "path": str(fpath),
                "md5": md5,
                "status": "ok",
            }
        except Exception as e:                          # 工单四：单图失败不阻塞整册
            logger.warning(f"[image_extractor] xref={xref} 提取失败: {e}")
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _vector_regions(page: "pymupdf.Page", raster_bboxes: List[Any]) -> List[Any]:
        """工单四：L2 —— 聚类矢量图元为候选图形区域（组织结构图/流程图等）

        策略：取 get_drawings 的矩形，按 L2_MERGE_GAP 迭代合并相交/邻近区域；
        过滤过小区域与与已提取位图高度重叠的区域（防止同一图双份）。
        """
        try:
            rects = [d["rect"] for d in page.get_drawings() if not d.get("fill_only")]
        except Exception:
            return []
        rects = [r for r in rects if r.width > 2 or r.height > 2]  # 工单四：去噪点
        if len(rects) < L2_MIN_PATHS:
            return []

        # 工单四：迭代合并（水平/垂直间距均 < L2_MERGE_GAP 视为同一图形区域）
        merged = list(rects)
        changed = True
        while changed:
            changed = False
            out = []
            while merged:
                cur = merged.pop()
                i = 0
                while i < len(merged):
                    o = merged[i]
                    # 工单四：间距 = 两矩形在轴上的空隙（相交时为负，天然覆盖相交情形）
                    h_gap = max(cur.x0, o.x0) - min(cur.x1, o.x1)
                    v_gap = max(cur.y0, o.y0) - min(cur.y1, o.y1)
                    if h_gap < L2_MERGE_GAP and v_gap < L2_MERGE_GAP:
                        cur = cur | o                   # 工单四：并集扩散
                        merged.pop(i)
                        changed = True
                    else:
                        i += 1
                out.append(cur)
            merged = out
        # 工单四：尺寸过滤 + 与位图重叠过滤
        keep = []
        for r in merged:
            if r.width < L2_MIN_REGION_W or r.height < L2_MIN_REGION_H:
                continue
            dup = False
            for bb in raster_bboxes:
                inter = pymupdf.Rect(bb) & r
                area_inter = max(inter.width, 0) * max(inter.height, 0)
                area_r = r.width * r.height
                if area_r > 0 and area_inter / area_r > L2_RASTER_OVERLAP:
                    dup = True
                    break
            if not dup:
                keep.append(r)
        return keep

    # ------------------------------------------------------------------
    def _render_clip(self, page: "pymupdf.Page", clip: Any, region_idx: int,
                     doc_id: str, img_dir: Path, seq: int) -> Optional[Dict[str, Any]]:
        """工单四：L2 —— 将矢量区域渲染为 PNG 并组装元数据"""
        try:
            pix = page.get_pixmap(clip=clip, dpi=L2_DPI)   # 工单四：含区域内的文字一并渲染
            png_bytes = pix.tobytes("png")
            md5 = hashlib.md5(png_bytes).hexdigest()
            fname = f"page_{page.number + 1:03d}_draw_{region_idx}.png"
            fpath = img_dir / fname
            fpath.write_bytes(png_bytes)
            return {
                "doc_id": doc_id,
                "page": page.number + 1,
                "image_index": region_idx,
                "bbox": [round(clip.x0, 2), round(clip.y0, 2),
                         round(clip.x1, 2), round(clip.y1, 2)],
                "width": pix.width,
                "height": pix.height,
                "format": "png",
                "extract_source": "L2_vector",             # 工单四：L2 矢量渲染标识
                "path": str(fpath),
                "md5": md5,
                "status": "ok",
            }
        except Exception as e:                             # 工单四：单区域失败不阻塞
            logger.warning(f"[image_extractor] L2 渲染失败 p{page.number + 1}: {e}")
            return None


def main() -> None:
    """工单四：命令行入口
    示例：
      python -m src.image_parser.image_extractor --pdf "../附件/招股说明书2.pdf" \
          --out "data/images/招股说明书2_images.json"
    """
    parser = argparse.ArgumentParser(
        description="工单四 PDF 图像提取（人工智能NLP-RAG-图像内容解析及检索优化）"
    )
    parser.add_argument("--pdf", nargs="+", required=True, help="PDF 路径（可多个）")
    parser.add_argument("--out", default=None,
                        help="清单 JSON 输出路径（单个 PDF 时生效；缺省 data/images/{doc_name}_images.json）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="图像输出根目录")
    parser.add_argument("--min-side", type=int, default=MIN_SIDE_PX, help="小图标过滤阈值(px)")
    parser.add_argument("--no-l2", action="store_true",
                        help="关闭 L2 矢量区域渲染（默认开启，仅关键词命中页触发）")
    args = parser.parse_args()

    extractor = ImageExtractor(out_dir=args.out_dir, min_side_px=args.min_side,
                               enable_l2=not args.no_l2)
    for i, pdf in enumerate(args.pdf):
        # 工单四：多 PDF 时 --out 仅对首个生效，其余按默认命名
        out_json = args.out if (args.out and len(args.pdf) == 1) else None
        result = extractor.extract_pdf(pdf, out_json=out_json)
        # 工单四：摘要输出（图像数量 + 前 3 张元数据，供验收核对）
        kept = result["stats"]["extracted"]
        print(f"\n=== {result['doc_id']} ===")
        print(f"图像数量: {kept}  (raw={result['stats']['raw_hits']}, "
              f"small={result['stats']['filtered_small']}, dup={result['stats']['duplicates']}, "
              f"failed={result['stats']['failed']})")
        for meta in result["images"][:3]:
            print(f"  {meta['image_id']} p{meta['page']} {meta['width']}x{meta['height']} "
                  f"{meta['format']} bbox={meta['bbox']} -> {meta['path']}")


if __name__ == "__main__":
    main()
