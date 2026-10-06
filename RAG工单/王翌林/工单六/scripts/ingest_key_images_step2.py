# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
scripts/ingest_key_images_step2.py —— OCR 兜底迷你入库（Qwen2-VL 就绪前的临时通道，新增文件）

对 3 张重点图（img_008 组织结构图 / img_011 img_012 IC市场图）仅跑 PaddleOCR
（不依赖 Qwen2-VL），以 ocr_text 构建 bge-m3 向量先行入库，验证检索链路；
Qwen2-VL 就绪后用 --rebuild 重跑全量覆盖。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

KEY_IDS = ("img_008", "img_011", "img_012")
MANIFEST = "data/images/招股说明书2_images.json"
OUT = "data/image_descriptions/招股说明书2_images_parsed.json"


class _StubVLM:
    """工单四：无 VLM 桩（caption 置空，仅走 OCR 通道）"""
    engine_name = "stub"

    def caption(self, image):
        return ""

    def generate(self, image, prompt, max_new_tokens=512):
        return ""


def main() -> None:
    from src.image_parser.image_parser import ImageParser
    from src.image_parser.image_embedding import build_image_text
    from src.image_parser.image_store import ImageStore

    ip = ImageParser(vlm_engine=_StubVLM(), enable_vqa=False,
                     enable_ocr=True, enable_clip=False)
    result = ip.parse_images(MANIFEST, OUT, image_ids=list(KEY_IDS))
    print(f"[ocr-mini] 解析 {result['stats']['total']} 张 -> {OUT}")

    store = ImageStore()
    store.ensure_collection()
    store.delete_by_doc_id("招股说明书2")        # 工单四：重跑去重（全量入库时重建覆盖）
    records = []
    for img in result["images"]:
        ocr = img.get("ocr_text", "")
        text = build_image_text("", ocr, "")
        records.append({
            "doc_id": "招股说明书2", "image_id": img["image_id"],
            "page": img["page"], "path": img["path"],
            "caption": "", "ocr_text": ocr, "vqa_text": "",
            "embedding": None, "clip_embedding": None,
            "metadata": {"ocr_only": True, "work_order": WORK_ORDER},
        })
        records[-1]["embedding"] = None
    # 工单四：文本嵌入
    from src.image_parser.image_embedding import ImageTextEmbedder
    emb = ImageTextEmbedder()
    for r in records:
        r["embedding"] = emb.embed_text(build_image_text(r["caption"], r["ocr_text"], ""))
    n = store.insert(records)
    print(f"[ocr-mini] 入库 {n} 条, rows={store.count()}")

    # 工单四：验收演示
    from src.image_parser.image_retriever import ImageRetriever
    hits = ImageRetriever(store=store).retrieve("组织结构图 销售部", top_k=5)
    print("\n=== 检索演示 '组织结构图 销售部' top5 ===")
    for i, h in enumerate(hits, 1):
        print(f"{i}. [{h['image_id']}] p{h['page']} score={h['final_score']} kw={h['kw_hits']}")
        print(f"   path: {h['path']}")
        print(f"   ocr: {h.get('ocr_text', '')[:60]}")


if __name__ == "__main__":
    main()
