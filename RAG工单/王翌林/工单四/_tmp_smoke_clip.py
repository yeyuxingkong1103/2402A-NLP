# 临时：Chinese-CLIP 冒烟（人工智能NLP-RAG-图像内容解析及检索优化）
import json

from src.image_parser.image_embedding import ImageClipEmbedder

clip = ImageClipEmbedder()
qv = clip.embed_query_text("组织结构图 销售部 销售处")
print("query vec:", None if qv is None else f"dim={len(qv)} head={qv[:3]}")

manifest = json.load(open("data/images/招股说明书2_images.json", encoding="utf-8"))
for iid in ("img_008", "img_012"):
    m = next(x for x in manifest["images"] if x["image_id"] == iid)
    iv = clip.embed_image_file(m["path"])
    print(iid, "image vec:", None if iv is None else f"dim={len(iv)} head={iv[:3]}")
    if qv and iv:
        import numpy as np
        print("  cosim:", float(np.dot(np.asarray(qv), np.asarray(iv))))
