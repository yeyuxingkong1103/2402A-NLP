# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
"""诊断脚本：把 RAGFlow 解析出来的 chunk 全部拉下来，看看解析质量

用途：每一轮优化前先看「解析到底丢没丢东西」，避免盲目调参。
输出：results/chunks_<标签>.json + 控制台摘要
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ragflow_client import RAGFlow

HERE = Path(__file__).parent
IDS = json.loads((HERE / ".ragflow_ids.json").read_text(encoding="utf-8"))

# 关心这些关键词：第7页图上的部件编号与尺寸记号
KEYS = ["X1", "X2", "X3", "h1", "h2", "部件4", "部件5", "6'", "6\"", "吉特勒", "图7", "第7页"]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "cur"
    c = RAGFlow()
    c.load_token()

    # 注意：GET /api/v1/datasets/{id} 在 RAGFlow v0.23 上返回 405，
    # 列表接口带 ?id= 才是对的
    code, j = c._req("GET", "/api/v1/datasets", headers=c._h(),
                     params={"id": IDS["ds_id"]})
    ds = (j.get("data") or [{}])[0]
    print("=== 知识库配置 ===")
    for k in ("name", "chunk_method", "parser_config", "embedding_model",
              "similarity_threshold", "vector_similarity_weight", "pagerank"):
        if k in ds:
            print(f"  {k}: {json.dumps(ds[k], ensure_ascii=False)}")

    chunks = c.chunks(IDS["ds_id"], IDS["doc_id"])
    print(f"\n=== chunk 总数：{len(chunks)} ===")

    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / f"chunks_{tag}.json"
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  已存 {out}")

    # 统计
    img_chunks = [ch for ch in chunks if "![image]" in ch.get("content", "")
                  or "<table" in ch.get("content", "")]
    print(f"  含图片/表格标记的 chunk：{len(img_chunks)}")
    print(f"  平均长度：{sum(len(ch.get('content','')) for ch in chunks)//max(len(chunks),1)} 字符")

    print("\n=== 关键词命中 ===")
    for k in KEYS:
        hits = [i for i, ch in enumerate(chunks) if k in ch.get("content", "")]
        print(f"  {k!r}: {len(hits)} 个 chunk {hits[:8]}")

    print("\n=== 前 3 个 chunk 预览 ===")
    for ch in chunks[:3]:
        txt = re.sub(r"\s+", " ", ch.get("content", ""))
        print(f"  [{ch.get('id','')[:8]}] {txt[:200]}")


if __name__ == "__main__":
    main()
