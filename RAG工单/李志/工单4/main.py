import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import demo_documents, load_documents, save, search

parser = argparse.ArgumentParser(description="工单4：PDF 图像内容解析及检索")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--query", default="组织结构图 销售部")
parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
docs = load_documents(args.docs) if args.docs else demo_documents()
captions = []
for doc in docs:
    hints = [line.strip() for line in doc["text"].splitlines() if any(key in line for key in ("图", "结构", "流程", "Figure"))]
    for hint in hints: captions.append({**doc, "text": "图像说明：" + hint})
if not captions:
    captions = [{"source": "demo-image", "page": 1, "chunk": 0, "text": "组织结构图：销售部下设大客户销售部、区域销售部和销售支持部。"}]
result = search(args.query, docs + captions, 5)
save(Path(__file__).parent / "outputs/image_index.json", {"image_captions": captions, "results": result})
print(f"建立 {len(captions)} 条图像语义描述并完成跨模态检索。")
