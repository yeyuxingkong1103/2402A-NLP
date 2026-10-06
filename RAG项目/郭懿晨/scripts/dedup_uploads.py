"""清理重复上传：按文件名分组，保留 indexed 文档，删除其余未构建的重复文档。

仅操作 state.json 和 data/ 下的孤儿 PDF 文件；不触碰 Qdrant（重复文档未构建、无向量点）。
"""
import json
from pathlib import Path

STATE = Path("data/state.json")


def main() -> None:
    data = json.loads(STATE.read_text(encoding="utf-8"))

    # 按文件名分组
    by_name: dict[str, list[tuple[str, dict]]] = {}
    for did, doc in data["documents"].items():
        by_name.setdefault(doc["file_name"], []).append((did, doc))

    removed: list[tuple[str, dict]] = []
    for name, docs in by_name.items():
        if len(docs) <= 1:
            continue
        indexed = [x for x in docs if x[1]["status"] == "indexed"]
        keep_ids = {x[0] for x in indexed} if indexed else {docs[0][0]}
        for did, doc in docs:
            if did not in keep_ids:
                removed.append((did, doc))

    if not removed:
        print("没有需要清理的重复文档。")
        return

    # 从 state.json 删除
    for did, _doc in removed:
        data["documents"].pop(did, None)
        for tid in list(data["tasks"]):
            if data["tasks"][tid].get("document_id") == did:
                data["tasks"].pop(tid, None)
        for cid in list(data["chunks"]):
            if data["chunks"][cid].get("document_id") == did:
                data["chunks"].pop(cid, None)

    # 删除孤儿 PDF（仅删除被删文档引用、且位于 data/ 下的文件）
    data_dir = Path("data").resolve()
    kept_paths = {d["file_path"] for d in data["documents"].values()}
    for _did, doc in removed:
        fp = Path(doc["file_path"])
        try:
            is_in_data = data_dir in fp.resolve().parents
        except Exception:
            is_in_data = False
        if fp.exists() and is_in_data and doc["file_path"] not in kept_paths:
            fp.unlink()
            print(f"  删除孤儿 PDF: {fp.name}")

    STATE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已删除 {len(removed)} 个重复文档: {[d for d, _ in removed]}")
    print(f"剩余文档: {list(data['documents'].keys())}")
    print(f"剩余任务: {len(data['tasks'])}，剩余 chunks: {len(data['chunks'])}")


if __name__ == "__main__":
    main()
