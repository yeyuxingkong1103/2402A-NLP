# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""把工单3 的 collection 复制到工单5 自己的 collection（避免两库互相覆盖）。

【与真实 API 对齐】工单3 的 collection 由 `build_index.py` 通过
`vector_store.ensure_collection(dim=embedder.get_dim())` 创建：向量维度
（bge-m3 → 1024）与距离度量（config.VECTOR_DISTANCE）都封在该函数内，payload
索引（source / block_type / table_id）也由它一并创建。本脚本复用同一个函数，
不另起一套向量参数，保证迁移出来的 collection 与重新建库的完全一致。

用法：
    python scripts/migrate_collection.py            # 幂等：目标已完整则跳过
    python scripts/migrate_collection.py --rebuild  # 目标点数据不完整时重建

入口必须走 `bootstrap.run_with_large_stack`：`embedder.get_dim()` 会触发
sentence_transformers 导入，而该导入在本机 Windows 主线程（约 1MB 栈）上会
间歇性 `access violation`（exit 139），与 build_index.py 同一套对策。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import bootstrap  # noqa: E402  —— 必须最先导入

from src import config, embedder, vector_store


def migrate(old: str, new: str, rebuild: bool = False) -> int:
    client = vector_store.get_client()
    names = {c.name for c in client.get_collections().collections}
    if old not in names:
        print(f"旧 collection 不存在：{old}（已有：{sorted(names)}）")
        return 0
    want = int(client.count(collection_name=old, exact=True).count)

    if new in names:
        have = vector_store.count(new)
        if have == want and not rebuild:
            print(f"新 collection 已存在且点数据完整，跳过：{new}（{have} 点）")
            return have
        if not rebuild:
            # 迁移中途崩溃 / Ctrl-C 后重跑会落在这里：目标 collection 已建但点不全，
            # 必须报错而不是静默成功，否则后续任务会一直检索不完整的数据。
            print(f"新 collection 已存在但点数不符：{new} 有 {have} 点，源有 {want} 点；"
                  f"请用 --rebuild 重建或删库后重跑", file=sys.stderr)
            return -1
        print(f"--rebuild：删除并重建 {new}（原有 {have} 点，源 {want} 点）", flush=True)
        # 复用建库同一条创建路径：维度取 embedding 模型自报值（失败时回退
        # config.EMBEDDING_DIM），距离取 config.VECTOR_DISTANCE。
        vector_store.ensure_collection(dim=embedder.get_dim(), name=new, recreate=True)
    else:
        vector_store.ensure_collection(dim=embedder.get_dim(), name=new)

    # 【必须重新取 client】本地模式下 recreate=True 会走 _purge_local_collection()，
    # 其第一步 close() 关闭缓存单例并置空（src/vector_store.py）——上面第 30 行缓存的
    # client 至此已失效，继续用会抛 "QdrantLocal instance is closed"，而此时目标已被
    # 清空。ensure_collection 内部已重建了新的 client，这里取回活句柄再进迁移循环。
    client = vector_store.get_client()

    old_count = want
    total, offset = 0, None
    while True:
        points, offset = client.scroll(
            collection_name=old, limit=512, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not points:
            break
        client.upsert(collection_name=new, points=points, wait=True)
        total += len(points)
        if offset is None:
            break
    if total != old_count:
        print(f"迁移不完整：{old} 有 {old_count} 点，仅迁移 {total} 点")
        return -1
    print(f"迁移完成：{old} -> {new}，共 {total} 点")
    return total


def main() -> int:
    rebuild = "--rebuild" in sys.argv[1:]
    try:
        code = migrate(config.GY3_COLLECTION, config.COLLECTION_OPT, rebuild=rebuild)
    finally:
        vector_store.close()  # 本地模式必须显式关闭，否则存储锁不释放
    return 0 if code >= 0 else 1


if __name__ == "__main__":
    # 与 build_index.py 相同：重量级导入必须放在 16MB 大栈线程里，避开
    # Windows 主线程栈溢出（access violation / exit 139）。
    sys.exit(bootstrap.run_with_large_stack(main))
