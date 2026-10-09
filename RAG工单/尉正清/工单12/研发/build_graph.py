# 工单编号：人工智能NLP-RAG项目-LightRAG优化
"""把两份招股说明书灌进 LightRAG，建成 neo4j 知识图谱（产出物 2）

流程就是工单描述的那三步：

    ① 切块 + 抽取知识   LightRAG 按 token 切块，逐块调大模型抽实体与关系
    ② 存进图谱 + 向量库 实体关系进 neo4j，实体/关系/块的向量进本地向量库
    ③ 动态更新          增量算法，重复灌同一份文档不会产生重复节点

⚠️ **跑在 `rag_gd1` 环境**，图存在 Docker 里的 neo4j。

耗时说明：招股书两份共约 84 万字符，LightRAG 会切成上千块，每块至少一次
大模型调用。默认的 deepseek-flash 是推理模型（单次 33 秒），本工单在
`lightrag_engine.llm_func` 里关掉了 reasoning（1.9 秒），否则跑不完。
详见 优化/过程问题记录.md 问题 1。

用法：
    python build_graph.py                # 增量灌入（已灌过的文档会跳过）
    python build_graph.py --reset        # 清空 neo4j 与本地缓存后重建
    python build_graph.py --no-images    # 不灌图表解析结果（第 5、6 题会答不出）
"""
import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import CACHE_ROOT, NEO4J_PASSWORD, NEO4J_URI, WORKING_DIR   # noqa: E402
from corpus import full_text, parse                                     # noqa: E402
from prospectus_docs import available, label                            # noqa: E402


def reset_graph():
    """清空 neo4j 里的图与 LightRAG 的本地工作目录。

    重跑建图必须清干净：LightRAG 认文档 id 做增量，文档内容变了而 id 没变时
    它会认为「已处理过」而跳过 —— 表现为灌了新解析结果但图谱纹丝不动。
    """
    from neo4j import GraphDatabase
    print(f"[重置] 清空 neo4j（{NEO4J_URI}）...")
    drv = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD))
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    drv.close()
    if Path(WORKING_DIR).exists():
        shutil.rmtree(WORKING_DIR)
    print("[重置] 完成")


def graph_stats():
    """数一下图里有多少实体和关系，建完图打印出来。"""
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD))
    out = {}
    with drv.session() as s:
        out["节点总数"] = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        out["实体"] = s.run(
            "MATCH (n) WHERE n.entity_id IS NOT NULL RETURN count(n) AS c"
        ).single()["c"]
        out["关系"] = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        out["实体类型"] = s.run(
            "MATCH (n) WHERE n.entity_type IS NOT NULL "
            "RETURN n.entity_type AS t, count(*) AS c ORDER BY c DESC"
        ).values()
    drv.close()
    return out


async def run(args):
    docs = available()
    if not docs:
        print("[错误] 没找到招股说明书 PDF，检查 prospectus_docs.CORPUS_DIR")
        return 1
    if args.reset:
        reset_graph()

    cache_dir = Path(CACHE_ROOT) / "images"
    from lightrag_engine import build_engine
    engine = build_engine()
    await engine.initialize_storages()

    try:
        for doc in docs:
            print(f"\n===== {label(doc)} =====", flush=True)
            _, page_texts = parse(doc, with_images=not args.no_images,
                                  cache_dir=cache_dir)
            text = full_text(page_texts)
            print(f"[灌入] {len(page_texts)} 页 / {len(text):,} 字符，"
                  f"交给 LightRAG 切块并抽取 ...", flush=True)
            t0 = time.time()
            # file_paths 只是取个名字用于溯源，内容以 text 为准
            await engine.ainsert(text, file_paths=doc["file"])
            print(f"[灌入] {doc['file']} 完成，耗时 {time.time() - t0:.0f}s", flush=True)
    finally:
        await engine.finalize_storages()

    st = graph_stats()
    print("\n===== 图谱统计 =====")
    for k in ("节点总数", "实体", "关系"):
        print(f"  {k}: {st[k]}")
    print("  实体类型分布（前 15）:")
    for t, c in list(st["实体类型"])[:15]:
        print(f"    {t:24} {c}")
    return 0



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="清空图后重建")
    ap.add_argument("--no-images", action="store_true", help="不灌图表解析结果")
    a = ap.parse_args()
    raise SystemExit(asyncio.run(run(a)))
