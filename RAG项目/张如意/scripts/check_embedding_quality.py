# -*- coding: utf-8 -*-
"""向量质量检查：入库前先看方向对不对。

做三件事：
  1. 术语对直接算余弦相似度（用户指定的 3 组）
  2. query -> 全库检索，看 top-k 排序合不合理
  3. **归一化前 vs 归一化后** 对比 —— 验证口语映射层到底有没有用

跑法：python scripts/check_embedding_quality.py
"""
import json
import os
import sys

EMB = "data/chunks/chunks_embedded.jsonl"   # 已编码的 chunk 向量库
MODEL = r"D:\zg6\bge-m3"                    # 本地 BGE-M3 模型目录
CAMP = json.load(open("configs/colloquial_map.json", encoding="utf-8"))  # 口语映射表
VERIFIED = [r for r in CAMP if r["status"] == "verified"]  # 只取已人工确认的条目参与归一化


def load():
    """读取已编码 chunk 的 JSONL，返回 dict 列表（含 dense_vector 与 meta 字段）。"""
    return [json.loads(l) for l in open(EMB, encoding="utf-8")]


def normalize(q, crop=None):
    """与 smoke_retrieval.py 同逻辑（单遍 + 链式）

    参数 q：原问句；crop：作物限定（用于过滤带作物条件的映射条目）。
    流程：① 收集全部命中位置并按（起点升序、同起点取最长）排序 ② 非重叠
    贪心选片，从右往左替换避免位置漂移 ③ 再做最多 3 轮链式归一（前面的
    替换可能暴露新的口语词）。返回归一化后的问句。
    """
    matches = []
    for r in VERIFIED:
        sp = r["spoken"]
        if not sp:
            continue
        if r["crop"] and crop and r["crop"] != crop:
            continue
        if r["crop"] is None and crop and any(
                x["spoken"] == sp and x["crop"] == crop for x in VERIFIED):
            continue
        start = 0
        while True:
            i = q.find(sp, start)
            if i < 0:
                break
            matches.append((i, len(sp), r))
            start = i + 1
    matches.sort(key=lambda m: (m[0], -m[1]))   # 按起点升序；同起点优先取更长匹配
    chosen, last_end = [], -1
    for i, ln, r in matches:
        if i < last_end:            # 与已选区间重叠 → 跳过（非重叠贪心）
            continue
        chosen.append((i, ln, r))
        last_end = i + ln
    out = q
    for i, ln, r in reversed(chosen):           # 从右往左替换，避免前面的替换使位置失效
        out = out[:i] + r["canonical"] + out[i + ln:]
    used = {id(r) for _, _, r in chosen}        # 已用条目不再参与链式，防 A→B→A 循环
    for _ in range(3):              # 链式归一最多 3 轮
        progressed = False
        for r in VERIFIED:
            if id(r) in used or not r["spoken"]:
                continue
            if r["crop"] and crop and r["crop"] != crop:
                continue
            if r["spoken"] in out:
                out = out.replace(r["spoken"], r["canonical"])
                used.add(id(r))
                progressed = True
        if not progressed:
            break
    return out


def main():
    """执行三段检查并打印报告。返回 0=正常，2=缺向量库文件。"""
    if not os.path.exists(EMB):
        print(f"缺 {EMB}"); return 2
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel

    rows = load()
    D = np.array([r["dense_vector"] for r in rows], dtype=np.float32)  # 全库向量矩阵
    print(f"载入 {len(rows)} 个已编码 chunk，矩阵 {D.shape}\n")

    model = BGEM3FlagModel(MODEL, use_fp16=False, devices="cpu")  # CPU 推理，检查够用

    def enc(texts):
        """批量编码文本为稠密向量（max_length=256，与建库时口径一致）。"""
        v = model.encode(texts, batch_size=8, max_length=256,
                         return_dense=True, return_sparse=False,
                         return_colbert_vecs=False)["dense_vecs"]
        return np.asarray(v, dtype=np.float32)

    # ---------------- 1. 术语对余弦 ----------------
    print("=" * 78)
    print("一、术语对余弦相似度（你指定的 3 组）")
    print("=" * 78)
    PAIRS = [
        ("黄瓜霜霉病", "黄瓜白粉病", "高", "同为黄瓜病害"),
        ("黄瓜霜霉病", "辣椒施肥", "低", "不同作物、不同事"),
        ("蚜虫防治", "腻虫怎么治", "高", "口语↔术语同义"),
        ("蚜虫防治", "蚜虫防治", "1.000", "自比对基准"),
    ]
    terms = []
    for a, b, _, _ in PAIRS:
        terms += [a, b]
    V = enc(list(dict.fromkeys(terms)))   # 术语去重后统一编码
    idx = {t: i for i, t in enumerate(dict.fromkeys(terms))}  # 术语 → 向量行号
    for a, b, want, why in PAIRS:
        s = float(np.dot(V[idx[a]], V[idx[b]]))
        print(f"  「{a}」 vs 「{b}」")
        print(f"      余弦 {s:.4f}   期望 {want}   （{why}）")

    # ---------------- 2. query -> 全库 ----------------
    print()
    print("=" * 78)
    print("二、query → 全库检索（看 top-3 排序）")
    print("=" * 78)
    QUERIES = [
        ("黄瓜霜霉病打什么药", "黄瓜"),
        ("辣椒叶子上有蚜虫", "辣椒"),
        ("大蒜什么时候施肥", "大蒜"),
        ("黄瓜白粉病", "黄瓜"),
    ]
    for q, crop in QUERIES:
        qv = enc([q])[0]
        sims = D @ qv                    # 与全库向量做内积相似度
        order = np.argsort(-sims)        # 相似度降序
        print(f"\n  「{q}」  (作物={crop})")
        for k in order[:3]:
            r = rows[k]
            mark = "✓" if r["meta"]["crop"] == crop else "✗跨作物"
            print(f"      {sims[k]:.4f}  {mark}  {r['meta']['subtype'][:14]:<16} {r['chunk_id']}")

    # ---------------- 3. 归一化前 vs 后 ----------------
    print()
    print("=" * 78)
    print("三、口语归一化到底有没有用（同一 query，前后各检索一次）")
    print("=" * 78)
    for q, crop in [("黄瓜起腻虫了打什么药", "黄瓜"), ("大蒜根蛆怎么防", "大蒜"),
                    ("黄瓜叶子上有鬼画符", "黄瓜")]:
        q2 = normalize(q, crop)          # 口语归一化（verified 条目 + 链式）
        print(f"\n  原句「{q}」  →  归一「{q2}」")
        for tag, text in (("归一前", q), ("归一后", q2)):
            qv = enc([text])[0]
            sims = D @ qv
            order = np.argsort(-sims)
            top = order[0]               # 只看 top-1 是否指向预期术语
            sub = rows[top]["meta"]["subtype"]
            rel = "命中" if sub in q2 or any(
                x["canonical"] == sub for x in VERIFIED if x["spoken"] in q) else "未必相关"
            print(f"      {tag}: {sims[top]:.4f}  {sub[:16]:<18} {rows[top]['chunk_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
