# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本脚本职责：在**真实候选池**上对比三种重排后端的耗时与排序一致性，回答一个问题 ——
#             「重排量化能不能作为既保准确率、又压缩响应时间的第三条路径？」
#
# 重要前提：线上 rerank 实际吃的是 FUSION_TOP_N=60 条候选（rrf_fuse 截断），
#          RERANK_POOL=14 只决定输出的截断条数、不限制送入模型的条数。
#          所以本脚本按 60 对/题来测，与线上负载一致；同时也顺势测一遍“减少送入条数”这条
#          从未被验证过的杠杆（《优化记录》§10.2 变的是输出条数，不是输入条数）。
#
# 用法：python scripts/bench_rerank_onnx.py [--repeat 5] [--onnx-dir D:/模型/bge-reranker-v2-m3-onnx]

import argparse                        # 命令行参数解析
import json                            # 读题目集
import statistics                      # 取中位数
import sys                             # 调整模块搜索路径
import time                            # 计时
from pathlib import Path               # 路径处理

import numpy as np                     # 数组运算与相关系数

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 项目根目录入 sys.path

from app import config as C            # 全局配置
from app import llm as L               # Query 理解（重排用的查询串由它产出）
from app import retrieval as R         # 线上召回与重排实现


# ============================ 候选池复现 ============================
def build_pools() -> list[dict]:                                       # 为 10 道题各复现一份真实候选池
    """照抄 retrieval.retrieve 的候选池构造过程，返回 [{id, query, fused}]。

    只复现到「RRF 融合之后、重排之前」这一步——那正是三种后端要比较的输入。
    """
    qs = json.loads((C.EVAL_DIR / "questions.json").read_text(encoding="utf-8"))["questions"]   # 题目集
    pools = []                                                         # 结果列表
    for item in qs:                                                    # 逐题复现
        q = item["question"]                                           # 原始问题
        qu = L.understand_query(q, "auto")                             # Query 理解（与线上同参）
        dense = R.search_dense(q, None, None)                          # 向量召回
        sparse = R.search_bm25(q, None)                                # BM25 召回
        rewritten = qu.get("rewritten")                                # 改写串
        if rewritten and (rewritten != q or qu.get("sub_questions")):  # 线上会补一轮
            d2 = R.search_dense(rewritten, qu.get("sub_questions"), None)   # 改写式向量召回
            s2 = R.search_bm25(rewritten, None)                        # 改写式关键词召回
            dense, sparse = R._merge_hits(dense, d2), R._merge_hits(sparse, s2)   # 合并去重
        for d in dense:                                                # 标注来源通道（与线上一致）
            d["source"] = "dense"                                      # 向量
        for s in sparse:                                               # 同上
            s["source"] = "bm25"                                       # 关键词
        fused = R.rrf_fuse([dense, sparse])                            # RRF 融合（截到 FUSION_TOP_N）
        pools.append({"id": item.get("id"), "question": q, "query": rewritten or q,
                      "expect": item.get("expect", []), "fused": fused})   # 记录
        print(f"  [{item.get('id')}] 候选 {len(fused)} 条")               # 进度
    return pools                                                       # 返回


# ============================ 三种后端打分 ============================
def score_torch_gpu(tok, model, query: str, cands: list[dict]) -> tuple[np.ndarray, float]:   # 线上后端
    """完全走 app.retrieval.rerank 的现有实现（GPU fp16），返回 (分数, 耗时)。"""
    import torch                                                       # 延迟导入
    copies = [{**c} for c in cands]                                    # 复制，避免污染原始候选
    t0 = time.time()                                                   # 计时
    R.rerank(query, copies, top_k=C.RERANK_POOL)                       # 走线上同一个函数
    dt = time.time() - t0                                              # 耗时
    return np.array([c["rerank_score"] for c in copies]), dt           # 按原序返回分数


def score_onnx(ort, tok, sess, query: str, cands: list[dict]) -> tuple[np.ndarray, float]:    # ONNX 后端
    """用 ONNX Runtime 复现 rerank 的打分逻辑：同样的截断长度、同样的批内长度排序。"""
    pairs = [[query, c["content"][:1200]] for c in cands]              # 与线上一致的输入构造
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))  # 按片段长度升序（线上同款）
    enc = tok([pairs[j] for j in order], padding=True, truncation=True,
              max_length=512, return_tensors="np")                     # 一次编码（与线上 max_length 一致）
    t0 = time.time()                                                   # 计时
    logits = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                             "attention_mask": enc["attention_mask"].astype(np.int64)})[0]   # 前向
    dt = time.time() - t0                                              # 耗时
    raw = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))                    # sigmoid → 0~1 相关度（同线上）
    scores = np.zeros(len(pairs))                                      # 按原始下标回填
    for k, j in enumerate(order):                                      # 位置 → 原始下标
        scores[j] = raw[k]                                             # 回填
    return scores, dt                                                  # 返回


# ============================ 一致性度量 ============================
def spearman(a: np.ndarray, b: np.ndarray) -> float:                   # 秩相关（避免引入 scipy）
    ra = np.argsort(np.argsort(a))                                     # a 的秩
    rb = np.argsort(np.argsort(b))                                     # b 的秩
    ra, rb = ra - ra.mean(), rb - rb.mean()                            # 中心化
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())                 # 分母
    return float((ra * rb).sum() / denom) if denom else 1.0            # 返回相关系数


def topk_set(scores: np.ndarray, k: int) -> set:                       # 取分数最高的 k 个下标集合
    return set(np.argsort(-scores)[:k].tolist())                       # 降序取前 k


def answer_index(cands: list[dict], expect: list[str]) -> int | None:   # 含答案片段在候选池中的下标
    """返回「含全部关键事实」的片段下标（0 起）；池子里找不到返回 None。"""
    norm = lambda t: "".join(ch for ch in (t or "") if ch not in " ,，%％").replace("％", "%")   # 归一化
    for i, c in enumerate(cands):                                      # 逐条检查
        if all(norm(e) in norm(c["content"]) for e in expect):          # 关键事实全部命中
            return i                                                   # 返回下标
    return None                                                        # 未找到


def answer_rank(scores: np.ndarray, cands: list[dict], expect: list[str]) -> int | None:   # 含答案片段的名次
    """返回「含全部关键事实」的片段在重排后的名次（0 起）；找不到返回 None。"""
    hit = answer_index(cands, expect)                                  # 先定位片段
    if hit is None:                                                    # 池子里根本没有
        return None                                                    # 返回 None
    order = np.argsort(-scores)                                        # 降序名次
    return int(np.where(order == hit)[0][0])                           # 命中片段的排名


def depth_measure(pool: dict, depths: list[int], repeat: int) -> list[dict]:   # 输入条数实扫
    """真跑重排：只把 RRF 前 N 条送进模型，量耗时，并看含答案片段是否还在最终 top-8。

    这是《优化记录》§10.2 当时**没有测**的那条杠杆——那里变的是 RERANK_POOL（输出截断），
    而 rerank 实际吃的是 rrf_fuse 截断后的 FUSION_TOP_N=60 条。
    送进模型的条数才是决定重排耗时的量。
    """
    cands = pool["fused"]                                              # 候选池（已按 RRF 降序）
    hit = answer_index(cands, pool["expect"])                          # 含答案片段在池中的位置
    out = []                                                           # 结果
    for n in depths:                                                   # 逐个深度
        if n > len(cands):                                             # 池子不够长
            continue                                                   # 跳过
        sub = cands[:n]                                                # 只取前 n 条送进重排
        copies = [{**c} for c in sub]                                  # 复制，避免污染
        times = []                                                     # 耗时样本
        for i in range(repeat + 1):                                    # 丢弃首次 + 重复
            t0 = time.time()                                           # 计时
            # 完全照抄 retrieve 的两步：先重排出 RERANK_POOL 条候选，再近重复去重截到 RERANK_TOP_K
            pool14 = R.rerank(pool["query"], copies, top_k=C.RERANK_POOL)   # 第一步
            if i > 0:                                                  # 丢弃首次
                times.append(time.time() - t0)                         # 记录
        kept = {id(c) for c in R.dedup_near(pool14)[:C.RERANK_TOP_K]}   # 第二步（与线上同一批 dict 对象）
        in_input = hit is not None and hit < n                          # 含答案片段是否还在送进去的这 n 条里
        out.append({"depth": n, "seconds": round(statistics.median(times), 3),
                    "answer_in_input": bool(in_input),                  # 答案是否还在输入里
                    "answer_in_top8": bool(in_input and id(copies[hit]) in kept)})   # 是否进了最终 top-8
    return out                                                         # 返回


def median_time(fn, repeat: int) -> tuple[np.ndarray, float, list[float]]:   # 计时：丢弃首次 + 取中位数
    """按「丢弃首次、重复 repeat 次取中位数」的协议计时，返回 (分数, 中位耗时, 全部耗时)。"""
    scores, times = None, []                                           # 结果与耗时
    for i in range(repeat + 1):                                        # 多跑一次用于丢弃
        scores, dt = fn()                                              # 执行
        if i > 0:                                                      # 丢弃首次（含预热与缓存抖动）
            times.append(dt)                                           # 记录
    return scores, statistics.median(times), times                     # 返回


def main() -> int:                                                     # 脚本主函数
    ap = argparse.ArgumentParser(description="对比 torch/ONNX 重排后端的耗时与排序一致性")   # 参数解析器
    ap.add_argument("--repeat", type=int, default=5, help="torch 侧计时重复次数（丢弃首次后取中位数）")   # 重复
    ap.add_argument("--onnx-repeat", type=int, default=1, help="ONNX 侧重复次数；CPU 上一次前向 20s 量级，默认不重复")
    ap.add_argument("--onnx-dir", default="D:/模型/bge-reranker-v2-m3-onnx", help="ONNX 产物目录")  # 产物目录
    ap.add_argument("--skip-onnx", action="store_true", help="只测 torch 基线")   # 跳过 ONNX
    ap.add_argument("--with-fp32", action="store_true", help="额外测 ONNX fp32（CPU 上约 40s/次，很慢）")
    args = ap.parse_args()                                             # 解析

    import onnxruntime as ort                                          # 延迟导入
    print(f"onnxruntime {ort.__version__}，可用 provider：{ort.get_available_providers()}")   # 环境信息

    print("[1/4] 复现 10 道题的真实候选池（需 Milvus 已建索引）")           # 进度
    pools = build_pools()                                              # 构造
    depth = [len(p["fused"]) for p in pools]                           # 各题候选数
    print(f"      候选数：min={min(depth)} max={max(depth)} 均值={sum(depth) / len(depth):.1f}")   # 汇总

    print("[2/4] 加载 torch 线上后端（GPU fp16）")                         # 进度
    tok_r, model = R._get_reranker()                                   # 复用线上单例（同一实现、同一设备）
    if model is None:                                                  # 加载失败
        print("[错误] torch 重排模型加载失败，无法对比。")                   # 提示
        return 2                                                       # 退出码 2
    print(f"      设备={C.EMBED_DEVICE}，候选池配置 FUSION_TOP_N={C.FUSION_TOP_N} / RERANK_POOL={C.RERANK_POOL}")

    sessions = {}                                                      # ONNX 会话表
    if not args.skip_onnx:                                             # 需要测 ONNX
        from transformers import AutoTokenizer                         # 延迟导入
        onnx_dir = Path(args.onnx_dir)                                 # 产物目录
        tok_o = AutoTokenizer.from_pretrained(str(onnx_dir))           # 与导出同一份分词器
        # 默认只测 int8：量化的产物就是它，而且 CPU 上一次 fp32 前向要 40s 量级，
        # 全量跑一遍要多花十几分钟。fp32 的 CPU 开销已由单点探针测过（见《优化记录》§10.6）。
        variants = [("onnx_int8_cpu", "model.int8.onnx")]              # 默认变体
        if args.with_fp32:                                             # 需要时才加上 fp32
            variants.insert(0, ("onnx_fp32_cpu", "model.onnx"))        # 插到前面
        for tag, name in variants:                                     # 逐个变体
            f = onnx_dir / name                                        # 文件路径
            if not f.exists():                                         # 不存在
                print(f"      跳过 {tag}：找不到 {f}")                    # 提示
                continue                                               # 下一个
            sessions[tag] = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])   # 建会话
            print(f"      已加载 {tag}")                                # 进度

    print("[3/4] 逐题测三种后端（丢弃首次 + 重复 {} 次取中位数）".format(args.repeat))   # 进度
    report = []                                                        # 逐题结果
    for p in pools:                                                    # 逐题
        row = {"id": p["id"], "n": len(p["fused"])}                    # 本行
        base_scores, base_med, base_all = median_time(                   # torch 基线
            lambda: score_torch_gpu(tok_r, model, p["query"], p["fused"]), args.repeat)   # 执行
        row["torch_gpu_fp16"] = base_med                               # 记录耗时
        row["base_candidates"] = base_all                              # 明细
        for tag, sess in sessions.items():                             # 逐个 ONNX 后端
            sc, med, allt = median_time(                               # 计时
                lambda s=sess: score_onnx(ort, tok_o, s, p["query"], p["fused"]), args.onnx_repeat)   # 执行
            row[tag] = med                                             # 耗时
            row[f"{tag}_candidates"] = allt                            # 明细
            row[f"{tag}_maxdiff"] = float(np.abs(sc - base_scores).max())   # 分数最大偏差
            row[f"{tag}_spearman"] = round(spearman(sc, base_scores), 4)    # 秩相关
            row[f"{tag}_top8_overlap"] = len(topk_set(sc, 8) & topk_set(base_scores, 8))   # 前 8 重合数
            row[f"{tag}_answer_rank"] = answer_rank(sc, p["fused"], p["expect"])   # 含答案片段名次
        row["torch_answer_rank"] = answer_rank(base_scores, p["fused"], p["expect"])   # 基线名次
        report.append(row)                                             # 收集
        print(f"  [{p['id']}] torch={base_med:.3f}s "                  # 打印本行
              + " ".join(f"{t}={row[t]:.3f}s" for t in sessions)       # ONNX 各后端耗时
              + f" | 前8重合 " + " ".join(f"{t}={row[f'{t}_top8_overlap']}/8" for t in sessions))

    print("[4/4] 汇总")                                                 # 进度
    def avg(key):                                                      # 求均值的小工具
        vals = [r[key] for r in report if r.get(key) is not None]       # 过滤空值
        return sum(vals) / len(vals) if vals else float("nan")          # 返回均值

    print(f"\n{'后端':<18}{'重排耗时均值':>14}{'分数最大偏差':>14}{'秩相关':>10}{'前8重合':>10}")   # 表头
    print(f"{'torch_gpu_fp16':<18}{avg('torch_gpu_fp16'):>13.3f}s{'-':>14}{'-':>10}{'-':>10}")   # 基线行
    for tag in sessions:                                               # ONNX 行
        print(f"{tag:<18}{avg(tag):>13.3f}s{avg(f'{tag}_maxdiff'):>14.2e}"
              f"{avg(f'{tag}_spearman'):>10.4f}{avg(f'{tag}_top8_overlap'):>9.1f}/8")
    hits_t = [r["torch_answer_rank"] for r in report if r["torch_answer_rank"] is not None]   # 基线名次
    print(f"\n含答案片段在重排后的名次（torch 基线）：{hits_t}")             # 逐题名次
    for tag in sessions:                                               # ONNX 名次对比
        hits = [r[f"{tag}_answer_rank"] for r in report]               # 逐题
        same = sum(1 for a, b in zip(hits, [r["torch_answer_rank"] for r in report]) if a == b)   # 名次一致数
        print(f"  {tag}：{hits}（与基线名次一致 {same}/{len(report)}）")   # 打印

    print("\n[补充] 送入重排的条数 FUSION_TOP_N vs 耗时 / 答案是否还在 top-8")   # 补充实验
    print("        （真跑重排，不复用分数；这是《优化记录》§10.2 没测过的那条杠杆）")
    depths = [14, 20, 30, 45, C.FUSION_TOP_N]                          # 待扫的送入条数
    sweep = {d: {"t": [], "in_input": 0, "in_top8": 0} for d in depths}   # 累计
    q_with_answer = 0                                                  # 池中含答案的题数
    detail = []                                                        # 逐题明细
    for p in pools:                                                    # 逐题
        rows = depth_measure(p, depths, args.repeat)                   # 实扫
        detail.append({"id": p["id"], "rows": rows})                   # 记录
        if any(r["answer_in_input"] for r in rows):                    # 池里确实含答案
            q_with_answer += 1                                         # 计数
        for r in rows:                                                 # 累计
            sweep[r["depth"]]["t"].append(r["seconds"])                # 耗时
            sweep[r["depth"]]["in_input"] += int(r["answer_in_input"])  # 答案仍在输入里
            sweep[r["depth"]]["in_top8"] += int(r["answer_in_top8"])    # 答案仍在最终 top-8
    print(f"{'送入条数':>10}{'重排耗时均值':>14}{'答案仍在输入':>16}{'答案仍在 top-8':>18}")   # 表头
    for d in depths:                                                   # 逐深度
        s = sweep[d]                                                   # 取累计
        if not s["t"]:                                                 # 无样本
            continue                                                   # 跳过
        print(f"{d:>10}{sum(s['t']) / len(s['t']):>13.3f}s"
              f"{s['in_input']:>12}/{q_with_answer}{s['in_top8']:>14}/{q_with_answer}")
    print("        （分母 = 候选池中确实含答案的题数）")                       # 说明

    out = C.EVAL_DIR / "results" / f"rerank_onnx_bench_{time.strftime('%Y%m%d_%H%M%S')}.json"   # 落盘路径
    out.write_text(json.dumps({"backends": report, "depth_sweep": detail},
                              ensure_ascii=False, indent=2), encoding="utf-8")   # 写文件
    print(f"\n逐题明细已落盘：{out}")                                      # 提示
    return 0                                                           # 成功退出码


if __name__ == "__main__":            # 仅在直接运行时执行
    raise SystemExit(main())          # 用返回值作为退出码
