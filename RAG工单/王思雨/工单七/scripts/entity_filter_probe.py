# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— 第四步方案③/C 的实测探针（零 API 成本，纯 GPU）
# 目的：把评审提的两条建议从"推测"变成"实测数字"，并暴露其失效边界：
#   ① 方案③「实体约束检索」：题面点名公司时，按元数据(code)过滤，看目标公司占比能提到多少；
#      ——关键边界：题面只写 `xx银行` 的题根本无公司名可识别，过滤不触发（Q01/Q02）。
#   ② 方案C「重排阈值截断」：先量 top-8 重排分的真实分布，再判断"0.85 阈值"是否成立。
# 纪律：探针先用已知正/反例自检（实体识别器必须"已知正例识别出、已知反例不误识别"），不通过即中止。
# 用法：D:/myconda1/python.exe scripts/entity_filter_probe.py
import sys, json, pathlib, collections, statistics                 # 读题集/统计
import numpy as np, jieba                                            # BM25 打分
ROOT = pathlib.Path(__file__).resolve().parent.parent                # 项目根
sys.path.insert(0, str(ROOT))                                        # 可 import app
from app import config as C                                          # 配置
jieba.setLogLevel(60)                                                # 静音

# 公司名 → 股票代码（含常见简称/全称，务必覆盖"中国邮政储蓄银行""中国太平洋保险"这类全称）
ALIAS = {"平安银行": "000001", "中国平安": "601318", "招商银行": "600036", "邮储银行": "601658",
         "中国邮政储蓄银行": "601658", "邮储": "601658", "中信证券": "600030", "中国人寿": "601628",
         "中国太保": "601601", "中国太平洋保险": "601601", "太保": "601601",
         "招商证券": "600999", "国泰君安": "601211"}

def detect(question):                                                # 从题面识别主体公司（最长匹配优先）
    hits = [a for a in ALIAS if a in question]                       # 题面出现的所有公司名/别名
    if not hits:                                                     # 一个都没有（如题面只写 xx银行）
        return None, []                                              # 无主体可识别 → 过滤不触发
    longest = max(len(h) for h in hits)                              # 最长匹配优先（避免"平安银行"被"中国平安"截断）
    top = sorted({ALIAS[h] for h in hits if len(h) == longest})       # 同长度并列则全部返回，供人工看歧义
    return (top[0] if len(top) == 1 else None), top                  # 唯一才敢用；并列≥2 视为歧义、不触发

def selfcheck():                                                     # 实体识别器自检：正例必识别、反例不误识别
    pos = [("平安银行的风险管理体系如何体现其应对宏观经济周期波动的能力？", "000001"),   # 已知正例：题面点名平安银行
           ("中国邮政储蓄银行2019年年报中，绿色贷款余额是多少？", "601658")]              # 已知正例：全称也要认得出
    neg = [("xx银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？", None),        # 已知反例：只有占位符
           ("分析这些银行和保险公司在面对经济周期波动时的共同策略与差异化策略。", None)]    # 已知反例：泛称、无主体
    ev = []                                                          # 证据
    ok = True                                                        # 初判通过
    for q, want in pos:                                              # 正例必须识别出期望代码
        got, _ = detect(q); ok &= (got == want); ev.append(("正", q[:14], got, want))   # 记录
    for q, want in neg:                                              # 反例必须识别为 None（不过滤）
        got, _ = detect(q); ok &= (got == want); ev.append(("反", q[:14], got, want))   # 记录
    # 占比函数也要自检：手造 3 条片段（2 条目标公司、1 条别的）→ 期望 2/3
    f = focus_of([{"company": "平安银行"}, {"company": "平安银行"}, {"company": "中国平安"}], "平安银行")
    ok &= (abs(f - 2 / 3) < 1e-9); ev.append(("正", "focus_of 2/3", round(f, 3), round(2 / 3, 3)))
    return ok, ev                                                    # 返回

def _bm25():                                                         # 取 BM25 索引（进程内缓存）
    import pickle                                                    # 落盘的索引
    if not hasattr(_bm25, "_d"):                                     # 只读一次
        _bm25._d = pickle.loads((C.DATA / "bm25.pkl").read_bytes())   # {bm25, recs}
    return _bm25._d                                                  # 返回

def retrieve_filtered(q, code, top_n):                               # 元数据过滤版检索：向量与 BM25 都限定该 code
    from app.rag_core import embed, milvus, fuse, rerank              # 复用生产件
    vh = [dict(h["entity"], score=float(h["distance"]), src="vec") for h in milvus().search(  # 向量通道加 code 过滤
        C.MILVUS_COLLECTION, data=[list(embed([q])[0])], limit=C.VEC_TOP_K,                    # 查询向量
        filter=f'code == "{code}"',                                                            # ← 元数据过滤（Milvus 服务端执行）
        output_fields=["text", "company", "year", "page", "code"],
        search_params={"metric_type": "COSINE"}, consistency_level="Strong")[0]]               # 强一致
    d = _bm25(); recs = d["recs"]                                      # BM25 侧：只在该 code 的分块上打分
    idx = [i for i, r in enumerate(recs) if r["code"] == code]         # 属于该公司的分块下标
    sc = d["bm25"].get_scores([w for w in jieba.lcut(q) if w.strip()]) # 全语料打分（BM25Okapi 不支持子集直接打分）
    bh = [dict(recs[i], score=float(sc[i]), src="bm25") for i in idx if sc[i] > 0]             # 仅保留该 code 且非零分
    bh = sorted(bh, key=lambda x: -x["score"])[: C.BM25_TOP_K]         # 取前 k
    return rerank(q, fuse(vh, bh), n=top_n)                            # 融合 + 重排

def focus_of(hits, company):                                         # 目标公司占比（与 sweep 同口径：按 company 判，不按 code）
    # ⚠ 踩过的坑：生产 vector_search 的 output_fields 里**没有 code**，只有 company；
    #    若这里按 h["code"] 统计，向量通道的片段会被全部漏掉，基线占比被系统性低估（曾把 Q03 的 0.12 算成 0.00）。
    return sum(1 for h in hits if h.get("company") == company) / len(hits) if hits else 0.0

def main():                                                          # 入口
    ok, ev = selfcheck()                                             # 先自检
    print("实体识别器自检:", "通过" if ok else "未通过")                 # 结论
    for tag, q, got, want in ev:                                     # 证据逐条打印
        print(f"  [{tag}] {q}… → {got} (期望 {want})")                 # 可审计
    if not ok:                                                       # 不过就停
        print("自检未通过 → 中止，不产出结论"); return None             # 未验证的探针等于没有探针
    from app.rag_core import retrieve                                 # 生产检索（不过滤）
    qs = json.loads((ROOT / "tests" / "questions.json").read_text(encoding="utf-8"))["questions"]   # 10 题
    rows = []                                                        # 逐题结果
    for q in qs:                                                     # 逐题
        code, cand = detect(q["question"])                           # 识别主体
        comp = C.CODE2NAME.get(code) if code else None                # 主体公司名（占比按公司名判）
        base_hits, _ = retrieve(q["question"], mode="hybrid")         # 基线检索（对比用）
        base_focus = focus_of(base_hits, comp) if comp else None      # 基线占比（仅对有唯一主体的题）
        filt_focus = None; filt_docs = None                           # 过滤后指标
        if code:                                                     # 有唯一主体才过滤
            fh = retrieve_filtered(q["question"], code, C.RERANK_TOP_N)   # 过滤检索
            filt_focus = focus_of(fh, comp)                           # 过滤后占比
            filt_docs = sorted({h["company"] for h in fh})            # 过滤后来源公司
        base_scores = [round(h.get("rscore", 0), 3) for h in base_hits]   # 基线 top-8 重排分（看分布）
        rows.append({"id": q["id"], "focal_code": code, "candidates": cand,   # 收行
                     "base_focus": base_focus, "filtered_focus": filt_focus,  # 占比前后
                     "base_docs": sorted({h["company"] for h in base_hits}),  # 基线来源公司
                     "filtered_docs": filt_docs,                           # 过滤后来源公司
                     "base_scores": base_scores})                          # 分数分布
    print("\n=== 方案③ 实体约束检索：目标公司占比（前 → 后） ===")
    for r in rows:                                                   # 逐题
        if r["focal_code"] is None:                                  # 未触发过滤
            print(f"  {r['id']} 主体=未识别/歧义 {r['candidates']} → **过滤不触发**（题面无公司名）")
        else:                                                        # 触发
            print(f"  {r['id']} 主体={r['focal_code']} 占比 {r['base_focus']:.2f} → {r['filtered_focus']:.2f}"
                  f"  来源 {r['base_docs']} → {r['filtered_docs']}")
    applied = [r for r in rows if r["focal_code"]]                    # 触发了过滤的题
    if applied:                                                      # 有样本才算均值
        b = statistics.mean(r["base_focus"] for r in applied)         # 过滤前均值
        f = statistics.mean(r["filtered_focus"] for r in applied)     # 过滤后均值
        print(f"\n触发过滤的 {len(applied)} 题：占比均值 {b:.3f} → {f:.3f}")
    print(f"\n未触发过滤：{sum(1 for r in rows if r['focal_code'] is None)}/{len(rows)} 题"
          f"（题面无公司名，方案③对它们无效）")
    print("\n=== 方案C 重排分分布（基线 top-8，看阈值可行性） ===")
    allsc = [s for r in rows for s in r["base_scores"]]                # 全部 80 条分数
    for r in rows:                                                    # 逐题
        print(f"  {r['id']}: {r['base_scores']}")                      # 打印分布
    print(f"\n80 条重排分：min={min(allsc):.3f} p10={np.percentile(allsc,10):.3f} "
          f"中位={statistics.median(allsc):.3f} max={max(allsc):.3f}")  # 分位
    print(f"低于 0.85 的条数：{sum(1 for s in allsc if s < 0.85)}/{len(allsc)}"
          f"（若此数极小，说明 0.85 阈值几乎不裁掉任何东西、也就没有效果）")
    print("\n=== 绝对阈值 0.85 vs 相对阈值(0.5×该题最高分) 的存活片段数 ===")
    for r in rows:                                                    # 逐题看两种阈值裁完还剩几条
        s = r["base_scores"]                                         # 该题 8 条分数
        abs_keep = sum(1 for x in s if x >= 0.85)                     # 绝对阈值存活
        rel_keep = sum(1 for x in s if x >= 0.5 * max(s))             # 相对阈值存活
        flag = "  ⚠ 绝对阈值会清空该题上下文！" if abs_keep == 0 else ""   # 危险的题标出来
        print(f"  {r['id']}: 绝对0.85 剩 {abs_keep}/8，相对0.5×max 剩 {rel_keep}/8{flag}")
    (ROOT / "results" / "entity_filter_probe.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写 results/entity_filter_probe.json")                   # 落盘
    return rows                                                      # 返回

if __name__ == "__main__":                                            # 直接运行
    main()
