# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— 检索结果问题分析（确定性口径，不依赖 LLM 判官）
# 为什么要这个脚本：RAGAS 分数受生成侧措辞影响，且判官与生成同源存在偏差。
# 这里用可核验的确定性指标定位检索问题：目标文档命中、目标页命中、模板噪声率、通道贡献。
# 模板噪声的判据与 ingest 的修复判据一致：某个「行首特征」在语料中成家族出现（≥8 次）即为模板。
# 用法：python scripts/analyze_retrieval.py [结果.json ...] [--index data/index.json]
import sys, json, pathlib, re, collections                            # 读结果、统计
ROOT = pathlib.Path(__file__).resolve().parent.parent                 # 项目根
HDR = lambda s: re.sub(r"\s+", "", s)[:24]                            # 行首特征：去空白取前 24 字（与 ingest 判据同一口径）
# 每题的目标文档与目标页（None = 不适用，不计入分母）；页码经 PDF 原文核对
TARGET = {"Q01": ("平安银行", 5), "Q02": ("招商银行", 5), "Q03": ("平安银行", 5), "Q04": (None, None),
          "Q05": ("招商银行", 53), "Q06": ("邮储银行", 61), "Q07": ("中国平安", 40),
          "Q08": ("中信证券", 31), "Q09": ("中国人寿", 12), "Q10": ("国泰君安", 16)}

def families(index_path):                                             # 统计语料中的「行首特征家族」：哪些行首出现 ≥8 次
    if not pathlib.Path(index_path).exists():                         # 索引不存在（如只看 raw 结果）
        return None                                                  # 无法判定模板，返回 None
    recs = json.loads(pathlib.Path(index_path).read_text(encoding="utf-8"))  # 分块表
    cnt = collections.Counter(HDR(l) for r in recs for l in r["text"].splitlines())  # 全语料行首频次
    return {k for k, v in cnt.items() if v >= 8 and len(k) == 24}     # 家族规模 ≥8 的行首集合

def analyze(res, fam):                                                # 对一次结果做确定性统计
    rows = res["questions"] if isinstance(res, dict) else res         # 兼容 eval_*.json（dict）与 raw_*.json（list）
    doc_hit = page_hit = 0; n_doc = n_page = 0                        # 命中计数与分母（只统计有目标的题）
    boiler = tot = 0                                                  # 模板噪声片段数 / 总片段数
    src = collections.Counter(); detail = []                          # 通道分布 / 逐题明细
    for r in rows:                                                    # 逐题
        comp, page = TARGET.get(r["id"], (None, None))                # 该题目标文档与目标页
        metas = r.get("ctx_meta", []) or []                           # 召回片段出处
        docs = {m["company"] for m in metas}                          # 命中的公司集合
        pages = {(m["company"], m["page"]) for m in metas}            # 命中的 (公司, 页) 集合
        if comp:                                                      # 有目标文档才计入分母
            n_doc += 1; doc_hit += comp in docs                       # 文档命中（真/假）
        if comp and page:                                             # 有目标页才计入分母
            n_page += 1; page_hit += (comp, page) in pages            # 页命中
        b = 0                                                         # 该题模板噪声数
        for c, m in zip(r["contexts"], metas):                        # 逐片段判定
            if fam and HDR(c.splitlines()[0] if c.splitlines() else "") in fam:  # 首行落在模板家族里
                b += 1                                                # 记一条模板噪声
            src[m["src"]] += 1                                        # 通道贡献
        boiler += b; tot += len(metas)                                # 累加
        detail.append({"id": r["id"], "doc_hit": comp in docs if comp else None, "n": len(metas), "boiler": b,
                       "docs": sorted(docs), "err": r.get("error", "")})  # 逐题记录
    return {"n_questions": len(rows), "avg_ctx": round(tot / max(len(rows), 1), 2), "doc_hit": doc_hit, "n_doc": n_doc,
            "page_hit": page_hit, "n_page": n_page, "boiler": boiler, "total_ctx": tot,
            "boiler_rate": round(boiler / max(tot, 1), 3), "src": dict(src), "detail": detail}  # 汇总

def main():                                                           # 入口
    args = [a for a in sys.argv[1:] if not a.startswith("--")]        # 位置参数：结果文件
    idx = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--index=")), str(ROOT / "data" / "index.json"))  # 分块表
    fam = families(idx)                                               # 模板家族
    print(f"模板家族判据来自 {idx}｜命中家族数 {len(fam) if fam else 'N/A'}")   # 口径可见
    files = args or sorted((ROOT / "results").glob("eval_*.json"))     # 默认取全部评估结果
    out = {}                                                          # 汇总
    for f in files:                                                   # 逐个分析
        res = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))  # 读结果
        a = analyze(res, fam)                                         # 统计
        ragas = {k: v for k, v in (res.get("ragas", {}) if isinstance(res, dict) else {}).items() if not k.startswith("per_sample_")}  # 指标
        out[pathlib.Path(f).name] = {"analysis": a, "ragas": ragas}   # 收进汇总
        print(f"\n=== {pathlib.Path(f).name} ===")
        print(f"题数 {a['n_questions']}｜平均召回 {a['avg_ctx']} 条｜目标文档命中 {a['doc_hit']}/{a['n_doc']}"
              f"｜目标页命中 {a['page_hit']}/{a['n_page']}｜模板噪声 {a['boiler']}/{a['total_ctx']} ({a['boiler_rate']})")
        print("通道来源:", a["src"])                                  # vec / bm25 / vec+bm25
        for d in a["detail"]:                                         # 逐题
            f1 = "N/A" if d["doc_hit"] is None else ("✓" if d["doc_hit"] else "✗")   # 文档命中标记
            print(f"  {f1} {d['id']} 召回{d['n']}条 模板{d['boiler']}条 来源{d['docs']}{' ERR:'+d['err'] if d['err'] else ''}")
        if ragas: print("RAGAS:", ragas)                              # 指标（无则跳过）
    (ROOT / "results" / "retrieval_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写 results/retrieval_analysis.json")

if __name__ == "__main__":                                            # 直接运行
    main()
