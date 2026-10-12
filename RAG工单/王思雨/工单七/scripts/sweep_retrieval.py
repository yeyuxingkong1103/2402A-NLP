# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— 第四步「优化」的实测台：只跑检索（不调大模型、不花 API 费用）
# 为什么要有它：第三步的 RAGAS 分数受生成措辞与判官同源偏差影响，且判官噪声底未定；要判断「调参有没有用」，
#               必须先有一个**确定性、可复现、零 API 成本**的检索质量尺子。本脚本就是这把尺子。
# 口径（与 scripts/analyze_retrieval.py 完全一致，不另立标准）：
#   ① 目标文档命中 doc_hit：召回片段里出现该题的目标公司（9 道单文档题，Q04 跨文档不计）
#   ② 目标页  命中 page_hit：召回片段里出现（目标公司, 目标页）（9 道有目标页的题）
#   ③ 模板噪声 boiler：首行落在「模板家族」的片段数（家族判据 = 全语料行首去空白后前 24 字出现 ≥8 次）
#   ④ 目标公司占比 focus：单文档题里，8 条片段中来自目标公司的比例（衡量"多答了别的公司"的杂音，越低越差）
# 用法：D:/myconda1/python.exe scripts/sweep_retrieval.py
# 纪律：本文件在跑配置之前会先用「已知答案」自检尺子（见 selfcheck），自检不过直接中止，不给结论。
import sys, json, os, pathlib, re, collections, statistics       # 读题集/结果/统计
ROOT = pathlib.Path(__file__).resolve().parent.parent             # 项目根
sys.path.insert(0, str(ROOT))                                     # 让脚本能 import app
from app import config as C                                       # 配置（改动会即时生效，retrieve 在调用时读 C.*）
HDR = lambda s: re.sub(r"\s+", "", s)[:24]                        # 行首特征：去空白取前 24 字（与 ingest/analyze 同口径）
TARGET = {"Q01": ("平安银行", 5), "Q02": ("招商银行", 5), "Q03": ("平安银行", 5), "Q04": (None, None),   # 目标文档/目标页（页码经 PDF 原文核对）
          "Q05": ("招商银行", 53), "Q06": ("邮储银行", 61), "Q07": ("中国平安", 40),
          "Q08": ("中信证券", 31), "Q09": ("中国人寿", 12), "Q10": ("国泰君安", 16)}

def load_questions():                                             # 读 10 题题面
    d = json.loads((ROOT / "tests" / "questions.json").read_text(encoding="utf-8"))   # 题集文件
    return d["questions"] if isinstance(d, dict) else d                               # 兼容两种写法

def families(idx_path=ROOT / "data" / "index.json"):              # 模板家族：行首前 24 字在语料中出现 ≥8 次
    recs = json.loads(pathlib.Path(idx_path).read_text(encoding="utf-8"))             # 分块表
    cnt = collections.Counter(HDR(l) for r in recs for l in r["text"].splitlines())   # 全语料行首频次
    return {k for k, v in cnt.items() if v >= 8 and len(k) == 24}                     # 家族集合

def score(hits, fam, target):                                     # 对一次检索的结果打分（纯确定性）
    docs = {h["company"] for h in hits}                           # 命中的公司集合
    pages = {(h["company"], h["page"]) for h in hits}             # 命中的 (公司,页) 集合
    boiler = sum(1 for h in hits if HDR(h["text"].splitlines()[0] if h["text"].splitlines() else "") in fam)  # 模板噪声片段数
    comp, page = target                                           # 该题目标
    focus = (sum(1 for h in hits if h["company"] == comp) / len(hits)) if comp and hits else None  # 目标公司占比
    return {"docs": sorted(docs), "pages": sorted(map(list, pages)), "boiler": boiler, "focus": focus,  # 逐题原始量（可审计；set 转 list 才能序列化）
            "doc_hit": (comp in docs) if comp else None,              # 目标文档是否命中
            "page_hit": ((comp, page) in pages) if (comp and page) else None}  # 目标页是否命中

def run_config(name, **over):                                     # 跑一套配置：临时改 C.* 再逐题检索
    old = {k: getattr(C, k) for k in over}                        # 备份旧值，跑完还原（不污染后续配置）
    for k, v in over.items():
        setattr(C, k, v)                                          # 即时生效：rag_core 在调用时读 C.*
    from app.rag_core import retrieve                             # 延迟导入（首次会加载模型）
    fam = families()                                              # 当前索引的模板家族
    qs = load_questions()                                         # 10 题
    rows = []                                                     # 逐题评分
    for q in qs:                                                  # 逐题
        hits, stat = retrieve(q["question"], mode="hybrid")       # 混合检索 + 重排
        s = score(hits, fam, TARGET.get(q["id"], (None, None)))   # 确定性打分
        top1 = max((h.get("rscore", 0) for h in hits), default=0) # 首条重排分（排序置信度）
        rows.append(dict(id=q["id"], **s, n=len(hits), top1=round(top1, 3), stat=stat))  # 收行（id 是字符串，须用关键字形式）
    for k, v in old.items():
        setattr(C, k, v)                                          # 还原配置
    nd = [r for r in rows if r["doc_hit"] is not None]            # 有文档目标的题
    npg = [r for r in rows if r["page_hit"] is not None]          # 有页面目标的题
    foc = [r["focus"] for r in rows if r["focus"] is not None]    # 有目标公司的题
    return {"name": name, "over": over,                           # 配置名与覆盖项
            "doc_hit": f"{sum(bool(r['doc_hit']) for r in nd)}/{len(nd)}",            # 目标文档命中率
            "page_hit": f"{sum(bool(r['page_hit']) for r in npg)}/{len(npg)}",        # 目标页命中率
            "boiler": sum(r["boiler"] for r in rows),                                  # 模板噪声总条数
            "total": sum(r["n"] for r in rows),                                        # 召回总条数
            "focus": round(statistics.mean(foc), 3) if foc else None,                  # 平均目标公司占比
            "top1": round(statistics.mean(r["top1"] for r in rows), 3),                # 平均首条重排分
            "rows": rows}                                                              # 明细

def selfcheck(fam):                                               # 尺子自检：已知合格全过 + 已知废全拦
    qs = load_questions()                                         # 题集
    from app.rag_core import retrieve                             # 复用真实链路
    hits, _ = retrieve(qs[4]["question"], mode="hybrid")           # Q05（招商银行零售指标）：已知应命中目标公司
    s = score(hits, fam, TARGET["Q05"])                            # 打分
    ok = (s["doc_hit"] is True) and (s["focus"] is not None and s["focus"] > 0)      # 正例：目标文档必须命中且占比>0
    bad = score([{"company": "不存在公司", "page": 999, "text": "无关内容无重复行"}], fam, TARGET["Q05"])  # 反例：错公司错页
    ok &= (bad["doc_hit"] is False) and (bad["page_hit"] is False)                   # 反例必须被拦下（两项都判否）
    return ok, {"正例_Q05_招商银行": s["doc_hit"], "反例_假公司假页": (bad["doc_hit"], bad["page_hit"])}  # 返回判据与证据

CONFIGS = [                                                       # 待对照的配置（第四步的优化候选）
    ("当前 0.7/0.3 top8", {}),                                     # 基线（现状）
    ("向量加权 0.85/0.15 top8", {"VEC_WEIGHT": 0.85, "BM25_WEIGHT": 0.15}),   # 更信任语义
    ("均衡 0.5/0.5 top8", {"VEC_WEIGHT": 0.5, "BM25_WEIGHT": 0.5}),           # 提高关键词权重
    ("关键词加权 0.3/0.7 top8", {"VEC_WEIGHT": 0.3, "BM25_WEIGHT": 0.7}),     # 更信任关键词（数字/指标场景）
    ("top6 0.7/0.3", {"RERANK_TOP_N": 6}),                                    # 少喂片段，降噪省 token
    ("top10 0.7/0.3", {"RERANK_TOP_N": 10}),                                  # 多喂片段，保召回
]

def main():                                                       # 入口
    fam = families()                                              # 先建家族表
    print(f"模板家族 {len(fam)} 个（判据：行首前 24 字全语料出现 ≥8 次）")
    ok, ev = selfcheck(fam)                                       # 尺子自检
    print("尺子自检:", "通过" if ok else "未通过", ev)              # 打印证据
    if not ok:                                                    # 自检不过
        print("自检未通过 → 中止，不产出任何结论（未验证的尺子等于没有尺子）")
        return None                                               # 直接退出
    out = []                                                      # 各配置结果
    for name, over in CONFIGS:                                    # 逐个配置
        r = run_config(name, **over)                              # 跑
        out.append(r)                                             # 收集
        print(f"\n=== {name} ===  目标文档 {r['doc_hit']}｜目标页 {r['page_hit']}｜模板噪声 {r['boiler']}/{r['total']}"
              f"｜目标公司占比 {r['focus']}｜均首条重排分 {r['top1']}")          # 汇总行
        for x in r["rows"]:                                       # 逐题
            dh = "N/A" if x["doc_hit"] is None else ("✓" if x["doc_hit"] else "✗")   # 命中标记
            print(f"  {dh} {x['id']} 召回{x['n']} 模板{x['boiler']} 占比{x['focus'] if x['focus'] is None else round(x['focus'],2)} 来源{sorted(x['docs'])}")
    p = ROOT / "results" / "sweep_retrieval.json"                 # 落盘
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写 {p}")                                          # 提示
    return out                                                    # 返回

if __name__ == "__main__":                                        # 直接运行
    main()
