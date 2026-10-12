# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
# 评估层：题集 PDF 三级降级解析 → 两条链路作答 → RAGAS 上下文精度/召回（逐指标串行）→ before/after 对比表
import argparse, json, re, time; from pathlib import Path                        # CLI / 落盘 / 正则切题 / 计时 / 路径对象
from . import config as C                                                       # 本单配置
def parse_pdfplumber(path):                                                     # 兜底解析器：pdfplumber（纯 Python，不依赖 GPU 与视觉模型）
    try:                                                                        # 依赖缺失或文件损坏都要能降级
        import pdfplumber; d = pdfplumber.open(path); r = [(i + 1, t) for i, p in enumerate(d.pages) if (t := (p.extract_text() or "").strip())]; d.close(); return r or None  # 逐页取文本（页码 1 起）后及时关闭句柄，全空视为失败
    except Exception as e: print(f"[parse] pdfplumber 失败：{type(e).__name__}"); return None  # 打印原因后交给下一级
def parse_pdf(path):                                                            # 三级降级解析（工单指定 MinerU 为主 + 备用方案），全失败抛可读异常
    p = Path(path)                                                              # 统一成路径对象
    if not p.exists(): raise RuntimeError(f"找不到题集 PDF：{p}。请把 sample_questions.pdf 放到 {C.DATA} 下，或用 --sample-pdf 指定绝对路径。")  # 容错 E2：明确报错并给出修复动作，绝不编造题集
    from 工单八.app.ingest import parse_mineru, parse_pymupdf                    # 复用 08 已验证的 MinerU（含中文路径补丁与 stderr 打印）与 pymupdf 解析器
    chain = ([(parse_mineru, "MinerU")] if C.C8.MINERU_ENABLED == "1" else []) + [(parse_pdfplumber, "pdfplumber"), (parse_pymupdf, "pymupdf")]  # MinerU 整份跑需数小时，MINERU_ENABLED=0 时跳过
    for fn, name in chain:                                                      # 依次尝试，先重后轻
        try: blocks = fn(str(p))                                                # 调用当前解析器
        except Exception as e: print(f"[parse] {name} 失败：{type(e).__name__}"); continue  # 单级失败不终止，打印后降级
        if blocks: print(f"[parse] 生效解析器={name}，{len(blocks)} 页"); return "\n".join(t for _, t in blocks), name  # 拿到内容即返回全文与生效解析器名
    raise RuntimeError(f"三级解析器均失败：{p}；文件可能损坏或为无文本层扫描件")     # 全失败才抛，信息可读
QA_PAT = (r"问题\s*[：:]\s*(.+?)\s*(?:参考答案|标准答案|答案|Reference(?:\s*Answer)?|Answer)\s*[：:]?\s*(.+?)"  # 问题段 + 答案段（本附件版式为「问题：/参考答案：」，**无编号**，故不匹配数字）
          r"(?=问题\s*[：:]|$)")                                                 # 前瞻到下一题或文末
def split_qa(text):                                                             # 从题集全文中切出 (问题, 参考答案) 对（版式已实测；切不出时由调用方自检中止）
    return [{"id": f"Q{i+1:02d}", "question": re.sub(r"\s+", " ", C.fix(q)).strip(), "ground_truth": re.sub(r"\s+", " ", C.fix(a)).strip()} for i, (q, a) in enumerate(re.findall(QA_PAT, re.sub(r"[ \t]+", " ", C.fix(text)), flags=re.S))]  # 先归一化字形（康熙部首→标准汉字）再折行内空白、切题
def load_questions(path=None):                                                  # 题集加载：优先读解析缓存，否则解析 PDF 并落缓存（避免重复解析）
    if C.PARSED_JSON.exists() and not path:                                     # 有缓存且未显式指定新路径
        d = json.loads(C.PARSED_JSON.read_text(encoding="utf-8")); print(f"[eval] 读题集缓存：{d['source']}（{d['parser']}）"); return d["questions"]  # 直接返回缓存
    text, parser = parse_pdf(path or C.SAMPLE_PDF); qs = split_qa(text)          # 解析 PDF 并切题
    if not qs: raise RuntimeError(f"题集解析出 0 题：版式与解析器口径不符。请人工核对 {C.PARSED_JSON}。")  # 自检：口径不符直接中止，不给任何结论
    C.PARSED_JSON.write_text(json.dumps({"source": str(path or C.SAMPLE_PDF), "parser": parser, "questions": qs}, ensure_ascii=False, indent=1), encoding="utf-8"); print(f"[eval] 解析出 {len(qs)} 题（{parser}）→ {C.PARSED_JSON.name}，请人工核对该 JSON 的题面与参考答案")  # 落缓存（可人工校对）并提示题数
    return qs                                                                   # 返回题集
def collect(which, qs):                                                         # 跑一条链路（before=08 原链路 / after=工单九优化链路），产出 RAGAS 原始输入
    from .retrieval import answer_before, answer_after                          # 延迟导入：避免评估模块一加载就拉起模型与客户端
    fn, rows = (answer_before if which == "before" else answer_after), []         # 选择链路
    for q in qs:                                                                # 逐题
        t0 = time.time()                                                        # 计时（作答失败时也要有时延）
        try: o, err = fn(q["question"], use_cache=False), ""                     # 关缓存：评估要测真实链路，不能被缓存掩盖
        except Exception as e: o, err = None, f"{type(e).__name__}: {e}"          # 单题失败不中断整轮
        cs = (o or {}).get("contexts", [])                                      # 片段列表
        rows.append({"id": q["id"], "question": q["question"], "ground_truth": q["ground_truth"], "answer": (o or {}).get("answer", ""), "contexts": [c["text"] for c in cs],  # 题面 / 参考答案 / 答案 / 纯文本上下文
                     "latency_ms": (o or {}).get("latency_ms", round((time.time() - t0) * 1000)), "error": err,  # 时延与错误如实记录
                     "ctx_meta": [{"company": c.get("company", ""), "page": c.get("page", 0), "src": c.get("src", "")} for c in cs]})  # 片段出处（供人工核查）
        print(f"[eval] {which} {q['id']} {rows[-1]['latency_ms']}ms contexts={len(rows[-1]['contexts'])} {err[:60]}")  # 进度可见
    return rows                                                                 # 返回结果行
def _judge0():                                                                  # 【优化·评估层】temperature=0 的判官（见 docs/04 §2.4：默认温度下同批 contexts 三次重跑 recall 极差 0.1875，单次结果不可信；置 0 后降到 0.0385）
    if not hasattr(_judge0, "j"): from openai import OpenAI; from ragas.llms import llm_factory; _judge0.j = llm_factory(C.C8.LLM_MODEL, provider="openai", client=OpenAI(api_key=C.C8.LLM_API_KEY, base_url=C.C8.LLM_BASE_URL, timeout=180, max_retries=3), max_tokens=4096, temperature=0.0)  # 单例只建一次；判官超时与 max_tokens 都要给足，截断会把样本打成 nan
    return _judge0.j                                                            # 返回判官
def ragas2(rows):                                                               # 【优化·评估层】只跑工单点名的两项，逐指标串行，nan 记 None 不入均值
    from ragas import EvaluationDataset, SingleTurnSample, evaluate; from ragas.run_config import RunConfig  # 0.4.x API / 并发与超时控制
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall  # 上下文精度 / 上下文召回（工单验收的两项）
    from 工单八.app.evaluate import _emb                                         # 复用 08 的嵌入单例（重复加载 bge-m3 会在第 6 次 CUDA OOM）；判官不用 08 的，改用上面的 _judge0
    ds = EvaluationDataset(samples=[SingleTurnSample(user_input=r["question"], response=r["answer"], reference=r["ground_truth"], retrieved_contexts=r["contexts"]) for r in rows]); out = {"n_samples": len(rows)}  # 组装样本 + 结果容器（含样本口径）
    for m in (LLMContextPrecisionWithReference(), LLMContextRecall()):          # 逐指标串行：四指标混跑会把并发翻倍触发限流，样本被打成 nan
        try: vals = evaluate(ds, metrics=[m], llm=_judge0(), embeddings=_emb(), run_config=RunConfig(max_workers=C.C8.EVAL_MAX_WORKERS, timeout=180, max_retries=3), raise_exceptions=False).to_pandas()[m.name].tolist()  # 不抛异常，失败样本落成 nan
        except Exception as e: vals = []; print(f"[ragas] {m.name} 失败：{type(e).__name__}: {e}")  # 整体失败（如 402 余额不足）如实打印
        clean = [float(v) for v in vals if v == v]                               # nan != nan，用它滤掉无效值
        out[m.name] = round(sum(clean) / len(clean), 4) if clean else None       # 有值才算均值，否则 None（绝不把 nan 当数字）
        out["valid_" + m.name] = f"{len(clean)}/{len(vals) or len(rows)}"         # 有效样本数必须上报
        out["per_" + m.name] = [None if v != v else round(float(v), 4) for v in vals]  # 逐题得分（便于人工核验，也可按 3 题口径再聚合）
        print(f"[ragas] {m.name} = {out[m.name]} 有效 {out['valid_' + m.name]}")  # 逐指标进度
    return out                                                                  # 返回指标集合
def compare(res):                                                               # 生成优化前后指标对比表（Markdown，可直接贴进 docs/04_优化.md）
    b, a = res["before"]["ragas"], res["after"]["ragas"]                        # 两组 RAGAS 指标
    L = ["| 指标 | 目标 | before(08 原链路) | after(工单九优化) | Δ | 达标 |", "|---|---|---|---|---|---|"]  # 表头
    for k, tgt in (("llm_context_precision_with_reference", 0.8), ("context_recall", 0.9)):  # 工单验收的两项与目标线（键名须与 ragas 指标自身 name 完全一致，写错会静默显示 None——本行实测踩到过）
        vb, va = b.get(k), a.get(k)                                             # 取值（可能为 None，表示样本全 nan）
        d = f"{va - vb:+.4f}" if (vb is not None and va is not None) else "—"; ok = "✅" if (va is not None and va >= tgt) else "❌"  # 增量与达标判定（卡在目标线附近时不得声称达标）
        L.append(f"| {k} | ≥{tgt} | {vb} | {va} | {d} | {ok} |")                 # 数据行
    for name, key in (("时延 p50(ms)", "p50_ms"), ("时延 max(ms)", "max_ms")):     # 时延对比
        L.append(f"| {name} | ≤{C.TARGET_MS} | {res['before'][key]} | {res['after'][key]} | — | {'✅' if res['after'][key] <= C.TARGET_MS else '❌'} |")  # 性能验收
    return "\n".join(L)                                                         # 返回表格文本
def main():                                                                     # 评估总入口：python -m app.eval_ragas --sample-pdf data/sample_questions.pdf
    ap = argparse.ArgumentParser(description="工单九 RAGAS 评估：before=08 原链路 / after=工单九优化链路")  # 参数解析
    ap.add_argument("--sample-pdf", default="", help="题集 PDF 路径（默认 data/sample_questions.pdf，或环境变量 SAMPLE_Q_PDF）"); ap.add_argument("--skip-ragas", action="store_true", help="只作答与统计时延，不调 RAGAS")  # 题集路径 / 跳过 RAGAS（先看检索效果，省钱省时）
    a = ap.parse_args(); qs = load_questions(a.sample_pdf or None)               # 解析参数并加载题集（默认走缓存，避免重复解析）
    C.EVAL.mkdir(parents=True, exist_ok=True); res = {}                          # 确保 eval/ 目录存在 + 汇总容器
    from .retrieval import retrieve_after; retrieve_after("预热")                 # 预热检索栈：否则先跑的 before 独自承担冷启动（实测 Q01 被记成 18276ms），两边时延不可比
    for which in ("before", "after"):                                           # 两条链路各跑一遍
        rows = collect(which, qs); lat = sorted(r["latency_ms"] for r in rows)     # 逐题作答 + 时延升序（便于取分位）
        res[which] = {"chain": "08 原链路（GraphRAG 基线）" if which == "before" else "工单九优化链路", "n": len(rows), "errors": sum(1 for r in rows if r["error"]),  # 链路标识与样本口径
                      "p50_ms": lat[len(lat) // 2] if lat else None, "max_ms": lat[-1] if lat else None, "questions": rows, "ragas": {} if a.skip_ragas else ragas2(rows)}  # 时延分位 + 逐题明细 + RAGAS 两指标
        (C.EVAL / f"{which}.json").write_text(json.dumps(res[which], ensure_ascii=False, indent=1), encoding="utf-8"); print(f"[eval] {which} → eval/{which}.json")  # 先落盘（长任务中途失败不至于全丢）并提示产物路径
    if not a.skip_ragas: print("\n" + compare(res))                              # 打印 before/after 对比表
if __name__ == "__main__":                                                      # 支持 python -m app.eval_ragas
    try: main()                                                                 # 正常执行评估
    except RuntimeError as e: print(f"[eval] 无法开始评估：{e}"); raise SystemExit(2)  # 题集缺失/解析失败时只给一行可读结论并以非 0 退出，不打印长栈（工单要求先提示放置，不编造题集）
