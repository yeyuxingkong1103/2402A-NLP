# -*- coding: utf-8 -*-
"""生成 collected 评估集的**人工复核清单**（单文件 HTML + CSV）。

## 为什么要这个东西

`eval/datasets/collect.jsonl` 的真实问法本身是独立的（来自 12316 热线，
不是从卡片反推的），但「这个问题该由哪张卡回答」是**模型标的**——
`annotated_by` 里挂着「待人工复核」。在复核之前，collected 的召回率
只能当参考值，不能当验收依据（见 eval/annotate_collected.py 开头）。

复核时最大的障碍是：**光看卡号没法判断标注对不对**。所以本工具把
每张期望卡的**实际内容**（防治对象、防治适期、用药方案、来源条款）摊出来，
复核者看到的是证据，不是编号。

## 设计取舍

- **系统实际结果默认折叠**。先让复核者独立判断「该由哪张卡回答」，
  再展开对照。否则系统返回什么就会被当成标答——那是循环论证，
  正是 CLAUDE.md 里明令禁止的。
- **判定口径与 eval/run_eval.py 完全一致**，不另起一套，避免两处口径漂移。
- 复核结果存浏览器 localStorage，刷新不丢；导出 JSON 后可用
  `--apply` 写回 collect.jsonl（自动备份原文件）。

## 用法

    # 服务开着（推荐）——Milvus Lite 库是单进程独占的，服务占用时本地直连会撞锁
    python eval/make_review.py --api http://127.0.0.1:8000
    python eval/make_review.py                 # 服务没开：本地直连
    python eval/make_review.py --no-sys        # 不跑检索（秒出，但没有系统结果）
    python eval/make_review.py --apply <导出.json>   # 把复核结果写回数据集
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime

# 项目根目录（本文件位于 eval/ 下，根目录是其上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 把 src/retrieve 加入 sys.path，便于 --no-api 模式下 import search 本地直连
sys.path.insert(0, os.path.join(ROOT, "src", "retrieve"))

DATASET = os.path.join(ROOT, "eval", "datasets", "collect.jsonl")   # 待复核数据集
CHUNKS = os.path.join(ROOT, "data", "chunks", "chunks.jsonl")       # 全量 chunk 索引（摊证据用）
OUT_HTML = os.path.join(ROOT, "eval", "review_collected.html")      # 输出：单文件复核页
OUT_CSV = os.path.join(ROOT, "eval", "review_collected.csv")        # 输出：表格版复核清单
SYS_CACHE = os.path.join(ROOT, "eval", "_syscache.json")            # 系统检索结果缓存
K = 5  # 检索截断位置，与 run_eval.py 的 top-K 保持一致


# ---------------------------------------------------------------- 数据装载
def load_chunks():
    """chunk_id -> chunk（含完整卡片内容，用于摊开证据）。"""
    idx = {}
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                idx[c["chunk_id"]] = c  # 以 chunk_id 为键建索引，O(1) 查卡
    return idx


def card_evidence(cid, chunks):
    """把一张卡压缩成复核者看得懂的「证据块」。"""
    c = chunks.get(cid)
    if c is None:
        return {"id": cid, "missing": True}  # 卡号在索引中不存在（标注可能写错了）
    cd = c["card"]
    # 提取用药方案条目（最多 6 条）：药剂/用量/用法/安全间隔/每季次数/兼治对象
    chem = []
    for ch in (cd.get("chemicals") or [])[:6]:
        chem.append({
            "product": ch.get("product"),
            "dose": ch.get("dose"),
            "method": ch.get("method"),
            "phi": ch.get("pre_harvest_interval"),       # 安全间隔期
            "max_uses": ch.get("max_uses_per_season"),   # 每季最多使用次数
            "targets": ch.get("companion_control"),      # 兼治对象
        })
    src = cd.get("source") or {}
    return {
        "id": cid,
        "missing": False,
        "title": cd.get("title"),
        "crop": cd.get("crop"),
        "std_no": cd.get("std_no"),        # 国标编号
        "std_name": cd.get("std_name"),    # 国标名称
        "pest_kind": cd.get("pest_kind"),  # 病害/虫害/草害
        "type_group": cd.get("type_group"),
        "subtype": cd.get("subtype"),      # 防治对象
        "trigger": cd.get("trigger"),      # 防治适期
        "stage": cd.get("growth_stage"),   # 生育期
        "section": c["meta"].get("source_section"),  # 来源条款号
        "page": c["meta"].get("source_page"),        # 来源页码
        "quote": src.get("quote"),         # 国标原文引文（回链用）
        "measures": [m.get("text") for m in (cd.get("measures") or [])][:6],  # 措施原文（最多 6 条）
        "chemicals": chem,
        # 是否附录B 卡：来源条款以「附录B」开头即为用药方案卡
        "is_appendix_b": str(c["meta"].get("source_section") or "").startswith("附录B"),
    }


# ---------------------------------------------------------------- 判定（口径同 run_eval.py）
def judge(case, status, got):
    """返回 (判定标签, 严重度, 说明)。标签用大白话，复核者不用读代码。"""
    fellback = status.startswith("fallback")  # 是否硬兜底（状态位以 fallback 开头）
    exp = set(case["expect_cards"])

    if case.get("expect_fallback"):
        # ---- 负例：期望系统说「暂无」----
        # 判断返回列表里有没有附录B 的具体用药方案卡（卡号含 -p2 页码段与 -c 方案段）
        gave_plan = any(("-p2" in cid and "-c" in cid) for cid in got)
        if not fellback and status != "ok_noplan" and gave_plan:
            return "漏兜底", "bad", "该说「暂无」却给了具体用药方案"  # 最危险：误导用药
        if status == "ok_noplan":
            return "放宽合格", "ok", "返回正文 + 「暂无用药方案」提示"  # 定稿允许的正确行为
        if fellback:
            return "兜底合格", "ok", "正确硬兜底"
        return "存疑", "warn", "既没兜底也没提示，复核时留意"

    if case.get("expect_noplan"):
        # ---- 期望放宽：语料只在正文提到、无用药方案 ----
        if status == "ok_noplan":
            return "放宽合格", "ok", "返回正文 + 「暂无用药方案」提示"
        if fellback:
            return "该放宽却兜底", "bad", "本可返回正文内容，却直接兜底了"
        return "状态异常", "warn", f"实际 {status}，期望 ok_noplan"

    # ---- 普通正例：期望召回指定卡片 ----
    if fellback:
        return "误兜底", "bad", "该答的答了「暂无」"
    if not exp:
        return "无期望值", "warn", "该用例没标期望卡，需复核是否漏标"
    hit = [cid for cid in got if cid in exp]  # top-K 里命中的期望卡
    r = len(hit) / len(exp)  # 命中比例（口径同 run_eval.py 的 recall）
    if r == 1:
        return "全部命中", "ok", f"{len(hit)}/{len(exp)} 张期望卡都在 top{K}"
    if r > 0:
        return "部分命中", "warn", f"只召回 {len(hit)}/{len(exp)} 张"
    return "未召回", "bad", f"top{K} 里没有一张期望卡"


# ---------------------------------------------------------------- 取系统实际结果
def fetch_via_api(cases, base, chunks):
    """走**正在运行的问答服务**取结果。

    两个好处：① Milvus Lite 的库是单进程独占的，服务开着时本地直连会撞锁；
    ② 走的是农户真实链路（含生成层），比本地直接调 search() 更贴近实际。

    注意：必须显式绕过系统代理（本机 127.0.0.1 走代理会不通）。
    """
    import urllib.request
    # 显式绕过系统代理：127.0.0.1 走代理会连不通
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(base.rstrip("/") + "/api/health", timeout=5) as r:
            json.loads(r.read().decode("utf-8"))  # 健康检查：确认服务在线
    except Exception as e:
        raise SystemExit(
            f"连不上问答服务 {base}（{e}）\n"
            f"  先启动：KMP_DUPLICATE_LIB_OK=TRUE python src/serve/app.py\n"
            f"  或去掉 --api 走本地直连（服务没开时才可行）")
    out = {}
    for i, c in enumerate(cases, 1):
        # 逐条调用问答接口，带上问句和作物约束
        body = json.dumps({"query": c["query"], "crop": c["crop"]}).encode("utf-8")
        req = urllib.request.Request(base.rstrip("/") + "/api/ask", data=body,
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=300) as r:
            d = json.loads(r.read().decode("utf-8"))
        ids = [x["chunk_id"] for x in d.get("chunks", [])]  # 系统实际返回的卡号
        out[c["query"]] = {
            "status": d["status"], "why": d.get("reason", ""),
            "normalized": d.get("normalized"), "intent": d.get("intent"),
            "got_ids": ids,
            "got": [card_evidence(cid, chunks) for cid in ids],  # 返回卡的证据块
        }
        print(f"  …{i}/{len(cases)} {d['status']}", flush=True)
    return out


def fetch_local(cases, chunks):
    """不开服务时退化为本地直连（需要 Milvus Lite 库没被别的进程占用）。"""
    from search import Searcher, search
    s = Searcher()
    out = {}
    for i, c in enumerate(cases, 1):
        # 直接调本地检索：返回 (状态, 归一化问句, 意图, 结果, 原因)
        st, q2, intent, res, why = search(s, c["query"], crop=c["crop"], topk=K)
        ids = [r["chunk_id"] for r in res]
        out[c["query"]] = {
            "status": st, "why": why, "normalized": q2,
            "intent": (intent or {}).get("intent") if intent else None,
            "got_ids": ids, "got": [card_evidence(cid, chunks) for cid in ids],
        }
        print(f"  …{i}/{len(cases)} {st}", flush=True)
    return out


# ---------------------------------------------------------------- 主流程
def build(no_sys=False, api=None, refresh=False):
    """主构建流程：读数据集 -> 取系统实际结果（带缓存）-> 逐条判定 -> 返回行数据。

    参数：no_sys 不跑检索；api 走问答服务地址；refresh 强制忽略缓存重跑。
    返回：列表，每项含问句、期望标注、期望卡证据块、系统结果与判定标签。
    """
    cases = [json.loads(l) for l in open(DATASET, encoding="utf-8") if l.strip()]
    chunks = load_chunks()

    sysres = {}
    if not no_sys:
        # 缓存：系统结果只跟「问句 + 当前检索逻辑」有关，改 HTML 模板时不必重跑
        if not refresh and os.path.exists(SYS_CACHE):
            try:
                cached = json.load(open(SYS_CACHE, encoding="utf-8"))
                # 只保留当前数据集中仍存在的问句（数据集更新后旧缓存自动失效）
                sysres = {q: v for q, v in cached.items() if q in {c["query"] for c in cases}}
                if sysres:
                    print(f"复用缓存 {len(sysres)} 条（--refresh 可强制重跑）")
            except Exception as e:
                print(f"缓存不可用（{e}），重新取数")
                sysres = {}

        todo = [c for c in cases if c["query"] not in sysres]  # 缓存未覆盖的问句
        if todo:
            print(f"取系统实际结果（{len(todo)} 条问句）…")
            fresh = (fetch_via_api(todo, api, chunks) if api
                     else fetch_local(todo, chunks))  # 有 api 走服务，否则本地直连
            sysres.update(fresh)
            # 落盘缓存，下次生成复核页时可复用
            json.dump(sysres, open(SYS_CACHE, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)

    rows = []
    for i, c in enumerate(cases, 1):
        sr = sysres.get(c["query"], {})
        # 没跑系统时给占位判定，避免复核页缺字段
        label, sev, detail = ("未跑检索", "warn", "本次生成未跑系统对照")
        if sr:
            label, sev, detail = judge(c, sr["status"], sr["got_ids"])  # 判定口径同 run_eval.py
        rows.append({
            "i": i,
            "query": c["query"],
            "crop": c["crop"],
            "origin": c.get("origin"),
            "tier": "高危" if c["is_high_risk"] else "普通",  # 分层展示
            "is_high_risk": c["is_high_risk"],
            "expect_cards": c["expect_cards"],
            "expect_noplan": bool(c.get("expect_noplan")),
            "expect_fallback": bool(c.get("expect_fallback")),
            "note": c.get("note", ""),
            "annotated_by": c.get("annotated_by", ""),
            "evidence": [card_evidence(cid, chunks) for cid in c["expect_cards"]],  # 期望卡的证据块
            "sys": sr,
            "label": label,
            "severity": sev,
            "label_detail": detail,
        })
    return rows


def write_csv(rows, path):
    """Excel 友好：utf-8-sig，否则中文乱码。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        # 表头：末尾四列留给复核者填写（人工结论/修正卡号/标志位修正/备注）
        w.writerow(["序号", "问句", "作物", "来源", "分层", "系统判定", "判定说明",
                    "期望卡号", "期望卡标题", "期望放宽+提示", "期望硬兜底",
                    "系统状态", "系统返回卡号", "标注依据",
                    "人工结论", "人工修正卡号", "人工标志位修正", "人工备注"])
        for r in rows:
            w.writerow([
                r["i"], r["query"], r["crop"], r["origin"], r["tier"],
                r["label"], r["label_detail"],
                " | ".join(r["expect_cards"]),
                " | ".join(e.get("title") or "?" for e in r["evidence"]),
                "是" if r["expect_noplan"] else "", "是" if r["expect_fallback"] else "",
                r["sys"].get("status", ""),
                " | ".join(r["sys"].get("got_ids", [])),
                r["note"], "", "", "", "",
            ])
    print(f"→ 表格版: {path}")


def write_html(rows, path):
    """把行数据 JSON 注入 HTML 模板，生成单文件复核页。"""
    data = json.dumps(rows, ensure_ascii=False)
    # 用数据替换模板里的 /*__DATA__*/null 占位符
    html = TEMPLATE.replace("/*__DATA__*/null", data)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"→ 复核页: {path}")


# ---------------------------------------------------------------- 回写
def apply_review(path):
    """把复核页导出的 JSON 写回 collect.jsonl（自动备份）。"""
    payload = json.load(open(path, encoding="utf-8"))
    edits = {e["query"]: e for e in payload.get("rows", payload if isinstance(payload, list) else [])}
    reviewer = payload.get("reviewer", "") if isinstance(payload, dict) else ""
    cases = [json.loads(l) for l in open(DATASET, encoding="utf-8") if l.strip()]

    bak = DATASET + ".bak"
    # 写备份：写回前先完整备份原 collect.jsonl，防止复核结果有误无法回滚
    with open(bak, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"原文件已备份 → {bak}")

    changed = dropped = 0  # changed=更新的条数，dropped=复核后删除的条数
    out = []
    for c in cases:
        e = edits.get(c["query"])  # 该问句是否有复核意见
        if not e:
            out.append(c)  # 没复核过的条目原样保留
            continue
        if e.get("review_verdict") == "drop":
            dropped += 1  # 复核判定「删掉这条用例」→ 直接丢弃
            continue
        # 修正后的期望卡号：按空白拆分成列表，去掉空串
        cards = [x.strip() for x in (e.get("expect_cards") or []) if x.strip()]
        if cards or e.get("expect_fallback"):
            c["expect_cards"] = cards  # 有修正卡号或期望兜底时才覆盖期望卡
        c["expect_fallback"] = bool(e.get("expect_fallback"))
        c["expect_noplan"] = bool(e.get("expect_noplan"))
        c["is_high_risk"] = bool(e.get("is_high_risk"))
        if e.get("review_comment"):
            # 追加复核意见到 note，保留原标注依据
            c["note"] = f"{c.get('note','')}｜复核意见：{e['review_comment']}"
        c["annotated_by"] = f"人工复核{('（' + reviewer + '）') if reviewer else ''}"  # 标注来源改为人工
        changed += 1
        out.append(c)

    with open(DATASET, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"已更新 {changed} 条，删除 {dropped} 条，剩余 {len(out)} 条")
    print("提示：复核后再跑一次 eval/run_eval.py，collected 的召回率才算可信。")


# ---------------------------------------------------------------- HTML 模板
TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>collected 评估集 · 人工复核清单</title>
<style>
  :root{--ink:#1a1a1a;--muted:#666;--line:#e3e3e3;--bg:#f7f8f7;
    --green:#2f7d4f;--green-l:#eef6f0;--danger:#c0392b;--danger-l:#fdf1ef;
    --danger-b:#e8b4ad;--warn:#b7791f;--warn-l:#fdf8ec;--blue:#2b5f9e;--blue-l:#eef3fa;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
  .wrap{max-width:1000px;margin:0 auto;padding:26px 20px 90px}
  h1{font-size:21px;margin:0 0 6px}
  .sub{color:var(--muted);font-size:13.5px;margin-bottom:16px}
  .bar{position:sticky;top:0;z-index:20;background:rgba(247,248,247,.96);
    backdrop-filter:blur(6px);border-bottom:1px solid var(--line);
    padding:11px 0;margin-bottom:18px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
  .prog{flex:1;min-width:150px;height:7px;background:#e6e8e6;border-radius:4px;overflow:hidden}
  .prog i{display:block;height:100%;width:0;background:var(--green);transition:width .25s}
  .progtxt{font-size:12.5px;color:var(--muted);white-space:nowrap}
  button{font:inherit;font-size:13.5px;padding:6px 12px;border:1px solid var(--line);
    border-radius:8px;background:#fff;cursor:pointer}
  button:hover{border-color:#bbb}
  button.on{background:var(--green);border-color:var(--green);color:#fff}
  input[type=text],textarea{font:inherit;font-size:13.5px;border:1px solid var(--line);
    border-radius:8px;padding:6px 9px;background:#fff;width:100%}
  textarea{resize:vertical;min-height:46px}
  .card{background:#fff;border:1px solid var(--line);border-radius:12px;
    padding:16px 18px;margin-bottom:14px}
  .card.done{border-color:#cfe4d6;background:#fcfdfc}
  .head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:9px;
    font-size:12.5px;color:var(--muted)}
  .num{font-weight:700;color:var(--ink)}
  .tag{padding:2px 8px;border-radius:999px;font-size:12px;background:#f1f2f1;color:#555}
  .tag.hr{background:var(--danger-l);color:var(--danger)}
  .tag.ok{background:var(--green-l);color:var(--green)}
  .tag.bad{background:var(--danger-l);color:var(--danger)}
  .tag.warn{background:var(--warn-l);color:var(--warn)}
  .q{font-size:16.5px;font-weight:600;margin:2px 0 10px;line-height:1.5}
  .note{font-size:13px;color:var(--muted);background:#fafbfa;border-left:3px solid var(--line);
    padding:8px 11px;border-radius:0 7px 7px 0;margin-bottom:12px;white-space:pre-wrap}
  .sec{font-size:12.5px;font-weight:700;color:#444;letter-spacing:.03em;
    margin:14px 0 7px;text-transform:uppercase}
  .ev{border:1px solid var(--line);border-radius:9px;margin-bottom:7px;overflow:hidden}
  .ev>summary{cursor:pointer;padding:9px 12px;font-size:13.5px;
    display:flex;gap:9px;align-items:center;list-style:none}
  .ev>summary::-webkit-details-marker{display:none}
  .ev>summary::before{content:"▸";color:var(--muted);font-size:12px}
  .ev[open]>summary::before{content:"▾"}
  .ev>summary:hover{background:#fafbfa}
  .ev .body{padding:4px 13px 13px;border-top:1px solid var(--line);
    background:#fcfcfc;font-size:13.5px}
  .kv{display:grid;grid-template-columns:88px 1fr;gap:3px 10px;margin:8px 0}
  .kv b{font-weight:600;color:#666;font-size:12.5px}
  .kp{padding:2px 7px;border-radius:5px;background:#eef1ee;font-size:12px;
    font-family:ui-monospace,Consolas,monospace;color:#444}
  .hit{background:var(--green-l);color:var(--green)}
  .mline{padding:5px 0;border-bottom:1px dashed #ececec}
  .mline:last-child{border-bottom:0}
  table.chem{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px}
  table.chem th{text-align:left;color:#666;font-weight:600;border-bottom:1px solid var(--line);
    padding:4px 6px}
  table.chem td{padding:4px 6px;border-bottom:1px solid #f2f2f2}
  .quote{font-size:12.5px;color:#555;background:#f7f8f7;border-radius:7px;
    padding:8px 10px;margin-top:8px;white-space:pre-wrap;max-height:150px;overflow:auto}
  .opts{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:6px 0 4px}
  .opt{padding:5px 12px;border:1px solid var(--line);border-radius:8px;background:#fff;
    cursor:pointer;font-size:13px}
  .opt:hover{border-color:#bbb}
  .opt.sel{background:var(--blue);border-color:var(--blue);color:#fff}
  .opt.sel[data-v=ok]{background:var(--green);border-color:var(--green)}
  .opt.sel[data-v=fix]{background:var(--warn);border-color:var(--warn)}
  .opt.sel[data-v=drop]{background:var(--danger);border-color:var(--danger)}
  .flags{display:flex;gap:14px;flex-wrap:wrap;margin:9px 0 4px;font-size:13.5px}
  .flags label{display:flex;gap:6px;align-items:center;cursor:pointer}
  .sysbox{border:1px dashed var(--line);border-radius:9px;margin-top:12px;
    background:#fbfbfa;padding:0 12px}
  .sysbox>summary{cursor:pointer;padding:10px 0;font-size:13px;color:var(--muted)}
  .sysbody{padding-bottom:12px}
  .warnbar{font-size:12.5px;color:#7a5510;background:var(--warn-l);border-radius:7px;
    padding:8px 10px;margin:8px 0}
  .gotlist{display:flex;flex-direction:column;gap:5px;margin-top:8px}
  .got{font-size:13px;padding:7px 10px;border-radius:7px;background:#fff;
    border:1px solid var(--line);display:flex;gap:9px;align-items:center;flex-wrap:wrap}
  .got.hit{border-color:#bcdcc6;background:var(--green-l)}
  .mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}
  .hide{display:none}
  .empty{text-align:center;color:var(--muted);padding:50px;font-size:14px}
</style>
</head>
<body>
<div class="wrap">
  <!-- 页面标题与统计副标题（render() 会填入「已复核 x/y · 待改 z 条」等） -->
  <h1>collected 评估集 · 人工复核清单</h1>
  <div class="sub" id="sub"></div>

  <!-- 顶部工具栏：筛选按钮（全部/仅待改/仅高危/仅未复核）+ 进度条 + 复核人输入 + 导出按钮 -->
  <div class="bar">
    <button id="f-all" class="on">全部</button>
    <button id="f-bad">仅待改</button>
    <button id="f-hr">仅高危</button>
    <button id="f-todo">仅未复核</button>
    <div class="prog"><i id="progi"></i></div>
    <span class="progtxt" id="progtxt"></span>
    <input type="text" id="reviewer" placeholder="复核人" style="width:96px">
    <button id="exp">导出复核结果</button>
    <button id="expcsv">导出 CSV</button>
    <button id="reset">清空</button>
  </div>

  <!-- 复核须知：提醒复核者先独立判断再看系统返回，避免循环论证 -->
  <div class="warnbar">
    <b>复核前请先自己判断</b>：期望卡的正确与否，只看问句 + 卡片内容。
    系统实际返回收在每条底部的折叠区里，<b>建议先想完再展开</b>——否则系统返回什么就会被当成标答，
    那是循环论证（CLAUDE.md 明令禁止）。复核结果自动存在本机浏览器，刷新不丢。
  </div>

  <!-- 复核卡片列表容器：由 render() 按筛选条件动态填充 -->
  <div id="list"></div>
</div>

<!-- 脚本区：ROWS 为行数据（生成时由 write_html() 把 /*__DATA__*/null 占位符替换为真实 JSON），下方为页面全部交互逻辑 -->
<script>
const ROWS = /*__DATA__*/null;
const LS = 'rag_review_collected_v1';
// 存储降级：file:// 或内嵌预览面板里 localStorage 可能被浏览器禁用，
// 直接调会抛异常把整页带崩。禁用时退回内存，仅提示「不会自动保存」。
const Store = (() => {
  let ok = true, mem = {};
  try { localStorage.setItem('__t','1'); localStorage.removeItem('__t'); } catch(e) { ok = false; }  // 探测 localStorage 是否可用
  return {
    ok,
    get: k => { try { return ok ? localStorage.getItem(k) : (mem[k] ?? null); } catch(e) { return mem[k] ?? null; } },
    set: (k,v) => { try { if(ok) localStorage.setItem(k,v); else mem[k]=v; } catch(e) { mem[k]=v; } }
  };
})();
// 复核状态：{ 问句: {review_verdict, expect_cards, ...} }，启动时从本地存储恢复
let state = JSON.parse(Store.get(LS) || '{}');
let filter = 'all', kw = '';  // 当前筛选 tab 与搜索关键词

// 工具函数：$ 按 CSS 选择器取元素；esc 做 HTML 转义（防注入，所有插值都过它）
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// st(q)：取（或首次初始化）某条问句的复核状态对象
function st(q){ return state[q] || (state[q] = {
  review_verdict:'', expect_cards:null, expect_fallback:null, expect_noplan:null,
  is_high_risk:null, review_comment:'' }); }

// save()：把复核状态持久化到本地存储并重绘页面
function save(){ Store.set(LS, JSON.stringify(state)); render(); }

// evidenceHTML(e)：渲染一张期望卡的「证据块」折叠面板——
// 展开后显示防治对象/防治适期/生育期/来源页码、措施原文、用药方案表格、国标原文引文；
// missing 时显示「卡片不存在于索引」（标注卡号写错时会触发）
function evidenceHTML(e){
  if(e.missing) return `<div class="ev"><summary><span class="mono">${esc(e.id)}</span>
    <span class="tag bad">卡片不存在于索引</span></summary></div>`;
  const chem = (e.chemicals||[]).length ? `<table class="chem">
    <tr><th>药剂</th><th>用量</th><th>用法</th><th>安全间隔</th><th>每季次数</th><th>兼治</th></tr>
    ${e.chemicals.map(c=>`<tr>
      <td>${esc(c.product||'—')}</td><td>${esc(c.dose||'—')}</td><td>${esc(c.method||'—')}</td>
      <td>${esc(c.phi||'—')}</td><td>${esc(c.max_uses||'—')}</td><td>${esc(c.targets||'—')}</td></tr>`).join('')}
    </table>` : '<div style="color:#888;font-size:12.5px">（无用药方案条目）</div>';
  return `<details class="ev"><summary>
      <span class="kp ${e.is_appendix_b?'hit':''}">${esc(e.id)}</span>
      <span>${esc(e.title||'')}</span>
      ${e.is_appendix_b?'<span class="tag ok">附录B 用药方案</span>':'<span class="tag">正文条款</span>'}
    </summary>
    <div class="body">
      <div class="kv">
        <b>标准</b><span>${esc(e.std_no||'')} ${esc(e.std_name||'')}</span>
        <b>防治对象</b><span>${esc(e.subtype||'—')}（${esc(e.pest_kind||'')}）</span>
        <b>防治适期</b><span>${esc(e.trigger||'—')}</span>
        <b>生育期</b><span>${esc(e.stage||'—')}</span>
        <b>来源</b><span>${esc(e.section||'')} · 第 ${esc(e.page ?? '?')} 页</span>
      </div>
      ${(e.measures||[]).length?`<div class="sec">措施原文</div>
        ${e.measures.map(m=>`<div class="mline">${esc(m)}</div>`).join('')}`:''}
      ${chem}
      ${e.quote?`<div class="sec">国标原文回链</div><div class="quote">${esc(e.quote)}</div>`:''}
    </div></details>`;
}

// cardHTML(r)：渲染单条复核卡片——
// 头部（序号/作物/分层/来源/系统判定标签）、问句、标注依据、
// 期望卡证据列表、复核操作区（结论三选一/修正卡号/标志位复选框/备注）、
// 底部折叠的「系统实际返回」（含状态位/归一化/意图/top-5 对照，命中期望卡高亮）
function cardHTML(r){
  const s = st(r.query);
  const verdict = s.review_verdict;
  const exp = (s.expect_cards ?? r.expect_cards).join(' ');
  const fb = s.expect_fallback ?? r.expect_fallback;
  const np = s.expect_noplan ?? r.expect_noplan;
  const hr = s.is_high_risk ?? r.is_high_risk;
  const sysHit = new Set(r.sys && r.sys.got_ids || []);
  const got = (r.sys && r.sys.got) || [];
  return `<div class="card ${verdict?'done':''}" data-q="${esc(r.query)}" data-sev="${r.severity}"
      data-hr="${r.is_high_risk}" data-todo="${verdict?'0':'1'}">
    <div class="head">
      <span class="num">#${r.i}</span>
      <span class="tag">${esc(r.crop)}</span>
      <span class="tag ${r.is_high_risk?'hr':''}">${esc(r.tier)}${r.is_high_risk?'（问用药）':''}</span>
      <span class="tag">来源 ${esc(r.origin||'—')}</span>
      <span class="tag ${r.severity}">系统：${esc(r.label)}</span>
      <span>${esc(r.label_detail)}</span>
      ${verdict?`<span class="tag ${verdict==='ok'?'ok':verdict==='drop'?'bad':'warn'}">已复核</span>`:''}
    </div>
    <div class="q">${esc(r.query)}</div>
    <div class="note">${esc(r.note||'（无标注依据）')}</div>

    <div class="sec">模型认为该由这些卡回答（点开核对内容）</div>
    ${r.evidence.map(evidenceHTML).join('')}

    <div class="sec">复核</div>
    <div class="opts">
      <span style="font-size:13px;color:#666;margin-right:2px">期望卡</span>
      <span class="opt ${verdict==='ok'?'sel':''}" data-v="ok" onclick="mark(${r.i},'ok')">正确</span>
      <span class="opt ${verdict==='fix'?'sel':''}" data-v="fix" onclick="mark(${r.i},'fix')">需改</span>
      <span class="opt ${verdict==='drop'?'sel':''}" data-v="drop" onclick="mark(${r.i},'drop')">删掉这条用例</span>
    </div>
    <input type="text" placeholder="修正后的卡号（空格分隔，可留空表示无）" value="${esc(exp)}"
      oninput="setf(${r.i},'expect_cards',this.value)">
    <div class="flags">
      <label><input type="checkbox" ${hr?'checked':''}
        onchange="setf(${r.i},'is_high_risk',this.checked)"> 高危（问的是用药）</label>
      <label><input type="checkbox" ${np?'checked':''}
        onchange="setf(${r.i},'expect_noplan',this.checked)"> 期望「正文 + 暂无用药方案」</label>
      <label><input type="checkbox" ${fb?'checked':''}
        onchange="setf(${r.i},'expect_fallback',this.checked)"> 期望硬兜底（语料完全没有）</label>
    </div>
    <textarea placeholder="备注：为什么改？问法是否超出国标范围？建议怎么改问句？"
      oninput="setf(${r.i},'review_comment',this.value)">${esc(s.review_comment||'')}</textarea>

    <details class="sysbox">
      <summary>▸ 系统实际返回（${esc(r.label)}）— 建议先自己判断完再展开</summary>
      <div class="sysbody">
        <div class="kv">
          <b>状态位</b><span class="mono">${esc((r.sys&&r.sys.status)||'—')}</span>
          <b>归一化后</b><span>${esc((r.sys&&r.sys.normalized)||'—')}</span>
          <b>意图</b><span class="mono">${esc((r.sys&&r.sys.intent)||'—')}</span>
          <b>判定原因</b><span>${esc((r.sys&&r.sys.why)||'—')}</span>
        </div>
        <div class="sec">top-${5} 返回</div>
        <div class="gotlist">
          ${got.length?got.map(g=>`<div class="got ${sysHit.has(g.id)?'hit':''}">
              <span class="kp">${esc(g.id)}</span>
              <span>${esc(g.title||'')}</span>
              <span class="tag">${esc(g.section||'')}</span>
              ${sysHit.has(g.id)?'<span class="tag ok">期望卡</span>':''}
            </div>`).join(''):'<div class="got">（无返回）</div>'}
        </div>
      </div>
    </details>
  </div>`;
}

// qOf(i)：行号 -> 问句（用于定位复核状态）
// mark(i,v)：设置复核结论；重复点击同一选项视为取消
// setf(i,k,v)：更新单个复核字段；expect_cards 按空白拆成数组，其余直接存值
function qOf(i){ return ROWS[i-1].query; }
function mark(i,v){ const s=st(qOf(i)); s.review_verdict = (s.review_verdict===v?'':v); save(); }
function setf(i,k,v){ st(qOf(i))[k] = (k==='expect_cards' ? String(v).split(/\s+/).filter(Boolean) : v); save(); }

// visible(r)：单条数据的筛选判断——
// 先按关键词匹配（问句/作物），再按当前 tab（all=全部 / bad=待改 / hr=高危 / todo=未复核）过滤
function visible(r){
  const s = state[r.query] || {};
  if(kw && !(r.query.includes(kw) || r.crop.includes(kw))) return false;
  if(filter==='bad' && !(r.severity==='bad' || s.review_verdict==='fix')) return false;
  if(filter==='hr' && !r.is_high_risk) return false;
  if(filter==='todo' && s.review_verdict) return false;
  return true;
}

// render()：按筛选条件重绘列表，并更新进度条与页头统计（已复核数/待改数/存疑数）
function render(){
  const rows = ROWS.filter(visible);
  $('#list').innerHTML = rows.length ? rows.map(cardHTML).join('')
    : '<div class="empty">这个筛选下没有条目</div>';
  const done = ROWS.filter(r=>state[r.query] && state[r.query].review_verdict).length;
  $('#progi').style.width = (done/ROWS.length*100)+'%';
  $('#progtxt').textContent = `${done} / ${ROWS.length} 已复核`;
  const bad = ROWS.filter(r=>r.severity==='bad').length;
  const warn = ROWS.filter(r=>r.severity==='warn').length;
  $('#sub').textContent = `${ROWS.length} 条真实农户问法（来源 12316 热线）· 系统判定待改 ${bad} 条 / 存疑 ${warn} 条 · `
    + `标注由模型初标，本页用于人工定稿`;
}

// download(name, text, mime)：用 Blob + 隐形 <a> 触发浏览器下载；
// 文本前加 BOM（\ufeff）保证 Excel 打开 CSV 时中文不乱码
function download(name, text, mime){
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob(['\ufeff'+text], {type:mime||'application/json'}));
  a.download = name; a.click(); URL.revokeObjectURL(a.href);
}

// 导出复核结果（JSON）：只导出已给出结论的条目，附带复核人与时间戳，
// 供 `python eval/make_review.py --apply <文件>` 写回 collect.jsonl
$('#exp').onclick = () => {
  const reviewer = $('#reviewer').value.trim();
  const rows = ROWS.filter(r=>state[r.query] && state[r.query].review_verdict).map(r=>{
    const s = state[r.query];
    return { query:r.query, crop:r.crop, review_verdict:s.review_verdict,
      expect_cards:(s.expect_cards ?? r.expect_cards), expect_fallback:(s.expect_fallback ?? r.expect_fallback),
      expect_noplan:(s.expect_noplan ?? r.expect_noplan), is_high_risk:(s.is_high_risk ?? r.is_high_risk),
      review_comment:s.review_comment||'' };
  });
  if(!rows.length){ alert('还没有复核任何条目'); return; }
  download('collect_review.json', JSON.stringify({
    reviewer, reviewed_at:new Date().toISOString(), dataset:'eval/datasets/collect.jsonl', rows }, null, 1));
};

// 导出 CSV：全部条目的简要复核表（问句/作物/分层/判定/期望卡/人工结论/修正卡/备注）
$('#expcsv').onclick = () => {
  const head = ['问句','作物','分层','系统判定','期望卡号','人工结论','人工修正卡','备注'];
  const esc2 = v => `"${String(v??'').replace(/"/g,'""')}"`;
  const lines = [head.map(esc2).join(',')];
  ROWS.forEach(r=>{
    const s = state[r.query] || {};
    lines.push([r.query, r.crop, r.tier, r.label, r.expect_cards.join(' '),
      s.review_verdict||'', (s.expect_cards||[]).join(' '), s.review_comment||''].map(esc2).join(','));
  });
  download('review_collected_export.csv', lines.join('\n'), 'text/csv');
};

// 清空本机所有复核记录（需二次确认，不可恢复）
$('#reset').onclick = () => { if(confirm('清空本机所有复核记录？不可恢复。')){ state={}; save(); } };
// 复核人姓名单独存一份（换导出文件时不丢），页面加载时回填
$('#reviewer').oninput = e => Store.set(LS+'_who', e.target.value);
$('#reviewer').value = Store.get(LS+'_who') || '';
// localStorage 被浏览器禁用时，在工具栏下方插入警告条提醒及时手动导出
if(!Store.ok){
  const b = document.createElement('div');
  b.className = 'warnbar';
  b.innerHTML = '<b>浏览器禁用了本地存储</b>：复核记录不会自动保存，刷新会丢。'
    + '请用「导出复核结果」及时落盘。';
  document.querySelector('.bar').after(b);
}
// 筛选 tab 切换：按按钮 id（f-all/f-bad/f-hr/f-todo）截取筛选类型，切换高亮并重绘
document.querySelectorAll('#f-all,#f-bad,#f-hr,#f-todo').forEach(b=>{
  b.onclick = () => {
    filter = b.id.slice(2);
    document.querySelectorAll('#f-all,#f-bad,#f-hr,#f-todo').forEach(x=>x.classList.toggle('on', x===b));
    render();
  };
});
render();
</script>
</body>
</html>
"""


def main():
    """命令行入口。

    --apply 有值时走回写流程（把复核 JSON 写回 collect.jsonl）后直接退出；
    否则执行 构建 -> 输出 HTML 复核页 + CSV 表格，并打印判定分布统计。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sys", action="store_true", help="跳过检索，秒出（但没有系统对照结果）")
    ap.add_argument("--api", default=None,
                    help="走正在运行的问答服务取结果，如 http://127.0.0.1:8000"
                         "（服务开着时本地直连会撞 Milvus 库锁，必须用这个）")
    ap.add_argument("--apply", default=None, help="把复核页导出的 JSON 写回 collect.jsonl")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存，强制重跑检索")
    a = ap.parse_args()

    if a.apply:
        apply_review(a.apply)  # 只回写，不生成复核页
        return 0

    rows = build(no_sys=a.no_sys, api=a.api, refresh=a.refresh)
    write_html(rows, OUT_HTML)
    write_csv(rows, OUT_CSV)

    # 打印系统判定分布，帮助复核者优先关注待改条目
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    print("\n系统判定分布：")
    for k, v in c.most_common():
        print(f"  {k}: {v}")
    bad = sum(1 for r in rows if r["severity"] == "bad")
    print(f"\n待改 {bad} 条 —— 复核时优先看这些\n")
    print(f"复核完成后：python eval/make_review.py --apply <导出的 collect_review.json>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
