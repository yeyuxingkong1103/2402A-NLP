#!/usr/bin/env python
# 工单编号：人工智能NLP-RAG-混合检索任务
# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单06 - 混合检索 / 工单05 - Query 理解优化
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""
演示截图作业台：把「要截图的终端输出」按顺序打出来。

【为什么要有这个脚本】截图手册里每张都要敲一条命令，而命令散在文档各处、
格式还各不相同（多行 python -c 容易粘错）。这个脚本把**所有终端类截图**
一次跑完，每段前面标好「第 N 张 · 存档名」，照着截就行。

用法：
    PY=~/rag-data/venv/bin/python
    $PY scripts/demo_shots.py --wo 06            # 工单06 的终端类截图（快，约 10 秒）
    $PY scripts/demo_shots.py --wo 05            # 工单05 的终端类截图
    $PY scripts/demo_shots.py --wo all --slow    # 加上耗时的（评测 6 分钟 / pytest 1 分钟）

【--slow 是什么】默认只跑"读一下就出结果"的项；评测与 pytest 要几分钟，
单独用 `--slow` 打开，免得每次截图前都要等。

【浏览器类的截图不在这里】第 10~14 张（工单06）与第 2~13 张（工单05）
是在浏览器里操作的，见 `docs/工单0X-演示截图清单.md`。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 配色（终端里让分隔更醒目）
B = "\033[1m"
DIM = "\033[2m"
G = "\033[32m"
Y = "\033[33m"
R = "\033[0m"


WO = "06"          # 当前工单号，由 main() 设置，只为了在标题里打印


def shot(n: int, name: str, note: str = "") -> None:
    """打印一段截图的标题。"""
    print(f"\n{B}{'=' * 74}{R}")
    print(f"{B}  工单{WO} · 第 {n} 张{R}  →  存为 {G}{name}{R}")
    if note:
        print(f"  {DIM}{note}{R}")
    print(f"{B}{'=' * 74}{R}\n")


def sub(cmd: list[str], *, timeout: int = 900) -> None:
    """跑一条子进程命令，原样把输出打出来（stdout + stderr 都收）。"""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        # 过滤掉每次都会出现的第三方噪声（jieba 的词典加载、pkg_resources 告警），
        # 否则它们会夹在每段截图内容里，拍出来很乱。
        noise = ("pkg_resources is deprecated", "UserWarning",
                 "import pkg_resources", "from pkg_resources import",
                 "Building prefix dict", "Loading model cost",
                 "Prefix dict has been built", "Dumping model to file cache",
                 "Loading model from cache")
        for line in out.splitlines():
            if any(k in line for k in noise):
                continue
            print(line)
    except subprocess.TimeoutExpired:
        print(f"{Y}（超时 {timeout}s，跳过）{R}")


def preflight_tail(n: int) -> None:
    """跑环境自检，**只打印末 n 行**。

    【为什么要截短】完整输出 100+ 行，`Alt+PrtSc` 只拍窗口可见部分 ——
    与其调字号，不如直接把要展示的那几行打出来，结果是确定的。
    tail 的窗口正好覆盖「第 9 节（工单05）+ 第 10 节（工单06）+ 汇总」。
    """
    try:
        r = subprocess.run([sys.executable, "scripts/preflight.py"],
                           capture_output=True, text=True, timeout=600)
        lines = [ln for ln in r.stdout.splitlines()
                 if "pkg_resources is deprecated" not in ln]
        print(f"{DIM}（只显示最后 {n} 行；完整输出 {len(lines)} 行）{R}")
        print("\n".join(lines[-n:]))
    except subprocess.TimeoutExpired:
        print(f"{Y}（环境自检超时，跳过）{R}")


def py(code: str) -> None:
    """跑一段内联 python（用当前解释器）。"""
    sub([sys.executable, "-c", code], timeout=600)


def latest_eval(profile: str) -> str | None:
    """取某剖面最新的评测报告。

    【必须排除 `eval-latest.json`】它只在单剖面模式更新；跑过 `--compare` 之后
    留在磁盘上的仍是旧内容（实测拿到过工单05 时期的报告，里面根本没有
    工单06 的 avg_recall_coverage 字段）。
    """
    fs = [x for x in glob.glob(os.path.expanduser("~/rag-data/eval/eval-*.json"))
          if x.endswith(f"-{profile}.json")]
    return max(fs, key=os.path.getmtime) if fs else None


# ======================================================================
# 工单06
# ======================================================================
def wo06(profile: str, slow: bool) -> None:
    print(f"\n\n{B}###### 工单06 · 混合检索 —— 终端类截图 ######{R}")

    shot(1, "工单6-01-环境自检.png", "要看见「10. 混合检索栈（工单06）」全 OK")
    preflight_tail(25)

    shot(2, "工单6-02-倒排索引统计.png", "证明倒排索引真的建起来了，不是拿稠密向量冒充")
    py("from app.core.fulltext import get_index\n"
       "idx = get_index()\n"
       "print('文档数', idx.n_docs, '｜词条', len(idx.postings), '｜模糊删除键', len(idx.deletes))\n"
       "print('字段平均长度', idx.stats()['field_avg_len'])")

    shot(3, "工单6-03-短语匹配有序性.png",
         "★ 顺序正确的有命中、打乱的是空 —— 这是短语匹配的铁证")
    py("from app.core.fulltext import get_index\n"
       "idx = get_index()\n"
       "for q in ['\"军用领域的收入\"', '\"军用收入领域\"', '\"大客户销售部\"', '\"收入领域军用\"']:\n"
       "    print(f'{q:22s} → {[h.page_label for h in idx.search(q, 5)]}')")

    shot(4, "工单6-04-布尔查询.png", "★ NOT 不是「对全库取补」，条数必须依次变化")
    py("from app.core.fulltext import get_index\n"
       "idx = get_index()\n"
       "for q in ['军用', '军用 AND 视频', '军用 AND NOT 视频']:\n"
       "    print(f'{q:20s} → {len(idx.search(q, 1000)):4d} 条')")

    shot(5, "工单6-05-模糊匹配不打折数字.png", "★ 错别字能命中；5,530~ 必须命中不了 5,520")
    py("from app.core.fulltext import get_index\n"
       "idx = get_index()\n"
       "for q in ['军用领城~', '5,520', '5,530~']:\n"
       "    print(f'{q:12s} → {[h.page_label for h in idx.search(q, 3)]}')")

    shot(6, "工单6-06-多字段检索.png", "title=章节标题 / abstract=前120字 / body=全文，BM25F 加权 3:2:1")
    py("from app.core.fulltext import get_index\n"
       "idx = get_index()\n"
       "for q in ['title:募集资金用途', 'body:募集资金用途']:\n"
       "    print(f'{q:22s} → {[h.page_label for h in idx.search(q, 5)]}')")

    if slow:
        shot(7, "工单6-07-三种模式对比.png", "★★ 约 5 分钟；工单点名要讲「两种检索各用什么技术」")
        sub([sys.executable, "scripts/eval.py", "--compare",
             "optimized,fulltext,hybrid", "--no-judge", "--no-norag"], timeout=1200)

        shot(8, "工单6-08-融合权重扫描.png", "★ 约 1 分钟（阶段一不调 LLM）")
        sub([sys.executable, "scripts/sweep_fusion.py"], timeout=900)
    else:
        shot(7, "工单6-07-三种模式对比.png", "（略过：加 --slow 才会跑，约 5 分钟）")
        shot(8, "工单6-08-融合权重扫描.png", "（略过：加 --slow 才会跑，约 1 分钟）")

    shot(9, "工单6-09-召回率达标.png", "★★核心：rag_rule_hits 16 与 avg_recall_coverage 1.0")
    f = latest_eval(profile)
    if not f:
        print(f"{Y}还没有 {profile} 的报告 —— 先跑 `scripts/eval.py --profile {profile}`"
              f"（或加 --slow 跑第 7 张的 --compare）{R}")
    else:
        s = json.load(open(f))["summary"]
        print(f"报告 {os.path.basename(f)}")
        for k in ["n", "profile", "retrieval_mode", "fusion", "reranker",
                  "rag_rule_hits", "rule_hit_rate", "avg_recall_coverage",
                  "recall_full_count", "recall_full_rate",
                  "ttft_under_3s", "ttft_p50_ms"]:
            print(f"  {k:24s} {s.get(k)}")

    shot(15, "工单6-15-Rebuild-Sparse.png",
         "★ 注意：这一步会**写库**（delete+insert，失败会自动回滚）")
    sub([sys.executable, "scripts/rebuild_sparse.py"], timeout=900)

    shot(16, "工单6-16-Milvus原生混合检索.png", "向量库原生的 hybrid_search + RRFRanker")
    py("import asyncio\n"
       "from app.core.vectorstore import VectorStore, lexical_sparse\n"
       "from app.core.embedder import Embedder\n"
       "async def main():\n"
       "    s = VectorStore(); v = await Embedder().embed_one('注册资本 军用领域收入')\n"
       "    hits = s.hybrid_search(v, lexical_sparse('注册资本 军用领域收入'), 3, ranker='rrf')\n"
       "    for h in hits: print(f'{h.page_label:10s} {h.doc_name[:12]:14s} {h.content[:34]!r}')\n"
       "asyncio.run(main())")

    if slow:
        shot(17, "工单6-17-测试通过.png", "约 1 分钟，末行应是 144 passed")
        sub([sys.executable, "-m", "pytest", "tests/", "-q"], timeout=900)
    else:
        shot(17, "工单6-17-测试通过.png", "（略过：加 --slow 才会跑）")

    shot(18, "工单6-18-回归闸门.png", "★ 倒排索引与重排器的专项测试")
    sub([sys.executable, "-m", "pytest", "tests/test_fulltext.py",
         "tests/test_rerank.py", "-q"], timeout=600)

    shot(19, "工单6-19-工单编号注释.png", "工单备注要求代码注释含工单编号")
    for fn in ["app/core/fulltext.py", "app/core/fusion.py",
               "app/core/rerank.py", "app/core/text_analysis.py"]:
        print(f"── {fn}")
        print(Path(fn).read_text(encoding="utf-8").splitlines()[0])


# ======================================================================
# 工单05
# ======================================================================
def wo05(slow: bool) -> None:
    print(f"\n\n{B}###### 工单05 · Query 理解优化 —— 终端类截图 ######{R}")

    shot(1, "工单5-01-环境自检.png", "要看见「9. Query 理解（多轮对话，工单05）」5 项全 OK")
    preflight_tail(20)

    shot(10, "工单5-10-每轮独立反馈.png",
         "先在浏览器第 3 轮答案下点 👍，这里才会出现记录（文件是首次提交时创建的）")
    fb = Path.home() / "rag-data" / "feedback" / "feedback.jsonl"
    if not fb.exists():
        print(f"{Y}文件还不存在：{fb}")
        print("→ 说明还没有成功提交过反馈。去「多轮对话」页在某一轮答案下点 👍，再回来跑这段。{R}")
    else:
        for line in fb.read_text(encoding="utf-8").splitlines()[-2:]:
            print(line)

    shot(14, "工单5-14-多轮评测汇总.png", "★★★核心：改写逐字 15/15、LLM 兜底 0 次、会话隔离 True")
    mt = Path.home() / "rag-data" / "eval" / "mt-latest.json"
    if not mt.exists():
        print(f"{Y}还没有多轮报告 —— 先跑 `scripts/eval_multiturn.py`（约 3 分钟）{R}")
    else:
        s = json.load(open(mt))["summary"]
        for k in ["n_scripts", "n_turns", "rewrite_exact_hits", "rewrite_exact_rate",
                  "rewrite_doc_rate", "rule_resolve_rate", "rewrite_llm_calls",
                  "rule_hit", "rule_hit_n", "rule_hit_rate", "ttft_under_3s",
                  "ttft_p50_ms", "ttft_p95_ms", "ttft_max_ms", "rewrite_ms_p50",
                  "session_expired_reported", "isolation_ok"]:
            print(f"  {k:24s} {s.get(k)}")

    shot(15, "工单5-15-多轮逐轮明细.png", "把终端最大化，16 行一屏刚好")
    fs = glob.glob(os.path.expanduser("~/rag-data/eval/mt-*.json"))
    if not fs:
        print(f"{Y}还没有多轮报告{R}")
    else:
        f = max(fs, key=os.path.getmtime)
        print(f"报告：{os.path.basename(f)}")
        print("剧本             轮 改写方式      逐字 命中  改写后的问题")
        for it in json.load(open(f))["items"]:
            y = "Y" if it["rewrite_exact"] else "N"
            h = "Y" if it["rule_hit"] else "-"
            print(f"{it['script_id']:16s} {it['turn']} {it['method']:12s} {y}    {h}    "
                  f"{it['rewritten'][:38]}")

    if slow:
        shot(16, "工单5-16-单轮16题回归.png", "★ 约 6 分钟，要看到 16/16 与 16/16")
        sub([sys.executable, "scripts/eval.py", "--profile", "optimized"], timeout=1200)
    else:
        shot(16, "工单5-16-单轮16题回归.png", "（略过：加 --slow 才会跑，约 6 分钟）")

    shot(18, "工单5-18-回归闸门测试.png",
         "★ 三条：既有 16 题零改写 / 5 轮剧本 / 会话隔离")
    sub([sys.executable, "-m", "pytest", "tests/test_multi_turn.py", "-v",
         "-k", "delivered or five_turn or isolation"], timeout=600)

    shot(19, "工单5-19-工单编号注释.png", "工单备注要求代码注释含工单编号")
    for fn in ["app/core/session.py", "app/core/query_understanding.py",
               "app/core/multiturn.py"]:
        print(f"── {fn}")
        print(Path(fn).read_text(encoding="utf-8").splitlines()[0])
    n = 0
    for pat in ("*.py", "*.js", "*.html", "*.css"):
        for d in ("app", "scripts", "tests"):
            for p in Path(d).rglob(pat):
                if "Query 理解优化任务" in p.read_text(encoding="utf-8", errors="ignore"):
                    n += 1
    print(f"── 全仓库带工单05 编号的源文件共 {n} 个")


# ======================================================================
# 工单07
# ======================================================================
def wo07(slow: bool) -> None:
    print(f"\n\n{B}###### 工单07 · 功能测试及评估 —— 终端类截图 ######{R}")

    shot(1, "工单7-01-环境自检.png",
         "要看见「11. ccf 语料（工单07)」全 OK：9 份 PDF / 页码模板 / 页数 / 实读覆盖率")
    preflight_tail(26)

    shot(2, "工单7-02-9份年报配置复核.png",
         "工单07 新增的复核脚本：4 条判据 × 9 份年报。全量约 8 分钟，这里抽样 40 页")
    sub([sys.executable, "scripts/verify_ccf_profiles.py", "--sample", "40"])

    shot(3, "工单7-03-语料入库状态.png",
         "库中应有 11 份文档：两份招股书 + 9 份年报")
    sub([sys.executable, "scripts/ingest_all.py", "--status"])

    shot(4, "工单7-04-取证工具.png",
         "出题时用它回到 PDF 上逐条核实参考答案与证据页（页码取自页脚实读）")
    sub([sys.executable, "scripts/lookup_ccf.py", "拨备覆盖率", "--limit", "4",
         "--context", "80"])

    if slow:
        shot(5, "工单7-05-10题评测.png",
             "工单07 核心产出：10 道题 × 两个剖面（约 8 分钟）")
        sub([sys.executable, "scripts/eval.py",
             "--questions", "eval/wo07_questions.json",
             "--compare", "optimized,hybrid"], timeout=2400)

        shot(6, "工单7-06-检索问题归因.png",
             "把「答不出来」拆成 语料缺口 / 检索侧漏召 / 生成侧漏答 三类（约 2 分钟）")
        sub([sys.executable, "scripts/diagnose_wo07.py"], timeout=1200)

        shot(7, "工单7-07-单元测试.png",
             "148 passed —— 含工单07 新增的 4 条量纲换算用例")
        sub([sys.executable, "-m", "pytest", "tests/", "-q"], timeout=1200)

        shot(8, "工单7-08-16题回归闸门.png",
             "★ 关键取证：新语料入同一 collection 后，招股书那 16 题必须仍是 16/16")
        sub([sys.executable, "scripts/eval.py", "--profile", "optimized",
             "--no-judge", "--no-norag", "--quiet"], timeout=2400)
    else:
        for n, name, note in (
            (5, "工单7-05-10题评测.png", "10 道题 × 两个剖面（约 8 分钟）"),
            (6, "工单7-06-检索问题归因.png", "三类归因（约 2 分钟）"),
            (7, "工单7-07-单元测试.png", "148 passed"),
            (8, "工单7-08-16题回归闸门.png", "16/16 不回归"),
        ):
            shot(n, name, note)
            print(f"{Y}（--slow 才跑；命令见 docs/工单07-演示截图清单.md 第 {n} 张）{R}\n")


# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="演示截图作业台（打印要截图的终端输出）")
    ap.add_argument("--wo", default="all", choices=["05", "06", "07", "all"])
    ap.add_argument("--profile", default="hybrid", help="第 9 张取哪个剖面的报告")
    ap.add_argument("--slow", action="store_true",
                    help="连耗时项一起跑（评测约 6 分钟、pytest 约 1 分钟）")
    args = ap.parse_args()

    print(f"{DIM}工作目录：{ROOT}{R}")
    if not args.slow:
        print(f"{Y}提示：默认跳过耗时项（评测/pytest）。要全跑加 --slow。{R}")

    global WO
    if args.wo in ("07", "all"):
        WO = "07"
        wo07(args.slow)
    if args.wo in ("06", "all"):
        WO = "06"
        wo06(args.profile, args.slow)
    if args.wo in ("05", "all"):
        WO = "05"
        wo05(args.slow)

    print(f"\n{B}{'=' * 74}{R}")
    print(f"{G}终端类截图的内容都打完了。浏览器类的（工单07 第 9~12 张、"
          f"工单06 第 10~14 张、工单05 第 2~13 张）见 docs/ 下的截图手册。{R}")
    print(f"{B}{'=' * 74}{R}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
