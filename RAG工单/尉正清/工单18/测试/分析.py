# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""质检结果抽查工具 —— 给工单任务四「结果分析」用

三个子命令，对应工单要求的三项分析：

    python 分析.py pairs  <缓存.jsonl>        版本冲突：看候选两两之间到底多像
    python 分析.py hits   <报告.json>         敏感信息：逐条打出来人工判误报
    python 分析.py tags   <报告.json> [N]     分类标签：随机抽 N 份人工核对

为什么单独写一个：报告里的列表是给人「看」的，而这个脚本是给人「判」的 ——
判完的结论要写进验证报告，所以得能重复跑。
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "研发" / "document_quality_assessment" / "scripts"))
import common   # noqa: E402
import dedup    # noqa: E402

common.utf8_console()


def load_records(path):
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def cmd_pairs(cache, root=None):
    """列出所有「正文够长」的文档两两之间的汉明距离，按距离升序。

    用途：看真实的版本对距离是多少、假对距离是多少，从而定 simhash_distance。

    `root` 用来只看某一个语料 —— 同一个 workdir 里可能攒了多个语料的记录
    （缓存是按文件路径做键的），不筛的话会把两个语料混着两两比较。
    """
    recs = [r for r in load_records(cache) if r.get("simhash") is not None]
    if root:
        recs = [r for r in recs if root.lower() in r["path"].lower()]
    print(f"参与比较 {len(recs)} 份\n")
    rows = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            d = dedup.hamming(recs[i]["simhash"], recs[j]["simhash"])
            rows.append((d, Path(recs[i]["path"]).name, Path(recs[j]["path"]).name))
    rows.sort()
    print("最像的 15 对（汉明距离越小越像，64 位）：")
    for d, a, b in rows[:15]:
        print(f"  距离 {d:>2}  相似度 {1 - d / 64:.1%}  {a}  <->  {b}")
    hist = {}
    for d, _, _ in rows:
        hist[d] = hist.get(d, 0) + 1
    print(f"\n距离分布（前 10 档）：{dict(sorted(hist.items())[:10])}")
    print(f"总对数 {len(rows)}")


def cmd_verify(report):
    """把「待确认的版本冲突列表」逐对打开，取著录项并排打印，供人工判定。

    这是工单任务四要的「人工抽查」—— 光看文件名判不了，得看内容。
    判据用的是专利首页的著录项：名称 (54)、申请人 (71)/(73)、申请日 (22)。
    """
    rep = json.loads(Path(report).read_text(encoding="utf-8"))
    pairs = rep["details"]["dup"]["sim_pairs"]
    print(f"待确认版本冲突 {len(pairs)} 对，逐对取著录项：\n")
    verdict = {"双申请（同技术，两种专利）": 0, "同标题不同申请人": 0,
               "标题不同": 0, "取不到": 0}
    for i, (a, b, sim, dist) in enumerate(pairs, 1):
        fa, fb = _fields(a), _fields(b)
        kind = _judge(fa, fb)
        verdict[kind] = verdict.get(kind, 0) + 1
        print(f"[{i}] {sim:<5} 距离{dist}  → {kind}")
        print(f"    {Path(a).name}\n      名称 {fa['title']}\n      申请人 {fa['applicant']}  申请日 {fa['date']}")
        print(f"    {Path(b).name}\n      名称 {fb['title']}\n      申请人 {fb['applicant']}  申请日 {fb['date']}")
    print("\n判定汇总：")
    for k, v in verdict.items():
        if v:
            print(f"  {k}: {v}/{len(pairs)}")


def _fields(pdf):
    """取专利首页著录项。**不能取「第一行非空文本」** —— 那是
    「(19)中华人民共和国国家知识产权局」这条页眉，两份不同专利完全一样。
    更坑的是这行在 2018 年机构改革后从「中华人民共和国国家知识产权局」
    缩成「国家知识产权局」，拿它比会把**机关更名**当成文档差异。
    """
    import re
    import fitz
    out = {"title": "", "applicant": "", "date": ""}
    try:
        with fitz.open(pdf) as doc:
            txt = doc[0].get_text("text") if doc.page_count else ""
    except Exception as e:
        out["title"] = f"<打不开：{e}>"
        return out
    flat = re.sub(r"[\s　]+", "", txt)
    m = re.search(r"\(54\)(?:发明|实用新型|外观设计)?名称(.{2,60}?)\(57\)", flat)
    if m:
        out["title"] = m.group(1)
    m = re.search(r"\((?:71|73)\)(?:申请人|专利权人)(.{2,40}?)(?:地址|\(72\))", flat)
    if m:
        out["applicant"] = m.group(1)
    m = re.search(r"\(22\)申请日(\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2})", flat)
    if m:
        out["date"] = m.group(1)
    return out


def _judge(fa, fb):
    """判定一对近似文档是什么关系。真阳性 / 误报二选一，好算准确率。

    ## 判据是「同申请人 + 同申请日」，不是标题

    实测 IMDR 语料里捞出来的近似对，全是**同日申请**：同一个申请人把同一项技术
    在**同一天**既报发明专利又报实用新型（中国特有做法）。两件的著录项、摘要、
    说明书几乎一样，只是文献种类码差个 A/U。

    **为什么不能拿标题判**：标题会因权利要求类型而不同 ——
    发明专利能写方法权利要求，实用新型不能，于是同一天的两件成了
        CN111873149A  一种瓷砖壁板加工模具、瓷砖壁板及其加工方法
        CN212602486U  一种瓷砖壁板加工模具以及瓷砖壁板
    拿标题判会把它划成「误报」，实际是同一项技术的两件。
    """
    if not fa["title"] or not fb["title"]:
        return "取不到"
    if fa["applicant"] and fa["applicant"] == fb["applicant"] \
            and fa["date"] and fa["date"] == fb["date"]:
        return "真阳性"
    return "误报"


def cmd_sweep(report, maxd=12, root=None):
    """扫一遍阈值：对每个候选 simhash_distance，算捞出来的对数和准确率。

    选定阈值不能拍脑袋 —— 这个命令回答「阈值定在几，准确率和条数各是多少」，
    于是可以按「准确率够高、条数人还能审完」来选。

    「取不到」= 首页著录项抽不出来（非专利文档，比如年报 .txt），
    没法判真伪，单独统计，不算进准确率的分母。
    """
    rep = json.loads(Path(report).read_text(encoding="utf-8"))
    cache = (Path(__file__).parent.parent / "研发" / "document_quality_assessment"
             / ".quality_work" / "parsed.jsonl")
    recs = [r for r in load_records(cache)
            if r.get("simhash") is not None and (r.get("chars") or 0) >= 200]
    if root:
        recs = [r for r in recs if root.lower() in r["path"].lower()]
    cand = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            d = dedup.hamming(recs[i]["simhash"], recs[j]["simhash"])
            if d <= maxd:
                cand.append((d, recs[i]["path"], recs[j]["path"]))
    print(f"参与比较 {len(recs)} 份，候选 {len(cand)} 对（距离 ≤ {maxd}）\n")
    judged = sorted((d, _judge(_fields(a), _fields(b))) for d, a, b in cand)
    print(f"{'阈值':<6}{'条数':<7}{'真阳性':<8}{'误报':<7}{'取不到':<8}{'准确率':<8}")
    for t in range(2, maxd + 1):
        sub = [v for d, v in judged if d <= t]
        if not sub:
            continue
        tp, fp = sub.count("真阳性"), sub.count("误报")
        unk = sub.count("取不到")
        acc = f"{tp / (tp + fp):.0%}" if tp + fp else "-"
        print(f"{t:<6}{len(sub):<7}{tp:<8}{fp:<7}{unk:<8}{acc:<8}")


def _strip(s):
    import re
    return re.sub(r"^一种|^本实用新型|^发明", "", re.sub(r"[\s　]+", "", s or ""))


def cmd_hits(report):
    """把待审核的敏感信息逐条打出来 —— 人工判误报就靠这个。

    工单验收要求「准确率与误报率」，所以每条要能看出：命中的是什么、
    在哪个文件的什么上下文里、像不像真的。
    """
    d = json.loads(Path(report).read_text(encoding="utf-8"))["details"]["sensitive"]
    hits = d["hits"]
    print(f"命中 {len(hits)} 条，类型分布 {d['summary']['by_type']}\n")
    for i, (src, typ, val, ctx) in enumerate(hits[:60], 1):
        print(f"[{i}] {typ:<8} {val}")
        print(f"    文件 {Path(src).name}")
        print(f"    上下文 …{ctx}…")
    if len(hits) > 60:
        print(f"\n（只打了前 60 条，共 {len(hits)} 条）")


def cmd_tags(report, n=20, seed=18):
    """随机抽 N 份文档，打印路径 + 标签 + 判定依据，用于人工核对准确率。"""
    d = json.loads(Path(report).read_text(encoding="utf-8"))["details"]
    docs = d["tags"]["docs"]
    pdf = {row[0]: row for row in d["pdf"]["uncertain"]}
    random.seed(seed)
    sample = random.sample(docs, min(n, len(docs)))
    print(f"从 {len(docs)} 份里随机抽 {len(sample)} 份（seed={seed}）\n")
    for path, tags, routes in sample:
        p = pdf.get(path)
        print(f"{Path(path).name}\n    标签 {tags}   路由 {routes}"
              + (f"\n    存疑 {p[2]} {p[3]}" if p else ""))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = sys.argv[2] if len(sys.argv) > 2 else ""
    if cmd == "pairs":
        cmd_pairs(arg, sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "hits":
        cmd_hits(arg)
    elif cmd == "verify":
        cmd_verify(arg)
    elif cmd == "sweep":
        cmd_sweep(arg, int(sys.argv[3]) if len(sys.argv) > 3 else 12,
                  sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "tags":
        cmd_tags(arg, int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    else:
        print(__doc__)
