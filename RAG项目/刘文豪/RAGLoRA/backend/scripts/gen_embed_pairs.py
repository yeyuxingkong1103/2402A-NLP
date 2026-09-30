# -*- coding: utf-8 -*-
"""为嵌入微调生成 (query, positive) 训练对。

思路：从已有语料**全量分块后打散**采样 chunk，用 LLM 为每条 chunk 生成
2 个"用户会怎么问"的问题。

⚠️ 43 题 QA 集（eval/qa_set.json）严格排除在训练对之外，避免数据泄漏导致指标虚高。
   防泄漏的实际粒度是 **chunk 级**（见 _build_leak_guard 的说明），
   且**不可能降到零重叠**（同一语料既做微调又做评测）。脚本会在日志中如实打印
   本次覆盖到哪一层、残留多少，不做"零泄漏"的虚假承诺。

退出码：
    0  正常完成
    1  LLM 预检失败（未产生任何训练对）
    2  产出显著低于目标（< MIN_YIELD_RATIO * 目标对数）
    3  连续生成失败达到阈值，判定 LLM 中途不可用而中止
    4  未采到任何 chunk（语料目录缺失/为空）
用法：
    python scripts/gen_embed_pairs.py                 # 续跑（默认）
    python scripts/gen_embed_pairs.py --fresh         # 清空重来
    python scripts/gen_embed_pairs.py --probe         # 只做采样分布统计，不调 LLM
    python scripts/gen_embed_pairs.py --out X.jsonl   # 输出到其它文件（验证用）
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config                      # noqa: E402
from app.core.logging import get_logger          # noqa: E402
from app.services import ingest, llm             # noqa: E402

log = get_logger("gen_pairs")

OUT = config.BACKEND_DIR / "eval" / "embed_pairs.jsonl"
QA_PATH = config.BACKEND_DIR / "eval" / "qa_set.json"

PER_CHUNK = 2          # 每条 chunk 生成几个问题
SAMPLE = {"kb_medical": 150, "kb_legal": 1050}   # 分层采样：按 636:14319 比例缩放

# ---------------------------------------------------------------- 阈值常量
# 集中管理，避免魔数散落在流程里（Important 2 / Minor 6 / Minor 7）
MIN_CHUNK_LEN = 80          # 过短的 chunk 不作为正例，也不计入采样池
MIN_Q_LEN = 6               # 生成问题的长度下界
MAX_Q_LEN = 100             # 生成问题的长度上界：超过基本是模型吐出的说明文字
PROMPT_CHARS = 1500         # 喂给 LLM 的 chunk 截断长度（也是正例的存储长度，见 Minor 8）
MAX_CONSEC_FAILS = 10       # 连续失败达到该次数即中止（非零退出）
MIN_YIELD_RATIO = 0.5       # 最终产出低于目标的该比例即判为异常（非零退出）
CHECKPOINT_EVERY = 50       # 每新增多少对落盘一次（按新增计数，不再用 len(rows) % 50）
CHECKPOINT_SECONDS = 120    # 距上次落盘超过该秒数也强制落盘
MAX_SOURCE_FILE_MATCHES = 5     # expect_source 命中超过这么多文件 → 视为通用前缀，不作为来源
SOURCE_EXCLUDE_MAX_FRACTION = 0.5   # 源文件级排除若会清空超过该比例的语料，则退化为 chunk 级
MIN_PER_FILE = 3                # 每个源文件至少取这么多条（候选足够时），保证小文件不被随机漏掉

EXIT_OK = 0
EXIT_PREFLIGHT = 1
EXIT_LOW_YIELD = 2
EXIT_LLM_ABORT = 3
EXIT_NO_SAMPLE = 4

PROMPT = """你是知识库检索测试员。请根据下面的知识片段，写出 {n} 个用户可能会提出的、需要这段内容才能回答的问题。

要求：
1. 每个问题独立成行，不要编号，不要解释
2. 问题要像真实用户的口语提问，不要照抄原文用词
3. 问题必须能且只能由这段内容回答

知识片段：
{text}
"""


def _qkey(s: str) -> str:
    """问题的归一化键：去掉所有空白与标点并转小写，用于去重。"""
    return re.sub(r"[\s\W_]+", "", s or "").lower()


# ---------------------------------------------------------------- QA 集读取
def _load_qa_items() -> dict[str, list[dict]]:
    """读回 qa_set.json 的原始分组。

    实际结构（用代码打印确认过，不是猜的）：
        {"_comment": str,
         "medical": [{"q","expect_keywords":[...],"expect_sources":[...]}, ...20 条],
         "legal":   [同上, 部分条目带 "corpus_gap": true / "gap_note"],
         "refusal": [{"q","role","note"}, ...3 条]}
    即：顶层是 dict，条目用 "q" 键；答案来源线索在 expect_keywords / expect_sources。
    """
    if not QA_PATH.exists():
        log.warning("评测集不存在，防泄漏无法生效: %s", QA_PATH)
        return {}
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                groups[k] = [it for it in v if isinstance(it, dict)]
    elif isinstance(data, list):
        groups["_all"] = [it for it in data if isinstance(it, dict)]
    return groups


def _load_qa_questions(groups: dict[str, list[dict]]) -> set[str]:
    """已用于评测的问题归一化文本。

    这是**弱防护**：prompt 明确要求模型"不要照抄原文用词"，指望生成的问题与
    评测问题字符串撞车本就不现实。保留它只是为了拦住模型偶尔的原文回显，
    真正起作用的是下方的 chunk 级防泄漏。
    """
    return {_qkey(it.get("q") or it.get("question") or "")
            for grp in groups.values() for it in grp
            if (it.get("q") or it.get("question"))}


def _build_leak_guard(collection: str, groups: dict[str, list[dict]],
                      pool: list[tuple[str, str]]) -> tuple[set[str], list[list[str]], str]:
    """构造 chunk 级防泄漏判据。

    返回 (需整文件排除的来源 stem 集合, 每个评测条目的期望关键词列表, 说明文本)。

    分级策略（按强度从高到低）：
      1. **来源文件级**：评测条目 expect_sources 里点名了语料中的某个文件
         （如"中华人民共和国民法典"），则该文件的全部 chunk 都不进训练集。
         这是最接近"同一 chunk 既当正例又当答案来源"的实质防护。
         - 只做精确/前缀匹配，避免"中华人民共和国"这种通用前缀命中全部 176 部法律；
           命中文件数 > MAX_SOURCE_FILE_MATCHES 的来源直接丢弃并记日志。
         - 若这样排除会清空超过 SOURCE_EXCLUDE_MAX_FRACTION 的语料
           （典型：kb_medical 只有 2 份 PDF，而 18/20 评测题的来源正是这两份），
           则不执行文件级排除，退化为第 2 级，并在日志中显式说明。
      2. **chunk 级关键词**：一条 chunk 同时命中某评测条目**全部**期望关键词
         （且关键词数 ≥ 2，或单个关键词长度 ≥ 4）时，判为该题的答案来源 chunk，排除。

    ⚠️ 诚实局限：本判据只能摘掉"答案来源本身"，无法摘掉**同一语料中的语义近邻**。
       kb_medical 的两份 PDF 是全国高血压指南，评测题也全部出自它们，
       排除答案 chunk 之后剩下的仍是同一批指南的其它段落，残留重叠无法避免。
    """
    role = "medical" if "medical" in collection else "legal"
    items = groups.get(role, [])
    if not items:
        log.warning("%s 在评测集中找不到对应分组（role=%s），本 collection 无 chunk 级防护",
                    collection, role)
        return set(), [], "无对应评测分组"

    stems = {stem for _, stem in pool}
    kw_sets: list[list[str]] = []
    named: list[str] = []
    unmatched: list[str] = []
    for it in items:
        kws = [str(k).strip() for k in (it.get("expect_keywords") or []) if str(k).strip()]
        if len(kws) >= 2 or (len(kws) == 1 and len(kws[0]) >= 4):
            kw_sets.append(kws)
        for src in (it.get("expect_sources") or []):
            src = str(src).strip()
            if not src:
                continue
            hits = {s for s in stems if s == src or s.startswith(src)}
            if not hits:
                unmatched.append(src)
            elif len(hits) > MAX_SOURCE_FILE_MATCHES:
                log.info("来源 %r 命中 %d 个文件（通用前缀），不作为来源标识", src, len(hits))
            else:
                named.extend(hits)

    named_set = set(named)
    total = len(pool)
    excluded_chunks = sum(1 for _, s in pool if s in named_set)
    note = (f"评测条目 {len(items)} 条、关键词规则 {len(kw_sets)} 条；"
            f"点名来源命中 {len(named_set)} 个文件 / {excluded_chunks} chunks")
    if unmatched:
        note += f"；语料中找不到的来源 {len(set(unmatched))} 个: {sorted(set(unmatched))}"

    if total and excluded_chunks / total > SOURCE_EXCLUDE_MAX_FRACTION:
        note += (f"；⚠️ 文件级排除将清空 {excluded_chunks}/{total} "
                 f"(>{SOURCE_EXCLUDE_MAX_FRACTION:.0%}) 的语料，已禁用文件级排除，"
                 f"退化为 chunk 级关键词排除（残留重叠不可避免）")
        return set(), kw_sets, note
    return named_set, kw_sets, note


def _chunk_leaked(text: str, kw_sets: list[list[str]]) -> bool:
    for kws in kw_sets:
        if all(k in text for k in kws):
            return True
    return False


# ---------------------------------------------------------------- 采样
# collection -> 原始语料目录
# ⚠️ kb_legal 的语料在 datasets/legal/chinese_law/ 子目录下（176 部法律 txt）。
#    ingest_all.py 入库用的就是这个子目录；datasets/legal/ 直接 glob 只能看到 2 个 PDF。
CORPUS_DIR = {
    "kb_medical": config.DATASETS_DIR / "medical",
    "kb_legal": config.DATASETS_DIR / "legal" / "chinese_law",
}


def _collect_pool(collection: str) -> list[tuple[str, str]]:
    """读**全部**原始语料文件并分块，返回 [(text, 来源文件名 stem)]。

    ⚠️ 刻意不读 Qdrant —— 后端运行时 Qdrant 嵌入式持有独占文件锁，
    另起进程访问会直接失败（README §七 已知限制）。
    这里改为直接读 datasets/ 下的原始语料并复用 ingest.build_chunks 分块，
    绕开文件锁，且能与服务端并行跑。

    Important 1：**遍历完全部文件**再返回。旧实现按"排序后的前若干文件"累计到
    limit*3 就 break，导致 kb_medical（第一份 PDF 就有 570 chunks）只采到第一份 PDF，
    kb_legal 只覆盖排序靠前的约 40 部法律。全量语料约 1.5 万条、文本十几 MB，
    内存完全放得下，不需要 reservoir sampling。

    Minor 9：空串/过短文本在**进入采样池之前**就丢掉，不参与任何计数。
    """
    root = CORPUS_DIR.get(collection)
    if not root or not root.exists():
        log.warning("语料目录不存在: %s", root)
        return []

    pool: list[tuple[str, str]] = []
    files = ingest.collect_files(root)
    for path in files:
        try:
            chunks, _ = ingest.build_chunks(path)
        except Exception as e:
            log.warning("分块失败 %s: %s", path.name, e)
            continue
        for c in chunks:
            t = (c.get("text") or "").strip()
            if len(t) <= MIN_CHUNK_LEN:      # 含空串
                continue
            pool.append((t, path.stem))
    log.info("%s 全量分块：%d 个文件 -> %d 条 chunk（>%d 字符）",
             collection, len(files), len(pool), MIN_CHUNK_LEN)
    return pool


def _stratified_select(kept: list[tuple[str, str]], limit: int) -> list[tuple[str, str]]:
    """按源文件分层抽取 limit 条（Important 1 的采样部分），**严格不超过 limit**。

    配额规则：
      - 基准配额 = round(limit * 该文件候选数 / 总候选数)，保持"大文件多采"的比例性；
      - 若 limit 足够大（>= 文件数 * MIN_PER_FILE），把配额不足的文件抬到 MIN_PER_FILE，
        保证小文件（如医疗的第二份 PDF、只有几条 chunk 的小法律）不被随机漏掉；
      - 超配时从配额最高的文件上削回，削到基准以下仍超配就继续削，直到恰好 == limit；
      - 配平后若仍有余量（round 取整导致欠配），按剩余容量补给候选最多的文件。
    覆盖率是确定性的，不靠运气；但 limit 小于文件数时只能覆盖 limit 个文件
    （此时会随机挑文件），这是硬约束，日志里会如实反映覆盖文件数。
    """
    by_file: dict[str, list[str]] = {}
    for text, stem in kept:
        by_file.setdefault(stem, []).append(text)
    for items in by_file.values():
        random.shuffle(items)

    n_files = len(by_file)
    total = len(kept)
    target = min(limit, total)
    if target >= n_files * MIN_PER_FILE:
        floor = MIN_PER_FILE
    elif target >= n_files:
        floor = 1
    else:
        floor = 0

    alloc = {stem: min(len(items), max(floor, round(target * len(items) / total)))
             for stem, items in by_file.items()}
    while sum(alloc.values()) > target:            # 超配：从配额最高的文件上削
        cands = [s for s in alloc if alloc[s] > floor]
        if not cands:
            cands = [s for s in alloc if alloc[s] > 0]
        if not cands:
            break
        top = max(alloc[s] for s in cands)
        alloc[random.choice([s for s in cands if alloc[s] == top])] -= 1
    spare = target - sum(alloc.values())           # 欠配：补给仍有剩余候选的文件
    while spare > 0:
        cands = [s for s in alloc if alloc[s] < len(by_file[s])]
        if not cands:
            break
        top = max(len(by_file[s]) - alloc[s] for s in cands)
        alloc[random.choice([s for s in cands if len(by_file[s]) - alloc[s] == top])] += 1
        spare -= 1

    sel: list[tuple[str, str]] = []
    for stem, items in by_file.items():
        sel.extend((t, stem) for t in items[:alloc[stem]])
    random.shuffle(sel)
    return sel


def _sample(collection: str, limit: int, groups: dict[str, list[dict]]):
    """全量收集 -> 防泄漏过滤 -> 按文件分层打散 -> 取 limit 条。

    返回 (选中列表 [(text, stem)], 统计 dict, 防泄漏说明)。
    """
    pool = _collect_pool(collection)
    if not pool:
        return [], {}, "语料为空"

    excl_stems, kw_sets, note = _build_leak_guard(collection, groups, pool)
    log.info("%s 防泄漏：%s", collection, note)

    kept, drop_src, drop_kw = [], 0, 0
    for t, stem in pool:
        if stem in excl_stems:
            drop_src += 1
        elif _chunk_leaked(t, kw_sets):
            drop_kw += 1
        else:
            kept.append((t, stem))

    sel = _stratified_select(kept, limit)
    stats = {
        "pool": len(pool),
        "dropped_source": drop_src,
        "dropped_keyword": drop_kw,
        "candidates": len(kept),
        "selected": len(sel),
        "by_file": {},
    }
    for _, stem in sel:
        stats["by_file"][stem] = stats["by_file"].get(stem, 0) + 1
    log.info("%s 采样：候选 %d -> 选中 %d（源文件级排除 %d，关键词级排除 %d，覆盖 %d 个文件）",
             collection, len(kept), len(sel), drop_src, drop_kw, len(stats["by_file"]))
    if len(sel) < limit:
        log.warning("%s 候选不足：目标 %d 条，实际只有 %d 条", collection, limit, len(sel))
    return sel, stats, note


def probe() -> int:
    """只做采样、不调 LLM 的分布探针（验证 Important 1）。"""
    groups = _load_qa_items()
    print(f"语料采样分布探针（不调用 LLM）  评测集: {QA_PATH}")
    for coll, limit in SAMPLE.items():
        sel, stats, note = _sample(coll, limit, groups)
        print(f"\n===== {coll}  目标 {limit} 条 =====")
        print(f"  全量池 {stats.get('pool')} | 源文件级排除 {stats.get('dropped_source')} "
              f"| 关键词级排除 {stats.get('dropped_keyword')} | 候选 {stats.get('candidates')} "
              f"| 选中 {stats.get('selected')}")
        print(f"  防泄漏: {note}")
        by = stats.get("by_file", {})
        print(f"  选中样本来自 {len(by)} 个文件：")
        for stem, n in sorted(by.items(), key=lambda kv: -kv[1]):
            print(f"    {n:5d}  {stem}")
    return EXIT_OK


def _preflight() -> None:
    """先探一次 LLM，避免"跑了很久才发现模型根本加载不了"。

    本机实测：内存/提交量紧张时 Ollama 加载 7B 会 cudaMalloc OOM。
    这里在启动即探活，失败就直接退出并报错。
    注意它只守得住启动那一刻，运行中途失效由 main() 的连续失败计数兜底。
    """
    try:
        llm.chat([{"role": "user", "content": "回复一个字：好"}],
                 temperature=0.0, max_tokens=8)
    except Exception as e:
        log.error("LLM 预检失败，脚本退出（不会产生任何训练对）: %s", str(e)[:300])
        log.error("请确认 Ollama 能加载 %s 后再重跑；"
                  "本机常见原因是可用内存/提交量不足。", config.LLM_MODEL)
        sys.exit(EXIT_PREFLIGHT)
    log.info("LLM 预检通过（%s 可正常响应）", config.LLM_MODEL)


# ---------------------------------------------------------------- 落盘 / 续跑
def _flush(rows: list[dict], out: Path) -> None:
    """整体落盘（先写临时文件再替换，避免中断留下半截 JSON 行）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    tmp.write_text(body + "\n" if body else "", encoding="utf-8")
    os.replace(tmp, out)


def _load_existing(out: Path) -> tuple[list[dict], set[str], set[str]]:
    """读回已有产出用于续跑（Important 3）。

    返回 (有效行, 已用问题键, 已用正例文本集合)。
    若进程在第 2000 对时死掉，重跑时不会 random.shuffle 后从空开始覆盖，而是接着写。
    """
    rows: list[dict] = []
    qkeys: set[str] = set()
    positives: set[str] = set()
    if not out.exists():
        return rows, qkeys, positives
    bad = 0
    for line in out.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            bad += 1
            continue
        if not isinstance(r, dict) or not r.get("query") or not r.get("positive"):
            bad += 1
            continue
        rows.append(r)
        qkeys.add(_qkey(r["query"]))
        positives.add(r["positive"])
    if bad:
        log.warning("续跑读取时跳过 %d 行无法解析的内容", bad)
    return rows, qkeys, positives


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成嵌入微调 (query, positive) 训练对")
    ap.add_argument("--fresh", action="store_true", help="清空输出文件，从头生成")
    ap.add_argument("--out", default=str(OUT), help=f"输出文件（默认 {OUT}）")
    ap.add_argument("--probe", action="store_true", help="只统计采样分布，不调用 LLM")
    args = ap.parse_args(argv)

    if args.probe:
        return probe()

    out = Path(args.out)
    _preflight()

    groups = _load_qa_items()
    questioned = _load_qa_questions(groups)
    target = sum(SAMPLE.values()) * PER_CHUNK

    if args.fresh and out.exists():
        out.unlink()
        log.info("--fresh：已清空 %s", out)
    rows, seen_q, done_pos = _load_existing(out)
    if rows:
        log.info("续跑：已有 %d 对，本次目标 %d 对 -> %s", len(rows), target, out)
    else:
        log.info("全新生成：目标 %d 对 -> %s", target, out)
    seen_q |= questioned      # 评测问题同样不进训练集

    t0 = time.time()
    new_since_ckpt = 0
    last_ckpt = time.time()
    consec_fail = 0
    total_fail = 0
    attempts = 0
    truncated = 0
    dup_q = 0

    sel_by_coll: dict[str, list[tuple[str, str]]] = {}
    for coll, limit in SAMPLE.items():
        sel, stats, _ = _sample(coll, limit, groups)
        if not sel:
            log.warning("%s 未采到任何 chunk", coll)
        sel_by_coll[coll] = sel
    if not any(sel_by_coll.values()):
        log.error("所有 collection 都未采到 chunk，检查语料目录: %s", CORPUS_DIR)
        return EXIT_NO_SAMPLE

    for coll, sel in sel_by_coll.items():
        for i, (text, stem) in enumerate(sel, 1):
            if text in done_pos:      # 续跑：该 chunk 已生成过，跳过
                continue
            attempts += 1
            body = text[:PROMPT_CHARS]          # Minor 8：正例与喂给 LLM 的文本保持一致
            if len(text) > PROMPT_CHARS:
                truncated += 1
            try:
                raw = llm.chat(
                    [{"role": "user",
                      "content": PROMPT.format(n=PER_CHUNK, text=body)}],
                    temperature=0.7,
                )
                consec_fail = 0
            except Exception as e:
                consec_fail += 1
                total_fail += 1
                log.warning("生成失败(%s %d/%d, 连续 %d 次): %s",
                            coll, i, len(sel), consec_fail, e)
                if consec_fail >= MAX_CONSEC_FAILS:
                    log.error("连续 %d 次生成失败，判定 %s 已不可用，中止；"
                              "已完成 %d 对，正在落盘", consec_fail, config.LLM_MODEL, len(rows))
                    _flush(rows, out)
                    log.error("已落盘 %d 对 -> %s（退出码 %d）", len(rows), out, EXIT_LLM_ABORT)
                    return EXIT_LLM_ABORT
                continue

            for line in raw.splitlines():
                q = line.strip().lstrip("0123456789.、) ").strip().strip('"').strip()
                if len(q) < MIN_Q_LEN or len(q) > MAX_Q_LEN:   # Minor 7：双向长度校验
                    continue
                k = _qkey(q)
                if not k or k in seen_q or k in questioned:     # Minor 5：问题去重
                    dup_q += 1
                    continue
                seen_q.add(k)
                rows.append({"query": q, "positive": body, "collection": coll})
                new_since_ckpt += 1

            # Minor 6：按"新增对数 + 时间"落盘，不再用 len(rows) % 50（会被不定条数追加跳过）
            if new_since_ckpt >= CHECKPOINT_EVERY or (time.time() - last_ckpt) >= CHECKPOINT_SECONDS:
                _flush(rows, out)
                log.info("已生成 %d 对（本次新增 %d）", len(rows), new_since_ckpt)
                new_since_ckpt = 0
                last_ckpt = time.time()

    _flush(rows, out)

    log.info("本次尝试 %d 条 chunk：失败 %d 次、问题去重丢弃 %d 条、"
             "chunk 超 %d 字符被截断 %d 条（截断后文本同时作为 positive）",
             attempts, total_fail, dup_q, PROMPT_CHARS, truncated)
    log.info("耗时 %.1f 分钟，产出 %d 对 -> %s", (time.time() - t0) / 60, len(rows), out)

    if len(rows) < target * MIN_YIELD_RATIO:
        log.error("产出显著低于预期：%d 对 < 目标 %d 的 %.0f%%；"
                  "请检查 Ollama 是否中途失效，可重跑续跑补齐（退出码 %d）",
                  len(rows), target, MIN_YIELD_RATIO * 100, EXIT_LOW_YIELD)
        return EXIT_LOW_YIELD

    print(f"完成：{len(rows)} 对 -> {out}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
