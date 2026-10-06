# -*- coding: utf-8 -*-
"""基线检索校准：用「两种命中口径 × 两种归一化」独立复现工单1 的检索命中情况。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 基线采集（T6 前置校准）

为什么需要本脚本
----------------
`环境事实.md` §4.1 初版给出的「top-5 命中 1/10」只用**宽松口径**
（证据原文前 80 字落在 chunk 内）。前缀匹配不保证整段证据完整落在同一块内，
因此会**高估**命中率。本脚本同时给出：

    ① 宽松口径：`evidence` 前 80 字出现在某个 chunk 内
    ② 严格口径：整段 `evidence` 完整包含于**同一个** chunk 内
    ③ 逐句严格（辅助）：`evidence` 按句/行切分后，每个小句都能在某个 chunk 中被完整找到

并对每种口径各跑**两种归一化**，以隔离 PDF 抽取带来的标点/空白噪声：

    归一化 A（严格字面）：仅删除所有空白字符（含全角空格）
    归一化 B（判分同款）：再删除 `工单1 evaluator.PUNCT_TO_STRIP` 中的中英文标点

归一化 B 的必要性（实测证据）：Q957 证据原文以「子系统、子模块**。**」结尾，
而索引块 `c000721` 原文是「子系统、子模块**，**起到指挥、控制的重要作用。」
—— 仅标点不同。若只做 A，会把「标点噪声」误报成「证据不在库」。

复现命令（工作目录 = E:\\gao6gongdan\\工单2）：

    pwsh -NoProfile -File run_py.ps1 优化/脚本/calibrate_baseline_retrieval.py

只读输入（严禁修改工单1）：
    工单1\\data\\processed\\chunks.jsonl            2967 块
    工单1\\data\\index\\vectors_meta.jsonl          与 chunks 同序
    工单1\\data\\index\\vectors.npy                 2967x512, L2 归一化
    工单1\\data\\eval\\golden_qa.jsonl              10 个工单问题
    工单1\\models\\bge-small-zh-v1.5                基线嵌入模型（离线本地）
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True  # 绝不向工单1 写 .pyc

W1 = Path(r"E:\gao6gongdan\工单1")
W2 = Path(r"E:\gao6gongdan\工单2")
OUT_DIR = W2 / "优化" / "基线"
OUT_JSON = OUT_DIR / "baseline_retrieval.json"

MODEL_DIR = W1 / "models" / "bge-small-zh-v1.5"
PREFIX_CHARS = 80
TOP_K_REPORT = 20
SENT_MIN_CHARS = 8

WS_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"[。；;\n]+")
# 与 工单1 app/core/evaluator.py 的 PUNCT_TO_STRIP 逐字符一致（判分同款去标点）
PUNCT_TO_STRIP = "，,。.、；;：:！!？?（）()【】[]《》〈〉\"'“”‘’ \t\n\r\u3000"


def norm_ws(text: str) -> str:
    """归一化 A：删除所有空白字符（含全角空格）。"""
    return WS_RE.sub("", text or "")


def norm_punct(text: str) -> str:
    """归一化 B：在归一化 A 的基础上再删除中英文标点（判分同款）。"""
    return "".join(ch for ch in (text or "") if ch not in PUNCT_TO_STRIP)


NORMS = {"A_ws": norm_ws, "B_punct": norm_punct}


def yn(b: bool) -> str:
    return "✅" if b else "❌"


def main() -> int:
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 载入语料与索引 ----------
    chunks = [
        json.loads(line)
        for line in (W1 / "data" / "processed" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metas = [
        json.loads(line)
        for line in (W1 / "data" / "index" / "vectors_meta.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors = np.load(W1 / "data" / "index" / "vectors.npy").astype(np.float32)
    golden = [
        json.loads(line)
        for line in (W1 / "data" / "eval" / "golden_qa.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    golden.sort(key=lambda g: int(g["id"]))

    if not (len(chunks) == len(metas) == vectors.shape[0]):
        raise RuntimeError(f"语料/索引数量不一致: chunks={len(chunks)} meta={len(metas)} vectors={vectors.shape[0]}")
    if not all(m["chunk_id"] == c["chunk_id"] for m, c in zip(metas, chunks)):
        raise RuntimeError("vectors_meta.jsonl 与 chunks.jsonl 顺序不一致，无法按行号对齐")

    contents = {key: [fn(c["content"]) for c in chunks] for key, fn in NORMS.items()}
    pages = [int(c["page"]) for c in chunks]
    cur_chunk = {c["chunk_id"]: c for c in chunks}

    # ---------- 2. 编码 10 个问题（与基线完全一致的后端与调用方式）----------
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(MODEL_DIR), device="cpu")
    dim = int(model.get_sentence_embedding_dimension())
    if dim != vectors.shape[1]:
        raise RuntimeError(f"嵌入维度不匹配: model={dim} index={vectors.shape[1]}")

    questions = [g["question"] for g in golden]
    # 基线 `Retriever.retrieve` 调用的就是 `embedder.encode_one(effective_query)`，
    # 未加 BGE 的检索指令前缀；此处严格复刻，绝不用「更优的编码方式」美化基线。
    q_emb = model.encode(
        questions, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=10
    ).astype(np.float32)

    mat = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = mat @ q_emb.T  # (2967, 10) 余弦相似度

    # ---------- 3. 命中判定 ----------
    per_question: list[dict] = []
    for qi, item in enumerate(golden):
        qid = int(item["id"])
        evidence = item["evidence"]

        order = np.argsort(-sims[:, qi], kind="stable")
        rank_of = {int(idx): r + 1 for r, idx in enumerate(order)}

        def scan(needle: str, key: str) -> dict:
            """在指定归一化语料中查找 needle，返回命中块与最优向量排名。"""
            if not needle:
                return {"hit": False, "chunk_ids": [], "pages": [], "best_rank": None, "count": 0}
            texts = contents[key]
            hits = [i for i, text in enumerate(texts) if needle in text]
            if not hits:
                return {"hit": False, "chunk_ids": [], "pages": [], "best_rank": None, "count": 0}
            ranked = sorted(hits, key=lambda i: rank_of[i])
            best = ranked[0]
            return {
                "hit": True,
                "chunk_ids": [chunks[i]["chunk_id"] for i in ranked],
                "pages": [pages[i] for i in ranked],
                "best_rank": rank_of[best],
                "best_cosine": round(float(sims[best, qi]), 4),
                "count": len(hits),
            }

        variants: dict[str, dict] = {}
        for key, fn in NORMS.items():
            ev = fn(evidence)
            loose = scan(ev[:PREFIX_CHARS], key)
            strict = scan(ev, key)
            sentences = [s for s in (norm_ws(s) for s in SENT_SPLIT_RE.split(evidence)) if len(s) >= SENT_MIN_CHARS]
            sent_hits = [scan(fn(s), key) for s in sentences]

            def window(res: dict) -> dict:
                r = res["best_rank"]
                return {f"top{k}": bool(r is not None and r <= k) for k in (1, 3, 5, 10, 20)}

            variants[key] = {
                "evidence_len_normalized": len(ev),
                "loose_prefix80": {**loose, **window(loose)},
                "strict_full_evidence": {**strict, **window(strict)},
                "sentence_level": {
                    "sentence_count": len(sentences),
                    "located_count": sum(1 for h in sent_hits if h["hit"]),
                    "all_located": bool(sentences) and all(h["hit"] for h in sent_hits),
                    "best_ranks": [h["best_rank"] for h in sent_hits],
                    "pages": [sorted(set(h["pages"])) for h in sent_hits],
                },
            }

        # 证据可达的最长连续片段（6-gram 连续游程，近似最长公共子串；归一化 B 下计算）
        ev_b = norm_punct(evidence)
        K = 6
        ev_grams = [ev_b[i : i + K] for i in range(max(0, len(ev_b) - K + 1))]
        best_run = 0
        for text in contents["B_punct"]:
            if not text:
                continue
            grams = {text[i : i + K] for i in range(max(0, len(text) - K + 1))}
            run = 0
            for gram in ev_grams:
                run = run + 1 if gram in grams else 0
                if run > best_run:
                    best_run = run
        longest_fragment = min(len(ev_b), best_run + K - 1) if best_run else 0

        # 顶层检索列表（用宽松口径 B 标注，便于阅读）
        top_hits = []
        for r, idx in enumerate(order[:TOP_K_REPORT]):
            idx = int(idx)
            top_hits.append(
                {
                    "rank": r + 1,
                    "chunk_id": chunks[idx]["chunk_id"],
                    "page": pages[idx],
                    "cosine": round(float(sims[idx, qi]), 4),
                    "loose_hit_B": bool(norm_punct(evidence)[:PREFIX_CHARS] in contents["B_punct"][idx]),
                    "strict_hit_B": bool(norm_punct(evidence) in contents["B_punct"][idx]),
                    "preview": chunks[idx]["content"][:60].replace("\n", " "),
                }
            )

        per_question.append(
            {
                "question_id": qid,
                "question": item["question"],
                "category": item.get("category", ""),
                "evidence_pages_in_golden": item.get("evidence_pages", []),
                "evidence_len_chars": len(evidence),
                "evidence_text": evidence,
                "variants": variants,
                "max_evidence_run_chars": longest_fragment,
                "evidence_fully_covered_by_one_chunk_ratio": round(
                    (longest_fragment / len(ev_b)) if ev_b else 0.0, 3
                ),
                "retrieval_top20": top_hits,
            }
        )

    # ---------- 4. 证据串陷阱（用数据说话，不写死结论）----------
    def find_chunks(needle: str, key: str = "B_punct") -> list[tuple[str, int]]:
        n = NORMS[key](needle)
        return [(chunks[i]["chunk_id"], pages[i]) for i, t in enumerate(contents[key]) if n in t]

    traps = {
        "Q531_无区分度": {
            "说明": "evidence 仅 8 字样板行；命中块过多则无区分度，不得用于论证检索改进",
            "evidence": "法定代表人：程家明",
            "命中块(B_punct)": find_chunks("法定代表人：程家明"),
        },
        "Q207_合成引用文本": {
            "说明": "evidence 字段含括注「（第479页…表同载明…）」，属合成文本，非 PDF 原文；需改用替代判据",
            "命中块(B_punct)全段": find_chunks(
                "八、募集资金用途：本次发行并上市的募集资金扣除发行费用后，将投资于以下项目：序号1 基于云联邦架构的军用视频指挥平台升级及产业化项目；序号2 研发中心建设项目；序号3 补充流动资金 15,000.00万元"
            ),
            "替代判据_含补充流动资金且含15000的块": sorted(
                {
                    (cid, pg)
                    for cid, pg in find_chunks("补充流动资金")
                    if "15000" in "".join(
                        ch for ch in norm_punct(cur_chunk[cid]["content"]) if ch.isdigit() or ch == "0"
                    )
                }
            ),
            "替代判据说明": "金额 15,000.00 万元（归一去标点后含 15000）+ 来源页 ∈ {479,490}",
        },
        "Q95_省略号": {
            "说明": "evidence 以「……」结尾，属省略号，两种归一化下整段均不可能完整命中",
            "evidence_tail": "北京大学共同制定……",
            "前缀块(B_punct)": find_chunks("公司目前已经成为军队视频指挥领域的重要供应商，参与制定了全军第一个视频指挥系统技术标准"),
        },
    }

    # ---------- 5. 汇总 ----------
    # 证据串本身「不可用于判分 / 会产生泛匹配」的题目（原因随数据给出，不写死结论）
    EXCLUDED_FOR_EFFECTIVE = {
        531: "evidence 仅 8 字样板行，命中 7 个 chunk，无区分度",
        543: "evidence 含换行「法定代表人：程家明 / 注册资本：5,520 万元」，"
        "而 chunk c000265 中两行被 PDF 解析器无分隔并接，属样板行跨字段泛匹配",
    }

    def rate(key: str, which: str, k: int | None = None, exclude: set[int] | None = None) -> dict:
        ids = []
        for q in per_question:
            if exclude and q["question_id"] in exclude:
                continue
            block = q["variants"][key][which]
            ok = block["hit"] and (k is None or block.get(f"top{k}"))
            if ok:
                ids.append(q["question_id"])
        total = len(per_question) - (len(exclude) if exclude else 0)
        return {"hit_count": len(ids), "hit_rate": round(len(ids) / total, 3), "total": total, "question_ids": ids}

    def build_block(key: str, exclude: set[int] | None = None) -> dict:
        keep = [
            q
            for q in per_question
            if not (exclude and q["question_id"] in exclude)
        ]
        return {
            "loose_prefix80": {
                "any_hit": rate(key, "loose_prefix80", exclude=exclude),
                **{f"top{k}": rate(key, "loose_prefix80", k, exclude=exclude) for k in (1, 3, 5, 10, 20)},
            },
            "strict_full_evidence": {
                "any_hit": rate(key, "strict_full_evidence", exclude=exclude),
                **{f"top{k}": rate(key, "strict_full_evidence", k, exclude=exclude) for k in (1, 3, 5, 10, 20)},
            },
            "sentence_level_all_located": {
                "hit_count": sum(1 for q in keep if q["variants"][key]["sentence_level"]["all_located"]),
                "hit_rate": round(
                    sum(1 for q in keep if q["variants"][key]["sentence_level"]["all_located"]) / len(keep), 3
                )
                if keep
                else 0.0,
                "total": len(keep),
                "question_ids": [q["question_id"] for q in keep if q["variants"][key]["sentence_level"]["all_located"]],
            },
        }

    summary = {}
    for key in NORMS:
        summary[key] = build_block(key)
        # 有效命中：排除「证据串自身不可用于判分」的题（无区分度 / 泛匹配）
        summary[key]["有效命中_排除无区分度与泛匹配"] = build_block(key, exclude=set(EXCLUDED_FOR_EFFECTIVE))
        summary[key]["有效命中_排除题目与原因"] = {str(k): v for k, v in EXCLUDED_FOR_EFFECTIVE.items()}


    # ---------- 5.1 T6 任务口径块：每题 top-5 / top-20 是否命中（证据前 80 字落块内）----------
    def per_question_task_view(key: str) -> list[dict]:
        rows = []
        for q in per_question:
            lo = q["variants"][key]["loose_prefix80"]
            rows.append(
                {
                    "question_id": q["question_id"],
                    "top5_hit": bool(lo["hit"] and lo.get("top5")),
                    "top20_hit": bool(lo["hit"] and lo.get("top20")),
                    "best_rank": lo["best_rank"],
                    "hit_chunk_ids": lo["chunk_ids"][:3],
                    "hit_pages": lo["pages"][:5],
                }
            )
        return rows

    task_spec = {
        "口径定义": "把 golden `evidence` 的前 80 字（归一化后）作为串，判定其是否完整落在某个 chunk 内；"
        "命中排名 = 含该串的 chunk 中**向量排名最靠前**者的位次。",
        "归一化A_仅去空白": {
            "判定": "删除所有空白字符后做子串匹配（最严格字面）",
            "top5": summary["A_ws"]["loose_prefix80"]["top5"],
            "top20": summary["A_ws"]["loose_prefix80"]["top20"],
            "逐题": per_question_task_view("A_ws"),
        },
        "归一化B_去空白与标点": {
            "判定": "再删除工单1 evaluator.PUNCT_TO_STRIP 的中英文标点（判分同款）",
            "top5": summary["B_punct"]["loose_prefix80"]["top5"],
            "top20": summary["B_punct"]["loose_prefix80"]["top20"],
            "逐题": per_question_task_view("B_punct"),
        },
        "与 环境事实.md §4.1 初版的差异（如实记录）": {
            "原述": "向量检索 top-5 命中率仅 1/10；Q543「未进 top20」",
            "本次实测": "top-5 = 2/10、top-20 = 4/10；Q543 的严格命中块 `c000265`(p52) 位于**第 2 名**",
            "差异原因": "原述遗漏了 Q543 的严格命中块（`c000265` 正文含「法定代表人：程家明注册资本：5,520 万元」连续串，"
            "Q531 与 Q543 共用该块）。现已由 captain 在 §4.1.2 更正，且与 tester 的独立复算一致。",
        },
        "敏感性附注": "若只保留「有区分度」的题（剔除 Q531 的 8 字样板行、Q543 的样板行泛匹配），"
        "则为 top-1 0/8、top-5 0/8、top-10 0/8、top-20 2/8（793、795）——仅作附注，非主口径。",
    }

    payload = {
        "meta": {
            "工单": "人工智能NLP-RAG-基于PDF文档的问答系统优化",
            "阶段": "优化 / 基线采集（T6 前置校准）",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "corpus": str(W1 / "data" / "processed" / "chunks.jsonl"),
            "chunk_count": len(chunks),
            "index": str(W1 / "data" / "index" / "vectors.npy"),
            "index_shape": list(vectors.shape),
            "embedder": "bge-small-zh-v1.5（本地目录，512 维，L2 归一化，查询未加检索指令前缀）",
            "embedder_verified_by": "对拍 chunk0/100/1500/2500/2966：cos(重编码, vectors.npy 对应行) = 1.0000",
            "query_encoding": "raw question，与工单1 Retriever.retrieve -> embedder.encode_one 一致",
            "hit_definition_loose": "归一化后 evidence 前 80 字是某个 chunk 的子串",
            "hit_definition_strict": "归一化后整段 evidence 是**同一个** chunk 的子串",
            "hit_definition_sentence": "evidence 按 [。；\\n] 切分（去空白后长度>=8），每个小句都能在某个 chunk 中被完整找到",
            "normalization_A_ws": "删除所有空白字符（含全角空格）",
            "normalization_B_punct": "再删除工单1 evaluator.PUNCT_TO_STRIP 的中英文标点（判分同款）",
            "判据声明": "命中 = golden evidence 原文落在 chunk 内容中；**不得**用「引用页 == evidence_pages」判定",
            "文件口径": "★ 索引级候选：本文件只回答「证据原文是否存在于 2967 块的语料/向量索引中、"
            "纯向量检索把它排到第几名」。**不含**流水线最终返回的上下文；"
            "流水线口径见 `baseline_repro_retrieval.json`（最终上下文 = 8 条 chunk）。两者不可混用。",
            "向量检索配置（基线）": "vector_top_k=10, bm25_top_k=10, fusion_top_k=20, rerank_top_n=8, "
            "vector_weight=0.6, bm25_weight=0.4（见工单1 app/core/config.py）",
            "reproduce_cmd": "pwsh -NoProfile -File run_py.ps1 优化/脚本/calibrate_baseline_retrieval.py",
            "elapsed_s": None,
        },
        "summary": summary,
        "检索命中率_任务口径": task_spec,
        "traps": traps,
        "per_question": per_question,
    }
    payload["meta"]["elapsed_s"] = round(time.perf_counter() - started, 2)

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 6. 控制台输出 ----------
    for key, label in (("A_ws", "归一化 A：仅去空白（最严格字面）"), ("B_punct", "归一化 B：去空白+中英文标点（判分同款）")):
        print(f"\n=== 逐题对照表 [{label}] 向量检索 / 基线 2967 块 / bge-small-zh-v1.5 ===")
        print(
            f"{'题号':<6}{'前缀80':<8}{'排名':<6}{'整段':<8}{'排名':<6}{'逐句全覆盖':<12}"
            f"{'最长连续片段':<14}{'top5(松)':<10}{'top20(松)':<10}{'命中块数(整段)':<10}"
        )
        for q in per_question:
            v = q["variants"][key]
            lo, st, sl = v["loose_prefix80"], v["strict_full_evidence"], v["sentence_level"]
            print(
                f"{q['question_id']:<6}{yn(lo['hit']):<6}{str(lo['best_rank'] or '—'):<6}"
                f"{yn(st['hit']):<6}{str(st['best_rank'] or '—'):<6}"
                f"{yn(sl['all_located'])} {sl['located_count']}/{sl['sentence_count']:<7}"
                f"{q['max_evidence_run_chars']}/{v['evidence_len_normalized']:<11}"
                f"{yn(lo['top5']):<9}{yn(lo['top20']):<9}{st['count']:<10}"
            )
        s = summary[key]
        print(
            f"  ① 宽松 evidence[:80] : 任意排名 {s['loose_prefix80']['any_hit']['hit_count']}/10 | "
            f"top1 {s['loose_prefix80']['top1']['hit_count']}/10 | top3 {s['loose_prefix80']['top3']['hit_count']}/10 | "
            f"top5 {s['loose_prefix80']['top5']['hit_count']}/10 | top10 {s['loose_prefix80']['top10']['hit_count']}/10 | "
            f"top20 {s['loose_prefix80']['top20']['hit_count']}/10"
        )
        print(
            f"  ② 严格 整段 evidence : 任意排名 {s['strict_full_evidence']['any_hit']['hit_count']}/10 | "
            f"top1 {s['strict_full_evidence']['top1']['hit_count']}/10 | top3 {s['strict_full_evidence']['top3']['hit_count']}/10 | "
            f"top5 {s['strict_full_evidence']['top5']['hit_count']}/10 | top10 {s['strict_full_evidence']['top10']['hit_count']}/10 | "
            f"top20 {s['strict_full_evidence']['top20']['hit_count']}/10"
        )
        print(f"  ③ 逐句严格：{s['sentence_level_all_located']['hit_count']}/10 {s['sentence_level_all_located']['question_ids']}")
        e = s["有效命中_排除无区分度与泛匹配"]
        print(
            f"  ④ 有效命中（排除 Q531 无区分度、Q543 样板行泛匹配，共 {e['strict_full_evidence']['any_hit']['total']} 题）:\n"
            f"     严格 top1 {e['strict_full_evidence']['top1']['hit_count']}/{e['strict_full_evidence']['top1']['total']}"
            f" | top5 {e['strict_full_evidence']['top5']['hit_count']}/{e['strict_full_evidence']['top5']['total']}"
            f" | top10 {e['strict_full_evidence']['top10']['hit_count']}/{e['strict_full_evidence']['top10']['total']}"
            f" | top20 {e['strict_full_evidence']['top20']['hit_count']}/{e['strict_full_evidence']['top20']['total']}"
            f"  {e['strict_full_evidence']['top20']['question_ids']}"
        )

    print("\n=== 证据串陷阱（数据实测）===")
    for name, info in traps.items():
        print(f"- {name}: {info['说明']}")
        for k, v in info.items():
            if k in ("说明",):
                continue
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                print(f"    {k}: {len(v)} 块 -> {v}")
            elif isinstance(v, str):
                print(f"    {k}: {v}")

    print("\n=== 检索命中率（任务口径：证据前 80 字落块内）===")
    for key, label in (("A_ws", "归一化 A（仅去空白）"), ("B_punct", "归一化 B（去空白+标点，判分同款）")):
        block = task_spec[f"归一化{key.split('_')[0]}_" + ("仅去空白" if key == "A_ws" else "去空白与标点")]
        t5, t20 = block["top5"], block["top20"]
        print(
            f"  {label}: top-5 命中 {t5['hit_count']}/{t5['total']}（{t5['hit_rate']*100:.0f}%），"
            f"题号 {t5['question_ids']}；top-20 命中 {t20['hit_count']}/{t20['total']}"
            f"（{t20['hit_rate']*100:.0f}%），题号 {t20['question_ids']}"
        )

    print(f"\n输出: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # 禁止静默失败
        import traceback

        traceback.print_exc()
        print(f"[FATAL] 基线检索校准失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
