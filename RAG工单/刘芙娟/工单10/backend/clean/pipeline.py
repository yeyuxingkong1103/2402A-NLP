"""清洗编排：把一份 content_list.json 变成 blocks.jsonl + dropped.jsonl。

处理次序（顺序本身是契约，不能随意调换）：
    校验 -> 逐块渲染(转换类) -> 剔除判定 -> 保护基线 -> 标题路径
         -> 修复合并 -> 编号 -> 保护复检 -> 落盘
"""

from __future__ import annotations

import os

from . import CLEAN_RULE_VERSION, rules_drop, rules_protect
from .errors import (
    CleanError,
    EXIT_EMPTY_OUTPUT,
    EXIT_PROTECT_REGRESSION,
)
from .io_utils import ensure_dir, write_jsonl, write_text
from .loader import load_content_list
from .paths import doc_file_name
from .report import Report
from .rules_convert import is_heading, render_block_text, to_page
from .rules_repair import (
    caption_like,
    detect_level_distortion,
    heading_level,
    looks_like_untagged_heading,
    merge_blocks,
    should_merge,
    split_glued_heading,
)
from .space_merge import merge_cjk_spaces

NO_SPACE_MERGE_TYPES = frozenset({"equation", "code"})


def clean_document(content_list_path: str, doc_id: str, opts) -> tuple[list, list, Report]:
    raw = load_content_list(content_list_path)
    file_name = doc_file_name(content_list_path)
    report = Report()
    convert_stats: dict[str, int] = {}

    # ---- 1) 逐块渲染 + 剔除判定 -------------------------------------------
    kept: list[dict] = []
    dropped: list[dict] = []
    drop_rule_by_idx: dict[int, str] = {}

    for raw_idx, block in enumerate(raw):
        btype = block.get("type", "")
        text, degraded = render_block_text(block, opts, convert_stats)
        heading = is_heading(block)

        # D1：词内空格合并（方程/代码除外，那里空格与反斜杠是语法）
        if btype not in NO_SPACE_MERGE_TYPES:
            text = merge_cjk_spaces(
                text,
                opts,
                lambda a, b, _i=raw_idx: _log_space_merge(report, _i, a, b),
            )

        if heading:
            convert_stats["heading_detect"] = convert_stats.get("heading_detect", 0) + 1

        decision = rules_drop.decide(block, text, opts)
        if decision is not None:
            dropped.append({
                "doc_id": doc_id,
                "block_id": "%s:x%06d" % (doc_id, raw_idx),
                "raw_idx": raw_idx,
                "page_idx": block["page_idx"],
                "page": to_page(block["page_idx"]),
                "drop_rule": decision.rule,
                "block_type": btype,
                "text": decision.text[:2000],
                "char_len": _char_len(decision.text),
            })
            drop_rule_by_idx[raw_idx] = decision.rule
            report.hit("drop", decision.rule)
            continue

        if degraded:
            report.note(
                "convert",
                "存在降级块（内容不完整但未被丢弃），已在 text 内显式标记",
                "块 raw_idx=%d 类型=%s：%s" % (raw_idx, btype, text[:80]),
            )

        kept.append({
            "raw_idx": raw_idx,
            "page_idx": block["page_idx"],
            "page": to_page(block["page_idx"]),
            "block_type": btype,
            "is_heading": heading,
            "caption_demoted": False,
            "degraded": degraded,
            "text": text,
            "text_level": block.get("text_level"),
        })

    report.hit("convert", "convert:page_number", len(raw))

    # ---- 1b) 修复：切出粘连标题（裁决 2-B） -------------------------------
    kept = _split_glued_headings(kept, opts, report)

    # ---- 1c) 修复：补标漏标标题 + 按编号重算层级（裁决 3-D） ---------------
    _tag_and_level(kept, raw, opts, report)

    # ---- 2) 保护基线（剔除前的原始渲染文本） ------------------------------
    before_kept = rules_protect.count_patterns([b["text"] for b in kept])
    for name, n in before_kept.items():
        report.hit("protect", rules_protect.RECORD_PREFIX + name, n)
    _audit_dropped(dropped, report)

    # ---- 3) heading_path ------------------------------------------------
    _build_heading_path(kept)

    # ---- 4) 修复：跨页断句合并（唯一会改正文的修复） ----------------------
    merged = _apply_merge(kept, drop_rule_by_idx, opts, report)

    # ---- 5) 编号与字段 ---------------------------------------------------
    blocks = []
    for seq, blk in enumerate(merged):
        text = blk["text"]
        record = {
            "doc_id": doc_id,
            "block_id": "%s:%06d" % (doc_id, seq),
            "file_name": file_name,
            "page_idx": blk["page_idx"],
            "page": blk["page"],
            "block_type": blk["block_type"],
            "is_heading": blk["is_heading"],
            "heading_path": blk["heading_path"],
            "text": text,
            "char_len": _char_len(text),
            "clean_rule_version": CLEAN_RULE_VERSION,
        }
        if blk.get("merged_from"):
            record["merged_from"] = blk["merged_from"]
            record["raw_text"] = blk["raw_text"]
        blocks.append(record)

    if not blocks:
        raise CleanError(
            EXIT_EMPTY_OUTPUT,
            "清洗后块数为 0：%s\n（全部内容都被剔除，不得写出空产物冒充成功）"
            % content_list_path,
        )

    # ---- 6) 保护复检 ------------------------------------------------------
    after = rules_protect.count_patterns([b["text"] for b in blocks])
    _check_citation_preserved(before_kept, after, report)
    killed = rules_protect.check_regression(before_kept, after)
    if killed:
        raise CleanError(
            EXIT_PROTECT_REGRESSION,
            "保护类判据被误杀（清洗前有、清洗后归零）：\n  " + "\n  ".join(killed),
        )

    # ---- 7) 只报告、不修复的两项（D2） ------------------------------------
    distortion = detect_level_distortion(raw)
    if distortion:
        report.note("repair", "标题层级说明", distortion)
    _report_heading_chain(blocks, report)

    _fill_convert_counts(report, convert_stats)
    _report_drop_ratio(report, len(raw), len(dropped))
    return blocks, dropped, report


# --------------------------------------------------------------------------

def _char_len(text: str) -> int:
    """非空白字符数。docs/04 §6 写的是「中文字符数」，但语料含整段英文时
    该口径会算成 0，误导下游分块的长度判据，故改用非空白字符数。"""
    return sum(1 for ch in text if not ch.isspace())


def _log_space_merge(report: Report, raw_idx: int, original: str, merged: str) -> None:
    report.hit("convert", "convert:cjk_space_merge")
    report.note(
        "convert",
        "D1 词内空格合并（每条均已留痕，可用 --no-space-merge 关闭）",
        "raw_idx=%d  %r -> %r" % (raw_idx, original, merged),
    )


def _audit_dropped(dropped: list[dict], report: Report) -> None:
    """FR-017 / FR-019：被剔除的块里若含紧急症状或剂量表述，必须报警。

    宁可误报也不放过——页眉页脚被误判成噪声只是浪费一次人工确认，
    而一块紧急转诊说明被静默丢掉，是知识库缺一块却无人知道。
    """
    hot = []
    for rec in dropped:
        hits = rules_protect.PROTECTED_PATTERNS["emergency:triage"].findall(rec["text"])
        hits += rules_protect.PROTECTED_PATTERNS["dosage:unit"].findall(rec["text"])
        if hits:
            hot.append((rec, sorted(set(hits))))

    if not hot:
        return
    report.hit("protect", "protection:dropped_content_risk", len(hot))
    for rec, hits in hot[:10]:
        report.note(
            "protect",
            "被剔除的块含紧急症状/剂量表述，须人工确认是否误杀",
            "%s [%s] %s …" % (rec["block_id"], rec["drop_rule"], rec["text"][:80]),
        )


def _check_citation_preserved(before: dict, after: dict, report: Report) -> None:
    """FR-015：引文标注计数必须一致（宪法原则 II）。"""
    n_before = before.get("citation:sup", 0)
    n_after = after.get("citation:sup", 0)
    if n_before != n_after:
        raise CleanError(
            EXIT_PROTECT_REGRESSION,
            "引文标注 <sup> 计数不一致：清洗前 %d 处，清洗后 %d 处。\n"
            "强制溯源引用不可协商，必须修复规则后重跑。" % (n_before, n_after),
        )
    report.hit("protect", "protection:citation_count_stable", n_after)


def _split_glued_headings(kept: list[dict], opts, report: Report) -> list[dict]:
    """裁决 2-B：把粘在段落尾部的章节标题切出来，还原为独立标题块。

    段落块额外带 raw_text，切出的标题参与 heading_path 构建。
    """
    if not getattr(opts, "repair", True):
        return kept

    out: list[dict] = []
    for blk in kept:
        if blk["is_heading"] or blk["block_type"] != "text":
            out.append(blk)
            continue
        parts = split_glued_heading(blk)
        if parts is None:
            out.append(blk)
            continue
        paragraph_text, heading_text = parts

        paragraph = dict(blk)
        paragraph["text"] = paragraph_text
        paragraph["raw_text"] = blk["text"]

        heading = dict(blk)
        heading["text"] = heading_text
        heading["is_heading"] = True
        heading["text_level"] = None
        heading["heading_source"] = "split"

        out.extend([paragraph, heading])
        report.hit("repair", "repair:split_glued_heading")
        report.note(
            "repair",
            "粘连标题已切出还原为标题块（段落带 raw_text 可逐字回溯）",
            "p%d  段落=…%s | 切出标题=%s"
            % (blk["page"], paragraph_text[-24:], heading_text),
        )
    return out


def _tag_and_level(kept: list[dict], raw: list[dict], opts, report: Report) -> None:
    """裁决 3-D：补标漏标标题、降级题注误标、按章节编号重算层级。"""
    if not getattr(opts, "repair", True):
        return

    for blk in kept:
        # 先降级：MinerU 把图/表题注标成了标题，留着会打断章节树
        if blk["is_heading"] and blk.get("heading_source") != "split":
            nxt = raw[blk["raw_idx"] + 1] if blk["raw_idx"] + 1 < len(raw) else None
            reason = caption_like(blk, nxt)
            if reason:
                blk["is_heading"] = False
                blk["caption_demoted"] = True
                report.hit("repair", "repair:demote_caption_heading")
                report.note(
                    "repair",
                    "图/表题注被 MinerU 误标为标题，已降级为正文块（否则会打断章节树）",
                    "p%d [%s] %s" % (blk["page"], reason, blk["text"][:50]),
                )
                continue

        if not blk["is_heading"] and blk["block_type"] == "text":
            if looks_like_untagged_heading(blk["text"]):
                blk["is_heading"] = True
                blk["heading_source"] = "number"
                report.hit("repair", "repair:retag_untagged_heading")
                report.note(
                    "repair",
                    "MinerU 漏标的标题已按章节编号补标",
                    "p%d  %s" % (blk["page"], blk["text"][:60]),
                )

        if not blk["is_heading"]:
            continue

        blk["level"] = heading_level(blk)
        original = blk.get("text_level")
        if original != blk["level"]:
            report.hit("repair", "repair:recompute_heading_level")
            if original is None:
                report.note(
                    "repair",
                    "标题无 text_level，层级由章节编号推出",
                    "p%d  层级=%d  %s" % (blk["page"], blk["level"], blk["text"][:50]),
                )


def _report_heading_chain(blocks: list[dict], report: Report) -> None:
    """抽样展示 heading_path，便于人工一眼判断章节树是否合理。"""
    shown = 0
    for blk in blocks:
        if blk["is_heading"] or not blk["heading_path"]:
            continue
        if len(blk["heading_path"]) >= 3 and shown < 4:
            report.note(
                "repair",
                "heading_path 抽样（层级由章节编号推出）",
                "%s  →  %s" % (blk["block_id"], " / ".join(blk["heading_path"])),
            )
            shown += 1


def _build_heading_path(kept: list[dict]) -> None:
    stack: list[tuple] = []
    for blk in kept:
        if blk["is_heading"]:
            level = blk.get("level") or blk.get("text_level") or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, blk["text"]))
            blk["heading_path"] = [t for _, t in stack]
        else:
            blk["heading_path"] = [t for _, t in stack]


def _apply_merge(kept, drop_rule_by_idx, opts, report) -> list[dict]:
    out: list[dict] = []
    for blk in kept:
        if out:
            gap = [
                rule for i, rule in drop_rule_by_idx.items()
                if out[-1]["raw_idx"] < i < blk["raw_idx"]
            ]
            if should_merge(out[-1], blk, gap, opts):
                prev = out[-1]
                report.hit("repair", "repair:cross_page_merge")
                report.note(
                    "repair",
                    "跨页/断句合并（唯一会改正文的修复；合并块带 raw_text 可逐字回溯）",
                    "p%d 块 raw_idx=%d + p%d 块 raw_idx=%d"
                    % (prev["page"], prev["raw_idx"], blk["page"], blk["raw_idx"]),
                )
                out[-1] = merge_blocks(prev, blk)
                continue
        out.append(blk)
    return out


def _fill_convert_counts(report: Report, stats: dict) -> None:
    mapping = {
        "backslash_unescape": "convert:backslash_unescape",
        "html_entity": "convert:html_entity",
        "table_to_markdown": "convert:table_to_markdown",
        "heading_detect": "convert:heading_detect",
    }
    for key, rule in mapping.items():
        if key in stats:
            report.hit("convert", rule, stats[key])


def _report_drop_ratio(report: Report, total: int, dropped: int) -> None:
    ratio = dropped / total if total else 0.0
    report.hit("drop", "drop:total_blocks", total)
    report.note(
        "drop",
        "丢弃比例 %.1f%%（%d/%d）；超过 40%% 须人工确认清洗规则" % (ratio * 100, dropped, total),
    )
    if ratio > 0.40:
        report.note(
            "drop",
            "【告警】丢弃比例超过 40%%，按 docs/04 §14 P5 必须人工确认后才可继续",
        )


# --------------------------------------------------------------------------

def process_one(content_list_path: str, doc_id: str, output_dir: str, opts) -> Report:
    blocks, dropped, report = clean_document(content_list_path, doc_id, opts)

    out_blocks = os.path.join(output_dir, "%s.blocks.jsonl" % doc_id)
    out_dropped = os.path.join(output_dir, "%s.dropped.jsonl" % doc_id)
    out_report = os.path.join(output_dir, "%s.report.txt" % doc_id)

    if getattr(opts, "dry_run", False):
        report.note("drop", "dry-run：未写出任何产物")
    else:
        ensure_dir(output_dir)
        write_jsonl(out_blocks, blocks)
        write_jsonl(out_dropped, dropped)
        write_text(out_report, report.render())

    report.note("drop", "输入：%s" % content_list_path)
    if not getattr(opts, "dry_run", False):
        report.note("drop", "输出：%s" % out_blocks)
        report.note("drop", "输出：%s" % out_dropped)
    report.note("drop", "保留 %d 块 / 丢弃 %d 块" % (len(blocks), len(dropped)))
    return report
