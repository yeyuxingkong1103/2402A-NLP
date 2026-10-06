#!/usr/bin/env python3
"""语料「版本盘点」：同一个文件名族里混着多个版本时，谁才是现行有效的那份？

真机背景（2026-09-21）：``knowledge/lawyer`` 里有 **411 个** ``X__<hash>.md`` 副本，
逐字节比对**没有一对相同** —— 抽样看差异，是 crawler 从 flk.npc.gov.cn 抓到了同一部
司法解释的**不同版本**（例如基名文件 制定日期=2020-12-29/时效性=3，
副本 = 2006-11-23/时效性=2）。法律 RAG 里这是**高危**数据问题：模型可能引到已失效的旧版。

本工具只做**只读盘点 + 建议**，不删任何文件：
* 按"去掉 ``__<hash>`` 的干净文件名"分族；
* 每族按 (时效性, 公布/制定日期) 排序，给出**建议保留**的那份；
* 输出 ``index/version_inventory.json``（供后续"版本过滤"使用）与一段人读摘要。

判定依据全部来自语料文件自带的头部元数据（crawler 写的），不猜：
``- 时效性：3``（3=现行有效，2=已修改，1=已废止 —— 以该库口径为准）、
``- 公布日期：YYYY-MM-DD`` / ``- 制定日期：YYYY-MM-DD``。
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

HASH_SUFFIX_RE = re.compile(r"__[0-9a-f]{6,}(?=\.md$)", re.IGNORECASE)
FIELD_RE = {
    "effect": re.compile(r"^-\s*(?:时效性|效力状态)\s*[：:]\s*(\S+)"),
    "promulgated": re.compile(r"^-\s*(?:公布日期|制定日期|发布日期)\s*[：:]\s*(\S+)"),
    "effective": re.compile(r"^-\s*(?:施行日期|实施日期|生效日期)\s*[：:]\s*(\S+)"),
    "number": re.compile(r"^-\s*(?:法规编号|发文字号)\s*[：:]\s*(\S+)"),
}
#: 时效性口径（以国家法律法规数据库为准，实测三档：3 现行有效 / 2 已修改 / 4 尚未生效；1 已废止）
_EFFECT_IN_FORCE = 3
_EFFECT_SUPERSEDED = 2
_EFFECT_PENDING = 4
_EFFECT_REPEALED = 1
_UNKNOWN_EFFECT = -1

#: 族的三种角色（写进盘点产物，检索侧据此决定"丢 / 留但要提示 / 留"）
ROLE_CURRENT = "current"
ROLE_PENDING = "pending"
ROLE_STALE = "stale"


def _parse_date(text: str) -> str:
    """把 ``2027-01-01`` 一类日期规整成可比较的 ``YYYY-MM-DD``；解析不出返回 ``""``。"""
    match = re.match(r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})", str(text or ""))
    if not match:
        return ""
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def role_of(meta: dict, today: str) -> str:
    """这一份版本**自己**属于哪种状态：现行有效 / 尚未生效 / 已过期。

    **为什么不能只看时效性数值**（真机踩到）：时效性里"数值越大越现行"的假设在
    2026 版《商标法》上失效 —— 它 2026-06-26 公布、**2027-01-01 才施行**，时效性=4
    （尚未生效），数值却最大，于是被当成"最现行"而把现行有效的 2019 版判成过期。
    所以主判据用**施行日期 vs 今天**；已修改/已废止由时效性码判定（那是"被取代"这件事，
    文件自己的日期看不出来）。
    """
    effective = _parse_date(meta.get("effective", ""))
    effect = int(meta.get("effect", _UNKNOWN_EFFECT))
    if effective:
        if effective > today:
            return ROLE_PENDING
    if effect == _EFFECT_PENDING:
        return ROLE_PENDING
    if effect in (_EFFECT_REPEALED, _EFFECT_SUPERSEDED):
        return ROLE_STALE
    return ROLE_CURRENT


def clean_family_name(name: str) -> str:
    return HASH_SUFFIX_RE.sub("", name)


def read_meta(path: Path) -> dict:
    meta: dict = {"effect": _UNKNOWN_EFFECT, "promulgated": "", "effective": "", "number": ""}
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for _ in range(20):
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                for key, pattern in FIELD_RE.items():
                    match = pattern.match(stripped)
                    if not match:
                        continue
                    value = match.group(1).strip()
                    if key == "effect":
                        try:
                            meta["effect"] = int(float(value))
                        except ValueError:
                            meta["effect"] = _UNKNOWN_EFFECT
                    else:
                        meta[key] = value
    except OSError as exc:  # noqa: BLE001 - 单文件读不了不该中断盘点
        meta["error"] = f"{type(exc).__name__}: {exc}"
    return meta


def rank_key(meta: dict, is_plain: bool, today: str) -> tuple:
    """排序键：**现行有效 > 尚未生效 > 已过期**，同类里施行日期新者优先。

    现行有效内部取"施行日期最晚"的那份 = 正在生效的那一版（用户 2026-09-23 拍板：
    回答现行问题用**现行有效版**，涉及尚未施行的新规要提示）。
    """
    role = role_of(meta, today)
    priority = {ROLE_CURRENT: 2, ROLE_PENDING: 1, ROLE_STALE: 0}.get(role, 1)
    return (priority, _parse_date(meta.get("effective", "")),
            meta.get("effect", _UNKNOWN_EFFECT), meta.get("promulgated", ""), int(is_plain))


def audit(corpus: Path, *, today: str | None = None) -> dict:
    today = today or date.today().isoformat()
    families: dict[str, list[dict]] = {}
    for path in sorted(corpus.rglob("*.md")):
        meta = read_meta(path)
        plain = not HASH_SUFFIX_RE.search(path.name)
        families.setdefault(clean_family_name(path.name), []).append({
            "file": path.name,
            "path": str(path.relative_to(corpus)).replace("\\", "/"),
            "is_plain": plain,
            "size": path.stat().st_size,
            **meta,
            "effective_date": _parse_date(meta.get("effective", "")),
            "role": role_of(meta, today),
        })

    multi = {name: rows for name, rows in families.items() if len(rows) > 1}
    stale_primary: list[dict] = []
    keep_map: dict[str, str] = {}
    pending_map: dict[str, str] = {}
    for name, rows in sorted(multi.items()):
        ordered = sorted(rows, key=lambda row: rank_key(row, row["is_plain"], today), reverse=True)
        # 族内定角色：**最新的"已生效"版本 = 现行版**，其余已生效的都是被取代的旧版；
        # "尚未生效"的版本单独留着（用户口径：涉新规要提示，不能因为不是现行版就藏起来）。
        current_index = next((i for i, row in enumerate(ordered)
                              if row["role"] == ROLE_CURRENT), None)
        if current_index is None:
            # 一族里没有任何已生效版本（例如这部法刚公布、还没施行）⇒ 让最新的那份可检索，
            # 否则整族会被当旧版滤掉、等于把新法藏了；它的"未生效"身份仍会走到答案提示里。
            for i, row in enumerate(ordered):
                if row["role"] == ROLE_PENDING:
                    row["role"] = ROLE_CURRENT
                    current_index = i
                    break
        for i, row in enumerate(ordered):
            if i != current_index and row["role"] == ROLE_CURRENT:
                row["role"] = ROLE_STALE
        keep_row = ordered[current_index] if current_index is not None else ordered[0]
        keep_map[name] = keep_row["file"]
        for row in ordered:
            if row["role"] == ROLE_PENDING and row["file"] != keep_row["file"]:
                pending_map.setdefault(name, row["file"])
        plain_row = next((row for row in rows if row["is_plain"]), None)
        if plain_row is not None and plain_row["file"] != keep_row["file"]:
            stale_primary.append({
                "family": name,
                "current_plain": {"file": plain_row["file"],
                                  "effect": plain_row["effect"],
                                  "promulgated": plain_row["promulgated"],
                                  "effective": plain_row.get("effective", ""),
                                  "role": plain_row.get("role")},
                "suggested_keep": {"file": keep_row["file"],
                                   "effect": keep_row["effect"],
                                   "promulgated": keep_row["promulgated"],
                                   "effective": keep_row.get("effective", ""),
                                   "role": keep_row.get("role")},
            })
    return {
        "corpus": str(corpus).replace("\\", "/"),
        "today": today,
        "files_total": sum(len(rows) for rows in families.values()),
        "families_total": len(families),
        "families_with_versions": len(multi),
        "stale_primary_count": len(stale_primary),
        "stale_primary": stale_primary,
        "keep_map": keep_map,
        "pending_map": pending_map,
        "families": {name: sorted(rows, key=lambda row: rank_key(row, row["is_plain"], today),
                                  reverse=True)
                     for name, rows in sorted(multi.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="语料版本盘点（只读，不删文件）")
    parser.add_argument("--corpus", default="knowledge/lawyer")
    parser.add_argument("--out", default="index/version_inventory.json")
    parser.add_argument("--sample", type=int, default=8, help="摘要里打印几个例子")
    parser.add_argument("--today", default="", help="按哪一天判'是否已施行'（默认今天，便于复现）")
    args = parser.parse_args()

    report = audit(Path(args.corpus), today=args.today or None)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"语料文件 {report['files_total']} 个 / 文件名族 {report['families_total']} 个 "
          f"（判定基准日 {report['today']}）")
    print(f"多版本族 {report['families_with_versions']} 个；"
          f"其中**基名文件不是最新版**的 {report['stale_primary_count']} 个")
    print(f"**尚未生效**（保留但需提示）的族 {len(report['pending_map'])} 个")
    for family, file in list(report["pending_map"].items())[: args.sample]:
        print(f"    [未生效] {family[:46]}… -> {file[:56]}")
    print(f"盘点结果已写：{target}")
    print("\n--- 样例（基名 vs 建议保留）:")
    for item in report["stale_primary"][: args.sample]:
        plain, keep = item["current_plain"], item["suggested_keep"]
        print(f"  【{item['family'][:42]}…】")
        print(f"      基名  {plain['file'][:56]}… 时效性={plain['effect']} "
              f"公布={plain['promulgated']} 施行={plain.get('effective') or '?'} 角色={plain.get('role')}")
        print(f"      建议  {keep['file'][:56]}… 时效性={keep['effect']} "
              f"公布={keep['promulgated']} 施行={keep.get('effective') or '?'} 角色={keep.get('role')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
