# -*- coding: utf-8 -*-
"""法规元数据核验回填：6 部留空 status 补官方标注 + 修正 ① 的公布日期（显式、可回滚）。

背景（为什么必须回填）：
本项目爬取的 11 份原始 HTML 里没有任何"时效性/有效性"字段（已逐份 grep 确认），
因此这 6 部的效力状态不能由抽取器产出、也不能规则推定（"有生效日无失效日 ⇒ 有效"
这类一律不算证据）。2026-09-21 用真实 Chromium 访问国家法律法规数据库（flk.npc.gov.cn）
与发布机关官网逐部取得官方时效性标注，证据链见 reports/metadata_verification_6laws.md。

映射口径（已确认）：flk「有效」/ 人社部「是否有效：有效」→ 词表「现行有效」。
revision_note 保留**原始标注文字**（不写只留映射后的值），格式：
    来源：flk 时效性标注＝有效｜<URL>｜抓取 <时间>；佐证：…

同时修正 law_id=76（ver_id=91）的 promulgation_date：2025-08-01（发布会日期）→
2025-07-31（法释〔2025〕12号 公告落款；flk 标注与 data/labor_law_raw/ 原始 HTML 三重印证：
原始 HTML 内载「法释〔2025〕12号」「2025年7月31日」「自2025年9月1日起施行」）。

经验（写入本脚本注释留档）：公布日/施行日这类信息**优先从 data/labor_law_raw/ 的原始
HTML 里取**（零网络、可复核）；外部数据库只用于"时效性标注"这类页面本身没有的字段。

用法：
    python scripts/migrations/backfill_status_6laws.py             # 预览，不写库
    python scripts/migrations/backfill_status_6laws.py --apply     # 实际写库
    python scripts/migrations/backfill_status_6laws.py --rollback  # 按最近一次记录回滚

回滚依据：--apply 时往 reports/ 写逐行变更记录（id / 原值 / 新值），
--rollback 只回退该记录里的行且校验"当前值仍是本脚本写入的值"，
因此**不会误伤**回填之后由抽取器/人工写入的值。

红线：只 UPDATE law_versions 的 status / revision_note / promulgation_date 三列、
只动下表列出的 6 行；不 DROP、不改结构、不动其它表。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 项目根（scripts/migrations → scripts → rag），供导入 scripts/_env 与词表常量
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# 词表常量必须取自 app/db/law_status.py（唯一定义处），脚本里不得另抄中文取值
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import pymysql

from app.db.law_status import EFFECTIVE
from scripts._env import load_project_env, mysql_config

load_project_env(PROJECT_ROOT)

REPORT_DIR = PROJECT_ROOT / "reports"
LOG_PREFIX = "law_status_backfill_6laws_"

# 回填目标：只认 (ver_id, status 必须为 NULL) 的行——保证幂等（重复跑为空操作）。
# note 里的标注文字是证据页原文（未改写），按裁决保留原始标注，不只写映射后的值。
BACKFILL_ROWS = (
    {
        "ver_id": 91,
        "law_id": 76,
        "title": "最高人民法院关于审理劳动争议案件适用法律问题的解释（二）",
        "note": (
            "来源：flk 时效性标注＝有效（公布2025-07-31 施行2025-09-01 司法解释 最高人民法院）"
            "｜https://flk.npc.gov.cn/search?searchContent=劳动争议案件适用法律问题的解释"
            "｜抓取 2026-09-21 21:40:24；"
            "佐证：最高法公告 法释〔2025〕12号 落款2025年7月31日"
            "｜https://www.court.gov.cn/zixun/xiangqing/472691.html；"
            "再证：data/labor_law_raw/f54ec9c0cf12aa1e1a0e9fa711ab288ca7400827777dc620524cab5c6065f021.html"
            " 载「法释〔2025〕12号」「2025年7月31日」「自2025年9月1日起施行」"
        ),
        # ① 的公布日期修正（同批执行，证据同上）
        "promulgation_date": {"from": "2025-08-01", "to": "2025-07-31"},
    },
    {
        "ver_id": 92,
        "law_id": 77,
        "title": "中华人民共和国劳动争议调解仲裁法",
        "note": (
            "来源：flk 时效性标注＝有效（公布2007-12-29 施行2008-05-01 法律 全国人民代表大会常务委员会）"
            "｜https://flk.npc.gov.cn/search?searchContent=劳动争议调解仲裁法"
            "｜抓取 2026-09-21 21:39"
        ),
    },
    {
        "ver_id": 95,
        "law_id": 80,
        "title": "工资支付暂行规定",
        "note": (
            "来源：人社部官网部门规章公开目录 是否有效＝有效（废止时间空；劳部发〔1994〕489号）"
            "｜https://www.mohrss.gov.cn/xxgk2020/fdzdgknr/zcfg/bmgz/index_13.html"
            "｜抓取 2026-09-21 21:47；"
            "佐证：中国政府网国家规章库转载页"
            "｜https://www.gov.cn/zhengce/2022-08/31/content_5711284.htm｜抓取 2026-09-21 21:49"
            "（flk 不收录部门规章，故取发布机关人社部法定公开字段）"
        ),
    },
    {
        "ver_id": 97,
        "law_id": 82,
        "title": "中华人民共和国劳动合同法实施条例",
        "note": (
            "来源：flk 时效性标注＝有效（公布2008-09-18 行政法规 国务院）"
            "｜https://flk.npc.gov.cn/search?searchContent=劳动合同法实施条例"
            "｜抓取 2026-09-21 21:40:55"
        ),
    },
    {
        "ver_id": 98,
        "law_id": 83,
        "title": "女职工劳动保护特别规定",
        "note": (
            "来源：flk 时效性标注＝有效（公布2012-04-28 行政法规 国务院）"
            "｜https://flk.npc.gov.cn/search?searchContent=女职工劳动保护特别规定"
            "｜抓取 2026-09-21 21:41:20"
        ),
    },
    {
        "ver_id": 99,
        "law_id": 84,
        "title": "职工带薪年休假条例",
        "note": (
            "来源：flk 时效性标注＝有效（公布2007-12-14 行政法规 国务院）"
            "｜https://flk.npc.gov.cn/search?searchContent=职工带薪年休假条例"
            "｜抓取 2026-09-21 21:41:58"
        ),
    },
)


def connect():
    """建立 MySQL 连接（pymysql，charset=utf8mb4 才能读写中文字面量）。"""
    return pymysql.connect(**mysql_config(), charset="utf8mb4")


def fetch_distribution(cur) -> list[tuple[str, int]]:
    """status 取值分布（NULL 显示为 <NULL>），按条数倒序。"""
    cur.execute(
        "SELECT IFNULL(status, '<NULL>') AS s, COUNT(*) FROM law_versions "
        "GROUP BY status ORDER BY 2 DESC, 1"
    )
    return list(cur.fetchall())


def print_distribution(cur, title: str) -> None:
    print(f"  {title}：")
    for value, count in fetch_distribution(cur):
        print(f"      {value:<16} {count} 行")


def print_rows(cur, title: str) -> None:
    """打印 6 行目标行的当前值（预览 / 校验用）。"""
    ids = tuple(r["ver_id"] for r in BACKFILL_ROWS)
    cur.execute(
        "SELECT id, law_id, status, promulgation_date FROM law_versions "
        f"WHERE id IN ({', '.join(['%s'] * len(ids))}) ORDER BY id",
        ids,
    )
    print(f"  {title}：")
    for row in cur.fetchall():
        print(
            f"      ver_id={row[0]}  law_id={row[1]}  status={row[2]!r}  prom={row[3]}"
        )


def latest_log_file() -> Path | None:
    logs = sorted(REPORT_DIR.glob(f"{LOG_PREFIX}*.json"))
    return logs[-1] if logs else None


def do_apply() -> int:
    """逐行 UPDATE，rowcount==1 才 commit；任一行不符预期即中止且不写任何行。"""
    conn = connect()
    cur = conn.cursor()
    applied: list[dict] = []
    try:
        print("== apply 预检 ==")
        print_rows(cur, "目标行当前值")
        print_distribution(cur, "迁移前 status 分布")

        # 预检：所有行必须 status IS NULL 且 prom 与脚本记录一致，否则中止
        for row in BACKFILL_ROWS:
            cur.execute(
                "SELECT status, promulgation_date, revision_note FROM law_versions WHERE id=%s",
                (row["ver_id"],),
            )
            hit = cur.fetchone()
            if hit is None:
                print(f"[FATAL] ver_id={row['ver_id']} 不存在，已中止（未写任何行）")
                return 1
            status, prom, note = hit
            if status is not None:
                print(
                    f"[FATAL] ver_id={row['ver_id']} status 已非空（{status!r}），"
                    "按『抽不到不覆盖』口径中止（未写任何行）"
                )
                return 1
            expected_prom = row.get("promulgation_date", {}).get("from")
            if expected_prom and str(prom) != expected_prom:
                print(
                    f"[FATAL] ver_id={row['ver_id']} promulgation_date={prom!r} "
                    f"与脚本记录 {expected_prom!r} 不一致，已中止（未写任何行）"
                )
                return 1
            applied.append(
                {
                    "ver_id": row["ver_id"],
                    "law_id": row["law_id"],
                    "old_status": status,
                    "old_promulgation_date": str(prom) if prom else None,
                    "old_revision_note": note,
                }
            )

        # 逐行写入：rowcount==1 才继续；任何一行不符即整批回退（不 commit）
        print("\n== 逐行写入 ==")
        for row in BACKFILL_ROWS:
            cur.execute(
                "UPDATE law_versions SET status=%s, revision_note=%s "
                "WHERE id=%s AND status IS NULL",
                (EFFECTIVE, row["note"], row["ver_id"]),
            )
            if cur.rowcount != 1:
                print(f"[FATAL] ver_id={row['ver_id']} status rowcount={cur.rowcount} != 1，回退整批")
                conn.rollback()
                return 1
            print(f"      ver_id={row['ver_id']}  status -> {EFFECTIVE}（rowcount=1）")

            fix = row.get("promulgation_date")
            if fix:
                cur.execute(
                    "UPDATE law_versions SET promulgation_date=%s "
                    "WHERE id=%s AND promulgation_date=%s",
                    (fix["to"], row["ver_id"], fix["from"]),
                )
                if cur.rowcount != 1:
                    print(
                        f"[FATAL] ver_id={row['ver_id']} 日期修正 rowcount={cur.rowcount} != 1，回退整批"
                    )
                    conn.rollback()
                    return 1
                print(
                    f"      ver_id={row['ver_id']}  promulgation_date "
                    f"{fix['from']} -> {fix['to']}（rowcount=1）"
                )
        conn.commit()
        print("\n== commit 完成 ==")
        print_distribution(cur, "迁移后 status 分布")
        cur.execute(
            "SELECT id, promulgation_date FROM law_versions WHERE id=91"
        )
        print(f"  验收 b) ver_id=91 promulgation_date = {cur.fetchone()[1]}")

        log_path = REPORT_DIR / f"{LOG_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_path.write_text(
            json.dumps(
                {"applied_at": datetime.now().isoformat(), "rows": applied},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  回滚记录已写：{log_path.name}")
        return 0
    finally:
        conn.close()


def do_rollback() -> int:
    """按最近一次 apply 记录回退；校验当前值仍是本脚本写入的值，不误伤后续写入。"""
    log_path = latest_log_file()
    if log_path is None:
        print("[FATAL] 找不到回滚记录（reports/law_status_backfill_6laws_*.json），无从回滚")
        return 1
    rows = json.loads(log_path.read_text(encoding="utf-8"))["rows"]
    conn = connect()
    cur = conn.cursor()
    try:
        print(f"== 按记录回滚：{log_path.name} ==")
        for row in rows:
            vid = row["ver_id"]
            cur.execute(
                "SELECT status, promulgation_date, revision_note FROM law_versions WHERE id=%s",
                (vid,),
            )
            status, prom, note = cur.fetchone()
            if status != EFFECTIVE or note != _note_of(vid):
                print(
                    f"[SKIP] ver_id={vid} 当前值已被其它来源改写"
                    f"（status={status!r}），不回退该行"
                )
                continue
            cur.execute(
                "UPDATE law_versions SET status=%s, revision_note=%s WHERE id=%s",
                (row["old_status"], row["old_revision_note"], vid),
            )
            assert cur.rowcount == 1, f"ver_id={vid} 回退 rowcount!=1"
            if row["old_promulgation_date"]:
                cur.execute(
                    "UPDATE law_versions SET promulgation_date=%s WHERE id=%s",
                    (row["old_promulgation_date"], vid),
                )
                assert cur.rowcount == 1, f"ver_id={vid} 日期回退 rowcount!=1"
            print(f"      ver_id={vid}  已回退")
        conn.commit()
        print_distribution(cur, "回滚后 status 分布")
        return 0
    finally:
        conn.close()


def _note_of(ver_id: int) -> str:
    """回滚校验用：本脚本会写入的 revision_note 原文。"""
    for row in BACKFILL_ROWS:
        if row["ver_id"] == ver_id:
            return row["note"]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="6 部 status 回填 + ① 公布日期修正")
    parser.add_argument("--apply", action="store_true", help="实际写库")
    parser.add_argument("--rollback", action="store_true", help="按最近一次记录回滚")
    args = parser.parse_args()
    if args.rollback:
        return do_rollback()
    if args.apply:
        return do_apply()
    # 默认：只读预览
    conn = connect()
    cur = conn.cursor()
    try:
        print("== 预览（未写库）==")
        print_rows(cur, "目标行当前值")
        print_distribution(cur, "当前 status 分布")
        print("\n加 --apply 实际写库；--rollback 按最近一次记录回滚")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
