# -*- coding: utf-8 -*-
"""批次 26：law_versions.status 英文取值 → 中文词表（显式、可回滚）。

背景（为什么必须迁）：
批次 26 往 law_versions.status 写了 5 行**英文**取值（4 行 effective、1 行 not_applicable），
来源是当时列注释里的英文词表（effective / amended / repealed / superseded）。
但抽取器产出的是**中文**，三个消费点（retrieval/context_builder.resolve_current_status、
db/vector_index_service 写 Milvus 的 is_current、retrieval/keyword_search._resolve_current）
也只按中文逐字比较——**英文取值不会被任何判断命中**。

后果是静默的：假设某行写成 repealed，它本该判"已失效"，
却会落进"未标注失效"分支、被判"现行有效"，检索时与有效法规一样返回。
现已把词表收敛到 app/db/law_status.py 单一来源，本脚本把这 5 行历史值迁回中文。

映射（不做任何规则推定；无法映射的取值一律中止，交人工裁决）：
    effective      → 现行有效    （页面标注现行有效 / 人工核对确认有效）
    amended        → 现行有效    （批次 26 裁决 Q1：amended = 现行文本曾经历修正 → 有效）
    repealed       → 已废止      （字面直译，无歧义）
    not_applicable → 不适用      （案例材料等非规范性文件）
    superseded     → 无对应取值  → 中止报错，不猜

用法：
    python scripts/migrations/migrate_law_status_vocabulary.py             # 预览，不写库
    python scripts/migrations/migrate_law_status_vocabulary.py --apply     # 实际写库
    python scripts/migrations/migrate_law_status_vocabulary.py --rollback  # 按最近一次记录回滚

回滚依据：--apply 时会往 reports/ 写一份逐行变更记录（id / 原值 / 新值），
--rollback 只回退该记录里的行且校验"当前值仍是迁移后的值"，因此**不会误伤**
迁移之后由抽取器正常写入的中文值。

红线：只 UPDATE law_versions.status 一列、只动映射命中的行；不 DROP、不改结构、不动其它表。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 项目根（scripts/migrations → scripts → rag），供下面导入 scripts/_env 使用
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# backend/ 也加入路径：词表常量必须取自 app/db/law_status.py（唯一定义处），
# 脚本里不得另抄一份中文取值
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import pymysql

from app.db.law_status import EFFECTIVE, NOT_APPLICABLE, REPEALED
from scripts._env import load_project_env, mysql_config

# 连接参数从 .env 读（顺带剔除 http(s)_proxy）；源码里不出现口令字面量
load_project_env(PROJECT_ROOT)

# 回滚记录的落盘目录与文件名前缀（放 reports/：项目约定报告不进 backend/）
REPORT_DIR = PROJECT_ROOT / "reports"
LOG_PREFIX = "law_status_vocabulary_migration_"

# 历史英文取值 → 中文取值。已列出的都是有裁决/字面依据的映射，
# 没把握的一律不进这张表（否则就是"规则推定"式回填）。
LEGACY_VALUE_MAPPING = {
    "effective": EFFECTIVE,
    "amended": EFFECTIVE,
    "repealed": REPEALED,
    "not_applicable": NOT_APPLICABLE,
}

# 已知英文词表里**无法映射**到新词表的取值：遇到即中止，交人工裁决。
# superseded = 被取代但无废止字样；新词表（现行有效/已废止/已失效/不适用）里
# 没有"已被替代"这一档，替它选任何一个都是替业务做判断。
UNMAPPABLE_LEGACY_VALUES = ("superseded",)


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
    """打印一份取值分布（迁移前后各调一次，便于肉眼对照）。"""
    print(f"  {title}：")
    for value, count in fetch_distribution(cur):
        print(f"      {value:<16} {count} 行")


def fetch_rows_in_scope(cur) -> list[tuple[int, str]]:
    """取出所有取值落在英文词表里的行（id, status），按 id 升序。

    范围只认"英文词表"这一个集合：中文取值、NULL 一律不动，
    保证脚本对当前库**幂等**（重复跑第二次为空操作）。
    """
    known = tuple(LEGACY_VALUE_MAPPING) + UNMAPPABLE_LEGACY_VALUES
    placeholders = ", ".join(["%s"] * len(known))
    cur.execute(
        f"SELECT id, status FROM law_versions WHERE status IN ({placeholders}) ORDER BY id",
        known,
    )
    return [(int(row[0]), str(row[1])) for row in cur.fetchall()]


def guard_unmappable(rows: list[tuple[int, str]]) -> None:
    """发现有无法映射的取值就中止（不写任何行）。"""
    bad = [(row_id, value) for row_id, value in rows if value in UNMAPPABLE_LEGACY_VALUES]
    if bad:
        print("\n[FATAL] 存在无法映射到新词表的英文取值，已中止（未写任何行）：")
        for row_id, value in bad:
            print(f"      id={row_id}  status={value!r}")
        print(
            "        新词表（现行有效/已废止/已失效/不适用）里没有「已被替代」这一档，\n"
            "        替它选任何一个都是替业务做判断。请人工确认后手工 UPDATE，"
            "或先把该取值补进 LEGACY_VALUE_MAPPING 并写明裁决依据。"
        )
        raise SystemExit(2)


def apply_migration(cur, conn) -> int:
    """执行迁移：逐行 UPDATE 并核对 rowcount==1，最后写回滚记录。"""
    rows = fetch_rows_in_scope(cur)
    if not rows:
        print("  [SKIP] 库里没有任何英文取值，无需迁移（本脚本幂等）")
        return 0

    guard_unmappable(rows)

    print(f"\n  将迁移 {len(rows)} 行（逐行 UPDATE，逐行核对 rowcount）：")
    changes: list[dict] = []
    for row_id, old_value in rows:
        new_value = LEGACY_VALUE_MAPPING[old_value]
        # WHERE 带上原值：万一并发改了同一行，rowcount 会变 0，当场暴露而不是静默覆盖
        cur.execute(
            "UPDATE law_versions SET status=%s WHERE id=%s AND status=%s",
            (new_value, row_id, old_value),
        )
        if cur.rowcount != 1:
            conn.rollback()
            raise SystemExit(f"[FATAL] id={row_id} 期望影响 1 行，实际 {cur.rowcount} 行，已回滚")
        print(f"      id={row_id:<4} {old_value!r} → {new_value!r}    rowcount=1 ✅")
        changes.append({"id": row_id, "from": old_value, "to": new_value})

    # 回滚记录：必须在 commit 之前写好并落盘，否则 rollback 无依据
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORT_DIR / f"{LOG_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(
        json.dumps(
            {
                "migrated_at": datetime.now().isoformat(timespec="seconds"),
                "table": "law_versions",
                "column": "status",
                "changes": changes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    conn.commit()
    print(f"\n  [OK] 已提交；回滚记录：{log_path.relative_to(PROJECT_ROOT)}")
    return len(changes)


def latest_log_path() -> Path:
    """找最近一次的回滚记录（没有就报错，不猜）。"""
    logs = sorted(REPORT_DIR.glob(f"{LOG_PREFIX}*.json"))
    if not logs:
        raise SystemExit(
            f"[FATAL] 未找到回滚记录（{REPORT_DIR}/{LOG_PREFIX}*.json）。\n"
            "        --rollback 只回退本脚本 --apply 记录过的行，不用猜测条件反向 UPDATE。"
        )
    return logs[-1]


def rollback(cur, conn, log_path: Path) -> int:
    """按记录回滚：先全量预检（当前值必须仍是迁移后的值），再统一执行。"""
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    changes = payload.get("changes", [])
    print(f"  回滚记录：{log_path.name}（{len(changes)} 行，迁移于 {payload.get('migrated_at')}）")

    # 预检：任一行当前值不等于迁移后的值 → 中止（保证全有或全无，不留半截状态）
    for item in changes:
        cur.execute("SELECT status FROM law_versions WHERE id=%s", (item["id"],))
        row = cur.fetchone()
        current = None if row is None else row[0]
        if current != item["to"]:
            raise SystemExit(
                f"[FATAL] id={item['id']} 当前值 {current!r} ≠ 记录的新值 {item['to']!r}，"
                "已中止（该行可能已被重导入或手工改过，请人工确认）"
            )

    for item in changes:
        cur.execute(
            "UPDATE law_versions SET status=%s WHERE id=%s AND status=%s",
            (item["from"], item["id"], item["to"]),
        )
        if cur.rowcount != 1:
            conn.rollback()
            raise SystemExit(f"[FATAL] 回滚 id={item['id']} 期望 1 行，实际 {cur.rowcount} 行，已回滚")
        print(f"      id={item['id']:<4} {item['to']!r} → {item['from']!r}    rowcount=1 ✅")
    conn.commit()
    print(f"\n  [OK] 已回滚 {len(changes)} 行")
    return len(changes)


def main() -> None:
    parser = argparse.ArgumentParser(description="law_versions.status 英文取值 → 中文词表")
    parser.add_argument("--apply", action="store_true", help="实际写库（缺省只预览）")
    parser.add_argument("--rollback", action="store_true", help="按最近一次记录回滚")
    args = parser.parse_args()

    conn = connect()
    cur = conn.cursor()
    print("=" * 100)
    print(f"law_versions.status {'回滚（中文 → 英文）' if args.rollback else '迁移（英文 → 中文）'}")
    print("=" * 100)
    print_distribution(cur, "迁移前 status 分布")

    print()
    if args.rollback:
        rollback(cur, conn, latest_log_path())
    elif args.apply:
        apply_migration(cur, conn)
    else:
        rows = fetch_rows_in_scope(cur)
        guard_unmappable(rows)
        if not rows:
            print("  [DRY-RUN] 没有需要迁移的行（本脚本幂等）")
        else:
            print(f"  [DRY-RUN] 将迁移 {len(rows)} 行（未写库，加 --apply 才执行）：")
            for row_id, old_value in rows:
                print(f"      id={row_id:<4} {old_value!r} → {LEGACY_VALUE_MAPPING[old_value]!r}")

    print()
    print_distribution(cur, "迁移后 status 分布")
    conn.close()
    print("\nLAW_STATUS_VOCABULARY_MIGRATION_DONE")


if __name__ == "__main__":
    main()
