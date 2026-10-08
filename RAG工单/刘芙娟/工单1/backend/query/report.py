"""只读自检：`verify` 子命令的实现。

⚠️ **本模块 MUST NOT 写任何文件。**

`verify` 是排障时**最先跑**的命令。它若会写文件，就不可能在出问题时
安全地使用 —— 而"不敢跑的自检"等于没有自检。

验收里有一条专门钉这个：跑 `verify` 前后，全部 jsonl 的 mtime 与字节数
必须完全不变（quickstart §5.4）。
"""

import argparse
import math

from . import (
    DIM,
    EXIT_DATA,
    EXIT_GATE,
    EXIT_OK,
    F_ERROR,
    F_STATUS,
    F_VECTOR,
    FIELD_ORDER,
    NORM_TOLERANCE,
    STATUS_FAILED,
    STATUS_OK,
)
from . import gate, store

__all__ = ["cmd_verify"]


def cmd_verify(args: argparse.Namespace) -> int:
    """只读自检：门禁 + 格式 + 自洽 + 向量 + 统计。"""

    failures = 0

    # ---- 1. 门禁 ----
    print("【门禁】")
    index_fp = gate.load_index_fingerprint()
    local_fp = gate.local_fingerprint()
    diffs = gate.compare_fingerprints(index_fp, local_fp)
    if diffs:
        failures += 1
        print("  ✗ 编码指纹不一致：")
        for name, expected, actual in diffs:
            print("      %s：索引侧 %r，查询侧 %r" % (name, expected, actual))
    else:
        for name in gate.GATE_FIELDS:
            print("  ✓ %-14s %r" % (name, local_fp.get(name)))
    print("  （基准：%s）" % gate.INDEX_MANIFEST.name)

    # ---- 2~4. 逐行检查 ----
    files = store.list_files()
    print("\n【文件】")
    if not files:
        print("  （没有留存文件）")
        print("\n" + "=" * 50)
        print("自检未通过：门禁%s" % ("有差异" if diffs else "通过，但无数据可查"))
        return EXIT_GATE if diffs else EXIT_OK
    print("  共 %d 个文件：%s" % (len(files), ", ".join(p.name for p in files)))

    total = ok_count = failed_count = 0
    problems: list[str] = []

    print("\n【逐行】")
    for path, lineno, record in store.iter_records(files):
        total += 1
        where = "%s:%d" % (path.name, lineno)

        # 字段齐全且顺序固定
        keys = list(record.keys())
        if keys != list(FIELD_ORDER):
            problems.append("%s 字段顺序/集合不符：%s" % (where, keys))

        status = record.get(F_STATUS)
        vector = record.get(F_VECTOR)
        error = record.get(F_ERROR)

        # 自洽约束（contracts/store.md §3）
        if status == STATUS_OK:
            ok_count += 1
            if not isinstance(vector, list):
                problems.append("%s status=ok 但 vector 不是数组" % where)
            elif len(vector) != DIM:
                problems.append(
                    "%s 维度不符：期望 %d，实际 %d" % (where, DIM, len(vector))
                )
            else:
                norm = math.sqrt(sum(float(x) * float(x) for x in vector))
                if abs(norm - 1.0) > NORM_TOLERANCE:
                    problems.append("%s 范数偏离 1：%.8f" % (where, norm))
            if error is not None:
                problems.append("%s status=ok 但 error 非空" % where)
        elif status == STATUS_FAILED:
            failed_count += 1
            if vector is not None:
                problems.append("%s status=failed 但 vector 非空" % where)
            if not error:
                problems.append("%s status=failed 但 error 为空" % where)
        else:
            problems.append("%s status 取值非法：%r" % (where, status))

    if problems:
        failures += 1
        print("  发现 %d 处问题：" % len(problems))
        for item in problems[:20]:
            print("      ✗ %s" % item)
        if len(problems) > 20:
            print("      …（另有 %d 处）" % (len(problems) - 20))
    else:
        print("  ✓ %d 行全部合法" % total)

    # ---- 5. 统计 ----
    print("\n【统计】")
    print("  总条数 %d | ok %d | failed %d" % (total, ok_count, failed_count))
    if failed_count:
        print("  （failed 是可重跑的工作队列，运行 `embed` 子命令处理）")

    print("\n" + "=" * 50)
    if failures or diffs:
        print("自检未通过。")
        return EXIT_GATE if diffs else EXIT_DATA
    print("自检通过。")
    return EXIT_OK
