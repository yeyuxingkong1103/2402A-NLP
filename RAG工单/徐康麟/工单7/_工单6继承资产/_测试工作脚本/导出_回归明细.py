# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 回归明细导出（供测试报告引用真实数字）。

只读 ``测试/测试数据/回归结果.json`` 与各产物文件的指纹，不调用模型。
"""

from __future__ import annotations

import hashlib
import json
import os
import time

FILES = [
    r"测试\测试数据\评估汇总.json",
    r"测试\测试数据\评估结果.jsonl",
    r"测试\测试数据\检索问题分析.md",
    r"测试\测试数据\评估结果.csv",
    r"测试\测试数据\关键词核对结果.json",
    r"测试\测试数据\回归结果.json",
    r"测试\测试数据\回归测试日志.txt",
    r"测试\测试数据\回归_评估框架自检.json",
    r"研发\data\index\index_manifest.json",
]


def fp(path: str) -> str:
    with open(path, "rb") as fh:
        blob = fh.read()
    return "sha256:%s  %d bytes  mtime %s" % (
        hashlib.sha256(blob).hexdigest()[:16], len(blob),
        time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(path))))


def main() -> int:
    with open(r"测试\测试数据\回归结果.json", encoding="utf-8") as fh:
        data = json.load(fh)
    print("回归生成时间:", data["生成时间"])
    print("汇总:", json.dumps(data["汇总"], ensure_ascii=False))
    warm = data.get("预热") or {}
    print("预热:", {k: warm.get(k) for k in ("index_chunks", "elapsed_ms")},
          "| multiturn:", warm.get("multiturn"))
    print("索引:", json.dumps(data.get("索引"), ensure_ascii=False))

    for group in data["分组"]:
        print()
        print("== %s  %s/%s  耗时 %ss" % (group["分组"], group["通过"], group["用例数"],
                                        round(group["耗时ms"] / 1000, 1)))
        for case in group["用例"]:
            obs = case.get("关键实测") or {}
            print("  %-10s %s %8.1fms  字%4d  页%s  覆盖率%s  判对%s  recall@10=%s  %s" % (
                case["用例编号"], "PASS" if case["通过"] else "FAIL", case["耗时ms"],
                case["答案字数"], case["引用页"], obs.get("关键词覆盖率", "-"),
                obs.get("是否判对(覆盖率>=0.6且未拒答)", "-"), obs.get("recall@10", "-"),
                case["生成方式"]))
            print("      引用语料=%s" % [c[:26] for c in case["引用语料"]])
            if case.get("改写策略"):
                print("      策略=%s 焦点=%s 改写=%s" % (
                    case["改写策略"], case["焦点实体"], str(case["改写后问句"])[:56]))
            for item in case.get("设计判据") or []:
                print("      判据[%s] %s => %s" % ("OK" if item["结果"] else "NG",
                                                 item["判据"], str(item["实测"])[:90]))
            for a in case.get("断言") or []:
                if not a["通过"]:
                    print("      失败断言: %s => %s" % (a["名称"], str(a["实测"])[:130]))

    print()
    print("== 产物指纹 ==")
    for path in FILES:
        try:
            print("  %s  %s" % (path, fp(path)))
        except OSError as exc:
            print("  MISS %s (%s)" % (path, exc))
    print()
    shots = sorted((f for f in os.listdir(r"测试\测试截图") if f.endswith(".png")),
                   key=lambda n: os.path.getmtime(os.path.join(r"测试\测试截图", n)))
    print("== 截图清单（%d 张）==" % len(shots))
    for name in shots:
        path = os.path.join("测试\\测试截图", name)
        with open(path, "rb") as fh:
            blob = fh.read()
        print("  %-24s %7d B  sha256:%s  %s" % (
            name, len(blob), hashlib.sha256(blob).hexdigest()[:12],
            time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(path)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
