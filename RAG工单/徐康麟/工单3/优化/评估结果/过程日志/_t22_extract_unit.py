# -*- coding: utf-8 -*-
"""t22 取值边界单测（补丁 6/7 的验收脚本，定稿版落地在过程日志内以便审计复跑）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用法：``run_py.ps1 优化/评估结果/过程日志/_t22_extract_unit.py``

覆盖 captain 的实测反证与本轮回归：取值必须**保留单位**、**不吞下一个编号项**、
**expects_numeric=True 时对非数值取值返回空**、**表格单元格与逐字段同口径**。
"""

import sys

sys.path.insert(0, r"E:\gao6gongdan\工单3\研发")
from app.core.generator import _bound_field_value, _extract_field_value  # noqa: E402

CAP = "hnology Co., Ltd. 3、注册资本：人民币5,000 万元 4、法定代表人：赵马克 5、公司住所：武汉市洪山区珞瑜路424 号"

CASES = [
    ("注册资本（文本框）", CAP, "注册资本", False, "人民币5,000万元"),
    ("注册资本（文本框·数值问法）", CAP, "注册资本", True, "人民币5,000万元"),
    ("法定代表人（文本框）", CAP, "法定代表人", False, "赵马克"),
    ("法定代表人（数值问法·应空）", CAP, "法定代表人", True, ""),
    ("注册资本（表格行·数值问法）", "| 注册资本 | 5,520.00 万元 | 法定代表人 | 程家明 |", "注册资本", True, "5,520.00万元"),
    ("发行股数（表格行·数值问法）", "| 发行股数 | 1,670 万股 |", "发行股数", True, "1,670万股"),
]

failed = []
for label, text, keyword, numeric, expect in CASES:
    value, style = _extract_field_value(text, keyword, expects_numeric=numeric)
    ok = value == expect
    print(f"  {'OK ' if ok else 'FAIL'} {label}: got={value!r}({style}) expect={expect!r}")
    if not ok:
        failed.append(label)
print("边界收敛:", repr(_bound_field_value("人民币5,000 万元 4、法定代表人：赵马克")))
if failed:
    raise SystemExit(f"单测失败 {len(failed)}/{len(CASES)}: {failed}")
print(f"单测通过 {len(CASES)}/{len(CASES)}")
