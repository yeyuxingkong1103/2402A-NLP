# -*- coding: utf-8 -*-
# 工单四 Step 1 验收辅助：扫描 Markdown 中 Mermaid 块的常见语法隐患
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import re
import sys

FILES = [
    "docs/01_图像解析方案.md",
    "docs/02_图像检索优化方案.md",
    "docs/03_图文融合RAG方案.md",
]

issues = 0
for fp in FILES:
    in_block = False
    for ln, line in enumerate(open(fp, encoding="utf-8"), 1):
        s = line.strip()
        if s.startswith("```mermaid"):
            in_block = True
            continue
        if in_block and s.startswith("```"):
            in_block = False
            continue
        if not in_block or not s:
            continue
        # 隐患1：[]节点文本内出现未加引号的半角括号
        for m in re.finditer(r"\[[^\]]*\]", s):
            if "(" in m.group() or ")" in m.group():
                print(f"[{fp}:{ln}] 半角括号需加引号: {s}")
                issues += 1
        # 隐患2：全角括号（mermaid 对其宽容，仅提示）
        if "（" in s or "）" in s:
            print(f"[{fp}:{ln}] 提示: 含全角括号 {s}")
print(f"ISSUES={issues}")
sys.exit(0)
