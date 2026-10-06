# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-PDF 文档的图像内容解析及检索优化
演示：图像类问题（id=5 组织结构图、id=6 IC市场增长图）+ 回归检查（id=1、2 表格类）
运行：python demo.py
"""
import time
from pathlib import Path

import rag

QUESTIONS = [
    ("5", "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？"),
    ("6", "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"),
    ("1", "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"),
    ("2", "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"),
]


def main():
    text_col, img_col = rag.get_collections()
    lines = ["# 工单04 演示结果：图像内容解析及检索", "",
             "图像解析链路：图表页检测 -> 2.5x 渲染 -> EasyOCR 图内文字识别 -> BLIP 图像描述(qwen译中) -> 图像内容块入库 -> 双路检索", ""]
    for qid, q in QUESTIONS:
        t0 = time.time()
        ans, hits = rag.answer(text_col, img_col, q)
        dt = time.time() - t0
        src = "; ".join(f"{rag.DOC_ALIAS[h['doc']]}p{h['page']}"
                        f"{'+图' if h['kind']=='image' else '+表' if h['kind']=='table' else ''}"
                        f"({h['score']})" for h in hits[:4])
        print(f"[{qid}] {dt:.1f}s  {q[:32]}…\n  -> {ans[:150]}\n  命中: {src}\n")
        lines += [f"## id={qid}", f"**Q**：{q}", "", f"**A**：{ans}", "",
                  f"耗时 {dt:.1f}s；命中：{src}", ""]
    Path("演示结果.md").write_text("\n".join(lines), encoding="utf-8")
    print("已保存 -> 演示结果.md")


if __name__ == "__main__":
    main()
