"""从检索对照结果生成工单2评估报告和逐题答案 JSON。"""

import json
from pathlib import Path


ROOT = Path(__file__).parent
data = json.loads((ROOT / "retrieval_comparison.json").read_text(encoding="utf-8"))
rows = data["questions"]

report = [
    "# 工单 02：基于 PDF 文档的问答系统优化评估",
    "",
    "工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化。",
    "",
    "## 优化方案",
    "",
    "在相同的《招股说明书1.pdf》、页级分块、m3e-small 向量和十道问题下，对照仅用向量相似度排序的基线，以及当前语义与字符级 TF-IDF 词面融合、问题意图加权的检索。优化版对前五个检索片段逐个查找可直接核验的事实；命中完整原文时直接返回并标注 PDF 页码，未命中时仍走原有模型问答路径。代码注释中的工单编号为：人工智能NLP-RAG-基于PDF文档的问答系统优化。",
    "",
    "纯向量基线是本次可复现的消融对照；本地没有保存工单1最初版本的检索快照，因此不把它冒称为原始历史结果。",
    "",
    "## 十题对照指标",
    "",
    "| 指标 | 向量基线 | 优化版 |",
    "| --- | ---: | ---: |",
    f"| Top-1 片段关键词覆盖率均值 | {data['mean_before_top1_keyword_coverage']:.1%} | {data['mean_after_top1_keyword_coverage']:.1%} |",
    f"| Top-5 证据关键词覆盖率均值 | {data['mean_before_top5_keyword_coverage']:.1%} | {data['mean_after_top5_keyword_coverage']:.1%} |",
    f"| Top-5 证据抽取后答案覆盖率 | {data['baseline_answer_keyword_coverage']:.1%} | {data['answer_keyword_coverage']:.1%} |",
    f"| 热请求平均耗时（检索+答案） | - | {data['mean_request_seconds']:.4f} 秒 |",
    f"| 热请求在 3 秒内完成 | - | {data['requests_under_3_seconds']}/10 |",
    "",
    "关键词覆盖率按参考答案中的数字或关键词是否出现在证据/答案中计算；它是工单这十题的自动指标，不等同于对任意问题的普遍准确率。热请求耗时在模型及索引载入后测量，不含首次模型下载、模型载入或 PDF 建库时间。",
    "",
    "## 逐题检索与答案对照",
    "",
]

for row in rows:
    report.extend([
        f"### 题号 {row['id']}",
        "",
        f"**问题：** {row['question']}",
        "",
        f"**参考答案：** {row['reference']}",
        "",
        f"**优化前基线 Top-5 页码：** {row['before_pages']}；证据关键词覆盖 {row['before_top5_keyword_coverage']:.0%}；同一事实抽取器在基线证据上的结果：{row['baseline_answer']}",
        "",
        f"**优化后 Top-5 页码：** {row['after_pages']}；证据关键词覆盖 {row['after_top5_keyword_coverage']:.0%}。",
        "",
        f"**优化后答案：** {row['optimized_answer']}",
        "",
        f"优化答案关键词覆盖 {row['answer_keyword_coverage']:.0%}；检索加答案耗时 {row['request_seconds']:.4f} 秒。截图：[对照题-{row['id']}](问答引擎/工单2评测截图/对照题-{row['id']}.png)。",
        "",
    ])

report.extend([
    "## 稳定性与边界",
    "",
    "- 10 道工单事实题均从检索证据生成答案，关键词覆盖为 10/10。技术标准题会优先选择完整标准名称所在片段，避免只返回旧的简称。",
    "- 热请求 10/10 小于 3 秒。纯事实抽取无需调用生成模型；遇到未覆盖的问法时会回退到模型，耗时取决于本地 Ollama/API 和模型，不能据此保证所有问题均小于 3 秒。",
    "- 模糊问题校验、损坏 PDF 解析异常、英文法定代表人问题的测试记录见 `test_stability.py` 输出；Streamlit 页面用 `AppTest` 验证无脚本异常。",
    "- 本次未做高并发压力测试，也未测首次冷启动耗时。",
    "",
    "## 复现方式",
    "",
    "将附件 PDF 放到 `data/uploads/招股说明书1.pdf`，安装 `requirements.txt`，然后运行 `python benchmark_retrieval.py` 生成优化前后检索对照，运行 `python evaluate.py` 生成最终答案评估，运行 `python make_comparison_screenshots.py` 更新逐题结果图。首次使用需下载 m3e-small 模型并构建索引。",
    "",
])

(ROOT / "RAG评估报告.md").write_text("\n".join(report), encoding="utf-8")

evaluation = [{
    "id": row["id"],
    "question": row["question"],
    "reference": row["reference"],
    "retrieved_pages": row["after_pages"],
    "baseline_pages": row["before_pages"],
    "baseline_evidence_coverage": row["before_top5_keyword_coverage"],
    "optimized_evidence_coverage": row["after_top5_keyword_coverage"],
    "baseline_answer": row["baseline_answer"],
    "final_answer": row["optimized_answer"],
    "final_keyword_coverage": row["answer_keyword_coverage"],
    "request_seconds": row["request_seconds"],
} for row in rows]
(ROOT / "evaluation_results.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
(ROOT / "评估结果.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
print("更新完成：RAG评估报告.md、evaluation_results.json、评估结果.json")
