# -*- coding: utf-8 -*-
"""
RAG 评估脚本
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

功能:
    1. 加载测试用例结果 (测试结果.json)
    2. 计算 RAG 评估指标 (对比 RAG vs 纯 LLM):
       - 是否含相关关键词 (覆盖度)
       - 检索 Top1 相似度
       - 响应时间达标率
    3. 生成评估报告 (测试报告.md)
"""
import os
import json
import re
from typing import List, Dict

# 评估用的关键词答案字典 (基于招股说明书原文的人工标注参考)
REFERENCE_ANSWERS: Dict[int, Dict] = {
    260: {
        "question": "军用领域收入",
        "keywords": ["6,464.51", "14,414.16", "18,780.67", "4,627.14", "万元"],
        "summary": "2016/2017/2018/2019H1 军方收入分别为 6,464.51 / 14,414.16 / 18,780.67 / 4,627.14 万元",
    },
    95: {
        "question": "参与制定的技术标准",
        "keywords": ["AVS", "标准", "编解码"],
        "summary": "公司参与制定了 AVS 系列音视频编解码国家标准",
    },
    33: {
        "question": "军方收入占主营业务比重",
        "keywords": ["82.10%", "97.31%", "94.84%", "94.34%"],
        "summary": "报告期内军方收入占主营业务收入比重分别为 82.10% / 97.31% / 94.84% / 94.34%",
    },
    34: {
        "question": "电子信息行业上游企业",
        "keywords": ["芯片", "元器件", "上游"],
        "summary": "上游涉及芯片、元器件、结构件等供应商",
    },
    957: {
        "question": "重要供应商领域",
        "keywords": ["国防", "军工", "军队", "指挥控制"],
        "summary": "公司在国防指挥控制领域已成为重要供应商",
    },
    793: {
        "question": "电子信息行业下游",
        "keywords": ["国防", "军队", "军工", "下游"],
        "summary": "下游主要为国防军工/军队用户",
    },
    795: {
        "question": "国家科技进步一等奖工程",
        "keywords": ["国家科技进步", "一等奖"],
        "summary": "参与的工程荣获国家科技进步一等奖",
    },
    543: {
        "question": "注册资本",
        "keywords": ["7,360", "注册资本"],
        "summary": "注册资本 7,360 万元 (发行后总股本 7,360 万股, 每股面值 1 元)",
    },
    531: {
        "question": "法定代表人",
        "keywords": ["法定代表人"],
        "summary": "法定代表人为公司董事长 (具体姓名见招股说明书)",
    },
    207: {
        "question": "补充流动资金占比",
        "keywords": ["补充流动资金", "%"],
        "summary": "募集资金中用于补充流动资金的比例见招股说明书募集资金运用章节",
    },
}


def keyword_coverage(answer: str, keywords: List[str]) -> float:
    """关键词覆盖率: 命中关键词数 / 总关键词数"""
    if not answer or not keywords:
        return 0.0
    hit = sum(1 for k in keywords if k in answer)
    return round(hit / len(keywords), 4)


def evaluate_one(result: Dict) -> Dict:
    """评估单条结果"""
    qid = result.get("id")
    ref = REFERENCE_ANSWERS.get(qid, {})
    keywords = ref.get("keywords", [])
    rag_cov = keyword_coverage(result.get("rag_answer", ""), keywords)
    llm_cov = keyword_coverage(result.get("llm_answer", ""), keywords)
    return {
        "id": qid,
        "question": result.get("question"),
        "reference_summary": ref.get("summary", ""),
        "rag_keyword_coverage": rag_cov,
        "llm_keyword_coverage": llm_cov,
        "rag_better": rag_cov >= llm_cov,
        "response_time": result.get("response_time"),
        "within_time_limit": result.get("within_time_limit"),
        "retrieval_count": result.get("retrieval_count"),
        "top_score": result.get("top_score"),
        "status": result.get("status"),
    }


def evaluate_all(results: List[Dict]) -> Dict:
    """评估全部结果并给出汇总"""
    items = [evaluate_one(r) for r in results]
    ok_items = [i for i in items if i["status"] == "ok"]
    rag_avg_cov = sum(i["rag_keyword_coverage"] for i in ok_items) / len(ok_items) if ok_items else 0
    llm_avg_cov = sum(i["llm_keyword_coverage"] for i in ok_items) / len(ok_items) if ok_items else 0
    rag_better_count = sum(1 for i in ok_items if i["rag_better"])
    in_time_count = sum(1 for i in ok_items if i["within_time_limit"])
    avg_time = sum(i["response_time"] for i in ok_items) / len(ok_items) if ok_items else 0

    return {
        "items": items,
        "summary": {
            "total": len(items),
            "success": len(ok_items),
            "rag_avg_coverage": round(rag_avg_cov, 4),
            "llm_avg_coverage": round(llm_avg_cov, 4),
            "rag_better_count": rag_better_count,
            "in_time_count": in_time_count,
            "avg_response_time": round(avg_time, 4),
        },
    }


def write_report(eval_result: Dict, report_path: str = None):
    """生成 Markdown 测试报告"""
    report_path = report_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "测试报告.md")
    s = eval_result["summary"]
    lines = []
    lines.append("# 测试报告 - 基于 PDF 文档的问答系统\n")
    lines.append("> 工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统\n")
    lines.append("## 一、评估汇总\n")
    lines.append(f"- 用例总数: {s['total']}")
    lines.append(f"- 成功运行: {s['success']}")
    lines.append(f"- 响应达标 (≤3s): {s['in_time_count']}/{s['success']}")
    lines.append(f"- 平均响应时间: {s['avg_response_time']}s")
    lines.append(f"- RAG 平均关键词覆盖率: {s['rag_avg_coverage']}")
    lines.append(f"- 纯 LLM 平均关键词覆盖率: {s['llm_avg_coverage']}")
    lines.append(f"- RAG 优于或等于纯 LLM 的题数: {s['rag_better_count']}\n")
    lines.append("## 二、逐题结果\n")
    lines.append("| ID | 问题 | RAG覆盖率 | LLM覆盖率 | RAG更优 | 响应(s) | 达标 | 检索数 | Top1分 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for it in eval_result["items"]:
        lines.append(
            f"| {it['id']} | {it['question'][:30]}... | "
            f"{it['rag_keyword_coverage']} | {it['llm_keyword_coverage']} | "
            f"{'是' if it['rag_better'] else '否'} | "
            f"{it['response_time']} | {'是' if it['within_time_limit'] else '否'} | "
            f"{it['retrieval_count']} | {it.get('top_score', 0)} |"
        )
    lines.append("\n## 三、结论\n")
    if s["rag_better_count"] >= s["success"] * 0.6:
        lines.append("RAG 模式在多数问题上优于纯 LLM, 验证了检索增强的价值。")
    else:
        lines.append("RAG 模式提升有限, 建议升级向量模型 (见《优化/性能优化说明.md》)。")
    if s["in_time_count"] == s["success"]:
        lines.append("全部用例响应时间达标 (≤3s), 满足工单性能要求。")
    else:
        lines.append(f"{s['success'] - s['in_time_count']} 个用例超时, 需进一步优化。")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"测试报告已生成: {report_path}")
    return report_path


def load_results(path: str = None) -> List[Dict]:
    """加载测试结果"""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "测试结果.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到测试结果: {path}, 请先运行 测试用例.py")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    results = load_results()
    eval_result = evaluate_all(results)
    print(json.dumps(eval_result["summary"], ensure_ascii=False, indent=2))
    write_report(eval_result)
