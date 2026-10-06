# -*- coding: utf-8 -*-
"""
14 题验收测试脚本
工单编号: 人工智能 NLP-RAG-PDF 文档的表格解析及检索优化

包含: 4 题新增 (武汉力源, 表格密集) + 10 题旧有 (武汉兴图新科)
"""
import os
import sys
import json
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

TEST_QUESTIONS = [
    # ============ 武汉力源 (新增, 表格密集) ============
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少, 占发行后总股本的比例是多少?",
     "ref": {"keywords": ["2000", "万股", "25.00%", "8000", "总股本"]}},
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目?",
     "ref": {"keywords": ["ADC/DAC", "接口", "射频", "补充流动资金", "项目"]}},
    {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁, 持股比例和本公司关系是什么?",
     "ref": {"keywords": ["武汉力源科技", "35", "控股股东", "赵彤", "赵刚"]}},
    {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些?",
     "ref": {"keywords": ["北京力源华创", "深圳力源微电子", "力源技术研究", "力源半导体"]}},
    # ============ 武汉兴图新科 (旧有) ============
    {"id": 260, "question": "报告期内, 武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少?",
     "ref": {"keywords": ["6,464.51", "14,414.16", "18,780.67", "4,627.14"]}},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准?",
     "ref": {"keywords": ["AVS", "标准", "编解码"]}},
    {"id": 33, "question": "报告期内, 武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少?",
     "ref": {"keywords": ["82.10%", "97.31%", "94.84%", "94.34%"]}},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书, 电子信息行业的上游涉及哪些企业?",
     "ref": {"keywords": ["芯片", "元器件", "上游"]}},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商?",
     "ref": {"keywords": ["国防", "军工", "军队"]}},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书, 电子信息行业的下游主要包括哪些行业?",
     "ref": {"keywords": ["国防", "军队", "下游"]}},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖?",
     "ref": {"keywords": ["国家科技进步", "一等奖"]}},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少?",
     "ref": {"keywords": ["7,360", "注册资本"]}},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁?",
     "ref": {"keywords": ["法定代表人"]}},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金?",
     "ref": {"keywords": ["补充流动资金"]}},
]


def keyword_coverage(answer, keywords):
    if not answer or not keywords:
        return 0.0
    hit = sum(1 for k in keywords if k in answer)
    return round(hit / len(keywords), 4)


def run_all():
    import qa_engine_v3
    results = []
    print(f"{'='*60}")
    print(f"V3 验收测试 ({len(TEST_QUESTIONS)} 题)")
    print(f"{'='*60}")

    for item in TEST_QUESTIONS:
        qid = item["id"]
        q = item["question"]
        ref_kws = item["ref"]["keywords"]

        print(f"\n[Q{qid}] {q[:50]}...")
        try:
            r = qa_engine_v3.answer_question(q)
            rag = r.get("rag_answer", "")
            llm = r.get("llm_answer", "")
            cov = keyword_coverage(rag, ref_kws)
            table_q = r.get("is_table_query", False)
            time_ok = r.get("within_time_limit", False)
            rt = r.get("response_time", 99)

            print(f"  表格查询: {table_q} | 覆盖: {cov} | 时间: {rt}s {'OK' if time_ok else 'OVER'}")
            print(f"  RAG: {rag[:100]}")

            results.append({
                "id": qid, "question": q,
                "rag_answer": rag, "llm_answer": llm,
                "keyword_coverage": cov,
                "is_table_query": table_q,
                "response_time": rt,
                "within_time_limit": time_ok,
            })
        except Exception as e:
            print(f"  [异常] {e}")
            results.append({"id": qid, "question": q, "error": str(e)})

    # 汇总
    ok = [r for r in results if "error" not in r]
    avg_cov = sum(r["keyword_coverage"] for r in ok) / len(ok) if ok else 0
    avg_time = sum(r["response_time"] for r in ok) / len(ok) if ok else 0
    in_time = sum(1 for r in ok if r["within_time_limit"])
    table_qs = [r for r in ok if r["is_table_query"]]
    table_cov = sum(r["keyword_coverage"] for r in table_qs) / len(table_qs) if table_qs else 0

    print(f"\n{'='*60}")
    print(f"汇总: 成功 {len(ok)}/{len(results)}")
    print(f"平均覆盖率: {avg_cov} (≥0.9 达标: {'是' if avg_cov >= 0.9 else '否'})")
    print(f"表格类问题覆盖率: {table_cov} ({len(table_qs)} 题)")
    print(f"平均响应: {avg_time:.3f}s | 达标 {in_time}/{len(ok)}")

    save_json(results)
    return results


def save_json(results, path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "V3验收结果.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {path}")


if __name__ == "__main__":
    run_all()
