# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化任务
模块：RAG vs LightRAG 检索对比
"""

import os
import sys
import json
import asyncio
import time

sys.path.insert(0, ".")
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import lightrag_build as lb
from lightrag_config import LightRAGConfig

cfg = LightRAGConfig()

QUESTIONS = [
    {"id": 5, "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？"},
    {"id": 6, "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"},
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"},
    {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？"},
    {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"},
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]


async def test_lightrag(questions, working_dir="./lightrag_test4"):
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=lb.llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=512, max_token_size=512, func=lb.embedding_func,
        ),
        llm_model_max_async=2,
        embedding_func_max_async=4,
    )
    await rag.initialize_storages()

    results = []
    for q in questions:
        t0 = time.time()
        try:
            answer = await rag.aquery(q["question"], param=QueryParam(mode="hybrid"))
        except Exception as e:
            answer = f"[ERROR] {e}"
        elapsed = time.time() - t0
        results.append({
            "id": q["id"],
            "question": q["question"],
            "lightrag_answer": str(answer)[:500],
            "lightrag_time": round(elapsed, 3),
        })
        print(f"  [{q['id']}] {elapsed:.2f}s")
    return results


async def main():
    print("=" * 70)
    print("RAG vs LightRAG 对比测试")
    print("=" * 70)
    print("\n测试 LightRAG...")
    lr_results = await test_lightrag(QUESTIONS)
    with open("lightrag_results.json", "w", encoding="utf-8") as f:
        json.dump(lr_results, f, ensure_ascii=False, indent=2)
    print("\n✅ LightRAG 结果已保存：lightrag_results.json")


if __name__ == "__main__":
    asyncio.run(main())
