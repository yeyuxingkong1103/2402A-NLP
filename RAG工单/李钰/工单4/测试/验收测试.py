# -*- coding: utf-8 -*-
"""
16 题验收测试脚本
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

2 题图像 + 4 题表格 + 10 题文本 = 16 题
"""
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING)

TEST_QUESTIONS = [
    # ======= 图像类 (新增, 2题) =======
    {"id": 5, "question": "武汉力源信息技术股份有限公司组织结构图中,销售部有几个部门构成,其中大客户销售部有几个销售处构成?",
     "type": "image", "ref_kws": ["销售部", "大客户销售部", "销售处"]},
    {"id": 6, "question": "武汉力源信息技术股份有限公司招股意向书中,从2008年中国IC市场应用结构与增长图中可以看出,增长率最快的是哪个行业?负增长的是哪个行业?",
     "type": "image", "ref_kws": ["增长", "最快", "负增长", "行业"]},
    # ======= 表格类 (工单3, 4题) =======
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少,占发行后总股本的比例是多少?",
     "type": "table", "ref_kws": ["万股", "比例"]},
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目?",
     "type": "table", "ref_kws": ["项目", "募集"]},
    {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁,持股比例和本公司关系是什么?",
     "type": "table", "ref_kws": ["关联方", "持股", "控制"]},
    {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些?",
     "type": "table", "ref_kws": ["关联方"]},
    # ======= 文本类 (旧有, 10题) =======
    {"id": 260, "question": "报告期内,武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少?",
     "type": "text", "ref_kws": ["万元", "收入"]},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准?",
     "type": "text", "ref_kws": ["标准"]},
    {"id": 33, "question": "报告期内,武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少?",
     "type": "text", "ref_kws": ["%"]},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书,电子信息行业的上游涉及哪些企业?",
     "type": "text", "ref_kws": []},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商?",
     "type": "text", "ref_kws": []},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书,电子信息行业的下游主要包括哪些行业?",
     "type": "text", "ref_kws": []},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖?",
     "type": "text", "ref_kws": ["一等奖"]},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少?",
     "type": "text", "ref_kws": ["注册资本"]},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁?",
     "type": "text", "ref_kws": ["法定代表人"]},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金?",
     "type": "text", "ref_kws": ["补充流动资金"]},
]


def keyword_coverage(answer, keywords):
    if not answer or not keywords:
        return 0.0
    hit = sum(1 for k in keywords if k in answer)
    return round(hit / len(keywords), 4)


def run_all():
    import qa_engine_v4
    results = []
    print(f"{'='*60}\nV4 验收测试 ({len(TEST_QUESTIONS)} 题)\n{'='*60}")

    for item in TEST_QUESTIONS:
        qid = item["id"]
        q = item["question"]
        tp = item["type"]
        ref_kws = item["ref_kws"]

        print(f"\n[Q{qid}][{tp}] {q[:40]}...")
        try:
            t0 = time.time()
            r = qa_engine_v4.answer_question(q)
            rt = time.time() - t0
            rag = r.get("rag_answer", "")
            cov = keyword_coverage(rag, ref_kws) if ref_kws else 1.0
            src = r.get("answer_source", "?")
            ok = r.get("within_time_limit", False)
            img_q = r.get("is_image_query", False)

            print(f"  来源: {src} | 覆盖: {cov} | 时间: {rt:.2f}s | 图像: {img_q}")
            print(f"  RAG: {rag[:80]}")

            results.append({
                "id": qid, "type": tp, "question": q,
                "rag_answer": rag, "answer_source": src,
                "keyword_coverage": cov, "response_time": round(rt, 3),
                "within_time_limit": ok,
            })
        except Exception as e:
            print(f"  [异常] {e}")
            results.append({"id": qid, "error": str(e)})

    # 汇总
    ok = [r for r in results if "error" not in r]
    by_type = {}
    for r in ok:
        t = r["type"]
        if t not in by_type:
            by_type[t] = {"count": 0, "cov_sum": 0, "time_sum": 0}
        by_type[t]["count"] += 1
        by_type[t]["cov_sum"] += r["keyword_coverage"]
        by_type[t]["time_sum"] += r["response_time"]

    total_avg = sum(r["keyword_coverage"] for r in ok) / len(ok) if ok else 0
    total_time = sum(r["response_time"] for r in ok) / len(ok) if ok else 0
    in_time = sum(1 for r in ok if r["within_time_limit"])

    print(f"\n{'='*60}")
    print(f"成功: {len(ok)}/{len(results)}")
    print(f"总平均覆盖率: {total_avg} {'达标(≥0.9)' if total_avg >= 0.9 else '未达标'}")
    print(f"平均响应: {total_time:.3f}s | 达标: {in_time}/{len(ok)}")
    for t, v in by_type.items():
        print(f"  [{t}] {v['count']}题, 平均覆盖 {v['cov_sum']/v['count']:.3f}")

    save_json(results)
    return results


def save_json(results, path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "V4验收结果.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果: {path}")


if __name__ == "__main__":
    run_all()
