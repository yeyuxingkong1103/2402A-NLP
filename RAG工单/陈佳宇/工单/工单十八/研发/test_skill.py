# -*- coding: utf-8 -*-
# test_skill.py 工单18 Skill单元测试
from agent_workflow import document_ingestion_workflow
import json

if __name__ == "__main__":
    test_document_dir = "./test_docs"
    result = document_ingestion_workflow(test_document_dir)
    print("====质检报告====")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    with open("./quality_inspect_report.json","w",encoding="utf‑8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print("\n✅报告输出 quality_inspect_report.json")
