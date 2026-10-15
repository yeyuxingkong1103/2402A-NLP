# 工单10：金融问答系统Docker部署

按设计、研发、测试、优化、部署五部分保存。

策略hybrid：Hit@5=70.0%，MRR@5=0.620，事实关键词覆盖代理=85.0%。
策略graph：Hit@5=70.0%，MRR@5=0.620，事实关键词覆盖代理=85.0%。
策略graph_optimized：Hit@5=70.0%，MRR@5=0.620，事实关键词覆盖代理=85.0%。
真实生成10题，完整稳态耗时794.15-1862.34ms；当前样本3秒目标通过。不包含模型冷启动。

一键启动：双击一键启动.cmd，打开http://127.0.0.1:8774 。

所有源PDF、嵌入模型、索引、图谱、图像、测试集与日志均位于本工单目录。本项目运行时不访问其他工单文件。LLM调用本机已有Ollama服务，其环境与模型名称单独记录；主目录无公共数据。

各项是否满足验收阈值以测试/实测结果与专项记录为准。未通过不冒称通过。

专项结果 容器验收记录.json
{
  "container": "rag-workorder10",
  "image": "rag-workorder10:local",
  "network": "host",
  "data_mount": "本工单研发/data:/app/data",
  "health": {
    "ok": true,
    "rebuilding": false,
    "error": null,
    "llm_configured": true,
    "manifest": {
      "documents": [
        {
          "name": "2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf",
          "sha256": "6c4585390b72124620e5a934367417a30ff30d7b7fd1436ed0c5f90c13e19730",
          "pages": 265,
          "tables": 12,
          "table_scope": "前30页含财务指标表格；正文全部页面",
          "empty_pages": []
        },
        {
          "name": "2020-02-21__中国平安保险集团股份有限公司__601318__中国平安__2019年__年度报告.pdf",
          "sha256": "304d8cfc4dd3600807ff040a19d082fb958afcb5fdb0f437e7b4f39d5f3e6ab3",
          "pages": 342,
          "tables": 0,
          "table_scope": "前30页含财务指标表格；正文全部页面",
          "empty_pages": [
            342
          ]
        },
        {
          "name": "2020-03-21__招商银行股份有限公司__600036__招商银行__2019年__年度报告.pdf",
          "sha256": "0593e63dece4248aa062797885c84fb31e6621e0b9025850633237b3e8cf7baf",
          "pages": 351,
          "tables": 13,
          "table_scope": "前30页含财务指标表格；正文全部页面",
          "empty_pages": []
        },
        {
          "name": "2020-03-26__中国邮政储蓄银行股份有限公司__601658__邮储银行__2019年__年度报告.pdf",
          "sha256": "dbb873735633943668bec493b071c599d72d7101c2bbd1e33223c20ca06d4add",
          "pages": 421,
          "tables": 11,
          "table_scope": "前30页含财务指标表格；正文全部页面",
          "empty_pages": []
        },
        {
          "name": "2021-03-19__中信证券股份有限公司__600030__中信证券__2020年__年度报告.pdf",
          "sha256": "69b60791628ca510d98c7405cd067485bfaaef91ef1c53f8d179e862074d3731",
          "pages": 370,
          "tables": 9,
          "table_scope": "前30页含财务指标表格；正文全部页面",
          "empty_pages": []
        },
        {
          "name": "2021-03-26__中国人寿保险股份有限公司__601628__中国人寿__2020年__年度报告.pdf",
          "sha256": "cd10e9a60065adf8c81f75cd3519143eaacf8a619a16a4e31908376d01591912",
          "pages": 235,
          "tables": 5,
          "table_scope": "前30页含财务指标表格；正文全部页面",
          "empty_pages": []
        },
        {
          "name": "2022-03-28__中国太平洋保险集团股份有限公司__601601__中国太保__2021年__年度报告.pdf",
          "sha256": "27897d840144c77dd3be76b23622c1cfffefb1c4848b895e29b2c4e02bbd550d",
          "pages": 264,
          "tables": 0,
          "table_scope": "前30页含财务指标表格；正文全部页面",
          "empty_pages": [
            135
          ]
        },
        {
          "name": "2022-03-28__招商证券股份有限公司__600999__招商证券__2021年__年度报告.pdf",
          "sha256": "db082915c6e01c47797fecafce13a77c2f6116c59bf05de951db545c038029fb",
          "pages": 278,
          "tables": 11,
          "table_scope": "前30页含财务指标表格；正文全部页面",
          "empty_pages": []
        },
        {
          "name": "2022-03-31__国泰君安证券股份有限公司__601211__国泰君安__2021年__年度报告.pdf",
          "sha256": "270dd50b295589493a44f2ea3460cfa1f952013b20d78d2967a717a1b45a86dd",
          "pages": 286,
          "tables": 13,
          "table_scope": "前30页含财务指标表格；正文全部页面",
          "empty_pages": []
        }
      ],
      "chunks": 6085,
      "embedding": "BAAI/bge-small-zh-v1.5",
      "embedding_status": "本地BGE",
      "dimension": 512,
      "table_scope": "财务摘要页表格，全文全页",
      "process_note": "第一次无限制find_tables在复杂年报页耗时过长，被取消；改为有界表格范围后重建"
    },
    "active_backend": "TF-IDF",
    "embedding_status": "未找到本地 BGE 模型，使用 TF-IDF 稀疏向量"
  },
  "answer": {
    "question": "平安银行2019年营业收入是多少？",
    "answer": "平安银行2019年营业收入为1,379.58亿元。\n\n[PDF第5页] [PDF第135页]",
    "mode": "rag",
    "generation": "llm",
    "query": {
      "intent": "财务数值",
      "language": "zh",
      "needs_clarification": false,
      "subquestions": [
        "平安银行2019年营业收入是多少？"
      ],
      "retrieval_query": "平安银行2019年营业收入？ 收入 销售额",
      "concepts": [
        "收入 销售额"
      ],
      "period_note": "报告期按来源文件定义，不使用当前年份推断",
      "session_id": "browser"
    },
    "sources": [
      {
        "id": "6c4585390b72-p1-c0",
        "document": "2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf",
        "sha256": "6c4585390b72124620e5a934367417a30ff30d7b7fd1436ed0c5f90c13e19730",
        "pdf_page": 1,
        "printed_page": "",
        "kind": "text",
        "text": " \n \n \n \n \n \n \n \n \n \n \n \n \n \n平安银行股份有限公司 \n \n2019 年年度报告 \n \n",
        "score": 0.05632118508219719,
        "lexical_score": 0.18773727118968964
      },
      {
        "id": "6c4585390b72-p5-c11",
        "document": "2020-02-1

## 验收范围说明
功能、材料和短时回归已交付；仍有部分指定问题完整耗时超过3秒。检索命中和关键词指标不是90%问答准确率或95%全事实召回率的证明。生产高并发、24小时稳定性及真人语音未全部验证；全部原始结果见本项目测试目录。
