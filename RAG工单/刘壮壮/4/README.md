# 工单 04 — PDF 文档的图像内容解析及检索优化

> 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

在工单 01/02/03 基础上，新增《招股说明书2.pdf》的图像内容解析，针对「组织结构图」「IC 市场应用结构与增长图」等图像类问题实现检索，使用多模态模型（多模态大模型 / CLIP）做图像语义解析。

## 实现思路

1. 用 pymupdf 提取 PDF 中的图像（带页码与所在页文本）
2. 图像语义解析：
   - **多模态大模型**（默认）：OpenAI 兼容视觉接口（gpt-4o / qwen-vl / glm-4v）生成自然语言描述
   - **CLIP**（可选）：用 CLIP 将图像与文本映射到同一向量空间，实现图文互检索
3. 图像描述块、正文块、表格块一起进入向量库与 BM25 索引

## 文件

```
4/
├── main.py          # 主程序：解析 + 图像语义解析 + 索引 + 问答
├── image_parser.py  # 图像提取 / 多模态描述 / CLIP 封装
├── table_parser.py  # 表格解析（沿用工单 03）
├── questions.py     # 图像 2 问 + 力源 4 问 + 兴图 10 问
├── rag_core.py
└── README.md
```

## 运行

```bash
pip install -r requirements.txt
cd 4
# 配置多模态大模型（.env 中设置 LLM_API_BASE/LLM_API_KEY/LLM_MODEL 为视觉模型）
python main.py
python main.py --query "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成？"
```

单独查看图像提取结果：

```bash
python image_parser.py "../附件/招股说明书2.pdf"
```

## 说明

- 未配置多模态大模型时，图像解析自动回退为「图像所在页正文」作为近似描述（文字型图表仍可命中）。
- 需真实图像语义理解时，请在 `.env` 配置视觉模型（如 `qwen-vl-plus`、`glm-4v`、`gpt-4o-mini`）。
- CLIP 实现位于 `image_parser.py` 的 `CLIPImageParser`（需 `transformers` + `torch`）。

## 验收要点

- 图像问题 id 5、6 检索结果准确
- 多模态模型实现图像语义解析
- 问答准确率 ≥ 90%，响应时间 ≤ 3 秒
