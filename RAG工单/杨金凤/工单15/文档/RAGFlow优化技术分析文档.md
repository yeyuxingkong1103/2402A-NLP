# 技术分析文档

---

## 1. RAGFlow 原有检索流程分析

### 1.1 完整检索链路

RAGFlow v0.19.1 从用户提问到返回答案，经历以下步骤：

```text
用户提问
    ↓ [API入口] api/apps/conversation_app.py :: stream()
    ↓ [对话服务] api/db/services/dialog_service.py :: chat() (generator)
    ↓ [问题向量化] rag/llm/embedding_model.py :: encode()
    ↓ [多路召回] rag/nlp/search.py :: Dealer.search()
    ├── 向量检索（ES KNN）
    └── 关键词检索（BM25）
    ↓ [分数融合] vector_similarity_weight * 向量分 + keyword_similarity_weight * 关键词分
    ↓ [重排] rag/llm/rerank_model.py (可选，本次未启用)
    ↓ [Prompt组装] 检索Top-K chunks拼入上下文
    ↓ [LLM生成] 通过 glm-4-flash 生成答案
    ↓ [结构化输出] conversation_service.py :: structure_answer() 包装为 dict
    ↓ SSE流式返回前端
```

### 1.2 关键代码位置

| 模块 | 文件路径 | 说明 |
| --- | --- | --- |
| 对话入口 | api/db/services/dialog_service.py:192 | `chat()` 生成器函数 |
| 检索器 | rag/nlp/search.py | `Dealer.search()` 混合检索 |
| 向量化 | rag/llm/embedding_model.py | bge-large-zh-v1.5 |
| 分块 | rag/app/paper.py | Paper 分块策略 |
| PDF解析 | deepdoc/parser/pdf_parser.py | DeepDoc OCR/版面分析 |
| 结果包装 | api/db/services/conversation_service.py:52 | `structure_answer()` |

### 1.3 输入输出格式约定

`chat()` 是生成器，yield 的对象必须是 dict，包含字段：

```python
{
    "answer": "...",       # 最终答案文本
    "reference": {},       # 引用块
    "prompt": "",          # Prompt（可选）
    "audio_binary": None   # 语音（可选）
}
```

任何返回非 dict 的 yield 都会在 `structure_answer()` 中抛 `TypeError: string indices must be integers`。

---

## 2. 瓶颈诊断

针对工单测试文档 `CN100347506C.pdf`，原有流程存在两个致命故障：

### 故障1：检索无关

问题3~6包含"第11页图3""编号13"等视觉引用，但原有查询理解仅做文本向量化。系统召回大量提及"部件""编号"的文本块，遗漏了真正包含图3的页面。

- **实测现象**：问题3返回"知识库未包含相关信息"。

### 故障2：图文割裂

即使召回到文本块，RAGFlow 默认将图片单独存储在 MinIO，不会把图片内容传给 LLM。LLM 收到的 Prompt 中只有文本，无法推理部件位置关系。

- **实测现象**：问题3首次回答为"我无法直接访问或查看文件中的图片"。

### 故障3：版本兼容性

本次部署过程中发现 RAGFlow v0.19.1 存在多个依赖版本冲突，需要修复才能正常运行（详见测试报告）。

---

## 3. 优化方案设计

### 3.1 总体思路：视觉问题旁路

保留 RAGFlow 原有 RAG 流程处理纯文本问题，新增视觉问题旁路处理图片引用问题：

```text
用户提问
    ↓ [视觉检测] vision_model.py :: is_visual_question()
    ├─ 检测到"第X页"或"图X" → 视觉旁路
    └─ 无视觉引用 → 原有 RAG 流程

    [视觉旁路分支]
    ↓ 从 MinIO 获取 PDF → PyMuPDF 抽指定页为 PNG (300 DPI)
    ↓ 调用 GLM-4V 视觉模型
    ↓ 直接返回答案（yield dict 到 generator）
```

### 3.2 技术要点

| 要点 | 实现方式 |
| --- | --- |
| 视觉引用识别 | 正则匹配 `第(\d+)页` 和 `图(\d+)` |
| 页码定位 | 从问题文本提取页号，默认第1页 |
| 图像抽取 | PyMuPDF `get_pixmap(dpi=300)` |
| 图像传输 | base64 编码 + data URI 格式 |
| 视觉模型 | 智谱 GLM-4V（走已配置的 API Key） |
| 结果注入 | 在 `chat()` 生成器 assert 后插入视觉分支 |

### 3.3 改动文件清单

| 文件 | 类型 | 行数 |
| --- | --- | --- |
| rag/llm/vision_model.py | 新增 | 约 130 行 |
| api/db/services/dialog_service.py | 修改 | 约 18 行 |
| rag/llm/vision_model.py 依赖 PyMuPDF | 新增依赖 | — |

### 3.4 为什么不用纯 RAGFlow 原生多模态

RAGFlow v0.19.1 的原生多模态支持不完整：

- `dialog_service.py` 虽然声明了 `LLMType.IMAGE2TEXT`，但代码里没有把图片内容传给 LLM 的逻辑
- 图片只作为引用元数据存在（`image_id`、`positions`），实际不参与语义推理
- 强行改造原生流程需要大改检索、重排、Prompt组装三大模块，风险高、周期长

视觉旁路方案以 100 行代码实现等效能力，改动最小、可回滚性最强。
