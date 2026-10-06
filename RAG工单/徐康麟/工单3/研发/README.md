# 研发/ —— 可运行代码与数据

> 工单：**人工智能NLP-RAG-PDF文档的表格解析及检索优化**
> 所有代码注释**必须中文**且含工单编号字符串：`人工智能NLP-RAG-PDF文档的表格解析及检索优化`

## 目录约定

| 路径 | 用途 |
| --- | --- |
| `app/core/` | 业务核心：解析、分块、索引、混合检索、生成、引用、配置、日志（**唯一业务实现，UI 共用**） |
| `app/ui/` | `streamlit_app.py`（真 Streamlit 应用，算力云可跑）+ `serve_fallback.py`（纯标准库 `http.server` 备用界面） |
| `app/models/` | 嵌入 / LLM 后端抽象与适配（ollama、openai 兼容、extractive 兜底） |
| `app/prompts/` | 提示词模板（含引用与「不清楚」约束） |
| `app/storage/` | 索引与缓存的读写封装（索引按嵌入模型分目录） |
| `data/raw/` | **语料**：`招股说明书1.pdf`、`招股说明书2.pdf`（已就位并校验哈希，见环境事实 §3）。严禁硬编码文件名，必须 `glob('*.pdf')` 自动发现 |
| `data/processed/` | 解析产物（文本块、表格块、归一化后的 Markdown） |
| `data/index/` | 检索索引（BM25 倒排、向量矩阵） |
| `data/eval/` | 评测集与评测结果 |
| `scripts/` | 数据准备、建索引、评测等命令行脚本 |
| `报告/` | Markdown 研发报告 |

## 统一入口

```powershell
# 工作目录 = E:\gao6gongdan\工单3
pwsh -NoProfile -File run_py.ps1 研发/scripts/你的脚本.py
```

## 硬性纪律

1. 所有函数入口/出口写结构化日志（函数名、输入摘要、输出摘要、耗时、异常堆栈）。
2. 禁止静默失败：`except` 必须 log 后传播或**显式降级**。
3. 答案必须带可回溯的真实引用 `[文件名: 页码]`，无依据统一回「不清楚」。
4. 禁止 pip install（本机断网）；缺失依赖见 `部署/配置/环境事实.md` §2.2 的替代方案。
