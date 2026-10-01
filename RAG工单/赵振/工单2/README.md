# 工单 02：基于 PDF 文档的问答系统优化

工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化。

本目录是在工单 1 系统上继续优化的独立提交包。主要变化是对向量检索结果融合字符级 TF-IDF 词面分数和问题意图权重，并在前五个证据片段中逐个抽取可核验事实，避免数字和技术名称被小模型改写。未匹配的问法仍走原模型回答。

## Windows 启动

1. 安装 Python 3.10 或更新版本，再运行 `python -m pip install -r requirements.txt`。
2. 把工单附件《招股说明书1.pdf》放到 `data/uploads/招股说明书1.pdf`。
3. 第一次启动时需要联网下载 `moka-ai/m3e-small` 向量模型；模型缓存完成后可离线建库和提问。
4. 双击或在 PowerShell 执行 `powershell -ExecutionPolicy Bypass -File .\启动网页.ps1`。
5. 浏览器打开 `http://127.0.0.1:8501`，在左侧点“解析并建立索引”，完成后输入问题。

工单十题的精确事实可由原文抽取后直接返回；其他问题会调用兼容 OpenAI API 的模型。默认调用本机 Ollama 的 `deepseek-r1:1.5b`，如需模型回退，请先安装并运行 Ollama。

## 工单 2 测试

- `python benchmark_retrieval.py`：同一批十题对比纯向量基线与优化检索。
- `python evaluate.py`：运行优化后的十题问答评估。
- `python test_stability.py`：检查模糊输入、PDF 解析失败处理和英文问题。
- `python test_app_smoke.py`：检查 Streamlit 页面和问答控件能正常启动。
- `python make_comparison_screenshots.py`：从评测 JSON 生成十题对照图与总览图。
- `RAG评估报告.md`：优化方案、前后指标、逐题答案和测试边界。
- `测试记录.md`：测试清单与截图链接。
- `演示脚本.md`、`演示视频.mp4`：演示流程和录屏截图串联视频。

前后对照是十题自动评测，不代表所有问题的普遍准确率；首次模型加载和建库耗时不计入热请求时间。索引、原始 PDF 和本机反馈文件均不提交。
