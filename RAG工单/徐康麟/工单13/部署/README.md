# 工单13 部署与复现说明

**工单编号**：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

---

## 1 运行环境

| 项 | 值 |
| --- | --- |
| 解释器 | `E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（Python 3.11.15，`--system-site-packages` 型） |
| 生成后端 | Ollama `127.0.0.1:11434`，模型 `qwen2.5:3b`（CPU 推理） |
| 嵌入后端 | 同 Ollama，`bge-m3:latest`（1024 维） |
| 索引 | `研发/data/index/`（8847 块，继承统一语料，**不要重建**） |
| 已装依赖 | pymupdf / numpy / jieba / scikit-learn / faiss / torch(CPU) / transformers / sentence-transformers / Pillow / fastapi / uvicorn / pytest / requests / openai / huggingface_hub |
| 缺失（本工单不需要） | pandas / matplotlib / loguru / cv2 / pytesseract / onnxruntime / gradio / streamlit / pdfplumber |

> 索引与语料**不要复制**，直接用绝对路径引用；需要新依赖时用
> `python -m pip install --target 研发/第三方库 --no-deps <包>`，禁止裸装（会写进只读的工单1）。

---

## 2 五条命令（把工单13 从零跑出来）

```powershell
$py = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
Set-Location 'E:\gao6gongdan\工单13'

# 1) 优化前（基线，全部优化开关关闭）
& $py 研发/scripts/分段计时剖析.py --profile baseline  --out 优化/评估结果/优化前.jsonl --summary 优化/评估结果/优化前_汇总.json --tag 优化前

# 2) 优化后（读 部署/配置/优化开关.json）
& $py 研发/scripts/分段计时剖析.py --profile optimized --out 优化/评估结果/优化后.jsonl --summary 优化/评估结果/优化后_汇总.json --tag 优化后

# 3) 生成对比表（csv + md）
& $py 研发/scripts/汇总对比.py

# 4) 生成中文命名截图（3 张）
& $py 测试/测试脚本/生成截图.py

# 5) 单元测试
& $py -m pytest 测试/test_核心链路.py -q --no-header -p no:cacheprovider
```

## 3 可选：定位瓶颈的其他手段

```powershell
& $py 研发/scripts/cProfile剖析.py      # 进程内热点函数
& $py 研发/scripts/采样剖析.py          # sys.setprofile 采样
& $py 研发/scripts/并发压测.py          # 标准库多线程负载
& $py 研发/scripts/阈值标定.py          # 重排预算等阈值等价性 A/B
```

## 4 配置

| 文件 | 作用 |
| --- | --- |
| `部署/配置/优化开关.json` | 11 组优化开关 + 阈值标定说明；**全部置 false = 基线行为** |
| `研发/app/core/perf.py` | 开关的读取与默认值（`DEFAULT_FLAGS`）、环境变量覆盖 `RAG_PERF__<KEY 大写>` |

优先级：**环境变量 > 配置文件 > 代码内置默认**。

## 5 关键运维事实（实测踩坑，务必遵守）

1. **对比实验必须在机器空闲时跑**：生成阶段占端到端 ~86%，CPU 被其它重活抢占时会被放大数倍。
   曾出现与另一工单并发导致 P95 从 1.6 s 恶化到 15 s 的污染测量，该数据已作废。
2. **不要用 `pwsh -File run_py.ps1 ... --out ...` 传参**：PowerShell 会把 `--out`/`--summary`
   误认成自身参数（与 `-OutVariable`/`-Summary` 歧义）并直接报错，参数静默丢失、退出码 1。
   直接用解释器调用最稳；若必须走 `run_py.ps1`，用 `-PyArgs '-m','pytest','测试','-q'` 这种形式。
3. **Ollama 一次只驻留 1 个模型**：跑完用 `POST /api/generate {"model":"<模型>","keep_alive":0}` 卸载；
   本工单流程会同时用到生成与嵌入两个模型，跑完请确认 `/api/ps` 为空。
4. **收工三查**：无遗留 python 进程、端口释放、清理 `__pycache__` / `.pytest_cache`。

## 6 目录导航

```
工单13/
├── 设计/    需求原文、需求分析、瓶颈定位方法论与优化方案、验收标准
├── 研发/    app/（RAG 引擎）、scripts/（5 个剖析脚本 + 汇总对比）、data/（索引只读引用）
├── 测试/    test_核心链路.py、测试脚本/生成截图.py、测试数据/、测试截图/（3 张中文命名）
├── 优化/    瓶颈排查与修复报告.md、性能优化说明.md、评估结果/（前后 jsonl + 汇总 + 对比表）
└── 部署/    本文件、配置/优化开关.json、日志/
```
