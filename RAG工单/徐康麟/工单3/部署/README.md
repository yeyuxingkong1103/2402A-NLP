# 部署/ —— 配置、脚本、日志与容器化

> 工单：**人工智能NLP-RAG-PDF文档的表格解析及检索优化**

## 目录约定

| 路径 | 用途 |
| --- | --- |
| `配置/` | `环境事实.md`（归档正本，实测数据）、`config.example.env`（全部 `RAG_*` 键 + 默认值 + 降级说明）、`requirements.txt`（**本机基线**：17 个精确版本，逐一与实测一致）、`requirements-cloud.txt`（**算力云额外依赖**，含 CUDA torch 覆盖步骤）、`verify_env_report.json`（T0 机器可读校验报告） |
| `脚本/` | 启动/评测/自检脚本（见下表），全部中文注释并含工单编号 |
| `日志/` | 运行日志与留痕：`app.log` / `error.log` / `rag_trace.jsonl`（产品日志）、`deploy.log`（部署脚本日志）、`eval/`（部署复跑的评测产物）、`verify_logs_report.{json,md}`、`docker_compose_config.txt` |
| `docker/` | `Dockerfile`、`docker-compose.yml`、`entrypoint.sh`、`.dockerignore`、`README.md` |
| `验证状态.md` | **实测状态总表**：哪些本机跑通、哪些只做静态校验、哪些未验证（含原因与云端自验命令） |

## 脚本清单

| 脚本 | 作用 | 本机实测 |
| --- | --- | --- |
| `脚本/run_app.ps1` | 启动入口：预检（真实 import 依赖/索引/语料/LLM 探测）→ 端口检查 → 启动 → 健康检查 → 运维信息；支持 `-Mode check|fallback|api|streamlit`、`-Background`、`-Stop`、`-EnvFile`、`-AllowDegraded` | ✅ check / api / fallback / 失败路径全部实测 |
| `脚本/run_app.sh` | 同上（Linux / 容器 / 算力云；`--mode/--background/--stop/--env-file/--allow-degraded`） | ⚠️ **未执行**（沙箱禁命名管道、WSL 拒绝访问） |
| `脚本/evaluate.ps1` | 一键评测：基座校验 →（可选重建索引）→ 14 题召回（全库/按文件过滤）→ 14 题答案（引用/首字）→ pytest 离线；产物写 `日志/eval/`，**不覆盖** `优化/评估结果/` | ✅ `-Level smoke` 与 `-Level full` 全绿 |
| `脚本/run_vllm.sh` | 算力云 GPU 起 vLLM 的 OpenAI 兼容服务（`--check/--background/--stop`） | ⚠️ 未执行（无 GPU、断网装不上 vllm） |
| `脚本/run_sglang.sh` | 同上，换 SGLang（默认端口 30000） | ⚠️ 未执行 |
| `脚本/verify_config_env.py` | T0 基座校验（解释器/依赖/PDF/Ollama 四段实测，纯标准库） | ✅ ✅11 ❌0 |
| `脚本/verify_logs.py` | 日志契约自检：公共字段/入口出口配对/耗时/堆栈/事件路由/坏行 + 写失败降级探针 | ✅ 退出码 0（`passed_with_findings`） |

## 快速开始（工作目录 = `E:\gao6gongdan\工单3`）

```powershell
# 1) 部署前自检（解释器 / 依赖真实 import / 索引 / 语料 / Ollama 探测）
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode check

# 2) 本机唯一可跑界面（纯标准库 http.server，无需 streamlit）
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode fallback -Port 8600 -Background
#    停止：pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Stop -Port 8600

# 3) FastAPI 服务（等价于 run_py.ps1 -m uvicorn app.main:app --app-dir 研发 --host 127.0.0.1 --port 8600）
pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode api -Port 8600 -Background

# 4) 一键评测 + 日志自检
pwsh -NoProfile -File 部署/脚本/evaluate.ps1 -Level full -KnownRedDeselect '测试/离线/test_t7_offline_judge.py::test_mandatory_negative_n1_partial_answer_must_be_wrong'
pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_logs.py --probe-degradation
```

- 默认解释器：`E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（Python 3.11.15）；覆盖方式 `$env:RAG_SCHEDULER_PYTHON`
- 环境变量样例：`部署/配置/config.example.env`（用 `-EnvFile` 注入子进程；**不含密钥**）
- **预热是硬要求**：`app/main.py` 的 startup、`serve_fallback.main()`、`streamlit_app._engine()` 三处均真实调用 `QAEngine.warmup()`
  （jieba 冷启动约 1 s、嵌入与后端探测；`serve_fallback.py --no-warmup` 仅调试用）
- **禁止**在本机 `pip install`（断网会挂死）；缺失依赖与降级路径见 `配置/环境事实.md` §2.2 与 `验证状态.md` §5

## 事实基线

环境与语料的一切结论以 `配置/环境事实.md`（归档正本）为准：
唯一可用解释器、缺失依赖清单与替代方案、两份 PDF 的哈希/页数、Ollama 模型与实测延迟、
优化前基线（准确率 0.50 / 首字 P95 57.3 s）。**不得**依据过时的提示词描述另做假设。

⚠️ 部署文档**不声称**本机可跑 `streamlit` / `chromadb`：本机两者都缺失且断网无法安装；
本机演示统一走 `-Mode fallback`；Streamlit 与容器路径只在算力云可用（**未在本机验证**，见 `验证状态.md` §4）。
