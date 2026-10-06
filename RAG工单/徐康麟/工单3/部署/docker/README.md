# 部署/docker/ —— 容器化说明

> 工单：**人工智能NLP-RAG-PDF文档的表格解析及检索优化**

## 一、文件

| 文件 | 说明 |
| --- | --- |
| `Dockerfile` | 镜像定义（python:3.11-slim + 核心依赖 + 代码；**不含**语料/索引/模型） |
| `docker-compose.yml` | 编排：`rag-app`（必起）+ 可选 `ollama` / `vllm` / `sglang`（profile 隔离） |
| `entrypoint.sh` | 容器入口：`api|fallback|streamlit|check` 四模式 + 可选等待 LLM 就绪 |
| `.dockerignore` | 构建上下文排除清单（**需复制到仓库根才生效**，原因见文件内注释） |

## 二、构建与运行（**工作目录 = 仓库根 `工单3`**）

```powershell
# ① 让 .dockerignore 生效（Docker 只认构建上下文根目录下的该文件）
Copy-Item 部署/docker/.dockerignore .dockerignore

# ② 构建
docker build -f 部署/docker/Dockerfile -t rag-pdf-qa:1.0 .
Remove-Item .dockerignore

# ③ 运行前自检（依赖/索引/语料/LLM 探测；退出码 0 = 通过）
docker run --rm -v "$PWD/研发/data:/app/研发/data:ro" rag-pdf-qa:1.0 check

# ④ 起服务（连宿主机的 Ollama；容器内 0.0.0.0:8600 映射到本机 8600）
docker run -d --name rag-app -p 8600:8600 `
  -e RAG_LLM__OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -v "$PWD/研发/data:/app/研发/data:ro" -v "$PWD/部署/日志:/app/部署/日志" `
  rag-pdf-qa:1.0 api
```

```bash
# Linux / 算力云（compose 更省事）
docker compose -f 部署/docker/docker-compose.yml up -d rag-app
docker compose -f 部署/docker/docker-compose.yml logs -f rag-app
docker compose -f 部署/docker/docker-compose.yml down
```

## 三、后端接线（可插拔：ollama → openai 兼容 → extractive）

| 场景 | 需要的环境变量 | 说明 |
| --- | --- | --- |
| 本机 Ollama（默认） | `RAG_LLM__OLLAMA_BASE_URL=http://host.docker.internal:11434` | 容器访问宿主机 Ollama |
| 同编排的 Ollama | `--profile local-llm` + `RAG_LLM__OLLAMA_BASE_URL=http://ollama:11434` | 需在容器内 `ollama pull bge-m3` / `qwen2.5:3b`（联网） |
| vLLM（GPU） | `RAG_LLM__BACKEND=openai`、`RAG_LLM__OPENAI_BASE_URL=http://vllm:8000/v1`、`RAG_LLM__OPENAI_MODEL=<model>` | 见 `../脚本/run_vllm.sh` 或 compose `--profile gpu` |
| SGLang（GPU） | 同上，但端口 `:30000/v1` | 见 `../脚本/run_sglang.sh` 或 `--profile gpu-sglang` |
| 无任何 LLM | 不设即可 | 自动降级 `extractive`（仍带引用，不静默失败） |

## 四、诚实声明（**未在本机验证**）

- 本机 Docker CLI 为 29.7.2，但**守护进程在本沙箱不可达**（`docker images` → `permission denied: npipe`），
  且本机断网无法拉取基础镜像 → **本 Dockerfile / compose 未在本机构建或运行过**。
- 本机可离线完成的校验只有：`docker-compose.yml` 的 YAML 语法（PyYAML 解析，见 `../验证状态.md`）。
- 由此产生的推论都**不写**：GPU 型号、镜像体积、容器内首字延迟等，均未实测，禁止编造。
- 算力云上的自验命令（复制即用）：
  ```bash
  docker compose -f 部署/docker/docker-compose.yml config -q   # 编排语法与变量替换检查
  docker build -f 部署/docker/Dockerfile -t rag-pdf-qa:1.0 .
  docker run --rm rag-pdf-qa:1.0 check
  ```
