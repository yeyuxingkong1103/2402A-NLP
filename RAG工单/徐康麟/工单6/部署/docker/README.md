# 部署/docker —— 容器化运行说明（**参考交付；本机未构建**）

> 工单编号：**人工智能NLP-RAG-图像内容解析及检索优化**
> 结论先行：本机 Docker Desktop 在运行，但 **docker CLI 走命名管道、沙箱拒绝**，
> 因此本目录的文件**未执行过 `docker build` / `docker compose up`**，只做过静态校验。
> 本机**原生运行**（Windows + `run_py.ps1`）是主交付路径，Docker 是**可选**交付，见 `部署/README.md`。

---

## 1. 文件清单

| 文件 | 作用 |
| --- | --- |
| `Dockerfile` | FastAPI 服务镜像（python:3.11-slim + `部署/docker/requirements.txt`；可选 `--build-arg WITH_MULTIMODAL=1` 装图像解析重依赖） |
| `docker-compose.yml` | 两个服务：`rag-api`（8600，FastAPI）与 `rag-web`（8123，纯标准库中文 Web，`profiles: [web]` 默认不起） |
| `requirements.txt` | API 运行依赖（版本取自本机实测，非臆造） |
| `requirements-multimodal.txt` | 离线图像解析重依赖（torch/transformers/Pillow/huggingface_hub），服务镜像默认不装 |

## 2. 实测到的构建/运行障碍（如实记录，勿绕过）

```powershell
$ docker version
Client: Version 29.7.2  Context: desktop-linux  …（客户端信息正常）
permission denied while trying to connect to the docker API at
npipe:////./pipe/dockerDesktopLinuxEngine
→ 退出码 1
```

* `\\.\pipe\dockerDesktopLinuxEngine` **存在**（`Test-Path` = True），但沙箱下**访问被拒**；
  需提权 `sandbox_permissions=danger-full-access` 才可能通过（本会话**不允许审批**，故未提权、未构建）。
* 已完成的**静态校验**（可复现）：

```powershell
# ① compose 语法/schema/卷路径解析（实测退出码 0）
pwsh -NoProfile -c "docker compose -f 部署/docker/docker-compose.yml config"          # → exit 0，73 行解析结果
pwsh -NoProfile -c "docker compose -f 部署/docker/docker-compose.yml config --services"
#   → rag-api                （默认）
pwsh -NoProfile -c "docker compose -f 部署/docker/docker-compose.yml --profile web config --services"
#   → rag-api / rag-web      （带 profile 才含 Web 服务）
# ② 依赖清单版本与本机一致：python -m pip show fastapi uvicorn …（见 部署/docker/requirements.txt 注释）
# ③ 镜像内启动命令与本机实测一致：部署/脚本/serve_api.py --host 0.0.0.0 --port 8600（本机已 HTTP 200 实测）
```

> **严禁**把上述静态校验写成「已构建成功 / 已启动容器」。真机构建请用下面命令并自行留痕。

## 3. 在**可访问 Docker 守护进程**的机器上运行

```bash
# 0) 前置：宿主机（或同网可达的机器）跑着 Ollama，且已拉取两个模型
#    ollama serve
#    ollama pull bge-m3:latest && ollama pull qwen2.5:3b

# 1) 构建（上下文 = 仓库根；本机路径含中文，建议在仓库根执行）
docker build -f 部署/docker/Dockerfile -t gongdan4-rag-api:local .

# 2) 只起 FastAPI（8600）
docker compose -f 部署/docker/docker-compose.yml up -d --build
curl http://127.0.0.1:8600/health          # 期望 {"status":"ok", ...}，HTTP 200
curl -X POST http://127.0.0.1:8600/ask -H "Content-Type: application/json" \
     -d '{"question":"武汉兴图新科电子股份有限公司注册资本是多少？","lang":"zh"}'

# 3) 一并起纯标准库中文 Web（8123）
docker compose -f 部署/docker/docker-compose.yml --profile web up -d

# 4) 收尾
docker compose -f 部署/docker/docker-compose.yml down
```

## 4. 设计取舍（都与本机实况有关）

1. **索引/语料/权重不进镜像**：`研发/data` 体积大（PDF + 向量 + 718MB 级权重），且语料属只读红线
   → 由 compose 以 **只读卷** 挂载（`/app/研发/data/**`）；容器只读、不改语料。
2. **嵌入与生成走宿主机 Ollama**：容器内 `127.0.0.1` 不是宿主机，故
   `RAG_LLM__OLLAMA_BASE_URL=http://host.docker.internal:11434`（Linux 宿主补 `extra_hosts: host-gateway`）。
3. **服务镜像不装 torch/transformers**：二者只在离线图像解析时用（运行时才 import），
   装进服务镜像只会让镜像膨胀数 GB。需要重建图像块时再 `--build-arg WITH_MULTIMODAL=1`。
4. **`rag-web` 用 profile 隔离**：验收要求「中文 Web 界面」，纯标准库实现依赖最少，
   默认不开、显式 `--profile web` 才起，避免与 8123 上已有的原生服务抢端口。
5. **健康检查等了 3 分钟**（`start_period: 180s`）：启动要加载 2676 块索引 + BM25 + 预热 Ollama 嵌入/生成，
   本机原生实测「解释器启动 → `/health` 可用」约 6.3 s，但容器冷启动（含首次模型加载）应留足余量。
