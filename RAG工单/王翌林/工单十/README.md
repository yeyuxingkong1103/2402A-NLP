# 金融问答系统部署任务 —— 工单十

> 工单编号：人工智能NLP-RAG-金融问答系统部署

本项目基于工单七复制而来，位于 `/home/dabaie/code/工单/工单十`。工单十不新增任何 RAG 能力，而是将既有金融问答系统（工单六 v6：FastAPI 8006 + Streamlit 8506）以 **Docker 容器**形式部署并验收：容器一键启动指定端口服务、named volume 持久化关键数据并支持容器间共享、自定义网络保证容器与宿主机 Milvus/外网 LLM 通信。

## 工单十新增内容

```
Dockerfile                     # python:3.10-slim 一体化镜像（模型 bind mount 复用，不入镜像）
requirements-docker.txt        # 精简 CPU 版运行期依赖（torch CPU wheel）
.dockerignore                  # .env/附件PDF/日志/截图不入镜像
docker-compose.yml             # rag-api(8006) + rag-ui(8506)；rag-v10-data 卷 + rag-v10-net 网络
scripts/start_docker_v10.sh    # build + up -d + 健康检查等待
scripts/stop_docker_v10.sh     # 停止容器（默认保留数据卷，--prune 才删卷）
scripts/test_deployed_v10.py   # 部署验收：问答测试 + 日志/卷持久化/容器互访/宿主Milvus 实证
tests/test_docker_deploy_v10.py# 12 个静态单测（不依赖 docker daemon）
docs/00_工单十任务说明.md       # 需求、验收标准对照、架构与关键决策
docs/10_Docker部署文档.md       # 部署/运维手册
docs/11_部署过程问题记录.md     # 实现步骤 + 7 个过程问题与解决
docs/deploy_v10_test_results.json # 验收实测结果（脚本生成）
docs/screenshots/               # 测试截图（Web 界面 + 终端实证）
```

## 工单要求对照

| 类别 | 要求 | 落地 |
| --- | --- | --- |
| 设计 | 技术组件 | [docs/技术组件.md](docs/技术组件.md) |
| 设计 | 技术架构 / 流程图 | [docs/技术架构与流程图.md](docs/技术架构与流程图.md) |
| 设计 | 思维导图 | [docs/思维导图.md](docs/思维导图.md) |
| 设计 | 接口文档 | [docs/API接口文档.md](docs/API接口文档.md) |
| 研发 | 代码 | `src/`、`app/`、`Dockerfile`、`docker-compose.yml` |
| 测试 | 截图（多截图） | [docs/screenshots/](docs/screenshots/) 共 9 张 |
| 部署 | linux shell 脚本 | `scripts/start_docker_v10.sh` / `stop_docker_v10.sh` / `test_deployed_v10.py` |
| 部署 | 安装脚本（conda/cuda 环境） | `scripts/install_conda_env.sh`（conda env + PyTorch cu121） |
| 部署 | 启动脚本 | `scripts/start_docker_v10.sh` |
| 部署 | 结束脚本 | `scripts/stop_docker_v10.sh` |

## 测试截图索引（docs/screenshots/）

| 文件 | 内容 |
| --- | --- |
| 01_fastapi_docs.png | FastAPI Swagger 文档页（/docs） |
| 02_health_check.png | /api/v6/health 健康检查返回（status=ok） |
| 03_streamlit_ui.png | Streamlit 金融问答界面 |
| 04_docker_ps.png | `docker ps` 双容器运行状态 |
| 05_docker_logs_api.png | rag-v10-api 日志（无 ERROR/Traceback） |
| 06_docker_logs_ui.png | rag-v10-ui 日志 |
| 07_docker_volume.png | named volume `rag-v10-data` |
| 08_docker_network.png | bridge 网络 `rag-v10-net` |
| 09_acceptance_result.png | 验收脚本 14/14 PASS 结果 |

## 快速部署

```bash
cd /home/dabaie/code/工单/工单十
bash scripts/start_docker_v10.sh     # 首次构建约10-20分钟
# FastAPI: http://localhost:8006/api/v6/health
# Streamlit: http://localhost:8506

# 验收（容器运行中执行，自动完成 down/up 持久化实证）
python scripts/test_deployed_v10.py

bash scripts/stop_docker_v10.sh      # 停止（数据卷保留）
```

## 验收对照

| 验收项 | 落地 |
| --- | --- |
| 部署后金融问答服务测试通过 | 健康检查 + 3 道金融题金标关键词校验（`test_deployed_v10.py`） |
| 容器启动/端口/日志无异常 | `docker compose up -d` 一键启动 8006/8506；`docker logs` 无 ERROR |
| 卷持久化 + 容器间数据共享 | named volume `rag-v10-data` 挂 `/app/data`，api/ui 同挂；down/up 实证数据不丢 |
| 网络配置正确 | bridge 网络 `rag-v10-net` 容器互访；host-gateway 访问宿主机 Milvus(19530)；外网 DeepSeek |

## 验收实测（2026-10-07）

`python scripts/test_deployed_v10.py` → **14/14 项全部通过**（`docs/deploy_v10_test_results.json`，`"passed": true`）：

- 服务测试：health_check 200；3 道金融题金标关键词全命中（邮储 2768.09/6.06、中信 543.83/149.02）；UI 200
- 容器管理：双容器 running 且日志无 ERROR/Traceback
- 卷持久化：down→up 后标记文件仍在，api/ui 共享同一 named volume
- 网络：rag-ui→rag-api:8006 HTTP 200；rag-api→宿主机 Milvus 19530 TCP 连通

## 质量

- 新增静态单测 12 用例全过；全量 pytest 回归 **286 passed / 1 skipped**（含工单七基线 274），无回归
- 部署验收 14/14 通过，数据见 [docs/deploy_v10_test_results.json](docs/deploy_v10_test_results.json)
- 新增代码注释均含工单编号：**人工智能NLP-RAG-金融问答系统部署**

---

# 附：被部署系统（工单六/七摘要）

- **工单六（混合检索）**：向量（bge-m3+Milvus+重排）/ 全文（倒排+布尔/短语/模糊）/ 混合（RRF、加权平均）三策略可配，LLM/TF-IDF/用户反馈三重排器；准确率 93.8%、召回 96.9%、平均响应 2206ms
- **工单七（功能测试及评估）**：9 份 A 股年报语料（7878 chunks）+ 10 题评估，答案准确率 10/10、doc_recall@5 0.96、MRR 0.95
- 语料共存于宿主机 Milvus `rag_chunks` 集合（招股书 1141 + 年报 7878 = 9019 chunks），部署后直接可用

工单六/七完整说明见 [docs/00_工单六任务说明.md](docs/00_工单六任务说明.md)、[docs/00_工单七任务说明.md](docs/00_工单七任务说明.md)。
