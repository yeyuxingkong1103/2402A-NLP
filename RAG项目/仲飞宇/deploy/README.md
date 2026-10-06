# 部署指南

需求里的「部署：在 Ubuntu / 算力云 / 腾讯云 / 阿里云上成功部署 + 部署脚本 + 部署文档」。
本目录集中存放部署相关的配置与脚本。

## 目录

| 文件 | 用途 |
|---|---|
| `nginx.conf` | 负载均衡 + 反向代理（多 uvicorn worker 横向扩展） |
| `docker-compose.multi-source.yml` | 多路召回扩展数据源：Neo4J / MongoDB / ClickHouse |
| `jmeter/` | Jmeter 压测脚本与说明（QPS） |
| `verify_prod_2026-09-21.log` | 第六节自证清单的**本机彩排留档**（真实命令输出，可直接对照验收） |

## 一、Ubuntu / 云服务器部署

在任意有 Python 3.10~3.12 的 Ubuntu 主机上：

```bash
# 1. 上传代码（git clone 或打包上传解压到项目目录）

# 2. 一键安装（检测环境 → 建 venv → 装依赖）
bash scripts/install.sh

# 3. 起三个服务化组件（.env.prod 假设它们都跑在本机 127.0.0.1）
sudo apt install -y mysql-server redis-server
sudo systemctl enable --now mysql redis

#    建库建用户 —— .env.prod 写的是 rag:rag@127.0.0.1:3306/rag_roleplay，
#    全新主机上这个库和用户都不存在，不建的话第 5 步自检会卡在「MySQL 不可达」
sudo mysql <<'SQL'
CREATE DATABASE IF NOT EXISTS rag_roleplay CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'rag'@'localhost' IDENTIFIED BY 'rag';
GRANT ALL PRIVILEGES ON rag_roleplay.* TO 'rag'@'localhost';
FLUSH PRIVILEGES;
SQL

#    向量库必须用独立 Milvus 服务，**不能用 Milvus Lite**：Lite 独占数据目录文件锁，
#    多 uvicorn worker 里第二个就起不来（详见 deploy/docker-compose.milvus.yml 头部注释）
docker compose -f deploy/docker-compose.milvus.yml up -d    # 首次约 30~90 秒起 healthy

# 4. 用生产配置
export APP_ENV=prod          # 后面每条命令都要带上它（含 seed），否则读到的是 .env 的 dev 配置
#    必须改 .env.prod 里这两项，否则 /chat 会 401：
#      LLM_API_KEY / EMBED_API_KEY = 硅基流动或 DeepSeek 的真实 key
#    云上到不了本机 Ollama，所以 .env.prod 走 openai_compat 在线 API，这是预期的。

# 5. 自检（--no-start 只查依赖不起服务）：三个组件 + 重排都应该 ✅
bash scripts/start_all.sh --no-start --strict

# 6. 若 RERANKER=bge，还要先做第二节（下模型 + 起重排服务），否则第 5 步会把它记为降级项

# 7. 灌入演示数据 + 启动
.venv/bin/python scripts/seed.py
bash scripts/start_all.sh --strict
```

算力云 / 腾讯云 / 阿里云：流程同上，另注意——
- 安全组放行 8000 端口（以及 Nginx 的 80 端口）；
- 想横向扩展（多 worker + nginx）就照第四节，**必须**用独立 Milvus 与 `MEMORY_BACKEND=redis`。

> **本节已在本机（WSL）实跑通过**（2026-09-21），不再是「纸面标准操作」：在 WSL 内
> `apt install mysql-server`（注意 apt 源要先 http→https 才连得上）+ 建库建用户之后，
> `APP_ENV=prod PORT=8100 bash scripts/start_all.sh --strict` **五项全 ok / exit 0**，
> 端到端 `/chat` 也出了有据答案（基线与判据见第六节）。云上照此执行即可。
>
> **`APP_ENV` 必须 export 出来**：`scripts/start_all.sh` 早期版本只读 `.env`、完全无视 `APP_ENV`，
> 于是 `APP_ENV=prod bash scripts/start_all.sh --strict` 会「脚本按 dev 依赖自检、应用按 prod 连组件」，
> 自检全绿而连的东西对不上。现已修成与应用同序的分层（`.env` ← `.env.${APP_ENV}` 覆盖），
> 并在启动横幅里打出当前生效的配置层。
>
> 同一类缺陷还有第二条轴，本机彩排时才暴露：脚本里那几行 `XXX="$(env_get XXX …)"` 赋值，
> 对**调用时已经导出的环境变量**同样生效，于是 `EMBED_MODEL=... bash scripts/start_all.sh` 传进去的值
> 会被 `.env` 里的值顶掉（只有出现在那几行赋值里的键中招）。现已把 `env_get` 改成环境变量优先、
> 与应用同序。当时的表现是 seed 走 `bge-m3` 正常、应用起来却报 `BAAI/bge-m3 404 model not found`。

## 二、BGE-rerank 精排（重排序从 score_fusion 升级）

BGE-rerank 需要独立服务 + 下载交叉编码模型（约 2.2GB）：

```bash
# 1) 首次：建独立 venv + 装 torch（CPU 版）+ sentence-transformers
python3 -m venv rerank_service/.venv
rerank_service/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
rerank_service/.venv/bin/pip install -r rerank_service/requirements.txt
# 2) 下载模型到 /mnt/d/models/bge-reranker-v2-m3（可重复执行，断点续传）
bash scripts/download_rerank_model.sh
# 3) 起重排服务（脚本会自动找到上一步下好的本地模型，不再联网）
bash scripts/run_rerank.sh
# 4) .env 里切到 bge
RERANKER=bge
```

**别让 `run_rerank.sh` 自己去 HuggingFace 下模型。** 本机实测（2026-09-19）走不通：

- 国内直连 `huggingface.co` 不通（HTTP 000）；走 hf-mirror 镜像时该 repo 按 xet 存储下载，
  但 CAS 重建接口 `cas-server.xethub.hf.co` 返回 **401 Unauthorized**，大文件反复中断；
- 更坑的是 `huggingface_hub` 的临时文件名带随机 uuid，进程一旦被强杀，已下部分
  **下次无法复用**（`huggingface_hub/file_download.py` 里写明 "could not be reused anyway"），
  1.2GB 下了一半等于白下。

`scripts/download_rerank_model.sh` 改用 modelscope 直链 + `curl -C -`，断点续传跨进程有效。
实测速度：modelscope 808KB/s、hf-mirror 570KB/s，全量约 47 分钟（6Mbps 链路）。

> 模型 ~2.2GB，小带宽机器首次下载较慢。服务不可用时 `BGEReranker` 会自动降级回
> score_fusion 融合分数排序，链路不断。
>
> 开启精排时召回池会按 `RERANK_POOL` 放宽（默认 20，即 4x TOP_K）再精排截断回 TOP_K。
> 精排只能从粗排池里挑人，池宽等于 TOP_K 时它就只能换顺序、捞不出粗排漏掉的候选。

## 三、多路召回扩展（多数据源）

默认已启用「Milvus 稠密 + BM25 关键词」两路，RRF 融合（`rrf_fusion` 天然支持 N 路）。
要再加数据源（Neo4J 图 / MongoDB 文档 / ClickHouse 分析 / 互联网）：

1. `docker compose -f deploy/docker-compose.multi-source.yml up -d` 起对应库；
2. 在 `app/core/retrieve/` 新增一个 route（返回 `list[id]` 的排序结果）；
3. 把它并入 `HybridRetriever.retrieve` 的 `rrf_fusion(*dense_lists, *bm25_lists, *new_lists)`。

## 四、压测与负载均衡

- 压测：见 `deploy/jmeter/README.md`（Jmeter，关注 QPS / 响应时间 / 错误率）。
- 负载均衡：`deploy/nginx.conf`，把流量分到多个 uvicorn worker。
  多 worker 时务必 `MEMORY_BACKEND=redis`（进程内记忆不跨 worker 共享）。
- 日志：`scripts/run_workers.sh` 会给每个 worker 注入 `WORKER_PORT`，**应用日志按端口分文件**
  （`logs/app-8000.log`、`logs/app-8002.log` …），与既有的 `logs/uvicorn-<port>.out` 同一套规矩。
  排查时按端口找文件，别只盯某一个：多个进程共写一个 `app.log` 时，按天轮转是「改名 + 新建同名
  文件」，跨天时输的那个进程**日志会直接丢**（实测，见 `app/core/logging_config.py` 注释）。
  单实例（`run.sh` / `start_all.sh`）不注入 `WORKER_PORT`，仍是 `logs/app.log`。
- 装完 nginx 后**跑一次 `bash scripts/verify_lb.sh` 自证分发真的生效**（它以非 0 退出即有问题）。
  为什么不能只靠肉眼看配置：`upstream` 不配 `zone` 时，默认 `worker_processes auto`
  会让**每个 worker 各自维护轮询指针、都从第一台上游起步**，而 nginx 本身不会报任何错。
  该脚本用静态判据（有没有 `zone`）+ 动态判据（实测落点分布）双重检查。

> 压测结果里的 worker 分发**不要指望从 Jmeter 的 .jtl 里读**。本机实测（JMeter 5.6.3）：
> `-Jjmeter.save.saveservice.responseHeaders=true`、`-q` 附加属性文件、乃至在 .jmx 里给
> ResultCollector 直接写 `<responseHeaders>true</responseHeaders>`，**三种方式都不落盘**
> 响应头（同一次 `-Jjmeter.save.saveservice.print_field_names=false` 却生效，说明属性机制本身没坏）。
> 要看分发就查 nginx 的 `$upstream_addr`，或用 `scripts/verify_lb.sh` 从 `X-Upstream` 头统计。

## 五、三环境

| 环境 | 启动方式 | 说明 |
|---|---|---|
| 开发 | `bash scripts/start_all.sh`（默认） | `.env`，本地 Ollama + Milvus Lite |
| 测试 | `APP_ENV=test .venv/bin/python -m pytest` | `.env.test`，全离线 dummy |
| 生产 | `APP_ENV=prod bash scripts/start_all.sh --strict` | `.env.prod`，服务化组件 |

## 六、部署后自证清单

「脚本跑完没报错」不等于「连的东西是对的」——第一节那两条 `APP_ENV` / `env_get` 的坑，
都曾经让自检全绿而链路是错的。所以部署完按下面逐条验，判据都是本机实跑过的
（完整留档见同目录 `verify_prod_2026-09-21.log`，里面是这些命令的真实输出）。

| # | 验什么 | 命令 | 通过判据 |
|---|---|---|---|
| 1 | 依赖自检 | `APP_ENV=prod bash scripts/start_all.sh --no-start --strict` | 五项 ✅ 且 **exit 0**；启动横幅打出的配置层是「`.env` ← 被 `.env.prod` 覆盖」 |
| 2 | 端到端 `/chat` | 见下方 curl | `http=200`、`sources` 非空、首条是相关原文，答案能在 `data/corpus/lawyer/*.md` 里逐条对上 |
| 3 | 短期记忆 | 同 `session_id` 追第二问 | 记得第一轮说过的、**知识库里没有**的事实 |
| 4 | 会话隔离 | 换 `session_id` 问同一句 | **答不出来**（明说资料里没有），而不是顺嘴编一个——这才是第 3 条的反证 |
| 5 | 负载均衡 | `bash scripts/verify_lb.sh` | exit 0；落点 ≥2 台且都在配置清单里 |

```bash
# 单发 /chat：务必给足超时。qwen3:8b 冷启动 + 检索实测 >2 分钟，
# 2026-09-21 用默认 120s 超时的客户端被 SIGKILL（curl exit 137），
# 而服务端其实答完了（app.log 里有 200、Redis 里也落了这一轮）——别把客户端的超时当成服务端故障。
curl -s -m 380 -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' \
  -d '{"question":"试用期最长可以约定多久？","role_id":"lawyer","session_id":"verify-1"}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["answer"][:200]); print([(round(s["score"],3), s["title"]) for s in d["sources"]])'

# 记忆是否真在用 Redis：键名格式 memory:{session_id}:{role_id}，一个来回 len=2
.venv/bin/python -c "import redis;r=redis.from_url('redis://127.0.0.1:6379/0',decode_responses=True);print({k:r.llen(k) for k in r.keys('memory:verify*')})"
```

**2026-09-21 本机彩排实测基线**（`APP_ENV=prod`，LLM/Embedding 用环境变量覆盖指回本地 Ollama）：

- 自检 5/5 ok、exit 0；`/chat` 200、**69s**（模型已加载时）、5 条 sources，
  首条 `hypertension_guide.md` 得分 **0.7299**、次条 0.5150 —— 分差明显，是精排在起作用的样子；
  纯 `score_fusion` 的 RRF 分数挤在同一量级、几乎不可分（实测见 `scripts/verify_rerank_e2e.py`）。
- 记忆对照：同一 `session_id` 答出「李四 / 138/88 mmHg」（这个事实知识库里没有），
  换 `session_id` 则答「资料中没有提到您的姓名或具体血压值」。
- 注意：本机彩排**没有**验证「真云厂商的 LLM/Embedding」这一环——`.env.prod` 里
  `LLM_API_KEY`/`EMBED_API_KEY` 是占位符、模型名是硅基流动那套，本机到不了也填不了 key，
  所以这一环只能上云后按第 2 条自证。
