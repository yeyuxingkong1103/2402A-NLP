# 工单18 部署说明：文档质量评估 Skill 集成进 RAGFlow

把「文档质量评估 Skill」装进 RAGFlow，并暴露成两个 HTTP 端点，让知识库入库前多一道质检关卡。

---

## 1. 这套东西长什么样

```
调用方（脚本/前端）
   │  POST /v1/document/quality-inspection      {"path": "...", "limit": 3}
   ▼
nginx（宿主 80）──► RAGFlow 容器 127.0.0.1:9380
                      │
                      │ api/apps/document_app.py 里新增的两个 route
                      ▼
                    subprocess 调 /ragflow/.venv/bin/python
                      /ragflow/document_quality_assessment/scripts/assess.py
                      │
                      ▼
                    产出 /ragflow/quality_report/quality_report.json
                                              /quality_report.html
```

| 组成部分 | 位置 |
|---|---|
| Skill 本体（纯 Python，不依赖 RAGFlow） | `研发/document_quality_assessment/`，入口 `scripts/assess.py`，阈值配置 `assessment_config.yaml` |
| 端点补丁 | `研发/patches/api/apps/document_app.py` → 容器内 `/ragflow/api/apps/document_app.py` |
| compose（含上面两条挂载，**无需改动**） | `研发/ragflow/docker/docker-compose.yml` |
| 本目录 | 部署脚本 + 本文档 |

两个端点：

| 方法 | 路径 | 返回 | 用途 |
|---|---|---|---|
| POST | `/v1/document/quality-inspection` | JSON（`data.summary` + `data.details`） | 给程序/流水线消费 |
| GET | `/v1/document/quality-inspection/report` | HTML 简报 | 给人看，浏览器直接打开 |

认证：两个端点都走 RAGFlow 的 `@login_required`，外部脚本用
`Authorization: Bearer <API_KEY>` 即可（登录 JWT 也行）。

请求体字段（POST）：

| 字段 | 必填 | 说明 |
|---|---|---|
| `path` | 是 | 要质检的**容器内**目录或单文件路径 |
| `limit` | 否 | 只处理前 N 份；不传或 0 = 全量 |
| `refresh` | 否 | `true` 则丢弃解析缓存重新解析 |
| `out` | 否 | 报告落盘目录，默认 `/ragflow/quality_report` |

---

## 2. 本目录文件

| 文件 | 干什么 |
|---|---|
| `install.sh` | 部署前置检查 + 往容器装 pymupdf。`--check` 只检查不动手 |
| `start.sh` | `docker compose up -d` + 等 HTTP 就绪，容器重建时提醒补装 pymupdf |
| `stop.sh` | `docker compose down`（`--volumes` 才删数据） |
| `quality_api.sh` | 质检 API 调用示例：path / limit / format 三个参数各演示一遍 |
| `DEPLOYMENT.md` | 本文档 |

所有脚本都是 LF 换行、`#!/usr/bin/env bash`，在 WSL / Git Bash / Linux 上都能跑。
换行符被 Windows 编辑器改成 CRLF 的话会报 `$'\r': command not found`，用
`dos2unix install.sh` 或 `sed -i 's/\r$//' install.sh` 修回来。

---

## 3. 前置条件

| 项 | 要求 | 怎么确认 |
|---|---|---|
| Docker | Engine + Compose v2（`docker compose`，不是 `docker-compose` 也行） | `docker compose version` |
| RAGFlow 镜像 | `infiniflow/ragflow:v0.23.1`（CPU 版），容器名 `docker-ragflow-cpu-1` | `docker images \| grep ragflow` |
| 宿主端口 | 80（nginx）与 9380（API 直连）可用 | `curl -s -o /dev/null -w '%{http_code}\n' http://localhost:9380/` |
| 已交付的基础部署 | 工单01-10 的 RAGFlow 已经在跑，本工单是在它上面加一层 | 容器在 `docker ps` 里 |
| conda 环境 `rag_gd` | 只在**本地命令行**跑 Skill 时用（`D:\Anaconda\envs\rag_gd\python.exe`，已有 pymupdf/pyyaml/python-docx）。**该环境是冻结的，不要往里装包** | `install.sh` 第 1 步会查 |
| API Key | 调用端点用 | 见第 4 步之三 |

> 为什么本地检查 `rag_gd`、而质检在容器里跑：容器里走的是 RAGFlow 自己的
> `/ragflow/.venv`，跟宿主 conda 环境没关系。宿主环境只用来在命令行直接跑 `assess.py` 调参。

---

## 4. 部署 5 步

### 第 1 步：起 RAGFlow

```bash
cd "工单18/部署"
bash start.sh
```

做三件事：`docker compose up -d`、等容器 running、轮询 `http://127.0.0.1:9380/`
直到 HTTP 有响应（默认最多等 300 秒，慢机器用 `WAIT=600 bash start.sh`）。

> **`.env` 不用管**：它里面有本机密钥，**不在版本库里**，所以克隆下来是没有的。
> 而 compose 用 `env_file: .env` 引用它，缺了会直接报错起不来 ——
> `start.sh` 会自动从 `.env.example` 复制一份（密钥为空）。
>
> 要调大模型的话，把 `DEEPSEEK_API_KEY` 填进 `研发/ragflow/docker/.env`，
> 或者直接 `export DEEPSEEK_API_KEY=...`（环境变量优先）。

期望输出：

```
==> docker compose up -d
==> 等容器进入 running（最多 60 秒）
    容器已运行（第 1 次探测）
==> 等 HTTP 服务就绪：http://127.0.0.1:9380/（最多 300 秒）
    HTTP 已响应（状态码 200，用时约 45s）
```

两条挂载（`document_app.py`、`document_quality_assessment`）已经在
`研发/ragflow/docker/docker-compose.yml` 里写好了，**不需要改 compose**。
挂载是容器创建时建立的：改了 compose 必须 `docker compose up -d --force-recreate` 才生效。

### 第 2 步：装容器侧依赖 + 全量检查

```bash
bash install.sh
```

四件事：查 `rag_gd`（只查不装）、查 docker/compose、**往容器装 pymupdf**、查 compose
引用的宿主路径是否都存在。

期望输出（关键几行）：

```
==> 1/4 检查 conda 环境 rag_gd（本地跑 Skill 用）
  [OK]   rag_gd 解释器：/d/Anaconda/envs/rag_gd/python.exe
  [OK]   依赖 fitz 可导入
==> 3/4 容器侧依赖：pymupdf（fitz）
  [注意] pymupdf 没装，开始安装（约 20MB，走宿主网络）
  [OK]   pymupdf 装好了，import fitz 通过
==> 4/4 检查 compose 引用的宿主路径
  [OK]   ../../document_quality_assessment
  [OK]   ../../patches/api/apps/document_app.py
 检查通过。下一步：bash quality_api.sh（需要先 export API_KEY）
```

手动等价命令（`install.sh` 装不上时用这条）：

```bash
docker exec docker-ragflow-cpu-1 /usr/local/bin/uv pip install --python /ragflow/.venv/bin/python pymupdf
```

### 第 3 步：取 API Key

用登录 token 换一个 API Key（token 从 RAGFlow 登录接口拿，或浏览器 F12 → Network 抄）：

```bash
export LOGIN_TOKEN='<登录 token>'
curl -sS -X POST http://localhost/v1/api/new_token \
     -H "Authorization: Bearer ${LOGIN_TOKEN}"
# 返回里的 api_key 就是后面要用的
export API_KEY='ragflow-xxxxxxxx'
```

**API Key 不要写进任何脚本、不要提交进 git。** 本目录的脚本一律从环境变量读：

```bash
API_KEY="${API_KEY:?请先设置 API_KEY}"
```

### 第 4 步：把要质检的数据挂进容器

端点收到的 `path` 是**容器内**路径。宿主机的目录没挂载进去的话，端点会直接报
「路径不存在」。挂载点在 compose 的 `ragflow-cpu` 服务里加一行：

```yaml
    volumes:
      # ... 原有的挂载 ...
      - /宿主机/数据目录:/data/manuals      # 新增：质检数据
```

然后重建容器让挂载生效：

```bash
docker compose -f 研发/ragflow/docker/docker-compose.yml up -d --force-recreate
bash install.sh        # 重建后 pymupdf 没了，必须重装
```

不想动 compose 的话，也可以直接对已经挂进去的目录跑质检，比如
`/ragflow/document_quality_assessment`（Skill 自己，适合冒烟测试）。

### 第 5 步：跑通质检

```bash
export API_KEY=ragflow-xxxxxxxx

LIMIT=3 bash quality_api.sh                     # 试跑 3 份，返回 JSON
DOC_PATH=/data/manuals LIMIT=0 bash quality_api.sh    # 全量
FORMAT=html bash quality_api.sh                 # 取 HTML 简报，存 ./quality_report.html
bash quality_api.sh --examples                  # 只打印全部 curl 示例
```

---

## 5. 验证方法（含期望输出）

### 5.1 端点已注册

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:9380/v1/document/quality-inspection
```

期望 `401` —— 说明路由在（没带 Key 被拦），补丁和挂载都生效了。
如果是 `404`：`document_app.py` 没挂进去，或容器没重启。
如果用宿主 80 端口访问（`http://localhost/...`），结果一样。

### 5.2 质检结果（JSON）

```bash
LIMIT=3 bash quality_api.sh
```

期望（数值随目录而变，字段名固定）：

```
==> POST http://localhost/v1/document/quality-inspection
    body: {"path": "/ragflow/document_quality_assessment", "limit": 3}
{
    "code": 0,
    "data": {
        "summary": {
            "root": "/ragflow/document_quality_assessment",
            "total_files": 3,
            "total_bytes": 27315,
            "parse_failed": 0,
            "ext": {".md": 2, ".yaml": 1},
            "pdf": {"text": 0, "scanned": 0, "mixed": 0, "unreadable": 0},
            "pdf_scanned_ratio": 0.0,
            "chars": {"total": 9124, "mean": 3041.3, "p50": 2900, "p90": 3300},
            "exact_dup_groups": 0,
            "redundant_copies": 0,
            "similar_pairs": 0,
            "sensitive_hits": 0,
            "uncertain_pdf": 0,
            "unreadable_pdf": 0,
            "by_tag": {...},
            "by_route": {...},
            "by_primary_route": {...}
        },
        "details": {"format": {...}, "pdf": {...}, "length": {...},
                    "duplicates": {...}, "sensitive": {...}, "tags": {...}}
    },
    "message": "success"
}
```

    --- 摘要 ---
    文件总数      : 3（27315 字节）
    PDF 类型分布  : {'text': 0, 'scanned': 0, 'mixed': 0, 'unreadable': 0}
    扫描件占比    : 0.0
    解析失败      : 0
    完全重复组    : 0（冗余副本 0）
    近似重复对    : 0
    敏感信息命中  : 0
    主路由分布    : {...}
    HTML 简报     : GET http://localhost/v1/document/quality-inspection/report?out=/ragflow/quality_report

判定要点：
- `code` 必须是 `0`。非 0 时 `message` 就是原因，直接看它。
- `total_files` 应等于 `limit`（或目录真实文件数）。
- 挑一个**扫描型 PDF** 目录再跑一次，`pdf.scanned` 应大于 0、`pdf_scanned_ratio` 明显大于 0
  —— 这是质检的核心能力（决定这批文件要不要走 OCR）。
- 往目录里放两份同样内容的文件再跑，`exact_dup_groups` 应为 1。

### 5.3 HTML 简报

```bash
FORMAT=html bash quality_api.sh
```

期望：

```
==> 先 POST 跑一次质检（否则可能还没有报告）
    HTTP 200
==> GET http://localhost/v1/document/quality-inspection/report?out=/ragflow/quality_report  ->  ./quality_report.html
    HTTP 200
    简报已存：./quality_report.html（18742 字节）
```

浏览器打开 `quality_report.html`，能看到格式分布、PDF 类型饼图、重复文件列表、
敏感信息命中列表。若返回 400 且 message 说「还没有报告」，说明 POST 那步没成功，先修 POST。

### 5.4 容器内的产物

```bash
docker exec docker-ragflow-cpu-1 ls -l /ragflow/quality_report
# quality_report.json  quality_report.html  .quality_work/
docker exec docker-ragflow-cpu-1 /ragflow/.venv/bin/python -c 'import fitz; print("fitz OK")'
```

---

## 6. 常见问题

### 6.1 `No module named 'fitz'` / 质检失败 exit=1

容器自带的 `/ragflow/.venv` **没有 pymupdf**，而 Skill 解析 PDF 必须用它。装：

```bash
docker exec docker-ragflow-cpu-1 /usr/local/bin/uv pip install --python /ragflow/.venv/bin/python pymupdf
# 或者：bash install.sh
```

端点返回的 message 里能看到 `ModuleNotFoundError: No module named 'fitz'`。
装完**不用**重启容器，下次请求就会用上（子进程每次现起）。

### 6.2 容器重建后端点又不好了

`docker compose down` 之后再 `up`、或 `up -d --force-recreate`、或换镜像，
容器可写层被丢掉，**pymupdf 跟着没**。这不是装错了，是设计如此（不动镜像，保持镜像干净）。

**记忆口诀：容器重建一次，就 `bash install.sh` 一次。**
`start.sh` 会在检测到容器 ID 变化时主动提醒。

### 6.3 路径要挂载进容器才能被端点看到

报错长这样：

```json
{"code": 102, "message": "路径不存在：/data/manuals（注意这是**容器内**的路径，宿主机路径要先挂载进容器）"}
```

`path` 是**容器内**路径。宿主机目录必须在 compose 里挂进容器，且重建容器后才生效
（见第 4 步）。自查：

```bash
docker exec docker-ragflow-cpu-1 ls /data/manuals
```

另一个隐蔽坑：compose 里写了挂载、但**宿主路径不存在**时 docker 不报错，
它会新建一个**空目录**顶上去 —— 结果容器里 Skill 代码是空的，端点报
「质检失败」。`bash install.sh` 的第 4 步就是专门查这个的。

### 6.4 `401` / `403`

- 没带 `Authorization` 头 → 401。
- Key 过期/抄错/带了多余空格 → 401。
- 同一账号在别处登出（token 失效）→ 重新取 Key。

```bash
curl -sS -o /dev/null -w '%{http_code}\n' -X POST \
     -H "Authorization: Bearer ${API_KEY}" \
     http://localhost/v1/document/quality-inspection
```

### 6.5 大目录超时（`质检超时（>1800s）`）

端点侧默认超时 1800 秒（`QUALITY_TIMEOUT`），1700 份实测约 1 分钟。超了就：

- 加 `limit` 分段跑；
- 或在 compose 里给环境变量加一行 `QUALITY_TIMEOUT: "7200"` 再重建容器；
- 或先跑一次 `refresh=false`（默认）复用解析缓存，第二次会快很多（缓存落在 `out/.quality_work`）。

### 6.6 改了 Skill 代码/配置要不要重启容器

不用。`document_quality_assessment` 是 bind mount，宿主改完立刻可见；
端点是每次请求现起子进程跑 `assess.py`。**但**改了
`document_app.py`（端点本身）或 compose 的挂载列表，就必须重建容器，
重建后记得重装 pymupdf（见 6.2）。

### 6.7 脚本报 `$'\r': command not found`

文件被 Windows 编辑器存成了 CRLF。修：

```bash
sed -i 's/\r$//' *.sh && chmod +x *.sh
```

---

## 7. 停止与清理

```bash
bash stop.sh              # 停并删容器（volume 与数据保留）
bash stop.sh --volumes    # 连 volume 一起删，知识库数据全没，需输入 yes 确认
```

只想暂停、连容器状态都留着（比如临时腾内存）：

```bash
docker compose -f 研发/ragflow/docker/docker-compose.yml stop    # 停
docker compose -f 研发/ragflow/docker/docker-compose.yml start   # 再起
```

`stop`/`start` 不重建容器，pymupdf 还在；`down` 之后再 `up` 就得重装。

---

## 8. 环境变量一览

脚本侧（本目录）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `API_KEY` | 无（必填） | 质检端点认证，从环境变量读 |
| `BASE_URL` | `http://localhost` | 宿主 80 端口（nginx）；直连 API 用 `http://localhost:9380` |
| `DOC_PATH` | `/ragflow/document_quality_assessment` | 容器内待质检路径 |
| `LIMIT` | `3` | 试跑份数，0 = 全量 |
| `FORMAT` | `json` | `json` / `html` |
| `OUT_DIR` | `/ragflow/quality_report` | 容器内报告目录 |
| `CONTAINER` | `docker-ragflow-cpu-1` | 容器名 |
| `COMPOSE_FILE` | `../研发/ragflow/docker/docker-compose.yml` | compose 路径 |
| `WAIT` | `300` | `start.sh` 健康检查最多等几秒 |
| `RAG_GD_PYTHON` | 自动探测 | 本地 `rag_gd` 解释器路径 |

端点侧（容器内，改这些要在 compose 里加环境变量再重建容器）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `QUALITY_SKILL_DIR` | `/ragflow/document_quality_assessment` | Skill 落点 |
| `QUALITY_PYTHON` | `/ragflow/.venv/bin/python` | 跑 Skill 的解释器 |
| `QUALITY_OUT_DIR` | `/ragflow/quality_report` | 报告默认落盘目录 |
| `QUALITY_TIMEOUT` | `1800` | 单次质检超时（秒） |
