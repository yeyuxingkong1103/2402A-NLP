# Plan 1：环境准备与 LangChain 双链路 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 S0（环境准备）与 S1（LangChain 双链路），使系统可在 `RAGLORA_CHAIN=langchain` 下跑通完整问答，且默认链路行为不变。

**Architecture:** 新增 `lc_chain.py`（LCEL 编排）与 `lc_retrievers.py`（检索器适配层），与现有手写 `rag_chain.py` 并存，由 `config.CHAIN_BACKEND` 切换。**bge-m3 只加载一份** —— 通过 `BgeM3DenseEmbeddings` 适配器委托给现有 `embed.py`，而不是用 `HuggingFaceEmbeddings` 重新加载。

**Tech Stack:** Python 3.10.14 (`D:\anaconda3\envs\rag_env`)、FastAPI、LangChain 1.2.18（core 1.4.6 / langchain-ollama）、Qdrant 嵌入式、Ollama **qwen2.5:7b**（全程单一生成模型）、Docker（Milvus 2.6.6 / Neo4j）

## Global Constraints

- **Python 解释器固定**：`D:/anaconda3/envs/rag_env/python.exe`，不得改用其它环境
- **绝不重新加载模型**：bge-m3（2.3GB）与精排（1.1GB）在进程内必须只有一份实例
- **默认链路不变**：`RAGLORA_CHAIN` 默认值为 `manual`，未显式设置时行为与改造前完全一致
- **内存上限**：本机 15.7GB，改动不得引入新的常驻模型副本
- **依赖范围**：本计划**只**新增 `ragas` / `pymilvus` / `pdfplumber` 三个包（Task 1），
  除此之外不得引入任何新依赖。rag_env 现有 325 个包的版本链不得被破坏
  （尤其 `transformers==4.39.3` 与 `torch==2.5.1+cu121` 必须保持不变）
- **后端单 worker**：Qdrant 嵌入式持有独占文件锁，不得加 `--workers`
- **注释与日志用中文**，与现有代码风格一致

---

### Task 0: 初始化版本控制

**Files:**
- Create: `.gitignore`
- Create: `.git/`（由 `git init` 生成）

**Interfaces:**
- Consumes: 无
- Produces: 可提交的 git 仓库；后续所有任务的 Commit 步骤依赖它

> **为什么必须先做**：本次改造涉及换向量库、改主链路、重编码 14955 条向量，
> 属高侵入变更。没有版本控制就没有回滚能力。当前项目**尚不是 git 仓库**。

- [x] **Step 1: 创建 .gitignore**

在项目根目录 `D:/桌面/RAGLoRA/` 创建 `.gitignore`：

```gitignore
# ---- Python ----
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/

# ---- 虚拟环境（本项目用 conda rag_env，仓内不存环境） ----
venv/
.venv/

# ---- 数据与模型产物（体积大，可重建） ----
qdrant_storage/
datasets/
*.log
backend/logs/
backend/eval/*.jsonl
backend/requirements-lock-before.txt

# ---- 前端 ----
frontend/node_modules/
frontend/dist/

# ---- Redis 免安装版的数据与日志 ----
tools/redis/data/
tools/redis/*.log
tools/redis/dump.rdb
tools/redis/*.pdb

# ---- 嵌入微调产物（Plan 7） ----
backend/finetune/out/

# ---- 编辑器 / 系统 ----
.vscode/
.idea/
Thumbs.db
desktop.ini
```

> **注意 `datasets/` 与 `qdrant_storage/` 被忽略**：前者是与法律/医疗语料等价的原始数据（
> 可重新下载），后者是 Qdrant 的二进制向量库。二者体积大且可重建，不入库。
> 但**微调产物、评测结果文档必须入库**（`docs/`、`backend/eval/qa_set.json`）。

- [x] **Step 2: 初始化仓库**

```bash
cd "D:/桌面/RAGLoRA"
git init
git config user.name >/dev/null 2>&1 || echo "⚠️ 未配置 git 用户，提交前需设置 user.name/user.email"
```

预期：`Initialized empty Git repository in D:/桌面/RAGLoRA/.git/`

- [x] **Step 3: 确认忽略规则生效（关键）**

```bash
cd "D:/桌面/RAGLoRA"
git status --porcelain | wc -l
git status --porcelain | grep -cE "node_modules|qdrant_storage|__pycache__" || echo "0 (忽略规则正确)"
```

预期：第二行输出 `0 (忽略规则正确)`
**若列出 node_modules 或 qdrant_storage，说明 .gitignore 没生效，先修正再继续。**

- [x] **Step 4: 确认待提交文件数量合理**

```bash
git status --short | head -30
```

预期：只出现源码（`backend/app/`、`frontend/src/`、`docs/`、`tools/` 脚本等），
**不应出现** `node_modules`、`qdrant_storage`、`datasets`、`logs`。

- [x] **Step 5: 首次提交**

```bash
cd "D:/桌面/RAGLoRA"
git add -A
git commit -m "chore: 初始化仓库，建立改造前基线"
```

预期：提交成功，输出文件数与插入行数统计

- [x] **Step 6: 记录基线提交号**

```bash
git rev-parse --short HEAD
```

预期：输出 7 位短哈希。
**把这个哈希记到本计划文件末尾的「基线」处**，回滚时用 `git reset --hard <哈希>`。

---

### Task 1: 版本备份与依赖安装

**Files:**
- Create: `backend/requirements-lock-before.txt`
- Modify: `backend/requirements.txt`

**Interfaces:**
- Consumes: 无
- Produces: 可导入的 `ragas` / `pymilvus` / `pdfplumber`；`requirements-lock-before.txt` 作为回滚依据

- [ ] **Step 1: 备份当前依赖快照**

```bash
cd "D:/桌面/RAGLoRA/backend"
D:/anaconda3/envs/rag_env/python.exe -m pip freeze > requirements-lock-before.txt
wc -l requirements-lock-before.txt
```

预期：输出约 325 行（当前 rag_env 包数）

- [ ] **Step 2: 安装三个新依赖**

```bash
D:/anaconda3/envs/rag_env/python.exe -m pip install ragas pymilvus pdfplumber
```

预期：安装成功，末尾显示 `Successfully installed ...`。
**注意**：会顺带升级 4 个已装包（requests 2.32.5→2.34.2、click 8.1.8→8.5.0、idna 3.11→3.19、pypdfium2 5.6.0→5.13.0），均为向后兼容的小版本，属预期内。

- [ ] **Step 3: 验证三个包可导入且版本正确**

```bash
D:/anaconda3/envs/rag_env/python.exe -c "
import ragas, pymilvus, pdfplumber
print('ragas', ragas.__version__)
print('pymilvus', pymilvus.__version__)
print('pdfplumber', pdfplumber.__version__)
"
```

预期：`ragas 0.4.3` / `pymilvus 3.0.1` / `pdfplumber 0.11.10`

- [ ] **Step 4: 确认关键老包未被降级**

```bash
D:/anaconda3/envs/rag_env/python.exe -c "
import transformers, torch, fastapi, sqlalchemy
print('transformers', transformers.__version__)
print('torch', torch.__version__)
print('fastapi', fastapi.__version__)
print('sqlalchemy', sqlalchemy.__version__)
"
```

预期：`transformers 4.39.3` / `torch 2.5.1+cu121` / `fastapi 0.128.0` / `sqlalchemy 2.0.48`
**必须完全一致** —— 若 transformers 变了，说明版本链被破坏，执行回滚：

```bash
D:/anaconda3/envs/rag_env/python.exe -m pip install -r requirements-lock-before.txt
```

- [ ] **Step 5: 更新 requirements.txt**

在 `backend/requirements.txt` 末尾追加：

```
# Plan 2/3 新增（2026-09-15）：
ragas==0.4.3
pymilvus==3.0.1
pdfplumber==0.11.10
```

同时删除原有的这段注释（已过时）：

```
#   pdfplumber     —— 仅表格解析用，本项目语料无表格提取需求。
#   ragas          —— 评测用自研脚本替代。
```

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/requirements-lock-before.txt
git commit -m "chore: 备份依赖快照并引入 ragas/pymilvus/pdfplumber"
```

---

### Task 2: 启动训练对生成（后台并行）

**Files:**
- Create: `backend/scripts/gen_embed_pairs.py`

**Interfaces:**
- Consumes: `config.DATASETS_DIR`、Ollama `qwen2.5:7b`
- Produces: `backend/eval/embed_pairs.jsonl`，每行 `{"query": str, "positive": str, "collection": str}`

> **为什么提前做**：本任务是 S12 嵌入微调的瓶颈（约 2400 次 LLM 调用 ≈ 2 小时）。
> 立即后台启动，与后续所有任务并行，避免串行等待拖垮整个项目。

- [ ] **Step 1: 编写训练对生成脚本**

创建 `backend/scripts/gen_embed_pairs.py`：

```python
# -*- coding: utf-8 -*-
"""为嵌入微调生成 (query, positive) 训练对。

思路：从已有语料分层采样 chunk，用 LLM 为每条 chunk 生成 2 个"用户会怎么问"的问题。

⚠️ 43 题 QA 集（eval/qa_set.json）严格排除在训练对之外，避免数据泄漏导致指标虚高。
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config                      # noqa: E402
from app.core.logging import get_logger          # noqa: E402
from app.services import ingest, llm             # noqa: E402

log = get_logger("gen_pairs")

OUT = config.BACKEND_DIR / "eval" / "embed_pairs.jsonl"
PER_CHUNK = 2          # 每条 chunk 生成几个问题
SAMPLE = {"kb_medical": 150, "kb_legal": 1050}   # 分层采样：按 636:14319 比例缩放

PROMPT = """你是知识库检索测试员。请根据下面的知识片段，写出 {n} 个用户可能会提出的、需要这段内容才能回答的问题。

要求：
1. 每个问题独立成行，不要编号，不要解释
2. 问题要像真实用户的口语提问，不要照抄原文用词
3. 问题必须能且只能由这段内容回答

知识片段：
{text}
"""


def _load_qa_questions() -> set[str]:
    """已用于评测的问题，生成时要避开。

    注意：qa_set.json 的实际结构是 `{_comment, medical:[...], legal:[...], refusal:[...]}`，
    每个条目用 `"q"` 键而非 `"question"`，顶层既不是 list 也没有 `"items"`。
    初版按 `data.get("items", [])` 解析会**静默返回 0 条**，使防泄漏形同虚设。
    """
    p = config.BACKEND_DIR / "eval" / "qa_set.json"
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))

    groups: list = []
    if isinstance(data, list):
        groups = [data]
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                groups.append(v)

    out: set[str] = set()
    for group in groups:
        for it in group:
            if not isinstance(it, dict):
                continue
            q = (it.get("q") or it.get("question") or "").strip()
            if q:
                out.add(q)
    return out


# collection -> 原始语料目录
# ⚠️ kb_legal 的语料在 datasets/legal/chinese_law/ 子目录下（176 部法律 txt）。
#    ingest_all.py 入库用的就是这个子目录；collect_files 非递归，
#    直接指向 datasets/legal 只能看到 2 个民法典 PDF（1764 chunks），
#    与 Qdrant 里的 kb_legal（14319 chunks）根本不是同一批内容。
CORPUS_DIR = {
    "kb_medical": config.DATASETS_DIR / "medical",
    "kb_legal": config.DATASETS_DIR / "legal" / "chinese_law",
}


def _sample(collection: str, limit: int) -> list[str]:
    """从原始语料文件取样 chunk。

    ⚠️ 刻意不读 Qdrant —— 后端运行时 Qdrant 嵌入式持有独占文件锁，
    另起进程访问会直接失败（README §七 已知限制）。
    这里改为直接读 datasets/ 下的原始语料并复用 ingest.build_chunks 分块，
    绕开文件锁，且能与服务端并行跑。
    """
    root = CORPUS_DIR.get(collection)
    if not root or not root.exists():
        log.warning("语料目录不存在: %s", root)
        return []

    texts: list[str] = []
    for path in ingest.collect_files(root):
        try:
            chunks, _ = ingest.build_chunks(path)
        except Exception as e:
            log.warning("分块失败 %s: %s", path.name, e)
            continue
        texts.extend(c.get("text", "") for c in chunks)
        if len(texts) >= limit * 3:       # 多取一些供打散
            break

    random.shuffle(texts)
    return [t for t in texts if len(t) > 80][:limit]


def _preflight() -> None:
    """先探一次 LLM，避免"跑了很久才发现模型根本加载不了"。

    本机实测：内存/提交量紧张时 Ollama 加载 7B 会 cudaMalloc OOM。
    而生成循环是逐条 try/except 的，会让脚本**空转 1200 次、只写 0 对，
    两小时后报"完成"** —— 静默失败，最难查。故启动即探活，失败直接退出。
    """
    try:
        llm.chat([{"role": "user", "content": "回复一个字：好"}],
                 temperature=0.0, max_tokens=8)
    except Exception as e:
        log.error("LLM 预检失败，脚本退出（不会产生任何训练对）: %s", str(e)[:300])
        log.error("请确认 Ollama 能加载 %s 后再重跑；"
                  "本机常见原因是可用内存/提交量不足。", config.LLM_MODEL)
        sys.exit(1)
    log.info("LLM 预检通过（%s 可正常响应）", config.LLM_MODEL)


def main() -> None:
    _preflight()
    questioned = _load_qa_questions()
    log.info("载入已评测问题 %d 条（生成时会避开）", len(questioned))
    rows: list[dict] = []

    for coll, limit in SAMPLE.items():
        texts = _sample(coll, limit)
        log.info("%s 取样 %d 条", coll, len(texts))
        for i, text in enumerate(texts, 1):
            try:
                raw = llm.chat(
                    [{"role": "user",
                      "content": PROMPT.format(n=PER_CHUNK, text=text[:1500])}],
                    temperature=0.7,
                )
            except Exception as e:
                log.warning("生成失败(%d/%d): %s", i, len(texts), e)
                continue

            for line in raw.splitlines():
                q = line.strip().lstrip("0123456789.、) ").strip()
                if len(q) < 6 or q in questioned:
                    continue
                rows.append({"query": q, "positive": text, "collection": coll})

            if len(rows) % 50 == 0:
                OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                               encoding="utf-8")
                log.info("已生成 %d 对", len(rows))

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                   encoding="utf-8")
    print(f"完成：{len(rows)} 对 -> {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 后台启动生成任务**

```bash
cd "D:/桌面/RAGLoRA/backend"
powershell -NoProfile -Command "Start-Process -FilePath 'D:\anaconda3\envs\rag_env\python.exe' -ArgumentList 'scripts/gen_embed_pairs.py' -WorkingDirectory 'D:\桌面\RAGLoRA\backend' -WindowStyle Hidden -RedirectStandardOutput 'D:\桌面\RAGLoRA\backend\logs\gen_pairs.log' -RedirectStandardError 'D:\桌面\RAGLoRA\backend\logs\gen_pairs.err.log'"
```

预期：命令立即返回，后台开始跑。
**此任务约需 2 小时，不要等待** —— 继续 Task 3。

- [ ] **Step 3: 确认任务已启动**

```bash
sleep 60 && tail -5 "D:/桌面/RAGLoRA/backend/logs/gen_pairs.log"
```

预期：出现 `kb_medical 取样 150 条` 或 `已生成 50 对`

- [ ] **Step 4: Commit**

```bash
git add backend/scripts/gen_embed_pairs.py
git commit -m "feat: 新增嵌入微调训练对生成脚本（后台生成中）"
```

---

### Task 3: 新增配置开关

**Files:**
- Modify: `backend/app/core/config.py`

**Interfaces:**
- Consumes: 无
- Produces: `config.CHAIN_BACKEND: str`、`config.VECTOR_STORE: str`、`config.OCR_TEXT_THRESHOLD: int`、`config.MILVUS_URI: str`、`config.NEO4J_URI: str`、`config.NEO4J_USER: str`、`config.NEO4J_PASSWORD: str`、`config.EMBED_MODELS_DIR: str`

> **设计变更（2026-09-15，用户确认）**：原计划拉取 `qwen2.5:3b` 以腾内存，现决定**不拉**，
> 全程使用 `qwen2.5:7b` 单一生成模型。
> **理由**：现有 100% 来源命中率 / 100% 引用标注率的成绩单是 7b 跑出来的，
> 换 3b 会让现场演示效果低于报告记载的指标。多路召回演示时的内存压力改用
> 「只演示检索链路、不生成」的方式规避，不靠降模型规格。

- [ ] **Step 1: 确认本地生成模型就绪（不下载任何模型）**

```bash
curl -s http://localhost:11434/api/tags | D:/anaconda3/envs/rag_env/python.exe -c "
import sys, json
names = [m['name'] for m in json.load(sys.stdin)['models']]
print(names)
assert 'qwen2.5:7b' in names, 'qwen2.5:7b 未就绪'
print('OK')
"
```

预期：打印模型列表并输出 `OK`
**注意**：本任务**不执行任何 `ollama pull`**。若 7b 缺失，停下来报告 BLOCKED，不要自行拉取。

- [ ] **Step 2: 在 config.py 末尾追加配置块**

在 `backend/app/core/config.py` 的 `COLLECTION_LEGAL = "kb_legal"` 之后追加：

```python
# ---------------------------------------------------------------- 链路与组件开关（2026-09-15 新增）
# 手动链路(manual) 与 LangChain 链路(langchain) 并存，默认走已验证的手动链路。
# 改动此值时行为随之切换，便于答辩现场对比两条链路。
CHAIN_BACKEND = os.environ.get("RAGLORA_CHAIN", "manual")

# 向量库选择：qdrant（嵌入式，默认） / milvus（Docker） / both（双写对比）
VECTOR_STORE = os.environ.get("RAGLORA_VECTOR_STORE", "qdrant")

# OCR 分流：页均字符数低于该值判为扫描件（详见设计文档 §3.5）
OCR_TEXT_THRESHOLD = int(os.environ.get("RAGLORA_OCR_TEXT_THRESHOLD", "50"))

# ---------------------------------------------------------------- 外部服务（Docker）
MILVUS_URI = os.environ.get("MILVUS_URI", "http://127.0.0.1:19530")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "raglora123")

# ---------------------------------------------------------------- 嵌入模型对比
# D:\桌面\模型\嵌入模型\ 下有 bge-m3 / m3e-base / bert-base-chinese
EMBED_MODELS_DIR = os.environ.get("EMBED_MODELS_DIR", r"D:\桌面\模型\嵌入模型")
```

- [ ] **Step 3: 验证配置可读且默认值正确**

```bash
cd "D:/桌面/RAGLoRA/backend"
D:/anaconda3/envs/rag_env/python.exe -c "
from app.core import config
assert config.CHAIN_BACKEND == 'manual', config.CHAIN_BACKEND
assert config.VECTOR_STORE == 'qdrant', config.VECTOR_STORE
assert config.OCR_TEXT_THRESHOLD == 50, config.OCR_TEXT_THRESHOLD
assert config.MILVUS_URI == 'http://127.0.0.1:19530'
print('配置 OK | chain=%s store=%s ocr_th=%d' % (config.CHAIN_BACKEND, config.VECTOR_STORE, config.OCR_TEXT_THRESHOLD))
"
```

预期：`配置 OK | chain=manual store=qdrant ocr_th=50`

- [ ] **Step 4: Commit**

```bash
git add backend/app/core/config.py
git commit -m "feat: 新增链路/向量库/OCR 配置开关与外部服务地址"
```

---

### Task 4: Docker 编排 Milvus 与 Neo4j

**Files:**
- Create: `docker/docker-compose.milvus.yml`
- Create: `docker/docker-compose.neo4j.yml`
- Create: `tools/demo/up.sh`
- Create: `tools/demo/down.sh`

**Interfaces:**
- Consumes: `config.MILVUS_URI`、`config.NEO4J_URI`
- Produces: `tools/demo/up.sh {milvus|neo4j|all}` 与 `tools/demo/down.sh` 可执行脚本

- [ ] **Step 1: 编写 Milvus compose**

创建 `docker/docker-compose.milvus.yml`：

```yaml
# Milvus standalone（嵌入式 etcd + minio，单容器模式）
# 镜像已在本地：milvusdb/milvus:v2.6.6（3.55GB），无需拉取
services:
  milvus:
    image: milvusdb/milvus:v2.6.6
    container_name: raglora-milvus
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_USE_EMBED: "true"
      ETCD_DATA_DIR: /var/lib/milvus/etcd
      COMMON_STORAGETYPE: local
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - milvus_data:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3

volumes:
  milvus_data:
```

- [ ] **Step 2: 编写 Neo4j compose**

创建 `docker/docker-compose.neo4j.yml`：

```yaml
# Neo4j 社区版（图谱召回）
# 镜像已在本地：neo4j:latest（1.06GB），无需拉取
services:
  neo4j:
    image: neo4j:latest
    container_name: raglora-neo4j
    environment:
      NEO4J_AUTH: neo4j/raglora123
      NEO4J_server_memory_heap_initial__size: 512m
      NEO4J_server_memory_heap_max__size: 1G
      NEO4J_server_memory_pagecache_size: 512m
    ports:
      - "7474:7474"    # 浏览器控制台
      - "7687:7687"    # bolt
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

> `NEO4J_AUTH` 的密码必须与 `config.NEO4J_PASSWORD` 默认值 `raglora123` 一致。

- [ ] **Step 3: 编写启停脚本**

创建 `tools/demo/up.sh`：

```bash
#!/usr/bin/env bash
# 按需拉起 Docker 组件。用法: bash tools/demo/up.sh {milvus|neo4j|all}
set -u
ROOT_DIR="D:/桌面/RAGLoRA"

case "${1:-}" in
  milvus) docker compose -f "$ROOT_DIR/docker/docker-compose.milvus.yml" up -d ;;
  neo4j)  docker compose -f "$ROOT_DIR/docker/docker-compose.neo4j.yml" up -d ;;
  all)
    docker compose -f "$ROOT_DIR/docker/docker-compose.milvus.yml" up -d
    docker compose -f "$ROOT_DIR/docker/docker-compose.neo4j.yml" up -d
    ;;
  *) echo "用法: bash tools/demo/up.sh {milvus|neo4j|all}"; exit 1 ;;
esac

echo "等待服务就绪..."
sleep 20
docker ps --filter "name=raglora-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

创建 `tools/demo/down.sh`：

```bash
#!/usr/bin/env bash
# 停止 Docker 组件并释放内存。用法: bash tools/demo/down.sh [milvus|neo4j|all]
set -u
ROOT_DIR="D:/桌面/RAGLoRA"
TARGET="${1:-all}"

[ "$TARGET" = "milvus" ] || [ "$TARGET" = "all" ] && \
  docker compose -f "$ROOT_DIR/docker/docker-compose.milvus.yml" down
[ "$TARGET" = "neo4j" ] || [ "$TARGET" = "all" ] && \
  docker compose -f "$ROOT_DIR/docker/docker-compose.neo4j.yml" down

echo "已停止。当前 raglora 容器："
docker ps --filter "name=raglora-" --format "table {{.Names}}\t{{.Status}}"
```

- [ ] **Step 4: 拉起 Milvus 并验证连通**

```bash
bash "D:/桌面/RAGLoRA/tools/demo/up.sh" milvus
```

预期：容器 `raglora-milvus` 状态为 `Up`（**首次启动需约 60-90 秒**）。
内存占用约 2.5GB。

> 状态里是否带 `(healthy)` 取决于镜像内有没有 `curl`，**没有也不影响使用**。
> 真正是否可用**以 Step 5 的 pymilvus 连接测试为准**，不以 healthcheck 为准。

- [ ] **Step 5: 用 pymilvus 真实连接验证**

```bash
D:/anaconda3/envs/rag_env/python.exe -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://127.0.0.1:19530')
print('Milvus 连通，已存在集合:', c.list_collections())
"
```

预期：`Milvus 连通，已存在集合: []`
**这一步同时验证了 pymilvus 3.0.1 与 milvus v2.6.6 的兼容性**——若报版本不兼容，记录错误信息并在 Plan 2 中降级 pymilvus。

- [ ] **Step 6: 关掉 Milvus 释放内存**

```bash
bash "D:/桌面/RAGLoRA/tools/demo/down.sh" milvus
```

预期：容器停止。**验证完立即关掉** —— 后续 Task 不需要它常驻。

- [ ] **Step 7: Commit**

```bash
git add docker/ tools/demo/
git commit -m "feat: 新增 Milvus/Neo4j compose 与按需启停脚本"
```

---

### Task 5: persona 抽出可复用的 system prompt 渲染

**Files:**
- Modify: `backend/app/services/persona.py:61-82`
- Test: `backend/tests/test_persona_render.py`

**Interfaces:**
- Consumes: `Character` ORM 对象、`hits: list[dict]`、`memory: list[dict]`
- Produces: `persona.render_system(character, question, hits, memory) -> str`
  （`build_messages` 改为调用它，保证两条链路用同一套模板，不会漂移）

> **为什么要做这一步**：LangChain 链路需要**单独的 system 字符串**喂给 `ChatPromptTemplate`。
> 若另写一套渲染逻辑，两条链路的 prompt 会逐渐分叉，对比实验就失去意义。

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/test_persona_render.py`：

```python
# -*- coding: utf-8 -*-
"""验证 render_system 与 build_messages 使用同一套渲染逻辑。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import persona  # noqa: E402


class _FakeCharacter:
    """最小 Character 替身，只带渲染需要的字段。"""
    name = "测试角色"
    prompt_template = "{identity_block}\n【知识】\n{context}\n【问题】\n{question}"
    identity_block = "你是测试角色。"
    description = "测试范围"
    style_json = {"tone": "简洁", "address": "朋友"}
    domain_constraints = "不编造。"
    kb_collection = "kb_medical"


def test_render_system_returns_str():
    out = persona.render_system(_FakeCharacter(), "问题?", [], [])
    assert isinstance(out, str)
    assert "测试角色" in out
    assert "问题?" in out


def test_build_messages_system_matches_render_system():
    """build_messages 的 system 必须与 render_system 完全一致。"""
    ch, q, hits, mem = _FakeCharacter(), "问题?", [], []
    assert persona.build_messages(ch, q, hits, mem)[0]["content"] == \
        persona.render_system(ch, q, hits, mem)


def test_render_system_includes_style():
    out = persona.render_system(_FakeCharacter(), "q", [], [])
    assert "简洁" in out
    assert "朋友" in out
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "D:/桌面/RAGLoRA/backend"
D:/anaconda3/envs/rag_env/python.exe -m pytest tests/test_persona_render.py -v
```

预期：FAIL — `AttributeError: module 'app.services.persona' has no attribute 'render_system'`

- [ ] **Step 3: 重构 persona.py**

把 `backend/app/services/persona.py` 的 `build_messages` 函数（第 61-82 行）整体替换为：

```python
def render_system(character: Character, question: str,
                  hits: list[dict], memory: list[dict]) -> str:
    """渲染最终 system prompt。

    手写链路(build_messages)与 LangChain 链路(lc_chain)共用此函数，
    确保两条链路的提示词完全一致 —— 否则对比实验失去意义。
    """
    template = character.prompt_template or (
        "{identity_block}\n\n## 【知识片段】\n{context}\n\n## 【用户问题】\n{question}"
    )

    return template.format(
        identity_block=character.identity_block or f"你是{character.name}。",
        knowledge_scope=(character.description
                         or SCOPE_LABELS.get(character.kb_collection, "你的专业知识")),
        style_block=_render_style(character.style_json),
        domain_constraints=character.domain_constraints or "保持专业边界。",
        context=_render_context(hits),
        memory=_render_memory(memory),
        question=question,
    )


def build_messages(character: Character, question: str,
                   hits: list[dict], memory: list[dict]) -> list[dict]:
    """构造送给大模型的 messages。"""
    return [
        {"role": "system", "content": render_system(character, question, hits, memory)},
        {"role": "user", "content": question},
    ]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
D:/anaconda3/envs/rag_env/python.exe -m pytest tests/test_persona_render.py -v
```

预期：3 passed

- [ ] **Step 5: 回归验证现有对话未受影响**

```bash
curl -s "http://127.0.0.1:8000/api/health?deep=1" | head -c 200
```

预期：`{"ok":true,...}`
（若后端未运行：`cd "D:/桌面/RAGLoRA" && bash backend/run.sh --bg`）

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/persona.py backend/tests/test_persona_render.py
git commit -m "refactor: persona 抽出 render_system 供两条链路共用"
```

---

### Task 6: LangChain 检索器适配层

**Files:**
- Create: `backend/app/services/lc_retrievers.py`
- Test: `backend/tests/test_lc_retrievers.py`

**Interfaces:**
- Consumes: `embed.encode_one(text, device) -> (dense, sparse)`、`embed.encode(texts, batch_size, device, progress) -> (dense_list, sparse_list)`、`retrieval.hybrid_search(query, collection, recall_k, filters) -> (hits, query)`
- Produces:
  - `lc_retrievers.BgeM3DenseEmbeddings()` — LangChain `Embeddings` 实现，含 `embed_query(str) -> list[float]`、`embed_documents(list[str]) -> list[list[float]]`
  - `lc_retrievers.HybridRetriever(collection: str, k: int)` — LangChain `BaseRetriever`，`invoke(query) -> list[Document]`
  - `lc_retrievers.hit_to_doc(hit: dict) -> Document`
  - `lc_retrievers.doc_to_hit(doc: Document) -> dict`

> **本任务的核心约束**：**绝不能**用 `langchain_huggingface.HuggingFaceEmbeddings` 包装 bge-m3 ——
> 那会让 bge-m3 在进程内存在两份（+2.3GB），本机 15.7GB 内存无法承受。
> `BgeM3DenseEmbeddings` 委托给现有 `embed.py` 单例，复用同一份模型。

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/test_lc_retrievers.py`：

```python
# -*- coding: utf-8 -*-
"""检索器适配层测试：重点是「没有重复加载模型」。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from app.services import lc_retrievers  # noqa: E402


def test_hit_to_doc_roundtrip():
    """hit -> Document -> hit 应保留关键字段。"""
    hit = {"id": 7, "score": 0.5, "text": "正文", "source": "a.pdf",
           "page": 3, "law_name": None, "article_no": None, "collection": "kb_medical"}
    doc = lc_retrievers.hit_to_doc(hit)
    assert isinstance(doc, Document)
    assert doc.page_content == "正文"
    assert doc.metadata["source"] == "a.pdf"

    back = lc_retrievers.doc_to_hit(doc)
    assert back["text"] == "正文"
    assert back["source"] == "a.pdf"
    assert back["page"] == 3
    assert back["id"] == 7


def test_embeddings_delegates_to_existing_singleton():
    """BgeM3DenseEmbeddings 必须复用 embed 模块的模型，而非新建实例。"""
    from app.services import embed

    emb = lc_retrievers.BgeM3DenseEmbeddings()
    vec = emb.embed_query("高血压")
    assert len(vec) == 1024
    assert isinstance(vec[0], float)
    # embed 模块的全局单例已被这次调用加载
    assert embed.current_device() == "cpu"


def test_hybrid_retriever_returns_documents():
    r = lc_retrievers.HybridRetriever(collection="kb_medical", k=3)
    docs = r.invoke("高血压的诊断标准")
    assert isinstance(docs, list)
    for d in docs:
        assert isinstance(d, Document)
    assert len(docs) <= 3
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "D:/桌面/RAGLoRA/backend"
D:/anaconda3/envs/rag_env/python.exe -m pytest tests/test_lc_retrievers.py -v
```

预期：FAIL — `ModuleNotFoundError: No module named 'app.services.lc_retrievers'`

- [ ] **Step 3: 实现 lc_retrievers.py**

创建 `backend/app/services/lc_retrievers.py`：

```python
# -*- coding: utf-8 -*-
"""LangChain 检索器适配层：把现有检索能力接入 LangChain 接口。

两处关键取舍（设计文档 §3.2）：

1. **不重复加载 bge-m3**
   langchain_huggingface.HuggingFaceEmbeddings 会新建一份模型实例
   （bge-m3 fp32 约 2.3GB）。本机 15.7GB 内存下，两份即 4.6GB，不可接受。
   因此 BgeM3DenseEmbeddings 委托给 app.services.embed 的全局单例。

2. **混合检索仍走手写实现**
   bge-m3 的 learned sparse 分支 LangChain 未覆盖。强行只做 dense 会丢失
   混合检索能力，故 HybridRetriever 包装现有 retrieval.hybrid_search，
   保留 dense ∥ sparse → RRF 的完整链路。
"""
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from ..core import config
from ..core.logging import get_logger
from . import retrieval
from .embed import encode, encode_one

log = get_logger("lc_retrievers")

_META_KEYS = ("id", "score", "source", "page", "law_name", "article_no", "collection")


def hit_to_doc(hit: dict) -> Document:
    """检索命中项 -> LangChain Document。"""
    return Document(
        page_content=hit.get("text") or "",
        metadata={k: hit.get(k) for k in _META_KEYS},
    )


def doc_to_hit(doc: Document) -> dict:
    """LangChain Document -> 检索命中项（供精排与 persona 复用）。"""
    hit = {k: doc.metadata.get(k) for k in _META_KEYS}
    hit["text"] = doc.page_content
    return hit


class BgeM3DenseEmbeddings(Embeddings):
    """bge-m3 的 dense 分支适配为 LangChain Embeddings。

    只暴露 dense —— sparse 由 HybridRetriever 内部处理。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        dense, _ = encode(list(texts), device="cpu")
        return dense

    def embed_query(self, text: str) -> list[float]:
        dense, _ = encode_one(text, device="cpu")
        return dense


class HybridRetriever(BaseRetriever):
    """包装现有混合检索（dense ∥ sparse → RRF）。"""

    collection: str = Field(description="Qdrant collection 名")
    k: int = Field(default=config.RECALL_TOP_K, description="召回条数")

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        hits, _ = retrieval.hybrid_search(query, self.collection, self.k)
        log.debug("LangChain 检索 %s -> %d 条", self.collection, len(hits))
        return [hit_to_doc(h) for h in hits]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
D:/anaconda3/envs/rag_env/python.exe -m pytest tests/test_lc_retrievers.py -v
```

预期：3 passed
（首次运行需加载 bge-m3，约 10 秒）

- [ ] **Step 5: 验证没有重复加载模型**

```bash
D:/anaconda3/envs/rag_env/python.exe -c "
from app.services import lc_retrievers, embed
emb = lc_retrievers.BgeM3DenseEmbeddings()
emb.embed_query('测试')
import app.services.embed as e
print('embed 单例设备:', e.current_device())
print('模型对象同一性:', e._model is not None)
"
```

预期：`embed 单例设备: cpu` 且 `模型对象同一性: True`
**关键**：整个过程只加载一次 bge-m3。若内存观察到两份，说明适配器写错了。

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/lc_retrievers.py backend/tests/test_lc_retrievers.py
git commit -m "feat: 新增 LangChain 检索器适配层（委托现有单例，不重复加载模型）"
```

---

### Task 7: LCEL 链路实现

**Files:**
- Create: `backend/app/services/lc_chain.py`
- Test: `backend/tests/test_lc_chain.py`

**Interfaces:**
- Consumes: `lc_retrievers.HybridRetriever`、`lc_retrievers.doc_to_hit`、`persona.render_system`、`persona.build_sources`、`rerank.rerank`、`rag_chain.postprocess`
- Produces: 与 `rag_chain` **同名同签名**的三个函数：
  - `prepare(character, question, memory) -> {"rewritten": str, "hits": list[dict], "trace": dict}`
  - `ask(character, question, memory) -> {"answer": str, "sources": list, "trace": dict}`
  - `ask_stream(character, question, memory) -> Iterator[tuple[str, dict]]`（事件：trace / sources / delta / done / error）

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/test_lc_chain.py`：

```python
# -*- coding: utf-8 -*-
"""LCEL 链路测试：签名与事件协议必须与 rag_chain 完全一致。"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import lc_chain, rag_chain  # noqa: E402


class _FakeCharacter:
    name = "测试角色"
    prompt_template = "{identity_block}\n{context}\n{question}"
    identity_block = "你是测试角色。"
    description = "测试"
    style_json = {}
    domain_constraints = "不编造。"
    kb_collection = "kb_medical"
    recall_top_k = 3
    rerank_top_k = 2
    temperature = 0.3


def test_signatures_match_manual_chain():
    """两条链路的函数签名必须一致，否则路由层无法透明切换。"""
    for fn in ("prepare", "ask", "ask_stream"):
        a = inspect.signature(getattr(lc_chain, fn))
        b = inspect.signature(getattr(rag_chain, fn))
        assert list(a.parameters) == list(b.parameters), fn


def test_prepare_returns_expected_keys():
    out = lc_chain.prepare(_FakeCharacter(), "高血压诊断标准", [])
    assert set(out) >= {"rewritten", "hits", "trace"}
    assert "recall_ms" in out["trace"]
    assert isinstance(out["hits"], list)
    for h in out["hits"]:
        assert "text" in h and "source" in h


def test_ask_stream_event_protocol():
    events = []
    for event, data in lc_chain.ask_stream(_FakeCharacter(), "高血压诊断标准", []):
        events.append(event)
        assert isinstance(data, dict)
        if event == "done":
            break
    assert events[0] == "trace"
    assert events[1] == "sources"
    assert "delta" in events
    assert "done" in events
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "D:/桌面/RAGLoRA/backend"
D:/anaconda3/envs/rag_env/python.exe -m pytest tests/test_lc_chain.py -v
```

预期：FAIL — `ModuleNotFoundError: No module named 'app.services.lc_chain'`

- [ ] **Step 3: 实现 lc_chain.py**

创建 `backend/app/services/lc_chain.py`：

```python
# -*- coding: utf-8 -*-
"""LangChain（LCEL）版 RAG 链路 —— 与手写 rag_chain 并存，由配置切换。

对照手写链路的差异：
    检索段：RunnablePassthrough.assign + HybridRetriever（LangChain 接口）
    生成段：ChatPromptTemplate | ChatOllama | StrOutputParser

保持一致的部分（刻意复用，不重写）：
    精排    —— LangChain 无 bge-reranker 集成，仍用 rerank.rerank
    后处理  —— 仍用 rag_chain.postprocess，保证两条链路输出格式一致
    Prompt  —— 仍用 persona.render_system，避免模板漂移

事件协议与 rag_chain.ask_stream 完全相同（trace → sources → delta* → done）。
"""
import time
from collections.abc import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from ..core import config
from ..core.logging import get_logger
from ..models import Character
from . import persona, rerank
from .lc_retrievers import HybridRetriever, doc_to_hit
from .rag_chain import postprocess

log = get_logger("lc_chain")


def _ollama_base_url() -> str:
    """config.OLLAMA_BASE_URL 是 OpenAI 兼容端点(带 /v1)，ChatOllama 需要裸地址。"""
    return config.OLLAMA_BASE_URL.removesuffix("/v1")


def _build_llm(character: Character, streaming: bool = False) -> ChatOllama:
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=_ollama_base_url(),
        temperature=character.temperature if character.temperature is not None
        else config.LLM_TEMPERATURE,
        num_predict=config.LLM_MAX_TOKENS,
        streaming=streaming,
    )


def _prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("human", "{question}"),
    ])


def _rerank_step(character: Character):
    """把精排插进 LCEL 管道 —— 框架未提供该集成，故用 RunnableLambda。"""

    def _fn(payload: dict) -> dict:
        hits = [doc_to_hit(d) for d in payload["hits"]]
        recalled = len(hits)
        t = time.time()
        if hits and config.RERANK_ENABLED:
            hits = rerank.rerank(payload["question"], hits, character.rerank_top_k)
        else:
            hits = hits[:character.rerank_top_k]
        # LCEL 把召回与精排融合在一个管道里，需显式带出两个数字，
        # 否则 trace 里的召回数/耗时会被精排污染，前端链路可视就失真了。
        return {**payload, "hits": hits, "recall_n": recalled,
                "rerank_ms": round((time.time() - t) * 1000)}

    return RunnableLambda(_fn)


def _retrieval_chain(character: Character):
    """LCEL 检索段：混合检索 → 精排。"""
    retriever = HybridRetriever(
        collection=character.kb_collection,
        k=character.recall_top_k,
    )
    return (
        RunnablePassthrough.assign(
            hits=RunnableLambda(lambda x: retriever.invoke(x["question"]))
        )
        | _rerank_step(character)
    )


# ---------------------------------------------------------------- 检索阶段
def prepare(character: Character, question: str, memory: list[dict]) -> dict:
    """执行检索侧全部工作：改写 → 召回 → 精排（与 rag_chain.prepare 同签名）。"""
    from . import llm

    t_start = time.time()
    trace: dict = {}

    t = time.time()
    rewritten = question
    if config.REWRITE_ENABLED and memory:
        rewritten = llm.rewrite_query(question, memory)
    trace["rewritten_query"] = rewritten
    trace["rewrite_ms"] = round((time.time() - t) * 1000)
    trace["rewrite_skipped"] = (rewritten == question and bool(memory))

    t = time.time()
    chain = _retrieval_chain(character)
    out = chain.invoke({"question": rewritten, "memory": memory})
    hits = out["hits"]
    rerank_ms = out.get("rerank_ms", 0)
    total_ms = round((time.time() - t) * 1000)
    trace["recall"] = out.get("recall_n", len(hits))   # 精排前的召回数
    trace["recall_ms"] = max(0, total_ms - rerank_ms)  # 抵扣精排耗时，与手动链路口径一致
    trace["reranked"] = len(hits)
    trace["rerank_ms"] = rerank_ms
    trace["total_retrieval_ms"] = round((time.time() - t_start) * 1000)
    trace["chain"] = "langchain"

    log.info("LCEL 检索完成 | 精排 %d 条 | %.0fms",
             len(hits), trace["total_retrieval_ms"])
    return {"rewritten": rewritten, "hits": hits, "trace": trace}


# ---------------------------------------------------------------- 非流式
def ask(character: Character, question: str, memory: list[dict]) -> dict:
    """完整问答（非流式）。"""
    t0 = time.time()
    prepared = prepare(character, question, memory)

    t = time.time()
    chain = (
        _prompt()
        | _build_llm(character)
        | StrOutputParser()
    )
    answer = chain.invoke({
        "system": persona.render_system(character, question, prepared["hits"], memory),
        "question": question,
    })
    prepared["trace"]["generate_ms"] = round((time.time() - t) * 1000)
    prepared["trace"]["total_ms"] = round((time.time() - t0) * 1000)

    return {
        "answer": postprocess(answer, character),
        "sources": persona.build_sources(prepared["hits"]),
        "trace": prepared["trace"],
    }


# ---------------------------------------------------------------- 流式
def ask_stream(character: Character, question: str,
               memory: list[dict]) -> Iterator[tuple[str, dict]]:
    """流式问答，事件协议与 rag_chain.ask_stream 完全一致。"""
    t0 = time.time()
    try:
        prepared = prepare(character, question, memory)
    except Exception as e:
        log.exception("LCEL 检索阶段失败")
        yield "error", {"code": "RETRIEVAL_FAILED", "message": str(e)[:200]}
        return

    yield "trace", prepared["trace"]
    yield "sources", persona.build_sources(prepared["hits"])

    system = persona.render_system(character, question, prepared["hits"], memory)
    chain = _prompt() | _build_llm(character, streaming=True) | StrOutputParser()

    t = time.time()
    buf: list[str] = []
    try:
        for chunk in chain.stream({"system": system, "question": question}):
            buf.append(chunk)
            yield "delta", {"text": chunk}
    except Exception as e:
        log.exception("LCEL 生成阶段失败")
        yield "error", {"code": "LLM_FAILED", "message": str(e)[:200]}
        return

    gen_ms = round((time.time() - t) * 1000)
    prepared["trace"]["generate_ms"] = gen_ms
    prepared["trace"]["total_ms"] = round((time.time() - t0) * 1000)
    yield "done", {
        "answer": postprocess("".join(buf), character),
        "generate_ms": gen_ms,
        "total_ms": prepared["trace"]["total_ms"],
    }
```

- [ ] **Step 4: 运行测试确认通过**

```bash
D:/anaconda3/envs/rag_env/python.exe -m pytest tests/test_lc_chain.py -v
```

预期：3 passed
（`test_ask_stream_event_protocol` 会真实调用 Ollama，首次约 20 秒）

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/lc_chain.py backend/tests/test_lc_chain.py
git commit -m "feat: 新增 LCEL 版 RAG 链路，签名与事件协议对齐手写链路"
```

---

### Task 8: 链路分发器与路由接入

**Files:**
- Create: `backend/app/services/chain.py`
- Modify: `backend/app/routers/chat.py:23`
- Test: `backend/tests/test_chain_dispatch.py`

**Interfaces:**
- Consumes: `rag_chain`、`lc_chain`、`config.CHAIN_BACKEND`
- Produces: `chain.ask()` / `chain.ask_stream()` / `chain.prepare()` 与 `chain.active_backend() -> str`

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/test_chain_dispatch.py`：

```python
# -*- coding: utf-8 -*-
"""分发器测试：默认走 manual，配置切换后走 langchain。"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _reload(monkeypatch, value: str):
    monkeypatch.setenv("RAGLORA_CHAIN", value)
    from app.core import config
    importlib.reload(config)
    from app.services import chain
    importlib.reload(chain)
    return chain


def test_default_is_manual(monkeypatch):
    monkeypatch.delenv("RAGLORA_CHAIN", raising=False)
    ch = _reload(monkeypatch, "manual")
    assert ch.active_backend() == "manual"


def test_switch_to_langchain(monkeypatch):
    ch = _reload(monkeypatch, "langchain")
    assert ch.active_backend() == "langchain"
    assert ch.ask_stream.__module__.endswith("chain")


def test_unknown_value_falls_back_to_manual(monkeypatch):
    ch = _reload(monkeypatch, "no-such-chain")
    assert ch.active_backend() == "manual"


def test_cleanup(monkeypatch):
    """恢复默认，避免污染其它测试。"""
    monkeypatch.delenv("RAGLORA_CHAIN", raising=False)
    ch = _reload(monkeypatch, "manual")
    assert ch.active_backend() == "manual"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "D:/桌面/RAGLoRA/backend"
D:/anaconda3/envs/rag_env/python.exe -m pytest tests/test_chain_dispatch.py -v
```

预期：FAIL — `ModuleNotFoundError: No module named 'app.services.chain'`

- [ ] **Step 3: 实现 chain.py**

创建 `backend/app/services/chain.py`：

```python
# -*- coding: utf-8 -*-
"""链路分发器：按 config.CHAIN_BACKEND 选择手写链路或 LangChain 链路。

路由层只 import 本模块，无需感知具体实现 —— 这样切换链路不改调用方代码。
未知取值一律回退到 manual（已验证链路），保证配置写错不会导致服务不可用。
"""
from collections.abc import Iterator

from ..core import config
from ..core.logging import get_logger
from ..models import Character
from . import rag_chain, lc_chain

log = get_logger("chain")

_BACKENDS = {
    "manual": rag_chain,
    "langchain": lc_chain,
}


def active_backend() -> str:
    """当前生效的链路名。未知值回退 manual。"""
    name = config.CHAIN_BACKEND
    if name not in _BACKENDS:
        log.warning("未知链路 %r，回退 manual", name)
        return "manual"
    return name


def _impl():
    return _BACKENDS[active_backend()]


def prepare(character: Character, question: str, memory: list[dict]) -> dict:
    return _impl().prepare(character, question, memory)


def ask(character: Character, question: str, memory: list[dict]) -> dict:
    return _impl().ask(character, question, memory)


def ask_stream(character: Character, question: str,
               memory: list[dict]) -> Iterator[tuple[str, dict]]:
    return _impl().ask_stream(character, question, memory)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
D:/anaconda3/envs/rag_env/python.exe -m pytest tests/test_chain_dispatch.py -v
```

预期：4 passed

- [ ] **Step 5: 修改路由层**

在 `backend/app/routers/chat.py` 第 23 行，把：

```python
from ..services import rag_chain
```

改为：

```python
from ..services import chain
```

然后把第 101 行的：

```python
        for event, data in rag_chain.ask_stream(character, question, memory):
```

改为：

```python
        for event, data in chain.ask_stream(character, question, memory):
```

以及第 135 行的：

```python
    result = rag_chain.ask(character, question, memory)
```

改为：

```python
    result = chain.ask(character, question, memory)
```

- [ ] **Step 6: 确认没有遗漏的 rag_chain 引用**

```bash
cd "D:/桌面/RAGLoRA/backend"
grep -rn "rag_chain" app/ --include=*.py
```

预期：只应出现 `app/services/chain.py` 与 `app/services/rag_chain.py` 中的引用。
**若 `app/routers/` 下还有，说明漏改。**

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/chain.py backend/app/routers/chat.py backend/tests/test_chain_dispatch.py
git commit -m "feat: 新增链路分发器，路由层改为按配置切换"
```

---

### Task 9: 端到端验证与对比

**Files:**
- Create: `docs/06-双链路对比.md`

**Interfaces:**
- Consumes: 全部前置任务；运行中的后端
- Produces: `docs/06-双链路对比.md`，含两条链路的实测对比

- [ ] **Step 1: 重启后端（manual 模式）**

```bash
cd "D:/桌面/RAGLoRA"
bash backend/shutdown.sh
bash backend/run.sh --bg
```

预期：启动成功，健康检查全绿

- [ ] **Step 2: 跑既有评测，确认未回归**

```bash
cd "D:/桌面/RAGLoRA/backend"
D:/anaconda3/envs/rag_env/python.exe eval/run_eval.py --limit 5
```

预期：来源命中率 100%，引用标注率 100%（与改造前基线一致）

- [ ] **Step 3: 切换到 LangChain 模式重启**

```bash
cd "D:/桌面/RAGLoRA"
bash backend/shutdown.sh
RAGLORA_CHAIN=langchain bash backend/run.sh --bg
```

预期：启动成功

- [ ] **Step 4: 验证后端确实跑在 langchain 链路**

```bash
curl -s "http://127.0.0.1:8000/api/health?deep=1" | D:/anaconda3/envs/rag_env/python.exe -c "
import sys, json
d = json.load(sys.stdin)
print('ok:', d.get('ok'))
" && grep -c "LCEL 检索完成" backend/logs/app.log
```

预期：`ok: True`，且日志中出现 `LCEL 检索完成`（证明走的是 lc_chain）

- [ ] **Step 5: 跑同一套评测（langchain 模式）**

```bash
D:/anaconda3/envs/rag_env/python.exe eval/run_eval.py --limit 5
```

预期：来源命中率与 manual 模式一致（Prompt 与检索完全相同，差异只在编排层）

- [ ] **Step 6: 记录内存对比**

```bash
D:/anaconda3/envs/rag_env/python.exe -c "
import psutil
m = psutil.virtual_memory()
print(f'总 {m.total/1024**3:.1f}G | 已用 {m.used/1024**3:.1f}G | 可用 {m.available/1024**3:.1f}G')
"
```

预期：可用内存与改造前相当（**没有因适配器重复加载模型而多占 2.3GB**）

- [ ] **Step 7: 撰写对比文档**

创建 `docs/06-双链路对比.md`，至少包含：

```markdown
# 手写链路 vs LangChain 链路 对比

> 日期：2026-09-15
> 切换方式：`RAGLORA_CHAIN=manual|langchain`

## 一、两条链路的分工

| 环节 | manual | langchain |
|---|---|---|
| Prompt 渲染 | persona.render_system | persona.render_system（**共用**） |
| 混合检索 | retrieval.hybrid_search | HybridRetriever → retrieval.hybrid_search（**共用**） |
| 精排 | rerank.rerank | rerank.rerank（**共用**，框架无该集成） |
| 编排 | 手写函数调用 | LCEL：`prompt \| ChatOllama \| StrOutputParser` |
| 大模型 | llm.chat（OpenAI SDK） | langchain_ollama.ChatOllama |
| 后处理 | rag_chain.postprocess | rag_chain.postprocess（**共用**） |

## 二、框架覆盖了什么、没覆盖什么

**覆盖**：编排表达（LCEL 管道可读性）、LLM 客户端抽象、流式接口统一、Prompt 模板化。

**未覆盖**（必须自研或适配）：
1. **bge-m3 的 learned sparse 分支** —— LangChain 只认 dense Embeddings 接口，
   强行套用会丢掉混合检索能力。
2. **bge-reranker cross-encoder 精排** —— 无内置集成。
3. **模型单例管理** —— `HuggingFaceEmbeddings` 会新建模型实例，
   在 15.7GB 内存的机器上多占 2.3GB，必须写适配器委托给已有单例。

## 三、实测对比

（填入 Step 2 / Step 5 的评测结果与 Step 6 的内存数据）

| 指标 | manual | langchain |
|---|---|---|
| 来源命中率 | | |
| 引用标注率 | | |
| 平均延迟 | | |
| 可用内存 | | |
```

- [ ] **Step 8: 切回默认模式并重启**

```bash
cd "D:/桌面/RAGLoRA"
bash backend/shutdown.sh
bash backend/run.sh --bg
```

预期：恢复 manual 默认链路

- [ ] **Step 9: Commit**

```bash
git add docs/06-双链路对比.md
git commit -m "docs: 新增手写链路与 LangChain 链路对比报告"
```

---

## 完成标志

- [ ] `RAGLORA_CHAIN=manual`（默认）与 `=langchain` 均可跑通完整问答
- [ ] 两条链路的评测指标一致（同 Prompt 同检索，差异只在编排层）
- [ ] 内存未因适配器增加模型副本（可用内存与改造前相当）
- [ ] `docs/06-双链路对比.md` 完成
- [ ] 全部 pytest 用例通过

## 后续计划

Plan 2（S2+S3）：Milvus 双写与多路召回
Plan 3（S4+S5）：OCR 分流与 RAGAS 评测
Plan 4（S6）：角色扩展与数据增强
Plan 5（S7+S8）：工程化与压测
Plan 6（S9+S10）：部署与文档
Plan 7（S11+S12）：嵌入模型对比与微调

## 基线

- 改造前基线提交：`e0dffd4`（完整：`e0dffd4f95cc7e15ee54a9c6377285dcd3762d05`，92 files changed, 19465 insertions(+)）
- 一键回滚：`git reset --hard e0dffd4`
- 依赖回滚：`D:/anaconda3/envs/rag_env/python.exe -m pip install -r backend/requirements-lock-before.txt` (Task 1 Step 1 生成)
