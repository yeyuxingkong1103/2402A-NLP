# 招股说明书图文问答系统 · 图像内容解析及检索优化

> **工单编号：人工智能NLP-RAG-图像内容解析及检索优化**
> 工单依据：《人工智能NLP-RAG项目-04-PDF文档的图像内容解析及检索优化任务工单V1.1-20250206.pdf》
> 技术栈：Python 3.11 · FastAPI + Vue · MinerU(+pymupdf/pdfplumber/paddleocr) · 语义分块 · bge-m3 · Milvus · BM25 · bge-reranker-v2-m3 · DeepSeek · Redis · RAGAS

---

## 1. 这个工单要解决什么

在工单 01/02 构建的 PDF 问答系统上，**新增《招股说明书2.pdf》**（武汉力源信息技术股份有限公司）的解析，
并用**多模态模型实现图像内容解析**，以提升检索精度，回答工单指定的 16 个问题。

### 为什么必须做图像解析：工单第 6 题只存在于图里

工单指定的 16 题中，**id=6 的答案完全在图上**（已实测核对）：

> **id=6**：从「2008年中国IC市场应用结构与增长图」可以看出，增长率最快的是哪个行业？负增长的是哪个行业？
> **答案**：增长率最快的是**汽车**（14.0%）；负增长的是**IC卡**（-2.0%）。

在《招股说明书2》全文的文本层与表格体字段中检索：**`IC卡` 出现 0 次、`-2.0` 出现 0 次**。
正文只写了"工业控制增长率 10.5%，位列第二，仅次于汽车电子"——推不出"汽车 14.0%"，更推不出"IC卡 -2.0%"。
**不做图像解析，这一题无解。**

<img src="docs/screenshots/图像题目-IC市场增长图.png" width="620" alt="2008年中国IC市场应用结构与增长图">

---

## 2. 图像内容解析怎么做的（工单核心）

三段式，全部实测验证过：

| 步骤 | 用的模型/方法 | 作用 | 实测 |
|---|---|---|---|
| ① 语义分类 | **Chinese-CLIP**（多模态模型，工单备注指定 CLIP 或多模态大模型） | 零样本判定图像语义类型 | 组织结构图 0.999、饼图 0.823、折线图 0.878 |
| ② 图内文字 | **PaddleOCR**（**放大 3×** 后识别） | 读出图里的文字与数字 | 1× 会把 `-2.0%` 读成 `2.0%`（负号丢失），3× 才稳定读出 |
| ③ 层级还原 | 像素几何：**横总线 + 下垂竖线** | 还原"谁下辖谁" | 正确还原「大客户销售部 下辖 6 个销售处」 |

第 ③ 步是自研的：组织结构图的画法有固定规律——父块下方引出一条**横总线**，总线上挂多条**下垂竖线**接到子块。
于是找出"挂着 ≥2 条下垂竖线的横线"，其横跨范围就是一组兄弟节点，正上方最近的块就是父节点。

两个实测踩坑（都写进了代码注释）：
- **子节点判定要用"与总线横向交叠"，不能用"中心落在总线内"**：最外侧节点中心常在总线端点之外
  （珠海销售处中心 x=486，落在总线起点 502 左侧 16px），用中心判定会漏掉它，答案从 6 个变成 5 个。
- **层级还原只能对 CLIP 判定为「组织结构图」的图启用**：饼图/柱状图的扇区与柱条会被误判成总线，
  实测在 IC 市场图上产出 8 条无意义关系。

解析产物直接变成可检索的文本块：

```
【所在位置】《招股说明书2》第 39 页，图前正文：截至本招股意向书签署日，本公司内部组织结构如下图：
【图像语义】类型：组织结构图（CLIP 置信度 0.999）
【图内文字】股东大会 | 薪酬与考核 | 委员会 | 监事会 | 董事会 | … | 成都销售处
【图内层级】销售部 下辖：电话及网络销售部、渠道销售部、大客户销售部、国际贸易部（共 4 个）
【图内层级】大客户销售部 下辖：珠海销售处、深圳销售处、北京销售处、武汉销售处、广州销售处、成都销售处（共 6 个）
```

---

## 3. 快速开始

### 3.1 环境准备

```bash
# Python 3.11，建议独立环境
pip install -r requirements.txt
```

需要提前准备好的本地资源：

| 资源 | 说明 |
|---|---|
| `D:/模型/bge-m3` | 向量模型（本地，不联网） |
| `D:/模型/bge-reranker-v2-m3` | 重排模型 |
| `D:/模型/chinese-clip-vit-base-patch16` | **多模态模型**，首次需下载（约 600MB） |
| `C:/paddleocr_models/{det,rec,cls}` | OCR 模型，必须放**纯英文路径** |
| `C:/mineru_models/langfix` | MinerU 中文路径补丁 |

```bash
# 下载中文 CLIP（工单备注要求的多模态模型）
python -c "from huggingface_hub import snapshot_download as d; d('OFA-Sys/chinese-clip-vit-base-patch16', local_dir='D:/模型/chinese-clip-vit-base-patch16')"

# PaddleOCR 模型复制到纯英文路径
# 用户名含中文时，Paddle 的 C++ 推理层打不开默认的 ~/.paddleocr/，必须另存一份
cp -r ~/.paddleocr/whl/{det,rec,cls} C:/paddleocr_models/
```

### 3.2 配置

```bash
cp .env.example .env
# 编辑 .env：填入 DEEPSEEK_API_KEY
```

> ⚠️ **路径一律用正斜杠**。写成 `D:\模型\bge-m3` 时 `.env` 会把 `\b` 解析成退格符，加载报"路径不存在"。

### 3.3 启动依赖服务

```bash
# 方式一：用自带的 compose（干净机器，会起 Milvus + Redis + backend + frontend）
docker compose up -d
# 注意：本机已有常驻的 rag-milvus / redis 占用 19530 / 6379 时，不要执行这条，
#      直接复用本机服务即可（下面方式二）。

# 方式二：复用已有的 Milvus(19530) 与 Redis(6379)
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 3.4 数据入库

```bash
python scripts/ingest.py                 # 完整入库：MinerU 解析 → 图像语义解析 → 语义分块 → 向量化 → 入库
python scripts/ingest.py --no-mineru     # 跳过 MinerU，用 pymupdf/pdfplumber 兜底（快速回归）
python scripts/ingest.py --only 招股说明书2  # 只重建指定文档
python scripts/ingest.py --no-images     # 关闭图像解析（仅用于对照实验，证明图像链路的价值）
```

入库耗时参考：两篇共 898 页，MinerU 解析约 13 分钟，图像语义解析约 5 分钟，分块与向量化约 4 分钟。

### 3.5 访问

| 地址 | 说明 |
|---|---|
| http://localhost:8000/ | 前端问答界面 |
| http://localhost:8000/docs | FastAPI 自动文档 |
| http://localhost:8000/api/health | 依赖健康状态 |

### 3.6 本地开发注意事项

- **本机 Windows 系统代理是开启的**（注册表 `ProxyEnable=1`，指向 `127.0.0.1:65532`）。
  它只影响**无头浏览器**：Chromium 会继承系统代理，把 `http://127.0.0.1:8000` 也送去代理，
  导致导航直接挂到超时。用 puppeteer 时**必须加 `--no-proxy-server`**。
  注意 **curl 不走系统代理**，命令行测着一切正常，极具迷惑性。
- 前端 Vue 走 CDN，本机实测**偶发** `ERR_CONNECTION_RESET`。此时页面样式正常但模板不渲染，
  满屏 `{{ res.answer }}` 占位符且不报错。已加同源兜底（`frontend/vue.global.prod.js`），
  正常情况下用不到；若兜底文件缺失，页面会退化成未渲染状态，先查它。
- 跑重任务（入库 / 评估 / 压测）时**确保只有一个进程占 GPU**：8G 显存下两个进程同时驻留模型会互相拖慢，
  实测把 P50 从 2428 ms 抬到 3112 ms，性能数据会失真。
- **任何"绕过服务端状态"的测试参数，都必须在服务端有对应开关**。
  踩过的实例：`test_questions.py --no-cache` 原先只在**测试进程**里给 `K.get/K.set` 打补丁，
  走 HTTP 时缓存在**服务端**、补丁够不着，于是 16 题全部命中缓存、耗时 3 ms。
  它**不会报错，只会安静地给出失真数据**。现已改为 `/api/ask` 的 `no_cache` 入参透传。
  同理，`scripts/eval_ragas.py` 的结果**必须连同 nan 比例一起看**（nan 比例 > 10% 时整轮作废，见 `NAN_GUARD`）。

---

## 4. 测试与评估命令

```bash
# 容错注入测试（真打断依赖，7 个场景）
python tests/test_faults.py

# 工单 16 题批量测试（走 HTTP，需先启动服务）
python scripts/test_questions.py
# 直连流水线，免起服务
python scripts/test_questions.py --direct
# 绕开缓存，测真实端到端耗时
python scripts/test_questions.py --direct --no-cache
# 判分器自检（用已知错误答案回验判分逻辑本身）
python scripts/test_questions.py --selfcheck

# RAGAS 评估
python scripts/eval_ragas.py
python scripts/eval_ragas.py --tag baseline
# ★重复评估：给出均值/标准差/95% 置信区间/单次达标概率，并统计作废轮数
#   该指标分辨力约 ±0.05，判定"是否达标"必须先看区间，不能凭单次测量
python scripts/eval_ragas.py --repeat 3 --target 0.85 --tag repeat3

# 有效代码行数统计
python scripts/count_lines.py
```

**结果文件**：`results/test_*.json`（逐题答案、来源、检索精确度、耗时）、`results/ragas_*.json`。

---

## 5. 实测结果

> 口径说明：以下数字均为本机实测（RTX 4060 Laptop 8G / 16G RAM），非估算。
> 命令、时间戳与原始 JSON 见 `results/` 与 `docs/optimization.md`。

### 5.1 工单 16 题准确率

| 指标 | 工单要求 | 实测 |
|---|---|---|
| 准确率 | ≥ 90%（16 题即 ≥15 题） | **16/16 = 100%**，连续 5 次复现 |
| 响应时间 | ≤ 3 秒 | 平均 **2151 ms** · P50 **1926 ms** · P95 4424 ms · 最大 4424 ms |
| 检索精确度 | 产出物要求"显示检索精确度" | 平均 **0.930** |

命令：`python scripts/test_questions.py --no-cache --tag mergerows`
结果文件：`results/test_20261008_115834_final_v3.json`

**三条必须一并说明的限定条件**（详见 `docs/optimization.md` §5、§7）：

1. **响应时间的达标口径**：工单只写"≤ 3 秒"、未指明分位数。本项目取 **P50 ≤ 3 秒**为达标线（实测 1926 ms）。
   **如按均值或最大值卡，则不达标**：均值 2151 ms，16 题里有 **1 题超 3 秒**（id=5，4424 ms，答案是长枚举）。
   超时来自**生成**而非检索——生成占端到端 **73%**，属外部 DeepSeek API 的固有延迟，截短答案可压下来但会牺牲完整性（并可能直接判错）。
2. **准确率的跑间波动**：10 次运行的分布为 **均值 96.25% / 最差 87.5% / 标准差 5.0 个百分点，8/10 达标**。
   但前 5 次跑在**改动中的代码**上；**最近连续 5 次均为 16/16**。申报口径为"实测 16/16、连续 5 次复现；历史最差单次 14/16"，不称"稳定 100%"。
3. **RAGAS 的 context_precision 是短板且噪声大**：实测 0.78 左右，裁判噪声底 ±0.05（§5.3）。详见 `docs/optimization.md` §7。

### 5.2 检索精确度口径（两个口径，不混说）

| 口径 | 定义 | 用途 |
|---|---|---|
| **在线代理口径** | 返回的 N 条上下文中，重排分 ≥ 0.5 的相关块占比 | `/api/ask` 实时返回的 `retrieval_precision` |
| **离线带标注口径** | 答案是否命中标准答案关键词 + 负样本未出现 + 枚举个数正确；来源是否落在正确文档与页码 | `scripts/test_questions.py` |

在线口径是**无标注代理指标**，线上没有标准答案可比，因此**它不等于准确率**，两者在 README 与文档中始终分开陈述。

### 5.3 RAGAS 评估（`python scripts/eval_ragas.py`）

裁判模型 deepseek-chat；嵌入 = 本地 bge-m3；答案与上下文取自同一问答链路。
**因为噪声底的存在（见下），这里给四轮，而不是只给最好的一轮：**

| 轮次 | faithfulness | answer_relevancy | context_precision | context_recall | 结果文件 |
|---|---|---|---|---|---|
| 原始 | 0.9875 | 0.9161 | 0.7880 | 1.0000 | `ragas_20261008_095729_final3.json` |
| 裁尾#1 | 0.9875 | 0.9244 | 0.7359 | 1.0000 | `ragas_20261008_110601_ctxfix.json` |
| 裁尾#2（同代码复测） | 0.9250 | 0.9572 | 0.7793 | 1.0000 | `ragas_20261008_112100_ctxfix_rep2.json` |
| 表级合并 | 0.9875 | 0.8953 | 0.7813 | 1.0000 | `ragas_20261008_115334_mergerows2.json` |

**读法**：`context_precision` 四轮极差 0.052 全在噪声带内；`context_recall` 稳定为 1.0；
**⚠ 这四轮跨越了三个代码版本，只能看逐轮值，不能跨版本求均值**（那样会得到一个没有物理意义的混合数）。
按版本分组、以及"0.85 是否达成"的初步判定，见 `docs/optimization.md` §7.6；
`faithfulness` 稳定在 0.93~0.99（**说明答案的事实性可靠，波动来自组织方式而非事实**）。
逐题得分已导出在结果文件的 `per_question` 字段。

> 首次运行时 `context_precision` 出 `nan`，根因是 RAGAS 默认 **`max_workers=16`** 并发猛打 DeepSeek 触发限流，
> 64 个作业里 18 个 `TimeoutError`；且 RAGAS 部分指标默认要 `n=3` 个生成，而 DeepSeek 只支持 `n=1`。
> 两处均已修（`RunConfig(max_workers=4, timeout=300)` + `bypass_n=True`），修复后 0 超时、四项指标齐全。

**⚠ 使用这套指标前必须知道的噪声底**：同一份代码连跑两次，`context_precision` 为 0.7359 与 0.7793
（**±0.043**），`faithfulness` 为 0.9875 与 0.9250（**±0.063**）；单题波动更大（id=2 在 0.167↔0.667 之间）。
**该指标的分辨力约 ±0.05，小于此幅度的差异不能当作改进或回退。** 详见 `docs/optimization.md` §7。

**另一个坑**：DeepSeek 会**瞬时 402（余额不足）**。第一次跑合并轮时 16 题里 10 题变 `nan`、聚合值 0.6682，
差点被误读成"改动让指标变差"——实际是 42 个 402 与改动无关。
**看 RAGAS 结果必须同时查 `per_question` 的 nan 比例与日志里的 402/超时计数。**

---

## 6. 容错与高可用（工单系统稳定性验收项）

`tests/test_faults.py` 用**真打断依赖**的方式验证 7 个场景，而不是读代码判断有没有 try：

| 场景 | 期望行为 |
|---|---|
| Milvus 不可用 | 退回 BM25 单通道作答 + 显式降级提示 |
| Redis 不可用 | 缓存静默失效，问答不受影响 |
| DeepSeek 不可用（含 402 余额不足 / 超时） | 返回检索原文 + "生成服务不可用"提示 |
| 向量模型不可用 | BM25 撑住，降级可见 |
| 重排模型不可用 | 按融合分兜底排序 |
| 索引缺失/损坏 | 提示重建索引，不抛栈 |
| 用户输入错误（空/超长/纯符号） | 中文友好提示 + 明确错误码 |

一条特别值得记的坑（已写进 `backend/rag.py` 注释）：
**`MilvusClient()` 在构造阶段就抛异常**。若不在 `load()` 里兜住，异常会直接冒到接口层，
向量检索内部的兜底分支根本走不到——结果是整个问答失败，而不是退回 BM25。

---

## 7. 部署

```bash
# 1) 准备挂载路径（模型不进镜像，用挂载）
#    D:/模型/bge-m3、D:/模型/bge-reranker-v2-m3、D:/模型/chinese-clip-vit-base-patch16
#    C:/paddleocr_models

# 2) 设置密钥（不写进镜像）
export DEEPSEEK_API_KEY=sk-xxxxxxxx

# 3) 启动
docker compose up -d
docker compose logs -f backend      # 观察预热日志

# 4) 入库（容器内执行）
docker compose exec backend python scripts/ingest.py
```

compose 定义了 5 个服务：`etcd` / `minio` / `milvus` / `redis` / `backend` / `frontend(nginx)`。
前端是 Vue CDN 单文件，nginx 直接静态托管并把 `/api` 反代到 backend。

---

## 8. 演示视频

**录制步骤**：

```bash
# 1) 先确保服务已启动且已完成入库
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 2) 安装录制依赖（首次）
cd frontend && npm init -y && npm i puppeteer-core@21    # 必须锁 21.x：22+ 移除了 recordVideo

# 3) 清缓存：不清的话演示里每题都命中缓存、耗时归零，指标就成假的了
python -c "import redis;r=redis.from_url('redis://localhost:6379/0',protocol=2);r.delete(*r.scan_iter('gd4:ans:*'))"

# 4) 自动录制
node ../docs/record_demo.mjs

# 产物：docs/video/工单四演示视频.webm  以及 docs/screenshots/*.png
```

**已产出的演示视频**：`docs/video/工单四演示视频.webm`
（309 帧 / 5 fps / **61.8 秒** / 1280×720 / 2.9 MB，回读验证过 309 帧全部可解码。
截图 9 张见 `docs/screenshots/`。）

本机录制踩到的三个坑（都已写进 `record_demo.mjs` 注释，换机器可能不需要）：
1. **Windows 系统代理**（注册表 `ProxyEnable=1` → `127.0.0.1:65532`）会被 Chromium 继承，
   把 `http://127.0.0.1:8000` 也送去代理，导航直接挂到超时。必须加 `--no-proxy-server`。
   注意 curl 不走系统代理，命令行测着是通的，**极具迷惑性**。
2. **喂给 ffmpeg 的管道输入必须显式写 `-vcodec mjpeg`**。省掉它会报
   `Output file does not contain any stream`（`-i -` 则报 `Protocol not found`），
   很像"管道输入不支持"，其实只是**管道不可寻址、ffmpeg 无法自行探测输入编码**。
   另外 `setInterval` 里并发截图会让多路写 stdin 把 JPEG 流写花、ffmpeg 直接退出，
   故加了写帧闸门；ffmpeg 的 stderr 也必须留出来，否则它启动即死也看不出原因。
3. 前端 Vue 走 CDN，偶发 `ERR_CONNECTION_RESET` 会让页面停在**未渲染**状态——满屏 `{{ res.answer }}`
   占位符但样式正常，肉眼不细看以为渲染好了，截出来全是废片且不报错。
   已给前端加**同源兜底**（CDN 仍是首选，仅在 `window.Vue` 缺失时回落 `vue.global.prod.js`，
   后端 `main.py` 也提供同源路由，nginx 则由静态目录直接提供）；
   录制脚本另加渲染闸门（判据是"页面里还有没有 `{{`"，因为 textarea 是静态节点、等它出现证明不了 Vue 挂载），
   未渲染就重载，最多 3 次，仍未渲染则明确报错退出——**宁可不出片，也不出废片**。

**视频展示的内容**（对应工单产出物二"展示系统的功能和操作流程"+"显示检索到的答案及检索精确度"）：

1. 首屏与界面布局
2. 左栏内置工单 16 题，点击即填入并检索
3. **id=6 图像独有题**：答案 + 来源标为「图像 · 饼图」+ 图表原图缩略图
4. **id=5 组织结构图**：图像块的【图内层级】展示 4 个部门 / 6 个销售处
5. 切换到 English 提问，英文作答
6. 依次演示其余工单问题
7. **指标卡特写**：响应时间 / 检索精确度 / 检索通道 / 检索与生成耗时分解
8. 同一问题重复提问 → 命中缓存，端到端降至毫秒级
9. 终端演示 `tests/test_faults.py` 7 个容错场景
10. 终端演示 `scripts/test_questions.py` 汇总（准确率 / 响应时间 / 检索精确度）

详细分镜表见 `docs/演示视频脚本.md`。

> 若无法播放 webm 或需要人工配音版，`docs/演示视频脚本.md` §四 给了完整讲解脚本（约 5 分钟）。

---

## 9. 项目结构

```
工单四/
├── config.py                 # 集中配置（密钥、路径、模型、检索与图像参数）
├── backend/
│   ├── main.py               # FastAPI 接口 + 前端托管
│   ├── rag.py                # 混合检索 + RRF + 生成 + 缓存 + 检索精确度 + 降级
│   ├── imaging.py            # ★图像内容解析：CLIP 语义 + OCR 文字 + 几何层级
│   ├── parse.py              # PDF 解析：MinerU 为主，pymupdf/pdfplumber/paddleocr 兜底
│   ├── chunking.py           # 语义分块
│   ├── store.py              # bge-m3 编码 + Milvus + bge-reranker 重排
│   └── cache.py              # Redis 缓存（键含源码结构指纹）
├── frontend/
│   ├── index.html            # Vue 3 CDN 单文件界面
│   └── vue.global.prod.js    # Vue 同源兜底副本（CDN 不可达时才用）
├── scripts/
│   ├── ingest.py             # 入库流程
│   ├── test_questions.py     # 工单 16 题批量测试与判分
│   ├── eval_ragas.py         # RAGAS 评估
│   └── count_lines.py        # 有效代码行数统计
├── tests/
│   ├── test_questions.json   # 16 题 + 标准答案 + 关键词 + 负样本 + 出处
│   └── test_faults.py        # 容错注入测试
├── docs/
│   ├── design.md             # 设计文档（架构、接口、页面、流程、非功能）
│   ├── optimization.md       # 优化记录（前后对比，含实测数据）
│   ├── 演示视频脚本.md        # 演示视频分镜与录制方法
│   ├── record_demo.mjs       # 录制脚本
│   ├── video/工单四演示视频.webm  # ★演示视频（63.2 秒）
│   └── screenshots/          # 演示截图 9 张
├── data/{raw,parsed}/        # 原始 PDF 与解析产物
├── results/                  # 测试与评估结果 JSON
├── docker-compose.yml / deploy/  # 部署
└── requirements.txt
```

---

## 10. 工单要求 vs 明确不做

### 做（工单四要求）

工单指定的 16 个检索问题 · 图像语义解析（CLIP + OCR + 几何层级）· 检索优化（混合检索 + RRF + 重排）·
准确率 ≥90% · 响应时间 ≤3 秒 · 容错机制 · 中英文问答 · 友好的交互界面 · 性能与资源 · 演示视频 · 代码注释含工单编号。

### 不做（工单四未要求）

- 用户登录、权限管理、管理后台、支付 —— 工单四全文无此要求
- 表格解析与结构化（行列定位、跨表消歧）—— 那是**工单三**的核心
- Query 改写与意图识别 —— **工单五**；混合检索调参 —— **工单六**
- Graph+RAG —— 工单八/九；Embedding 微调 —— 工单十一
- 流式输出 —— 工单四未要求（首字延迟非本工单验收项）
- 并发 QPS 门槛承诺 —— 工单四未给并发数值指标，故只如实报告实测资源曲线，不设达标线

---

## 11. 关于代码行数（如实申报）

工单建议"有效 Python 代码总行数控制在 300 行以内"，并给出退路：
"如果无法同时满足，优先保证核心逻辑有效代码 ≤300 行，其他用配置、脚本、文档补充"。

实测（`python scripts/count_lines.py`）：

| 口径 | 行数 |
|---|---|
| 核心逻辑（`config.py` + `backend/main.py`） | 140 |
| 全部 Python 有效代码行 | 1283 |

**如实说明：整体超出 300 行建议值**，核心逻辑 140 行达标。
原因：工单四要求的功能面本身很宽（PDF 解析 + **图像多模态解析** + 语义分块 + 向量库 + BM25 +
混合检索 + 重排 + 大模型生成 + Redis 缓存 + 容错 + 测试 + RAGAS 评估 + 部署），
且工单同时要求"每一行有效代码都要有详细注释"，压缩空间被这两条要求夹住。
若需进一步压缩，可优先砍掉的**非工单必需项**是：`scripts/count_lines.py`（统计工具）与
`tests/test_faults.py` 的部分场景（但容错是工单明确的验收项，不建议删）。

---

## 12. 常见问题

| 现象 | 原因与处置 |
|---|---|
| `未配置 DEEPSEEK_API_KEY` | `.env` 未填密钥，参考 `.env.example` |
| 加载模型报"路径不存在" | `.env` 里路径用了反斜杠，`\b` 被解析成退格符。改成正斜杠 |
| MinerU 解析失败（fasttext 打不开模型） | 用户名含中文，需设置 `MINERU_LANGFIX_DIR` |
| PaddleOCR 报 `Cannot open file ... pdmodel` 但文件存在 | 同样是中文用户名，模型必须复制到纯英文路径 |
| 首次提问特别慢（十几秒） | 本地模型加载。服务启动时已预热；若直接调脚本请先 `R.warmup()` |
| 检索结果里出现"八维教育 / 人工智能刘敏 / rengongzhinengliumin" | PDF 斜向水印不在文本层，MinerU 读图必然读进单元格。已用水印词表过滤 + `rag.py` 的 `clean()` 清洗紧贴中文的小写拉丁碎片（实测修掉"赵马克gzh"，全库 1336 处），见 `docs/optimization.md` §3.2 |
| 答案里出现"资料中未找到相关信息" | 检索未召回。检查 `docs/optimization.md` 的融合池大小说明 |
| 无头浏览器导航本地页面超时 | Windows 系统代理被 Chromium 继承。加 `--no-proxy-server` |
| 演示视频没生成但脚本不报错 | 精简 ffmpeg 的 image2pipe 不认管道输入。已改用 `scripts/make_video.py`（cv2）编码 |
| 截图/录屏里满屏 `{{ res.answer }}` | 前端 Vue 的 CDN 偶发 `ERR_CONNECTION_RESET`，页面停在未渲染状态。录制脚本已加重载闸门 |
| 跑 RAGAS 时 context_precision 是 nan | RAGAS 默认 `max_workers=16` 打满限流 + DeepSeek 不支持 `n>1`。已用 `RunConfig(max_workers=4, timeout=300)` 与 `bypass_n=True` 修掉 |
| RAGAS 报 `Can't instantiate abstract class HuggingfaceEmbeddings` | RAGAS 0.4 起该类变抽象基类，须经 `LangchainEmbeddingsWrapper` 使用本地 bge-m3 |
