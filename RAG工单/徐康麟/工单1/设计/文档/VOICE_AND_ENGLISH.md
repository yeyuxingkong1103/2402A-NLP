# 语音输入与英文问答（工单追加需求）

> 本文档说明两项**追加需求**的实现方式、使用步骤与限制：
> **① 语音输入**、**② 英文问答**。
> 基础系统的架构与部署见 `TECH_DOC.md`，操作步骤见 `USER_MANUAL.md`。

---

## 1. 需求来源与实现状态

| 追加需求 | 状态 | 主要实现文件 |
| --- | --- | --- |
| 语音输入 | ✅ 已实现（两种后端，自动降级） | `app/core/asr.py`、`app/ui/streamlit_app.py` |
| 英文问答 | ✅ 已实现（跨语言检索 + 英文作答） | `app/core/language.py`、`app/core/english_answer.py` |

> 说明：原始需求说明书写的是“不需要语音输入”，工单原件写的是“支持语音输入”。
> 按追加要求，本系统**同时支持**语音与文字、中文与英文，语音为可选功能，
> 未部署语音服务时不影响文字问答。

---

## 2. 语音输入

### 2.1 交互流程

```
用户录音/上传音频
   └─ Streamlit 控件采集音频字节
       └─ app/core/asr.py: SpeechRecognizer.transcribe()
           ├─ 后端 api   : POST {RAG_ASR__BASE_URL}/audio/transcriptions
           ├─ 后端 local : 本地 transformers Whisper
           └─ 后端 none  : 抛出带中文原因的异常（绝不静默失败）
               └─ 识别文本写入 session_state.pending_question
                   └─ 进入与文字提问完全相同的问答链路
```

识别结果会先显示在界面上（含识别语种），再自动作为问题提交，
因此**语音与文字走的是同一条链路**，引用、多轮、反馈等能力完全一致。

### 2.2 后端选择

| 后端 | 适用场景 | 依赖 | 说明 |
| --- | --- | --- | --- |
| `api` | **算力云 4090 推荐** | `openai` 客户端 | 调 vLLM 起的 Whisper 服务；本机的 `run_local_asr.py` 也走这条 |
| `local` | 本机离线验证 | `transformers` + `torch` | GPU/CPU 均可（本机 RTX 2060 约 1~2 秒）；**只支持 WAV** |
| `none` | 两者都不可用 | — | 界面给出中文修复提示，文字提问仍可用 |

配置项（可用环境变量覆盖，无需改代码）：

```bash
export RAG_ASR__BACKEND=auto      # auto / api / local
export RAG_ASR__BASE_URL=http://127.0.0.1:8001/v1
export RAG_ASR__MODEL=openai/whisper-large-v3
export RAG_ASR__LOCAL_MODEL_DIR=models/whisper-tiny
```

`auto` 的优先级：**api > local**。原因是 4090 的显存应优先留给 LLM，
Whisper 单独进程便于按需启停。

### 2.3 启动语音服务

```bash
# 另开一个终端（与主环境隔离，避免 CUDA 冲突）
bash 研发/scripts/run_whisper.sh

# 自检
curl http://127.0.0.1:8001/v1/models
```

### 2.4 界面控件说明

| 环境 | 录音方式 | 说明 |
| --- | --- | --- |
| Streamlit ≥ 1.40 | `st.audio_input` | 浏览器内直接录音 |
| Streamlit < 1.40 | `st.file_uploader` | 上传已录好的音频文件（本机实测版本为 1.37，走这条路径） |

两条路径功能等价，`部署/环境配置/requirements.txt` 已标注 `streamlit>=1.40`。

### 2.5 已知限制

1. **本地后端只支持 WAV**：本机没有 `ffmpeg`，无法解码 mp3/m4a。
   非 WAV 会得到明确的中文提示，并建议改用 API 后端。
   （`app/core/asr.py::_decode_audio` 用标准库 `wave` 解码，
   刻意绕开 ffmpeg，因此离线也能跑通 WAV。）
2. **首次加载慢**：本地 Whisper 首次加载约 10 秒（CPU），
   故界面只在用户真正上传音频后才加载模型。
3. **小模型识别质量有限**：本机用的是 `whisper-tiny`，
   中文识别精度一般；算力云上请用 `whisper-large-v3`。
4. 语音提问的**首字延迟包含转写时间**，因此不计入“首字 < 3 秒”的
   文字问答指标（该指标衡量的是检索+生成链路）。

---

## 3. 英文问答

### 3.1 核心难点与方案

语料是**中文招股书**，因此英文问答要解决两个问题：
英文问题如何命中中文证据、中文证据如何变成英文答案。

```
英文问题
  ├─ ① 语言检测（app/core/language.py::detect_language）
  ├─ ② 英文意图识别（detect_english_intent）→ 选出中文问句模板
  ├─ ③ 中文检索查询构造（build_chinese_query）
  │     主查询 = 中文问句模板（干净问句，不拼关键词）
  │     补充变体 = 中文关键词 / 去公司名版本
  ├─ ④ 中文混合检索（向量 + BM25，多查询变体加权合并）
  └─ ⑤ 英文答案构造（app/core/english_answer.py）
        ├─ 模板路径：意图 → 英文句式 + 中文事实抽取（默认，毫秒级）
        └─ 模型路径：本地小模型润色（可选，见 3.4）
```

### 3.2 为什么要用「中文问句模板」而不是关键词

这是实测踩出来的结论。同一份索引下：

| 检索查询 | 最高余弦 | 命中的页面 |
| --- | --- | --- |
| `注册资本`（裸关键词） | 0.539 | 第 170、286 页（无关） |
| `注册资本是多少`（完整问句） | 0.694 | 第 22 页（**正确**） |
| `武汉兴图新科电子股份有限公司的注册资本是多少 注册资本` | 0.539 | 释义表（错误） |

原因：语料里的相关段落本身是完整句子，用词组检索会与
“释义/基本用语”这类同样含该词组的噪声段落竞争；
而把公司全称拼进查询，会把语义方向进一步拉向高频出现公司名的定义性章节。

因此最终方案是：
- **主查询用干净的中文问句模板**（`CHINESE_QUERY_TEMPLATES`，13 个意图）；
- 关键词与“去公司名版本”作为**独立变体**参与多路召回；
- 多路结果按**相对语义置信度加权**合并（见下一条）。

### 3.3 为什么多路召回要按置信度加权

每个变体的 top1 都会被归一化成 1.0，若直接取最大值合并，
一个“词袋式关键词变体”会与真正贴题的变体等权。
实测后果：`供应商 客户 武汉兴图新科电子股份有限公司`
（余弦 0.829，高于贴题变体的 0.762）把“释义/基本情况”页顶到第一名，
导致“公司在哪个领域已经成为重要供应商”答错。

现在的做法：先算每个变体的最高原始余弦，再按
`(cosine / best_cosine)²` 加权后合并。语义方向更准的变体自然占优。

### 3.4 英文答案为什么默认不用模型翻译

**数字与单位绝不能出错。** 实测本地 0.6B 模型会把

```
5,520 万元  ->  "5,520 million yuan"     （错，差 100 倍）
```

因此金额一律走 `app/core/number_utils.py` 的**确定性换算**：

```
5,520 万元  ->  "RMB 55.20 million (5,520.00 ten-thousand yuan)"
```

英文答案由「意图 → 英文句式模板 + 从中文原文抽取事实」生成，
数字、比例、日期原样保留，术语走 `GLOSSARY_ZH_EN` 术语表。
该路径是**毫秒级**，天然满足首字 < 3 秒。

若需要更自然的行文，可启用本地模型润色（可选，默认仅用于模板未覆盖的问题）：

```bash
export RAG_LANGUAGE__ALLOW_LOCAL_TRANSLATION=true
export RAG_LANGUAGE__QUERY_BRIDGE=model   # 查询也用模型翻译（较慢，CPU 上 1~10 秒）
```

### 3.5 引用格式

英文回答的引用为 `[Page: N]`（中文回答为 `[页码: N]`），
由 `Citation.label(language)` 按回答语言生成。

### 3.6 实测效果（本机，抽取式路径）

| 指标 | 结果 |
| --- | --- |
| 英文问题准确率 | **5/5 = 100%** |
| 英文回答引用正确率 | **100%** |
| 英文问答首字延迟 | 平均 1.68ms，最大 3.74ms（预算 3000ms） |
| 中文问答（回归） | **10/10 = 100%**，未受影响 |
| 英文无关问题兜底 | 回复 `I don't know`（不编造） |

复现命令：

```bash
python 研发/scripts/evaluate.py --no-llm --english
# 输出中 [extractive] 为中文组，[rag_en] 为英文组
```

### 3.7 已知限制

1. **术语表覆盖范围有限**：`GLOSSARY_ZH_EN` 覆盖招股书高频术语
   （注册资本、法定代表人、军用领域、上下游、技术标准、科技进步奖等）。
   开放性问题会走「本地模型润色」，模型不可用时给出带页码的依据提示，
   **不会编造英文内容**。
2. **中文专有名词可能残留**：如工程名称“某情报、指挥、控制与通信网络一体化工程”，
   原文本身是脱敏名称（招股书用“某”代替），术语表不做臆造翻译，
   英文答案中会保留该名称并附英文解释。
3. **人名**：`程家明` 通过术语表映射为 `Cheng Jiaming`；
   未收录的人名会输出原文并标注“Chinese name as in the prospectus”。
4. **未接入真正的多语言嵌入模型**：本机尝试下载 `BAAI/bge-m3`
   时被上游返回 403，因此跨语言检索靠“中文问句模板 + 术语表”桥接，
   而非多语言向量对齐。若算力云上能下载 `bge-m3`，
   可直接改 `RAG_EMBEDDING__MODEL_NAME=BAAI/bge-m3` 并重建索引，
   跨语言召回会进一步改善（届时可把 `RAG_LANGUAGE__QUERY_BRIDGE` 设为 `glossary` 仍生效）。

---

## 4. 配置项汇总

```yaml
# 语音识别
asr:
  backend: auto                  # auto / api / local
  base_url: http://127.0.0.1:8001/v1
  model: openai/whisper-large-v3
  local_model_dir: models/whisper-tiny
  timeout: 60.0

# 多语言问答
language:
  default_answer_language: auto  # auto 跟随提问语言；也可强制 zh / en
  query_bridge: glossary         # glossary（默认，毫秒级）/ model / auto
  translation_model_dir: models/Qwen3-0.6B
  allow_local_translation: true
  translation_max_tokens: 96
```

环境变量覆盖示例：

```bash
export RAG_ASR__BACKEND=api
export RAG_LANGUAGE__DEFAULT_ANSWER_LANGUAGE=en
export RAG_LANGUAGE__QUERY_BRIDGE=model
```

---

## 5. 相关测试

| 测试文件 | 覆盖内容 | 用例数 |
| --- | --- | --- |
| `测试/tests/offline/test_asr.py` | WAV 解码（绕开 ffmpeg）、立体声下混、非 WAV 报错、后端选择、无后端时中文报错 | 7 |
| `测试/tests/online/test_bilingual.py` | 语言检测、英文意图识别、问句模板完整性、金额换算精确性、术语替换、英文端到端 5 题、引用格式、中文回归 | 23 |

运行：

```powershell
$env:PYTHONPATH="E:\gao6gongdan\工单1"
& ".gao6gongdan-src\python.exe" -m pytest 测试/tests/offline/test_asr.py 测试/tests/online/test_bilingual.py -v
```
