# 工单7 部署与演示

> 工单编号：**人工智能NLP-RAG-功能测试及评估**
> 本目录负责「怎么跑起来、怎么看结果、怎么演示」。

---

## 一、目录结构

```
部署/
├── 界面/测试评估台.py          标准库 http.server 单文件 Web 界面（三标签页）
├── 脚本/
│   ├── 环境.ps1                统一环境变量（解释器 + UTF-8 + 语料/索引/日志路径）
│   ├── 一键跑测试评估.ps1       解压 → 建索引 → 冒烟 → 评估 → 回归 → 截图 → 视频
│   ├── 启动测试评估台.ps1       启动 Web 界面（日志 → 部署/日志/界面.log）
│   └── 生成演示视频.py          PIL 逐帧 + ffmpeg 合成演示视频
├── 演示视频/                   工单7_功能测试及评估演示.mp4（56 s）+ 关键帧/
├── 日志/                       运行日志（app.log / rag_trace.jsonl / 界面.log）
└── _工单6继承部署资产/          工单6 的历史部署资料（界面/docker/说明），本工单不复用
```

---

## 二、启动 Web「测试评估台」（两版，共用同一后端）

### ① Gradio 版（任务书原计划设计）

```powershell
Set-Location E:\gao6gongdan\工单7
. 部署\脚本\环境.ps1
& $env:GD7_PY 部署\界面\gradio_app.py --port 7862
# 参数：--host / --port / --no-warmup / --share
```

- 依赖：**gradio 6.29.1**（装在 `研发/第三方库`，由 `工具/不依赖pip装包.py` 安装；本机 `pip install` 会 PermissionError）
- 未安装时的提示：脚本会打印「请先运行 `python 工具/不依赖pip装包.py gradio --with-deps`」并以退出码 3 结束

### ② 标准库兜底版（零依赖）

```powershell
pwsh -NoProfile -File 部署\脚本\启动测试评估台.ps1                 # 默认 http://127.0.0.1:7861
pwsh -NoProfile -File 部署\脚本\启动测试评估台.ps1 -Port 7862      # 换端口
```

> **两版共用同一套后端**：`测试评估台.EngineBox`（唯一 QAEngine + 后台预热 + 串行问答）与
> `build_eval_payload` / `build_flow_payload` / `build_health_payload`。
> `gradio_app.py` 通过 `importlib` 加载标准库版当后端模块，因此**换界面不换逻辑、口径不会漂移**。
> 标准库版保留为「任何环境都能跑」的兜底。

### 三个标签页

| 标签页 | 内容 |
| --- | --- |
| **问答测试** | 输入问题 → 真实调用 `QAEngine.answer()` → 展示答案、是否拒答与原因、引用页、Top-10 检索片段（chunk_id/文件/页码/类型/分数/前 200 字）、检索/生成/首字/总耗时、生成方式 |
| **评估看板** | 读 `测试/测试数据/评估结果.jsonl`、`评估汇总.json`、`检索问题分析.md`，表格展示 17 题的覆盖率/判对/recall@5/@10/泄漏/耗时 + 汇总指标 + 问题分析摘要 |
| **测试流程演示** | 一键逐步展示「问题提取 → 索引与检索 → 评估打分 → 结果输出」四步，每步显示真实产物路径与摘要统计 |

### 实测验证证据（本机真实执行）

| 验证项 | 结果 |
| --- | --- |
| **Gradio 版** `GET http://127.0.0.1:7862/` | **HTTP 200**（37 165 B），页面含「人工智能NLP-RAG-功能测试及评估」 |
| **Gradio 版** `GET /config` | 检出三个标签页：**问答测试 / 评估看板 / 测试流程演示** |
| **Gradio 版** `_load_eval` | HTTP 200，0.06 s，返回汇总 Markdown + 17 行表格 + 问题分析 |
| **Gradio 版** `_load_flow` | HTTP 200，0.05 s，返回四步流程 |
| **Gradio 版** `_on_ask`（真实问答） | HTTP 200，**2.34 s**，「平安银行2019年实现营业收入1,379.58亿元，净利润281.95亿元」，`llm_grounded`，含 Top-10 证据表 |
| 标准库版 `GET /` | **HTTP 200**，25 860 B，三大标签页文案齐全 |
| 标准库版 `POST /api/ask` | **HTTP 200**，返回真实答案 + 引用页 + 分段耗时 |
| 停止后端口释放 | 7861 / 7862 均无残留监听（验证完即关） |

> ⚠️ **一处判断失误的更正（2026-10-08）**：本仓库最初写「沙箱封网、装不了 Gradio」，
> 据此只做了标准库替代版。经复测，**网络完全可用**（Python/OpenSSL 对 pypi、清华源、hf-mirror 全部 200），
> 失败的只是 `curl.exe` / `Invoke-WebRequest` 的 Windows schannel 凭证；
> 真正受限的是 `pip install --target`（PermissionError）。
> 现已用 `工具/不依赖pip装包.py` 补齐 **gradio / pandas / pdfplumber** 与 **718 MB 中文 CLIP 权重**，
> 并按原计划实现 Gradio 界面与图像内容检索。详见 `交付物核验.md` §6.2 与 `设计/工单7环境事实.md` §2。

**踩坑（写进代码注释）**：`研发/第三方库` 必须放在 `PYTHONPATH` **最后** ——
它含 numpy 2.x，而 venv 的 faiss 1.7.4 是针对旧 numpy 编译的 C 扩展，放前面会让
`import faiss` 抛 `_ARRAY_API not found`，向量检索整条链路失败。该目录里的 numpy 2.4.6 已移除。

---

## 三、一键跑通全流程

```powershell
pwsh -NoProfile -File 部署\脚本\一键跑测试评估.ps1                       # 全部步骤
pwsh -NoProfile -File 部署\脚本\一键跑测试评估.ps1 -SkipExtract -SkipIndex  # 语料/索引已就绪时
pwsh -NoProfile -File 部署\脚本\一键跑测试评估.ps1 -Only eval            # 只跑某一步
```

步骤与产物：

| 步骤 | 命令 | 主要产物 |
| --- | --- | --- |
| `extract` | `测试/解压语料.py` | `研发/data/raw/*.pdf`（9 份） |
| `images` | `研发/scripts/extract_images.py --all` | `研发/data/processed/image_blocks.jsonl`（175 条）+ `研发/data/index/images_clip.npy`（196×512） |
| `index` | `研发/scripts/build_index.py --force --no-seed --include-images --batch-size 32` | `研发/data/index/bge-m3_1024/`（**8847 块** = 文本 7325 + 表格 1417 + 图像 105） |
| `smoke` | `测试/冒烟自检.py` | 控制台（索引/检索/生成/多轮/异常兜底） |
| `eval` | `研发/评估/运行评估.py` | `测试/测试数据/评估结果.{jsonl,csv}`、`评估汇总.json`、`检索问题分析.md` |
| `regress` | `测试/回归测试/run_all.py` | `测试/测试数据/回归结果.json`、`回归测试日志.txt` |
| `screens` | `测试/生成截图.py` | `测试/测试截图/*.png`（10 张） |
| `video` | `部署/脚本/生成演示视频.py` | `部署/演示视频/工单7_功能测试及评估演示.mp4` |

> 图像链路（`images` 步）**不需要 cv2/onnxruntime** —— 实测只依赖 `PIL + pymupdf + numpy`，
> 加中文 CLIP 权重即可（先跑 `测试/下载中文CLIP权重.py`，718 MB，HF 镜像优先）。
> 若跳过 `images` 步，`index` 步必须补 `--allow-empty-images`，否则「图像块数=0」会被判为硬失败
> （一键脚本已把该步骤的这种退出码从「失败判定」中排除）。
> 想更快可在 `index` 步加 `--fast-table`（跳过「无绘制线且文本极短」页的表扫描）。

---

## 四、演示视频

`部署/演示视频/工单7_功能测试及评估演示.mp4`

- 时长 **56.00 s**（ffprobe 实测）｜1120 帧 @ 20 fps｜1600×900｜H.264｜1.75 MB
- 五个镜头：①问题提取（`sample_questions.pdf` → 17 题 / 103 关键词）②语料与索引规模（9 份年报 / 2812 页 / 8847 块（含图像 105））
  ③RAG 检索与回答示例（真实取一道判对的题，展示答案/引用页/分段耗时/Top-5 命中）
  ④评估判分表（17 题逐题覆盖率/判对/recall/泄漏/耗时/生成方式）⑤汇总指标与检索问题分析（含 TOP5 根因）
- 关键帧同时落盘在 `部署/演示视频/关键帧/`，便于截图引用
- **诚实边界**：本片是**图文讲解片**（数据面板 + 旁白字幕），**不是屏幕操作录像**，**无音轨**（本机无可用 TTS）；
  画面里每个数字都从真实产物文件读取（`评估结果.jsonl` / `评估汇总.json` / `检索问题分析.md` /
  `测试问题集.jsonl` / `index_manifest.json`），文件缺失时脚本直接打印 FATAL 而不是编数。
  编码用本机 `C:\Program Files\EVCapture\ffmpeg.exe`（PATH 里没有 ffmpeg）。

重新生成：

```powershell
& $env:GD7_PY 部署\脚本\生成演示视频.py
& $env:GD7_PY 部署\脚本\生成演示视频.py --frames-only    # 无 ffmpeg 时只出帧序列
```

---

## 五、已知限制

1. **无 gradio/streamlit**：界面用标准库实现（见上）；如需 Gradio，本机需先联网安装。
2. **图像检索缺口**：索引 `chunk_by_modality.image = 0`（缺 cv2/onnxruntime + CLIP 权重）。
3. **Ollama 是 CPU 单实例**：界面只在人工点击时单题调用，内部串行化；与其它任务并发时会互相拖慢，
   响应时间跨轮波动明显（实测同一份代码 1.19 s ~ 2.7 s 平均）。
4. **必须预热**：不预热时冷启动首问约 7.5 s（必违约），界面启动时调用 `warmup(with_generate=True)`。
5. **索引缺失时**：界面会显示「数据未就绪」与生成命令，不会伪造数据。
