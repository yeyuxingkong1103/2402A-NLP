# 工单14 部署与复现说明

**工单编号**：人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单

---

## 1 任务构成与产出对应

| 任务 | 要求 | 产出 |
| --- | --- | --- |
| **任务一** | 梳理 RAGFlow deepdoc 技术实现（3 问） | `设计/RAGFlow技术分析.md`（735 行，源码级 + 精确行号） |
| **任务二** | 上传 `CN100342976C.pdf`，测试 6 题的答案 | `测试/六题实证与根因分析.md`、`研发/评估/六题结果.json`、`测试/测试截图/` |

## 2 环境与依赖

| 项 | 值 |
| --- | --- |
| 解释器 | `E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（Python 3.11.15） |
| 本工单实际用到 | `pymupdf`（渲染/文本层探针）、`Pillow` + `numpy`（自研像素 OCR） |
| **本机缺失** | `cv2` / `pytesseract` / `paddleocr` / `easyocr` / `onnxruntime` —— **无任何 OCR 引擎** |
| 视觉模型 | **无**（Ollama 13 个模型均为文本/嵌入模型） |
| Docker | 未使用（`docker` CLI 走 npipe 需提权，本工单**未实跑 RAGFlow 全栈**） |

> 因此本工单的解析链路是**自研等价实现**（渲染 → 文本层探针 → 像素级标记识别 → 视觉读图），
> 而非 RAGFlow 端到端运行；这一点在 `测试/六题实证与根因分析.md` §4 已如实标注。

## 3 三条复现命令

```powershell
$py = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
Set-Location 'E:\gao6gongdan\工单14'

# 1) 解析实证（文本层探针 + 渲染 8 页 + 附图标记识别）→ 研发/data/解析实证.json
& $py 研发/scripts/演示_扫描件解析实证.py --pages 2,7

# 2) 问题集与实证产物核对（6 题答案键、标记覆盖）
& $py 研发/scripts/核对问题集.py

# 3) 测试截图 + 单元测试
& $py 测试/测试脚本/生成截图.py
& $py -m pytest 测试/test_核心链路.py -q --no-header -p no:cacheprovider
```

## 4 目录导航

```
工单14/
├── 设计/    工单需求原文.md、RAGFlow技术分析.md（任务一·三问）
├── 研发/    pipeline/font_ocr.py（自研像素 OCR）、scripts/（实证+核对）、
│            data/_render/（8 页 PNG）、data/解析实证.json、评估/六题结果.json
├── 测试/    六题实证与根因分析.md、test_核心链路.py（9 用例）、
│            测试数据/测试问题集.jsonl、测试脚本/生成截图.py、测试截图/（3 张中文命名）
├── 优化/    信息丢失根因与修复方向见 测试/六题实证与根因分析.md §3
└── 部署/    本文件、交付物核验.md
```

## 5 关键运维事实（实测踩坑）

1. **本机无 OCR 引擎**：`cv2`/`pytesseract`/`paddleocr`/`easyocr`/`onnxruntime` 全部缺失。
   自研 `font_ocr.py` 用「字体模板匹配 + 连通域」替代，**实测所需 14 个附图标记只恢复 7 个**，
   缺的正是 `6'`、`6''`、`X1`、`X2`、`X3`、`h1`、`h2`（带撇号字符与紧邻尺寸符号是难点）。
2. **该 PDF 是纯扫描件**：8 页全为整页位图，`page.get_text()` 返回 0 字符。
   任何纯文本抽取路径都拿不到内容，必须先渲染再走图像/视觉路径。
3. **视觉通道是本机唯一可行的图内问答手段**：把页面渲染为 PNG 后逐页读图，
   实测 6/6 题全部答对（含 4 道需图内空间信息的题）。**但该方式不可规模化**，
   修生产链路请参考 `测试/六题实证与根因分析.md` §3.3 给出的 4 条方案（图块坐标入索引／
   图块独立模态／接入 VLM／专用符号 OCR 增强）。
4. **PowerShell 传参注意**：`pwsh -File run_py.ps1 ... --out ...` 会把 `--out`/`--summary`
   误认成 PowerShell 自身参数并报错，直接用解释器调用最稳。
