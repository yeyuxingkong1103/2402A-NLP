# =============================================================================
# 工单：人工智能NLP-RAG-图像内容解析及检索优化
# 用途：本仓库（E:\gao6gongdan\工单4）所有脚本 / 测试的统一 Python 入口。
#       镜像 工单3\run_py.ps1 写法，参数默认值已按工单4 修正。
#
# 环境事实（实测，勿凭提示词臆测）：
#   本机并【不存在】名为 gao6gongdan 的 conda 环境（conda 命令本身报错不可用）。
#   唯一可用解释器是工单1 的 venv：
#       E:\gao6gongdan\工单1\.venv\Scripts\python.exe   （Python 3.11.15）
#   已装：pymupdf(fitz) / numpy / jieba / scikit-learn / faiss / torch(CPU版) /
#         transformers / sentence-transformers / Pillow / fastapi / uvicorn /
#         pytest / requests / openai / huggingface_hub
#   缺失：cv2、pytesseract、paddleocr、easyocr、onnxruntime、gradio、streamlit、
#         pdfplumber、pandas、loguru、matplotlib、open_clip、clip、timm、accelerate
#   —— 对应能力一律自实现或改用替代方案。
#
# ⚠️ 网络与 Docker（工单4 实测，最关键的一条）：
#   本机【有网】，但 DSH 沙箱在 workspace-write 模式下【封禁外网出口】，
#   pypi / huggingface / baidu 全部 SSL 握手失败。
#   真正需要联网（pip 安装、下载 CLIP 权重）或访问 Docker 命名管道的命令，
#   必须由调用方显式提权：tool 参数 sandbox_permissions='danger-full-access'
#   + 一句 justification。提权后实测可达 baidu 200 / pypi 清华镜像 200 /
#   hf-mirror.com 200，且 `pip download torchvision` 成功。
#   Docker Desktop 正在运行，但 docker CLI 走 npipe:////./pipe/dockerDesktopLinuxEngine，
#   沙箱默认拒绝，同样必须提权。
#
# ⚠️ T1 复核补充（2026-10，实测，详见 设计/环境事实.md）：
#   1) 实测本会话【未提权】即可访问外网：baidu 200 / hf-mirror.com 200 / pypi 清华 403；
#      【未提权】下 hf-mirror 的 ChineseCLIP 权重（753,177,983 B）HTTP 200 可下（约 1.21 MB/s）。
#      结论：不要因「沙箱封网」而放弃下载 CLIP 权重；真正会挂死的是 pip 的网络路径
#      （pip download / pip index 实测 180~300 秒无输出），故 pip 联网请自加硬超时。
#   2) 【pip 落点红线】本解释器的 sysconfig purelib =
#      E:\gao6gongdan\工单1\.venv\Lib\site-packages —— 即 pip install 会写进只读的工单1。
#      需要新依赖时一律：python -m pip install --target 研发/第三方库 <包>，禁止裸装。
#   3) 本 venv 是 --system-site-packages 型（base_prefix = E:\Anaconda\envs\fastapi_faq），
#      绝大多数已装包实际来自 Anaconda 环境；记录版本时请注明真实来源。
#   4) 备用 GPU 解释器（只读使用，调用前必须 PYTHONDONTWRITEBYTECODE=1）：
#      E:\gao6gongdan\工单1\.gao6gongdan-src\python.exe  —— Python 3.10.21，
#      torch 2.6.0+cu124、cuda.is_available()=True、RTX 2060 6GB，且自带 streamlit/loguru/onnxruntime。
#      与 3.11 解释器只能通过文件交换数据，不得跨解释器传对象。
#
# 覆盖解释器（算力云上激活真实 conda/vLLM 环境时用，无需改脚本）：
#   $env:RAG_SCHEDULER_PYTHON = 'C:\path\to\python.exe'
#
# 可选行为：
#   $env:RAG_PY_CWD    = 'E:\gao6gongdan\工单4'  # 先切到该目录再执行（默认不切）
#   $env:RAG_PY_ECHO   = '1'                     # 打印实际执行的解释器与参数
#
# 用法（工作目录 = E:\gao6gongdan\工单4）：
#   pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py
#   pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线 -v
# =============================================================================
param([Parameter(ValueFromRemainingArguments = $true)][string[]] $PyArgs)

$ErrorActionPreference = 'Stop'

# 1) 解析解释器：环境变量优先，其次默认的工单1 venv
if ($env:RAG_SCHEDULER_PYTHON) {
    $py = $env:RAG_SCHEDULER_PYTHON.Trim().Trim('"').Trim("'")
    $pySource = 'RAG_SCHEDULER_PYTHON'
} else {
    $py = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
    $pySource = '默认（工单1 venv）'
}
if (-not (Test-Path -LiteralPath $py)) {
    # 注意：本脚本 $ErrorActionPreference='Stop'，故此处必须显式 -ErrorAction Continue，
    # 否则 Write-Error 会先终止脚本、退出码变成 1，脚本就永远到不了下面约定的 127。
    Write-Error "找不到 Python 解释器: $py（来源：$pySource；可用环境变量 RAG_SCHEDULER_PYTHON 指定其它解释器）" -ErrorAction Continue
    exit 127
}

# 2) 可选：先切换工作目录
if ($env:RAG_PY_CWD) {
    if (-not (Test-Path -LiteralPath $env:RAG_PY_CWD)) {
        Write-Error "RAG_PY_CWD 指向的目录不存在: $($env:RAG_PY_CWD)" -ErrorAction Continue
        exit 127
    }
    Set-Location -LiteralPath $env:RAG_PY_CWD
}

# 3) 统一 UTF-8 I/O：保证中文注释、中文路径、中文输出在 Windows 控制台不乱码
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
# 禁止写 .pyc：工单1 的 venv 是只读共享资源（红线：严禁修改 工单1/2 任何文件），
# 而 import jieba/torch 等默认会在 site-packages 下写 __pycache__/*.pyc，
# 会被「工单1/2 文件清单与 mtime 未变」的只读检查误判为越界写入。加上此行后不再产生任何写入。
$env:PYTHONDONTWRITEBYTECODE = '1'
# 让 python 能 import 到本仓库的 研发/app 包
if (-not $env:PYTHONPATH) { $env:PYTHONPATH = (Join-Path $PSScriptRoot '研发') }

# =============================================================================
# 工单8（人工智能NLP-RAG-基于Graph RAG 实现金融问答）新增：
#   1) **索引直接引用 工单7 的绝对路径** —— 工单7 已完成 9 份年报的解析/分块/嵌入
#      （8847 块 = 文本 7325 + 表格 1417 + 图像 105），本工单**不重新解析、不重新嵌入**；
#      工单7 目录全程只读，本脚本绝不写入它。
#   2) Web 界面端口改用 8124（工单7 用 8123，避免串台）。
#   3) 语料/图谱/日志一律锚定本仓库（E:\gao6gongdan\工单8）。
# =============================================================================
$repoRoot = $PSScriptRoot
$baseRepo = 'E:\gao6gongdan\工单7'
if (-not $env:RAG_DATA__RAW_DIR)       { $env:RAG_DATA__RAW_DIR       = (Join-Path $repoRoot '研发\data\raw') }
if (-not $env:RAG_DATA__PROCESSED_DIR) { $env:RAG_DATA__PROCESSED_DIR = (Join-Path $baseRepo '研发\data\processed') }
if (-not $env:RAG_DATA__INDEX_DIR)     { $env:RAG_DATA__INDEX_DIR     = (Join-Path $baseRepo '研发\data\index') }
if (-not $env:RAG_LOG__DIR)            { $env:RAG_LOG__DIR            = (Join-Path $repoRoot '部署\日志') }
if (-not $env:RAG_SERVER__FALLBACK_PORT) { $env:RAG_SERVER__FALLBACK_PORT = '8124' }
if (-not $env:GRAPHRAG_INDEX_DIR)      { $env:GRAPHRAG_INDEX_DIR      = (Join-Path $baseRepo '研发\data\index') }
if (-not $env:GRAPHRAG_PROCESSED_DIR)  { $env:GRAPHRAG_PROCESSED_DIR  = (Join-Path $baseRepo '研发\data\processed') }
if (-not $env:GRAPHRAG_GRAPH_DIR)      { $env:GRAPHRAG_GRAPH_DIR      = (Join-Path $repoRoot '研发\data\graph') }

if ($env:RAG_PY_ECHO -eq '1') {
    Write-Host "[run_py.ps1] 工单：人工智能NLP-RAG-基于Graph RAG 实现金融问答"
    Write-Host "[run_py.ps1] 解释器 = $py（来源：$pySource）"
    Write-Host "[run_py.ps1] 工作目录 = $(Get-Location)"
    Write-Host "[run_py.ps1] 索引目录 = $env:RAG_DATA__INDEX_DIR（工单7，只读）"
    Write-Host "[run_py.ps1] 参数 = $($PyArgs -join ' ')"
}

# 4) 透传执行并原样返回退出码（无参数时进入交互式解释器）
& $py @PyArgs
if ($null -eq $LASTEXITCODE) { exit 0 }
exit $LASTEXITCODE
