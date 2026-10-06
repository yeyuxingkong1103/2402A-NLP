# =============================================================================
# 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 用途：本仓库（E:\gao6gongdan\工单3）所有脚本 / 测试的统一 Python 入口。
#
# 环境事实（实测，勿凭提示词臆测）：
#   本机并【不存在】名为 gao6gongdan 的 conda 环境（conda 命令本身报错不可用）。
#   唯一可用解释器是工单1 的 venv：
#       E:\gao6gongdan\工单1\.venv\Scripts\python.exe   （Python 3.11.15）
#   该 venv 已装 pymupdf(fitz) / numpy / jieba / scikit-learn / faiss / torch(cpu) /
#   transformers / sentence-transformers / openai / fastapi / uvicorn / pytest 等。
#   缺失且本机断网无法安装：streamlit、loguru、pdfplumber、chromadb、langchain、
#   rank_bm25、ragas、pandas、gradio、markdown、bs4、lxml
#   —— 对应能力一律自实现或改用替代方案，禁止 pip install（本机断网会挂死）。
#
# 覆盖解释器（算力云上激活真实 conda/vLLM 环境时用，无需改脚本）：
#   $env:RAG_SCHEDULER_PYTHON = 'C:\path\to\python.exe'
#
# 可选行为（都不设时为最小行为，与工单2 入口一致）：
#   $env:RAG_PY_CWD    = 'E:\gao6gongdan\工单3'  # 先切到该目录再执行（默认不切）
#   $env:RAG_PY_ECHO   = '1'                     # 打印实际执行的解释器与参数
#
# 用法（工作目录 = E:\gao6gongdan\工单3）：
#   pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_config_env.py
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

# 2) 可选：先切换工作目录（仓库内脚本按仓库根目录相对路径调用时才需要）
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

if ($env:RAG_PY_ECHO -eq '1') {
    Write-Host "[run_py.ps1] 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化"
    Write-Host "[run_py.ps1] 解释器 = $py（来源：$pySource）"
    Write-Host "[run_py.ps1] 工作目录 = $(Get-Location)"
    Write-Host "[run_py.ps1] 参数 = $($PyArgs -join ' ')"
}

# 4) 透传执行并原样返回退出码（无参数时进入交互式解释器）
& $py @PyArgs
if ($null -eq $LASTEXITCODE) { exit 0 }
exit $LASTEXITCODE
