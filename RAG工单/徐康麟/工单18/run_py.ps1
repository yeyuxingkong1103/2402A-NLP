# =============================================================================
# 工单：人工智能NLP-RAG-实现文档质量评估Skill 并集成至智能体工作流工单
# 用途：本仓库（E:\gao6gongdan\工单18）所有脚本 / 测试的统一 Python 入口。
#
# 环境事实（实测，来自 _共享约定.md §2，勿凭提示词臆测）：
#   唯一可用解释器 = 工单1 的 venv：
#       E:\gao6gongdan\工单1\.venv\Scripts\python.exe   （Python 3.11.15，
#        --system-site-packages 型，base = E:\Anaconda\envs\fastapi_faq）
#   已装：pymupdf(fitz) / PyYAML / numpy / scipy / jieba / scikit-learn / faiss /
#         torch(CPU) / transformers / sentence-transformers / Pillow /
#         fastapi / uvicorn / pytest / requests / openai / huggingface_hub
#   缺失：pandas / python-docx / cv2 / pytesseract / paddleocr / easyocr /
#         onnxruntime / matplotlib / loguru —— 本工单一律不依赖，docx 用标准库 zipfile 自解析。
#   ⚠️ pip 落点红线：禁止裸 pip install（会写进只读的工单1）。
#   ⚠️ 沙箱 workspace-write：只能写 E:\gao6gongdan 内。
#
# 覆盖解释器：$env:RAG_SCHEDULER_PYTHON = 'C:\path\to\python.exe'
#
# 用法（工作目录 = E:\gao6gongdan\工单18）：
#   pwsh -NoProfile -File run_py.ps1 研发/scripts/run_inspection.py --folder <dir> --limit 20
#   pwsh -NoProfile -File run_py.ps1 -m pytest 测试/单元测试 -q
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
    Write-Error "找不到 Python 解释器: $py（来源：$pySource）" -ErrorAction Continue
    exit 127
}

# 2) 统一 UTF-8 I/O，保证中文注释、中文路径、中文输出不乱码
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
# 禁止写 .pyc：工单1 的 venv 是只读共享资源
$env:PYTHONDONTWRITEBYTECODE = '1'
# 让 python 能 import 到本仓库的 研发/ 包（document_quality_assessment / integration）
$repoRoot = $PSScriptRoot
if (-not $env:PYTHONPATH) { $env:PYTHONPATH = (Join-Path $repoRoot '研发') }

# 3) 路径一律锚定本仓库
if (-not $env:DQA_CONFIG)  { $env:DQA_CONFIG  = (Join-Path $repoRoot '研发\assessment_config.yaml') }
if (-not $env:DQA_LOG_DIR) { $env:DQA_LOG_DIR = (Join-Path $repoRoot '部署\日志') }

if ($env:RAG_PY_ECHO -eq '1') {
    Write-Host "[run_py.ps1] 工单：人工智能NLP-RAG-实现文档质量评估Skill 并集成至智能体工作流工单"
    Write-Host "[run_py.ps1] 解释器 = $py（来源：$pySource）"
    Write-Host "[run_py.ps1] 工作目录 = $(Get-Location)"
    Write-Host "[run_py.ps1] 参数 = $($PyArgs -join ' ')"
}

# 4) 透传执行并原样返回退出码
& $py @PyArgs
if ($null -eq $LASTEXITCODE) { exit 0 }
exit $LASTEXITCODE
