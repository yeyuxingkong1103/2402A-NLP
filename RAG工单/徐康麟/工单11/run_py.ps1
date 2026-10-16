# =============================================================================
# 工单：人工智能NLP-RAG 项目-Embedding 模型微调任务（工单11）
# 用途：本仓库（E:\gao6gongdan\工单11）所有脚本 / 测试的统一 Python 入口。
#       照抄 工单7\run_py.ps1 写法，解释器默认值一致。
#
# 环境事实（实测，勿凭臆测；详见 设计/环境事实.md）：
#   唯一可用解释器：E:\gao6gongdan\工单1\.venv\Scripts\python.exe （Python 3.11.15，
#   --system-site-packages 型，base = E:\Anaconda\envs\fastapi_faq）
#   已装：torch(CPU) / transformers / sentence-transformers / numpy / jieba /
#         scikit-learn / faiss / Pillow / pytest / requests / huggingface_hub / pymupdf
#   缺失：accelerate / peft / datasets / pandas / matplotlib / loguru / streamlit
#   —— 因此本工单【自实现】微调循环（原生 nn.Module + AdamW），【自实现】PIL 折线图。
#
#   [pip 落点红线] 禁止裸 pip install（会写进只读的工单1）。
#   需要新依赖一律：python -m pip install --target 研发/第三方库 --no-deps <包>，且自加硬超时。
#
# 覆盖解释器：$env:RAG_SCHEDULER_PYTHON = 'C:\path\to\python.exe'
# 用法（工作目录 = E:\gao6gongdan\工单11）：
#   pwsh -NoProfile -File run_py.ps1 研发/src/generate_dataset.py
#   pwsh -NoProfile -File run_py.ps1 -m pytest 测试 -q
# =============================================================================
param([Parameter(ValueFromRemainingArguments = $true)][string[]] $PyArgs)

$ErrorActionPreference = 'Stop'

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

if ($env:RAG_PY_CWD) {
    if (-not (Test-Path -LiteralPath $env:RAG_PY_CWD)) {
        Write-Error "RAG_PY_CWD 指向的目录不存在: $($env:RAG_PY_CWD)" -ErrorAction Continue
        exit 127
    }
    Set-Location -LiteralPath $env:RAG_PY_CWD
}

# 统一 UTF-8 I/O + 禁止写 .pyc（工单1 是只读共享资源）
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
if (-not $env:PYTHONPATH) { $env:PYTHONPATH = (Join-Path $PSScriptRoot '研发') }

# CPU 训练：限制线程数，避免与其它并行工单抢资源（可用环境变量覆盖）
if (-not $env:WO11_NUM_THREADS) { $env:WO11_NUM_THREADS = '8' }

$repoRoot = $PSScriptRoot
if (-not $env:WO11_ROOT) { $env:WO11_ROOT = $repoRoot }

if ($env:RAG_PY_ECHO -eq '1') {
    Write-Host "[run_py.ps1] 工单：人工智能NLP-RAG 项目-Embedding 模型微调任务"
    Write-Host "[run_py.ps1] 解释器 = $py（来源：$pySource）"
    Write-Host "[run_py.ps1] 工作目录 = $(Get-Location)"
    Write-Host "[run_py.ps1] 参数 = $($PyArgs -join ' ')"
}

& $py @PyArgs
if ($null -eq $LASTEXITCODE) { exit 0 }
exit $LASTEXITCODE
