# =============================================================================
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 用途：本仓库所有脚本的统一 Python 入口。
#
# 环境说明（重要）：
#   本机并不存在名为 gao6gongdan 的 conda 环境；实际可用的是工单1 的 venv
#   （Python 3.11.15，由 conda 环境 fastapi_faq 派生，--system-site-packages），
#   已装好 torch / transformers / sentence-transformers / fastapi / pymupdf /
#   jieba / scikit-learn / pytest / numpy / openai 等依赖。
#   若在算力云上激活了真实 conda 环境，只需设置环境变量 RAG_SCHEDULER_PYTHON
#   指向该解释器，无需修改任何脚本。
#
# 用法（工作目录 = 工单2）：
#   pwsh -File run_py.ps1 -m pytest 测试/离线 -v
#   pwsh -File run_py.ps1 测试/用户/simulate_user_v2.py
# =============================================================================
param([Parameter(ValueFromRemainingArguments = $true)][string[]] $PyArgs)

$ErrorActionPreference = 'Stop'
if ($env:RAG_SCHEDULER_PYTHON) {
    $py = $env:RAG_SCHEDULER_PYTHON
} else {
    $py = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
}
if (-not (Test-Path $py)) {
    Write-Error "找不到 Python 解释器: $py（可用环境变量 RAG_SCHEDULER_PYTHON 指定）"
    exit 127
}
& $py @PyArgs
exit $LASTEXITCODE
