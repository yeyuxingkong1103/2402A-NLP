# =============================================================================
# 工单编号：人工智能NLP-RAG-功能测试及评估
# 文件：部署/脚本/环境.ps1 —— 统一环境变量（本工单所有脚本/测试的公共前置）
#
# 用法（在任意脚本开头 dot-source 它）：
#     . "$PSScriptRoot\..\..\部署\脚本\环境.ps1"        # 或绝对路径
#     & $env:GD7_PY 研发/评估/运行评估.py
#
# 作用：
#   1) 指定唯一可用解释器（工单1 venv，Python 3.11.15）；
#   2) 统一 UTF-8 I/O + 禁止写 .pyc（工单1 venv 是只读共享资源）；
#   3) 把语料/索引/日志路径钉在本仓库（工单7），避免继承到指向 工单6 的残留变量；
#   4) 导出 $env:GD7_PY / $env:GD7_ROOT 供调用方使用。
# =============================================================================

$ErrorActionPreference = 'Stop'

$GD7_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # ...\工单7
if (-not (Test-Path -LiteralPath $GD7_ROOT)) { $GD7_ROOT = 'E:\gao6gongdan\工单7' }

$GD7_PY = if ($env:RAG_SCHEDULER_PYTHON) { $env:RAG_SCHEDULER_PYTHON } else { 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $GD7_PY)) { throw "找不到 Python 解释器：$GD7_PY" }

$env:GD7_ROOT = $GD7_ROOT
$env:GD7_PY = $GD7_PY

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
# PYTHONPATH 顺序很重要（实测踩坑）：
#   「研发」必须在最前，**「研发\第三方库」必须放在最后**。
#   原因：第三方库里有 numpy 2.4.6，而 venv 自带的 faiss 是针对较旧 numpy 编译的 C 扩展；
#   把第三方库放前面会让 numpy 2.x 覆盖旧版 → `import faiss` 抛
#   `AttributeError: _ARRAY_API not found` → 向量检索整条链路失败。
#   放最后则：venv 的 numpy/faiss 优先，第三方库只补 venv 缺失的包（gradio/pandas/pdfplumber）。
$GD7_SITE = Join-Path $GD7_ROOT '研发\第三方库'
$env:GD7_SITE = $GD7_SITE
$env:PYTHONPATH = @((Join-Path $GD7_ROOT '研发'), $GD7_SITE) -join ';'

$env:RAG_DATA__RAW_DIR = (Join-Path $GD7_ROOT '研发\data\raw')
$env:RAG_DATA__PROCESSED_DIR = (Join-Path $GD7_ROOT '研发\data\processed')
$env:RAG_DATA__INDEX_DIR = (Join-Path $GD7_ROOT '研发\data\index')
$env:RAG_LOG__DIR = (Join-Path $GD7_ROOT '部署\日志')

Write-Host "[环境.ps1] 工单：人工智能NLP-RAG-功能测试及评估"
Write-Host "[环境.ps1] 仓库 = $GD7_ROOT"
Write-Host "[环境.ps1] 解释器 = $GD7_PY"
