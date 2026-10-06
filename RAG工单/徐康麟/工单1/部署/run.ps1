# =============================================================================
# 统一启动入口 —— 基于 PDF 文档的 RAG 问答系统（工单1）
# =============================================================================
# 目录结构（按交付阶段分类）：
#
#     工单1/
#     ├─ 设计/                文档、规格说明、自检工具
#     │  ├─ 文档/             README / TECH_DOC / USER_MANUAL / DEMO_CHECKLIST …
#     │  └─ 规格说明/         check_local_env.py、dump_pdf.py 等辅助脚本
#     ├─ 研发/                源码与开发脚本
#     │  ├─ app/              Python 包（core / ui / storage / models / prompts）
#     │  └─ scripts/          build_index.py、evaluate.py、run_*.sh、setup_env.ps1
#     ├─ 测试/                pytest 用例（offline / online / user）+ pytest.ini
#     ├─ 优化/                评估结果（eval_results：CSV / 报告 / 明细）
#     ├─ 部署/                环境配置与启动入口（本脚本）
#     │  └─ 环境配置/         requirements.txt、environment.yml
#     ├─ data/                运行产物：语料、分块、索引、标准答案（与阶段无关）
#     ├─ logs/                日志（app.log / error.log / rag_trace.jsonl）
#     └─ models/              本地模型（嵌入 / 语音 / 翻译）
#
# 用法（在**项目根目录**执行）：
#     pwsh -File 部署\run.ps1 web          # 启动网页界面（默认）
#     pwsh -File 部署\run.ps1 index        # 建索引
#     pwsh -File 部署\run.ps1 eval         # 评估（中文 10 题 + 英文 5 题）
#     pwsh -File 部署\run.ps1 test         # 跑全部测试
#     pwsh -File 部署\run.ps1 doctor       # 环境自检
#     pwsh -File 部署\run.ps1 ask "问题"    # 命令行提问
#     pwsh -File 部署\run.ps1 envs         # 环境自检（含依赖/模型/端口清单）
#
# 可选参数：
#     -Python <路径>   指定解释器（默认自动探测）
#     -Port   <端口>   web 子命令的端口（默认 8501）
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('web', 'index', 'eval', 'test', 'doctor', 'ask', 'envs', 'llm', 'asr', 'check')]
    [string]$Command = 'web',

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest,

    [string]$Python = '',
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# ---------------------------------------------------------------------------
# 路径解析：本脚本位于 <root>/部署/run.ps1
# ---------------------------------------------------------------------------
$ProjectRoot = Split-Path -Parent $PSScriptRoot      # <root>
$SourceRoot = Join-Path $ProjectRoot '研发'
$AppEntry = Join-Path $SourceRoot 'app\main.py'

function Write-Head($text) {
    Write-Host ''
    Write-Host ('=' * 66) -ForegroundColor DarkGray
    Write-Host " $text" -ForegroundColor Cyan
    Write-Host ('=' * 66) -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 解释器探测：优先显式传入，其次项目内克隆环境，最后系统 python
# ---------------------------------------------------------------------------
function Resolve-Python {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-Path $Explicit)) { throw "指定的解释器不存在: $Explicit" }
        return (Resolve-Path $Explicit).Path
    }
    $candidates = @(
        (Join-Path $ProjectRoot '.gao6gongdan-src\python.exe'),   # 本机克隆环境（含全部依赖）
        (Join-Path $ProjectRoot '.venv\Scripts\python.exe')       # 备选虚拟环境
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    $found = Get-Command python -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    throw '未找到可用的 Python 解释器，请用 -Python <路径> 指定'
}

$PythonExe = Resolve-Python -Explicit $Python

# 让 `import app.xxx` 生效：源码根与项目根都要在 PYTHONPATH 上
$env:PYTHONPATH = @($SourceRoot, $ProjectRoot) -join [IO.Path]::PathSeparator

Write-Head "RAG 问答系统 · $Command"
Write-Host "  项目根目录: $ProjectRoot"
Write-Host "  源码目录  : $SourceRoot"
Write-Host "  解释器    : $PythonExe"

Push-Location $ProjectRoot
try {
    switch ($Command) {
        'web' {
            Write-Host "  界面地址  : http://127.0.0.1:$Port" -ForegroundColor Green
            & $PythonExe $AppEntry web --port $Port --address 127.0.0.1
        }
        'index' {
            & $PythonExe (Join-Path $SourceRoot 'scripts\build_index.py') @Rest
        }
        'eval' {
            & $PythonExe (Join-Path $SourceRoot 'scripts\evaluate.py') --no-llm --english @Rest
        }
        'test' {
            & $PythonExe -m pytest (Join-Path $ProjectRoot '测试\tests') -q @Rest
        }
        'doctor' {
            & $PythonExe $AppEntry doctor
        }
        'ask' {
            if (-not $Rest -or $Rest.Count -eq 0) { throw 'ask 需要提供问题文本，例如: run.ps1 ask "注册资本是多少？"' }
            & $PythonExe $AppEntry ask @Rest
        }
        'envs' {
            & $PythonExe (Join-Path $ProjectRoot '设计\规格说明\check_local_env.py')
        }
        'llm' {
            # 本地 CPU 推理服务（无 GPU 时替代 vLLM，仅供联调）
            & $PythonExe (Join-Path $SourceRoot 'scripts\run_local_llm.py') --port 8000 @Rest
        }
        'asr' {
            # 本地语音识别服务（无 GPU 时替代 vLLM Whisper，仅供联调）
            & $PythonExe (Join-Path $SourceRoot 'scripts\run_local_asr.py') --port 8001 @Rest
        }
        'check' {
            # 端到端联调：LLM 流式生成 + 语音识别 + 中英双语 + 兜底
            & $PythonExe (Join-Path $SourceRoot 'scripts\check_integration.py') @Rest
        }
    }
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($code -ne 0) { Write-Host "`n[退出码 $code]" -ForegroundColor Yellow }
exit $code
