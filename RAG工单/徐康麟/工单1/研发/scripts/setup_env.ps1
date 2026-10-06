# =============================================================================
# gao6gongdan 环境一键构建脚本（Windows / conda）
# =============================================================================
# 本脚本有两种使用场景，请按需选择：
#
# 【场景 A：算力云 4090 部署（推荐，网络正常）】
#   直接按 requirements.txt 安装到全新的 gao6gongdan 环境：
#       conda create -n gao6gongdan python=3.11 -y
#       conda activate gao6gongdan
#       pip install -r 部署/环境配置/requirements.txt
#   vLLM / SGLang 单独装，见 研发/scripts/run_vllm.sh
#
# 【场景 B：本机 Windows 开发环境（外网极慢/受限）】
#   本脚本用「克隆已有 conda 环境 + 补齐缺失包」的方式离线构建，
#   避免长时间下载。前提是本机已有含 pymupdf / chromadb / torch 的环境。
#
# 用法（PowerShell，需在项目根目录执行）：
#     pwsh -File 研发/scripts/setup_env.ps1 -SourceEnv sglang -TargetDir .\.gao6gongdan-src
# =============================================================================

param(
    [string]$SourceEnv = "sglang",
    [string]$TargetDir = ".\.gao6gongdan-src",
    [string]$CondaRoot = "E:\Anaconda"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------
# 1. 克隆基础环境
# ---------------------------------------------------------------------------
Write-Step "1/4 克隆基础环境 $SourceEnv -> $TargetDir"
$src = Join-Path (Join-Path $CondaRoot "envs") $SourceEnv
if (-not (Test-Path $src)) { throw "源环境不存在: $src" }
if (Test-Path $TargetDir) { Remove-Item $TargetDir -Recurse -Force }
# 注意：必须带 /E 递归子目录，否则会产生大量空目录（空目录会被 Python 当成
# 命名空间包，从而遮蔽真实模块，导致 import 报 "no attribute __version__"）
robocopy $src $TargetDir /E /NFL /NDL /NJH /NJS /MT:16 /R:1 /W:1 | Out-Null
Write-Host "克隆完成"

# ---------------------------------------------------------------------------
# 2. 清理空目录（关键修复步骤，勿删）
# ---------------------------------------------------------------------------
Write-Step "2/4 清理空目录（防止命名空间包遮蔽）"
$site = Join-Path $TargetDir "lib\site-packages"
if (-not (Test-Path $site)) { $site = Join-Path $TargetDir "Lib\site-packages" }
$removed = 0
Get-ChildItem $site -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $files = Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue
    if (-not $files -or $files.Count -eq 0) { Remove-Item $_.FullName -Recurse -Force; $removed++ }
}
Write-Host "清理空目录: $removed 个"

# ---------------------------------------------------------------------------
# 3. 从本机其它环境补齐缺失的纯 Python 包
# ---------------------------------------------------------------------------
Write-Step "3/4 补齐缺失包"
$copies = @(
    @{ Name = "loguru";           From = "$CondaRoot\envs\fastapi_ds\Lib\site-packages"; Version = "loguru-0.7.3.dist-info" },
    @{ Name = "win32_setctime";   From = "$CondaRoot\envs\fastapi_ds\Lib\site-packages"; Version = "win32_setctime-1.2.0.dist-info" },
    @{ Name = "jieba";            From = "$CondaRoot\Lib\site-packages";                Version = "jieba-0.42.1.dist-info" }
)
foreach ($item in $copies) {
    foreach ($pkg in @($item.Name, $item.Version)) {
        $from = Join-Path $item.From $pkg
        if (-not (Test-Path $from)) { Write-Host "  跳过（源不存在）: $pkg"; continue }
        $to = Join-Path $site $pkg
        if (Test-Path $to) { Remove-Item $to -Recurse -Force }
        robocopy $from $to /E /NFL /NDL /NJH /NJS /R:1 /W:1 | Out-Null
        Write-Host "  已复制: $pkg"
    }
}

# ---------------------------------------------------------------------------
# 4. 自检
# ---------------------------------------------------------------------------
Write-Step "4/4 环境自检"
$python = Join-Path $TargetDir "python.exe"
$probe = @'
import warnings, sys
warnings.filterwarnings("ignore")
mods = ["numpy","pymupdf","loguru","jieba","sqlalchemy","pydantic",
        "chromadb","sentence_transformers","torch","transformers","openai"]
ok = 0
for m in mods:
    try:
        mod = __import__(m); ok += 1
        print(f"  OK   {m} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"  FAIL {m}: {type(exc).__name__}: {exc}")
print(f"\n结果: {ok}/{len(mods)} 可用 | Python {sys.version.split()[0]}")
sys.exit(0 if ok >= len(mods) - 1 else 1)
'@
$probeFile = Join-Path $env:TEMP "gao6gongdan_env_probe.py"
Set-Content -Path $probeFile -Value $probe -Encoding UTF8
& $python $probeFile
if ($LASTEXITCODE -ne 0) { Write-Host "`n环境自检未全部通过，请检查上面的 FAIL 项。" -ForegroundColor Yellow }
else { Write-Host "`n环境就绪。使用方式：`n  & '$python' scripts\build_index.py`n  & '$python' -m pytest tests\offline -v" -ForegroundColor Green }
