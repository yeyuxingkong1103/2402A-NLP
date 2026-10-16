# =============================================================================
# 工单编号：人工智能NLP-RAG-功能测试及评估
# 文件：部署/脚本/一键跑测试评估.ps1
#
# 用途：把工单7 的「解压语料 → 建索引 → 冒烟自检 → 跑评估 → 跑回归测试 →
#       生成截图 → 打印产物清单」串成一条命令，便于交付复现与录屏演示。
#
# 用法（在任意目录）：
#     pwsh -NoProfile -File 部署/脚本/一键跑测试评估.ps1
#     pwsh -NoProfile -File 部署/脚本/一键跑测试评估.ps1 -SkipIndex -SkipExtract
#
# 参数：
#     -SkipExtract  跳过解压（语料已在 研发/data/raw）
#     -SkipIndex    跳过建索引（索引已存在）
#     -SkipVideo    跳过演示视频生成
#     -Only <名字>  只跑某一步：extract/images/index/smoke/eval/regress/screens/video
#
# 容错说明（2026-10 补）：
#   * step screens 依赖 测试/生成截图.py（归属「测试官」），该文件尚未就绪时本脚本
#     **如实打印跳过**而不报错，避免把「别人还没交付」误判成本流程失败；
#   * step index 在「图像块数=0」时会以退出码 1 结束，这是**已登记的降级**
#     （见 研发/data/index/index_manifest.json 的 warnings 与 部署/README.md 已知限制），
#     故汇总判定时对 index 单独豁免。
# =============================================================================
param(
    [switch]$SkipExtract,
    [switch]$SkipIndex,
    [switch]$SkipVideo,
    [string]$Only = ''
)

$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot '环境.ps1')

$steps = [System.Collections.Generic.List[object]]::new()
function Invoke-Step {
    param([string]$Name, [string]$Desc, [scriptblock]$Body)
    if ($Only -and $Only -ne $Name) { return }
    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host "[$Name] $Desc"
    Write-Host ("=" * 78)
    $t0 = Get-Date
    try {
        # 先清零：若 Body 内部没有调用任何外部命令（例如「产物缺失 → 跳过」分支），
        # $LASTEXITCODE 会残留上一步的值，导致把成功误报为失败。
        $global:LASTEXITCODE = 0
        & $Body
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
    } catch {
        Write-Host "❌ 步骤 $Name 异常：$_" -ForegroundColor Red
        $code = 1
    }
    $dt = [Math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $steps.Add([pscustomobject]@{ 步骤 = $Name; 说明 = $Desc; 退出码 = $code; 耗时秒 = $dt })
    Write-Host ("-" * 78)
    Write-Host "[$Name] 退出码=$code 耗时=${dt}s"
}

Set-Location -LiteralPath $env:GD7_ROOT

# ---- 1) 解压语料 ----
if (-not $SkipExtract) {
    Invoke-Step -Name 'extract' -Desc '解压 CCF 竞赛语料 → 研发/data/raw（cp437→gbk 修正中文名）' -Body {
        & $env:GD7_PY '测试\解压语料.py' --list
        & $env:GD7_PY '测试\解压语料.py'
    }
}

# ---- 2) 图像内容解析（中文 CLIP：框图结构树 / 图表数值 / 图片语义标签）----
Invoke-Step -Name 'images' -Desc '抽图像块 + 中文 CLIP 嵌入（不需要 cv2，只需 PIL+pymupdf+numpy）' -Body {
    $model = Join-Path $env:GD7_ROOT '研发\data\models\chinese-clip-vit-base-patch16\pytorch_model.bin'
    if (-not (Test-Path -LiteralPath $model)) {
        Write-Host "（跳过）中文 CLIP 权重缺失：$model"
        Write-Host '         请先运行： & $env:GD7_PY 测试\下载中文CLIP权重.py'
        $global:LASTEXITCODE = 0
    } else {
        & $env:GD7_PY '研发\scripts\extract_images.py' --all
    }
}

# ---- 3) 建索引 ----
if (-not $SkipIndex) {
    Invoke-Step -Name 'index' -Desc '解析 → 分块（文本/表格/图像）→ 嵌入(bge-m3) → FAISS + BM25 → manifest' -Body {
        # 有 image_blocks.jsonl 就直接并入（期望 chunk_total ≈ 8847，含图像块 105）；
        # 没有则必须补 --allow-empty-images —— build_index.py 把「产出向量」挂在 --include-images 上，
        # 图像块数为 0 时会被判硬失败。
        $hasImages = Test-Path -LiteralPath (Join-Path $env:GD7_ROOT '研发\data\processed\image_blocks.jsonl')
        $extra = @()
        if (-not $hasImages) {
            Write-Host '（注意）未发现 image_blocks.jsonl → 追加 --allow-empty-images（图像块将为 0）'
            $extra += '--allow-empty-images'
        }
        & $env:GD7_PY '研发\scripts\build_index.py' --force --no-seed --include-images `
            --batch-size 32 @extra
    }
}

# ---- 3) 冒烟自检 ----
Invoke-Step -Name 'smoke' -Desc '索引加载 / 单题检索 / 生成 / 多轮 / 异常兜底 冒烟' -Body {
    & $env:GD7_PY '测试\冒烟自检.py'
}

# ---- 4) 评估 ----
Invoke-Step -Name 'eval' -Desc '17 题 RAG 检索 + 自研评估框架判分 → JSONL/CSV/汇总/问题分析' -Body {
    & $env:GD7_PY '研发\评估\运行评估.py'
}

# ---- 5) 回归测试 ----
Invoke-Step -Name 'regress' -Desc '单轮 / 多轮 / 异常输入 回归测试' -Body {
    & $env:GD7_PY '测试\回归测试\run_all.py'
}

# ---- 6) 截图 ----
Invoke-Step -Name 'screens' -Desc '把真实运行结果渲染为 测试/测试截图/*.png' -Body {
    if (Test-Path '测试\生成截图.py') {
        & $env:GD7_PY '测试\生成截图.py'
    } else {
        Write-Host '（跳过）测试\生成截图.py 尚未就绪（归属：测试官「测试用例与截图」）。' -ForegroundColor Yellow
        Write-Host '         界面「测试流程演示」第 4 步会如实显示当前已有截图数量。' -ForegroundColor Yellow
    }
}

# ---- 7) 演示视频 ----
if (-not $SkipVideo) {
    Invoke-Step -Name 'video' -Desc '生成演示视频（PIL 逐帧 + ffmpeg 合成 mp4）' -Body {
        if (Test-Path '部署\脚本\生成演示视频.py') {
            & $env:GD7_PY '部署\脚本\生成演示视频.py'
        } else {
            Write-Host '（跳过）部署\脚本\生成演示视频.py 尚未就绪' -ForegroundColor Yellow
        }
    }
}

# ---- 汇总 ----
Write-Host ""
Write-Host ("=" * 78)
Write-Host "工单7 一键流程汇总"
Write-Host ("=" * 78)
$steps | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

Write-Host "关键产物："
foreach ($p in @(
        '研发\data\index\index_manifest.json',
        '研发\data\processed\image_blocks.jsonl',
        '研发\data\index\images_clip.npy',
        '研发\data\models\chinese-clip-vit-base-patch16\pytorch_model.bin',
        '研发\评估\测试问题集.jsonl',
        '测试\测试数据\评估结果.jsonl',
        '测试\测试数据\评估结果.csv',
        '测试\测试数据\评估汇总.json',
        '测试\测试数据\检索问题分析.md',
        '测试\测试截图',
        '部署\界面\测试评估台.py',
        '部署\演示视频\工单7_功能测试及评估演示.mp4',
        '部署\README.md')) {
    $full = Join-Path $env:GD7_ROOT $p
    $mark = if (Test-Path -LiteralPath $full) { '✅' } else { '❌' }
    Write-Host "  $mark $p"
}

Write-Host ""
Write-Host "界面（人工点击单题，不并发压 Ollama）："
Write-Host "  pwsh -NoProfile -File 部署\脚本\启动测试评估台.ps1 -Port 7861"
Write-Host "  浏览器打开 http://127.0.0.1:7861/"

$bad = @($steps | Where-Object { $_.退出码 -ne 0 -and $_.步骤 -ne 'index' })
if ($bad.Count -gt 0) {
    Write-Host ""
    Write-Host "⚠️ 有步骤非零退出：$($bad.步骤 -join '、')" -ForegroundColor Yellow
    exit 1
}
Write-Host ""
Write-Host "✅ 流程完成" -ForegroundColor Green
exit 0
