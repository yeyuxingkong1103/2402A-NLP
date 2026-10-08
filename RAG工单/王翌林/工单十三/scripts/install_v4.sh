#!/bin/bash
# ================================================================
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# scripts/install_v4.sh —— 工单四环境安装（依赖 + MySQL + Milvus rag_images + 解析入库）
#
# 在工单三 install_v3.sh 基础上增量扩展：
#   1. 新增 PyMuPDF / PaddleOCR / transformers（含 Chinese-CLIP、BLIP 实现）/ sentence-transformers
#   2. 新增 MySQL 业务库初始化（复用工单二 src/db.py，幂等）
#   3. 新增 Milvus rag_images collection 初始化（不影响 rag_chunks / rag_tables）
#   4. 运行 PDF 文本/表格入库（复用工单三脚本，已有数据自动跳过）
#   5. 运行图像提取 + 多模态解析 + 图像入库（Qwen2-VL 未就绪时自动 OCR-only）
#
# 用法：
#   bash scripts/install_v4.sh                    # 完整安装（幂等，可重复执行）
#   bash scripts/install_v4.sh --skip-ingest      # 只装环境不入库
# ================================================================
set -e

echo "================================================"
echo "工单四：PDF图像内容解析及检索优化 - 环境安装"
echo "工单编号：人工智能NLP-RAG-图像内容解析及检索优化"
echo "================================================"

# 工单四：项目根目录（全部路径加引号，处理中文路径"工单四/附件"）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "[INFO] 项目根目录: $PROJECT_ROOT"

# 工单四：PDF 附件目录（两个招股说明书所在位置）
ATTACH_DIR="${ATTACH_DIR:-/home/dabaie/code/工单/附件}"
SKIP_INGEST=false
[ "${1:-}" = "--skip-ingest" ] && SKIP_INGEST=true

# ================= 0. GPU 检查 =================
echo ""
echo "[STEP 0] 检查 CUDA / GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    echo "[OK] GPU: $GPU_NAME"
    export RAG_EMBED_DEVICE=cuda
else
    echo "[WARN] 未检测到 nvidia-smi，将使用 CPU（速度较慢）"
    export RAG_EMBED_DEVICE=cpu
fi
export RAG_EMBED_BATCH_SIZE=8
export RAG_EMBED_MAX_SEQ=512

# ================= 1. Python 环境 =================
echo ""
echo "[STEP 1] 配置 Python 环境..."
PYTHON_BIN="/home/dabaie/code/my_project/.venv/bin/python"
if [ -x "$PYTHON_BIN" ]; then
    echo "[OK] 使用现有 venv: $PYTHON_BIN"
else
    echo "[INFO] 创建项目内 venv .venv ..."
    python3 -m venv "$PROJECT_ROOT/.venv"
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
    echo "[OK] venv 创建完成"
fi
$PYTHON_BIN --version

# ================= 2. 安装依赖 =================
echo ""
echo "[STEP 2] 安装 Python 依赖（已安装的自动跳过）..."
$PYTHON_BIN -m pip install --quiet --upgrade pip

# 工单四：基础依赖（文本/表格/服务，与工单三一致）
$PYTHON_BIN -m pip install --quiet \
    pdfplumber PyMuPDF rank_bm25 sentence-transformers \
    fastapi "uvicorn[standard]" streamlit pydantic loguru \
    pymilvus pymysql sqlalchemy python-dotenv jieba \
    FlagEmbedding httpx openai \
    || echo "[WARN] 部分基础依赖安装失败，请检查网络/源"

# 工单四：图像解析依赖
#   - PyMuPDF：图像提取（L1 位图 + L2 矢量渲染）
#   - paddlepaddle + paddleocr：图中文字 OCR（3.x，需 enable_mkldnn=False）
#   - transformers：Chinese-CLIP（ChineseCLIPModel）与 BLIP（BlipFor*）经 transformers 加载
#   - sentence-transformers：bge-m3 文本向量 / bge-reranker 重排
$PYTHON_BIN -m pip install --quiet \
    "paddlepaddle>=3.0" "paddleocr>=3.0" \
    "transformers>=4.40" \
    || echo "[WARN] 图像依赖安装失败（PaddleOCR/transformers），请手动执行 scripts/install_paddleocr_step3.sh"

# 工单四：依赖自检
$PYTHON_BIN - << 'PYEOF'
mods = {"fitz": "PyMuPDF", "paddleocr": "PaddleOCR", "transformers": "transformers",
        "sentence_transformers": "sentence-transformers", "pymilvus": "pymilvus"}
for m, name in mods.items():
    try:
        __import__(m)
        print(f"[OK] {name}")
    except ImportError:
        print(f"[MISS] {name} 未安装")
PYEOF

# ================= 3. 检查 .env =================
echo ""
echo "[STEP 3] 检查 .env 配置..."
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cat > "$PROJECT_ROOT/.env" << 'ENVEOF'
# 工单四：环境变量（人工智能NLP-RAG-图像内容解析及检索优化）
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MILVUS_HOST=localhost
MILVUS_PORT=19530
MYSQL_HOST=localhost
MYSQL_PORT=3307
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=rag_pdf_qa
RAG_EMBED_DEVICE=cuda
RAG_EMBED_BATCH_SIZE=8
RAG_EMBED_MAX_SEQ=512
ENVEOF
    echo "[WARN] .env 已创建，请编辑填入真实 API Key / MySQL 密码"
else
    echo "[OK] .env 已存在"
fi

if [ "$SKIP_INGEST" = true ]; then
    echo ""
    echo "[DONE] --skip-ingest：环境安装完成，跳过初始化与入库"
    exit 0
fi

# ================= 4. 初始化 MySQL（复用工单二 src/db.py，幂等） =================
echo ""
echo "[STEP 4] 初始化 MySQL（建库 + 建表，幂等）..."
$PYTHON_BIN - << 'PYEOF' || echo "[WARN] MySQL 初始化失败（服务未启动或账号密码错误），可在启动 MySQL 后重跑本脚本"
from dotenv import load_dotenv  # 工单四：显式加载 .env
load_dotenv()
from src.db import create_database_if_not_exists, create_tables, test_connection
ver = test_connection()
if ver:
    print(f"[OK] MySQL 连接正常: {ver}")
    create_database_if_not_exists()
    create_tables()
else:
    print("[WARN] MySQL 连接失败")
PYEOF

# ================= 5. 检查 Milvus 并初始化 rag_images =================
echo ""
echo "[STEP 5] 检查 Milvus 连接..."
if ! $PYTHON_BIN -c "
from pymilvus import connections
connections.connect(host='localhost', port='19530')
print('[OK] Milvus 连接正常')
" 2>&1 | tail -1; then
    echo "[WARN] Milvus 未启动，请先启动后重跑：sudo systemctl start milvus"
fi

# ================= 6. PDF 文本/表格入库（工单三链路，已有数据自动跳过） =================
echo ""
echo "[STEP 6] 检查文本/表格库数据量..."
NEED_TEXT_INGEST=$($PYTHON_BIN - << 'PYEOF' 2>/dev/null || echo NEED
from dotenv import load_dotenv
load_dotenv()
from pymilvus import MilvusClient
c = MilvusClient(uri="http://localhost:19530")
rows = {n: (c.get_collection_stats(n).get("row_count", 0) if n in c.list_collections() else 0)
        for n in ("rag_chunks", "rag_tables")}
print("SKIP" if rows["rag_chunks"] > 100 and rows["rag_tables"] > 50 else "NEED")
PYEOF
)

if [ "$NEED_TEXT_INGEST" = "SKIP" ]; then
    echo "[OK] rag_chunks / rag_tables 已有数据，跳过文本与表格入库（如需重建请用工单三脚本 --rebuild）"
else
    echo "[INFO] 开始表格入库（init_milvus_v3.py）..."
    $PYTHON_BIN scripts/init_milvus_v3.py 2>&1 | tail -3 || true
    echo "[INFO] 开始文本入库（ingest_text_v3.py）..."
    $PYTHON_BIN scripts/ingest_text_v3.py 2>&1 | tail -3 || true
fi

# ================= 7. 图像提取 + 多模态解析 + 图像入库 =================
echo ""
echo "[STEP 7] 图像提取与入库（工单四新增链路）..."
for PDF_NAME in 招股说明书1 招股说明书2; do
    PDF_FILE="$ATTACH_DIR/$PDF_NAME.pdf"
    IMG_JSON="$PROJECT_ROOT/data/images/${PDF_NAME}_images.json"
    PARSED_JSON="$PROJECT_ROOT/data/image_descriptions/${PDF_NAME}_images_parsed.json"

    # 工单四：7.1 图像提取（PyMuPDF L1 位图 + L2 矢量渲染）
    if [ -f "$IMG_JSON" ]; then
        echo "[OK] $PDF_NAME 图像清单已存在，跳过提取"
    elif [ -f "$PDF_FILE" ]; then
        echo "[INFO] 提取 $PDF_NAME 图像（矢量图经渲染提取）..."
        $PYTHON_BIN -m src.image_parser.image_extractor --pdf "$PDF_FILE" --out "$IMG_JSON" || \
            echo "[WARN] $PDF_NAME 图像提取失败，请检查 PDF"
    else
        echo "[WARN] 未找到 $PDF_FILE，跳过 $PDF_NAME（可用 ATTACH_DIR=... 指定附件目录）"
    fi

    # 工单四：7.2 多模态解析（Qwen2-VL caption/VQA + PaddleOCR；VLM 缺失自动降级 OCR-only）
    if [ -f "$PARSED_JSON" ]; then
        echo "[OK] $PDF_NAME 图像解析结果已存在，跳过解析"
    elif [ -f "$IMG_JSON" ]; then
        echo "[INFO] 解析 $PDF_NAME 图像（VLM 未就绪时自动 OCR-only）..."
        $PYTHON_BIN -m src.image_parser.image_parser --images "$IMG_JSON" --out "$PARSED_JSON" || \
            echo "[WARN] $PDF_NAME 图像解析失败，可后续单跑 image_parser"
    fi
done

# 工单四：7.3 图像入库 rag_images（--rebuild 幂等：先删后插，不影响 rag_chunks/rag_tables）
echo "[INFO] 初始化 rag_images 并入库（init_milvus_v4.py --rebuild）..."
$PYTHON_BIN scripts/init_milvus_v4.py --rebuild 2>&1 | tail -5 || \
    echo "[WARN] rag_images 入库失败，请检查 Milvus"

# ================= 8. 库存统计 =================
echo ""
echo "[STEP 8] Milvus 库存统计..."
$PYTHON_BIN - << 'PYEOF'
from dotenv import load_dotenv
load_dotenv()
from pymilvus import MilvusClient
c = MilvusClient(uri="http://localhost:19530")
for n in ("rag_chunks", "rag_tables", "rag_images"):
    rows = c.get_collection_stats(n).get("row_count", 0) if n in c.list_collections() else 0
    print(f"  {n}: {rows} 行")
PYEOF

echo ""
echo "================================================"
echo "[DONE] 工单四环境安装完成！"
echo "  GPU:        ${GPU_NAME:-CPU only}"
echo "  Python:     $PYTHON_BIN"
echo "  启动服务:   bash scripts/start_v4.sh"
echo "  停止服务:   bash scripts/stop_v4.sh"
echo "  对比评估:   python scripts/evaluate_v4.py"
echo "================================================"
