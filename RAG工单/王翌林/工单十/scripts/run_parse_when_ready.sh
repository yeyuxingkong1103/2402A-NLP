#!/bin/bash
# 工单四 Step 1（新框架）：模型就绪后自动执行两册 PDF 图像多模态解析
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 说明：先解析重点图（组织结构图/IC市场图）快速验收，再全量解析招股说明书2、招股说明书1
PY=/home/dabaie/code/my_project/.venv/bin/python
DIR=/home/dabaie/models/Qwen2-VL-2B-Instruct
cd /home/dabaie/code/工单/工单四

echo "[waiter] 等待 Qwen2-VL 下载完成..."
while true; do
    n=$(ls "$DIR" 2>/dev/null | grep -c 'safetensors$')
    inc=$(ls "$DIR" 2>/dev/null | grep -c incomplete)
    if [ "$n" -ge 2 ] && [ "$inc" -eq 0 ]; then
        echo "[waiter] 模型就绪 $(date)"
        break
    fi
    sleep 60
done

echo "[waiter] Step A：招股说明书2 重点图（img_008 组织结构图 / img_011 img_012 IC市场图）..."
$PY -m src.image_parser.image_parser \
    --images "data/images/招股说明书2_images.json" \
    --out "data/image_descriptions/招股说明书2_images_parsed.json" \
    --image-ids img_008 img_011 img_012 \
    2>&1 | tail -35

echo "[waiter] Step B：招股说明书2 全量解析..."
$PY -m src.image_parser.image_parser \
    --images "data/images/招股说明书2_images.json" \
    --out "data/image_descriptions/招股说明书2_images_parsed.json" \
    2>&1 | tail -12

echo "[waiter] Step C：招股说明书1 全量解析..."
$PY -m src.image_parser.image_parser \
    --images "data/images/招股说明书1_images.json" \
    --out "data/image_descriptions/招股说明书1_images_parsed.json" \
    2>&1 | tail -12

echo "[waiter] 两册解析全部完成 $(date)"
