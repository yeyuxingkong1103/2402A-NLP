#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <pdf_path> <source_file>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

PDF_PATH="$1"
SOURCE_FILE="$2"
SOURCE_BASENAME="$(basename "$SOURCE_FILE")"
SOURCE_STEM="${SOURCE_BASENAME%.*}"
SAFE_SOURCE_STEM="$(printf '%s' "$SOURCE_STEM" | tr -c '[:alnum:]_.-' '_')"

OUTPUT_ROOT="${INGEST_OUTPUT_ROOT:-$PROJECT_ROOT/data/processed/$SAFE_SOURCE_STEM}"
PARSE_DIR="$OUTPUT_ROOT/parse"
CHUNK_DIR="$OUTPUT_ROOT/chunks"
VECTOR_DIR="$OUTPUT_ROOT/vectors"

PDF_BASENAME="$(basename "$PDF_PATH")"
PDF_STEM="${PDF_BASENAME%.*}"
MARKDOWN_PATH="$PARSE_DIR/$PDF_STEM.md"
PARENTS_PATH="$CHUNK_DIR/parents.json"
CHILDREN_PATH="$CHUNK_DIR/parent_child_chunks.jsonl"

mkdir -p "$PARSE_DIR" "$CHUNK_DIR" "$VECTOR_DIR"

echo "[1/3] Parse PDF with MinerU"
echo "  pdf: $PDF_PATH"
echo "  output: $PARSE_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/parse_pdf.py" "$PDF_PATH" "$PARSE_DIR"

echo "[2/3] Build parent-child chunks"
echo "  markdown: $MARKDOWN_PATH"
echo "  parents: $PARENTS_PATH"
echo "  children: $CHILDREN_PATH"
"$PYTHON_BIN" "$SCRIPT_DIR/parent_child_chunking.py" \
  --input "$MARKDOWN_PATH" \
  --parents "$PARENTS_PATH" \
  --children "$CHILDREN_PATH"

echo "[3/3] Embed child chunks with BGE-m3"
echo "  chunks: $CHILDREN_PATH"
echo "  output: $VECTOR_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/embed_chunks.py" "$CHILDREN_PATH" "$VECTOR_DIR"

echo "Ingest outputs written under: $OUTPUT_ROOT"
echo "Milvus global primary key design: ${SOURCE_FILE}:<child_id>"
