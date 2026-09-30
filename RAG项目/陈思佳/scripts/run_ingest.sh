#!/usr/bin/env bash
set -euo pipefail

python -m src.edu_rag_ingest.ingestion.pipeline --config configs/crawler.yaml
