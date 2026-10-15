#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../研发/ragflow-main/docker"
docker compose -p rag-wo15 -f docker-compose.yml up -d
