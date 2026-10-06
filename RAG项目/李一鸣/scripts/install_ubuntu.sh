#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Please run as root: sudo bash scripts/install_ubuntu.sh"
  exit 1
fi

apt-get update
apt-get install -y python3 python3-venv python3-pip build-essential curl git unzip docker.io docker-compose-plugin
systemctl enable --now docker

PROJECT_DIR="${PROJECT_DIR:-/opt/rag-roleplay-system}"
mkdir -p "$PROJECT_DIR"
echo "Base environment installed. Put the project at: $PROJECT_DIR"
echo "Then run: cd $PROJECT_DIR && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
