#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Please run as root: sudo bash scripts/install_centos.sh"
  exit 1
fi

yum install -y python3 python3-pip git unzip docker
systemctl enable --now docker
python3 -m pip install --upgrade pip
echo "CentOS base environment installed. Install Docker Compose v2 if it is not available on this host."
