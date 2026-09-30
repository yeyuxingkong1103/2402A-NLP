#!/bin/bash
# ================================================
# PersonaRAG · 多角色智能顾问系统 - Ubuntu/CentOS 一键安装脚本
# 用法：chmod +x install.sh && ./install.sh
# ================================================

set -e  # set -e：任何一条命令返回非零状态码就立即退出脚本，不让错误滚雪球

echo "=========================================="  # 打印分隔线
echo "  PersonaRAG · 多角色智能顾问系统 - 安装脚本"               # 脚本标题
echo "=========================================="  # 打印分隔线

# ---------- 0. 检查操作系统 ----------
echo "[0/6] 检查操作系统 ..."  # 打印当前步骤
if [ -f /etc/os-release ]; then  # 如果 /etc/os-release 文件存在（Linux 标准发行版标识文件）
    . /etc/os-release  # source 该文件，加载 NAME、VERSION、ID 等变量到当前 shell
    echo "  系统：$NAME $VERSION"  # 打印操作系统名称和版本（如 "Ubuntu 22.04"）
    OS_ID=$ID  # 提取发行版 ID（如 ubuntu、centos、debian），后续用来区分包管理器
else  # 文件不存在（可能是非标准 Linux 或 macOS）
    echo "  [警告] 无法识别操作系统，假设是 Ubuntu"  # 打印警告
    OS_ID="ubuntu"  # 兜底默认为 ubuntu（最常见）
fi  # 结束 if

# ---------- 1. 检查 Python ----------
echo "[1/6] 检查 Python ..."  # 打印当前步骤
if command -v python3 &>/dev/null; then  # command -v 检查 python3 命令是否存在，&>/dev/null 丢弃输出
    PY_VER=$(python3 --version 2>&1 | awk '{print $2}')  # 执行 python3 --version，取第二列（版本号）
    echo "  Python: $PY_VER"  # 打印版本号
else  # python3 不存在
    echo "  [安装] Python3 ..."  # 提示正在安装
    if [ "$OS_ID" = "ubuntu" ] || [ "$OS_ID" = "debian" ]; then  # Ubuntu/Debian 用 apt-get
        apt-get update && apt-get install -y python3 python3-pip python3-venv  # 更新源 + 安装 python3、pip、venv
    else  # CentOS/RHEL 用 yum
        yum install -y python3 python3-pip  # 安装 python3 和 pip
    fi  # 结束 if
fi  # 结束 if

# ---------- 2. 创建 conda 环境（如果有 conda）或 venv ----------
echo "[2/6] 创建虚拟环境 ..."  # 打印当前步骤
if command -v conda &>/dev/null; then  # 检查是否安装了 conda
    echo "  检测到 conda，创建 conda 环境 ..."  # 提示
    conda create -n persona_rag python=3.10 -y  # 创建名为 persona_rag 的 conda 环境，Python 3.10，-y 自动确认
    echo "  激活环境：conda activate persona_rag"  # 提示用户手动激活
    PIP="conda run -n persona_rag pip"  # 用 conda run 在指定环境内执行 pip（脚本内不 activate 也能装到对的环境）
else  # 没有 conda
    echo "  未检测到 conda，使用 venv ..."  # 提示用 venv
    python3 -m venv venv  # 在当前目录创建 venv 虚拟环境
    source venv/bin/activate  # 激活 venv
    PIP="pip"  # 激活后直接用 pip 即可
fi  # 结束 if

# ---------- 3. 安装 Redis、MySQL、Milvus ----------
echo "[3/6] 安装基础服务 ..."  # 打印当前步骤

# Redis 安装
if ! command -v redis-cli &>/dev/null; then  # 检查 redis-cli 命令是否存在（不存在说明没装 Redis）
    echo "  [安装] Redis ..."  # 提示正在安装 Redis
    if [ "$OS_ID" = "ubuntu" ] || [ "$OS_ID" = "debian" ]; then  # Ubuntu/Debian
        apt-get install -y redis-server  # 安装 redis-server 包
    else  # CentOS/RHEL
        yum install -y redis  # 安装 redis 包
    fi  # 结束 if
    systemctl enable redis-server || systemctl enable redis  # 设置开机自启（服务名可能是 redis-server 或 redis）
    systemctl start redis-server || systemctl start redis  # 立即启动（同上兼容两种服务名）
    echo "  Redis 已启动"  # 提示完成
else  # redis-cli 已存在
    echo "  Redis 已安装"  # 提示跳过
fi  # 结束 if

# MySQL 安装
if ! command -v mysql &>/dev/null; then  # 检查 mysql 命令是否存在
    echo "  [安装] MySQL ..."  # 提示正在安装
    if [ "$OS_ID" = "ubuntu" ] || [ "$OS_ID" = "debian" ]; then  # Ubuntu/Debian
        apt-get install -y mysql-server  # 安装 mysql-server 包
    else  # CentOS/RHEL
        yum install -y mysql-server  # 安装 mysql-server 包
    fi  # 结束 if
    systemctl enable mysqld || systemctl enable mysql  # 设置开机自启（服务名可能是 mysqld 或 mysql）
    systemctl start mysqld || systemctl start mysql  # 立即启动
    # 创建数据库（utf8mb4 支持 emoji，unicode_ci 大小写不敏感排序）
    mysql -e "CREATE DATABASE IF NOT EXISTS rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"  # 执行 SQL 建库
    echo "  MySQL 已启动，数据库 rag 已创建"  # 提示完成
else  # mysql 已存在
    echo "  MySQL 已安装"  # 提示跳过
fi  # 结束 if

# Milvus 安装（Docker 方式最简单，手动编译太复杂）
if ! docker ps | grep -q milvus; then  # 检查 Docker 容器里有没有正在跑的 milvus
    echo "  [安装] Milvus（Docker 方式）..."  # 提示正在安装
    if ! command -v docker &>/dev/null; then  # 如果连 Docker 都没装
        echo "  [安装] Docker ..."  # 提示正在安装 Docker
        curl -fsSL https://get.docker.com | sh  # 用 Docker 官方一键安装脚本
        systemctl enable docker  # 设置开机自启
        systemctl start docker  # 立即启动
    fi  # 结束 if
    # 下载 Milvus 官方的 docker-compose 配置文件（v3.0.0 standalone 单机版）
    wget -O docker-compose.yml https://github.com/milvus-io/milvus/releases/download/v3.0.0/milvus-standalone-docker-compose.yml  # 下载
    docker compose up -d  # 后台启动 Milvus 及其依赖（etcd + minio）
    echo "  Milvus 已启动（端口 19530）"  # 提示完成
else  # milvus 容器已在运行
    echo "  Milvus 已运行"  # 提示跳过
fi  # 结束 if

# ---------- 4. 安装 Ollama + 模型 ----------
echo "[4/6] 安装 Ollama + bge-m3 ..."  # 打印当前步骤
if ! command -v ollama &>/dev/null; then  # 检查 ollama 命令是否存在
    echo "  [安装] Ollama ..."  # 提示正在安装
    curl -fsSL https://ollama.com/install.sh | sh  # 用 Ollama 官方一键安装脚本
    systemctl enable ollama  # 设置开机自启
    systemctl start ollama  # 立即启动
fi  # 结束 if
echo "  拉取 bge-m3 模型 ..."  # 提示正在拉取模型
ollama pull bge-m3  # 从 Ollama 仓库下载 bge-m3 向量模型（约 567M）

# ---------- 5. 安装 Python 依赖 ----------
echo "[5/6] 安装 Python 依赖 ..."  # 打印当前步骤
$PIP install -r requirements.txt  # 用之前确定的 pip 命令安装所有 Python 依赖

# ---------- 6. 配置环境变量 ----------
echo "[6/6] 配置环境变量 ..."  # 打印当前步骤
cat > .env << 'EOF'  # 用 heredoc 生成 .env 文件（'EOF' 带引号 = 不做变量展开）
# DeepSeek API（必填）
deepseek_api_key1=your_api_key_here  # DeepSeek 的 API Key，请替换成你的真实 key
deepseek_base_url=https://api.deepseek.com/v1  # DeepSeek 接口地址

# MySQL
MYSQL_HOST=localhost  # MySQL 地址
MYSQL_PORT=3306  # MySQL 端口
MYSQL_USER=root  # MySQL 用户名
MYSQL_PASSWORD=  # MySQL 密码（请填入实际密码）
MYSQL_DB=rag  # 数据库名

# Redis
REDIS_HOST=localhost  # Redis 地址
REDIS_PORT=6379  # Redis 端口

# Milvus
MILVUS_URI=http://localhost:19530  # Milvus 服务地址

# 服务
API_HOST=0.0.0.0  # FastAPI 监听地址（0.0.0.0 = 对外开放）
API_PORT=8066  # FastAPI 端口
EOF  # heredoc 结束标记
echo "  .env 文件已生成，请编辑填入实际值"  # 提示用户修改

echo "=========================================="  # 打印分隔线
echo "  安装完成！"  # 完成提示
echo "  1. 编辑 .env 填入 API Key 和数据库密码"  # 第一步
echo "  2. 运行：./start.sh 启动服务"  # 第二步
echo "  3. 打开：http://localhost:8066/docs"  # 第三步
echo "=========================================="  # 打印分隔线
