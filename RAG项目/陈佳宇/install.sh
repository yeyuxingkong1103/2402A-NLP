#!/bin/bash
# 基于RAG的角色扮演系统 - 环境安装脚本
# 支持 Ubuntu 22.04 / WSL Ubuntu

echo "===== 开始安装环境 ====="

# 1. 安装系统依赖
echo "[1/5] 安装系统依赖（MySQL、Redis、Milvus依赖）..."
sudo apt update
sudo apt install -y mysql-server redis-server python3 python3-pip python3-venv

# 2. 启动MySQL和Redis
echo "[2/5] 启动MySQL和Redis..."
sudo service mysql start
sudo service redis-server start

# 3. 创建Python虚拟环境
echo "[3/5] 创建Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 4. 安装Python依赖
echo "[4/5] 安装Python依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. 初始化数据库
echo "[5/5] 初始化MySQL数据库..."
sudo mysql << 'SQL'
CREATE DATABASE IF NOT EXISTS rag_character CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE rag_character;
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS characters (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    description TEXT,
    system_prompt TEXT NOT NULL,
    collection_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT IGNORE INTO characters (name, description, system_prompt, collection_name) VALUES
('Alice', '20岁温柔女孩，喜欢星空和诗歌', '你是Alice，20岁，性格温柔，喜欢星空和诗歌，喜欢安静，不喜欢吵闹。', 'alice_knowledge'),
('医生', '专业医疗咨询助手', '你是一位专业的医生，请根据医学知识回答用户问题，给出专业建议。', 'doctor_knowledge'),
('律师', '专业法律咨询助手', '你是一位专业律师，请根据法律条文回答用户问题，给出法律建议。', 'lawyer_knowledge');
INSERT IGNORE INTO users (username, password) VALUES ('xiaoming', '123456');
SQL

echo "===== 环境安装完成 ====="
echo "下一步：bash run.sh 启动服务"
