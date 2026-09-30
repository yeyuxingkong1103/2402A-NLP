#!/bin/bash
# ================================================
# RAG 角色扮演系统 - 启动脚本
# 用法：chmod +x start.sh && ./start.sh
# ================================================

set -e  # set -e：任何命令失败立即退出，避免在缺服务的情况下继续启动

echo "[启动] RAG 角色扮演系统 ..."  # 打印启动提示

# ---------- 1. 加载环境变量 ----------
if [ -f .env ]; then  # 如果当前目录下有 .env 文件
    # cat .env 读取文件内容 → grep -v '^#' 去掉注释行 → xargs 把每行拆成 key=value → export 逐条导出为环境变量
    export $(cat .env | grep -v '^#' | xargs)  # 加载到当前 shell 环境
    echo "  环境变量已从 .env 加载"  # 提示完成
fi  # 结束 if

# ---------- 2. 激活 Python 虚拟环境 ----------
# 优先用 conda，其次 venv
if command -v conda &>/dev/null && conda env list | grep -q rag_roleplay; then  # 如果有 conda 且存在 rag_roleplay 环境
    echo "  激活 conda 环境：rag_roleplay"  # 提示
    # conda activate 在 bash 脚本里不能直接用，需要先初始化 shell hook
    eval "$(conda shell.bash hook)"  # 在当前 shell 中初始化 conda 的 bash 钩子
    conda activate rag_roleplay  # 激活环境
elif [ -f venv/bin/activate ]; then  # 如果没有 conda 但有 venv
    echo "  激活 venv ..."  # 提示
    source venv/bin/activate  # 激活 venv 虚拟环境
fi  # 结束 if

# ---------- 3. 检查依赖服务是否在线 ----------
echo "  检查依赖服务 ..."  # 提示正在检查

# 检查 Milvus：curl 访问 19530 端口，-s 静默，> /dev/null 丢弃输出，2>&1 合并错误流
if ! curl -s http://localhost:19530 > /dev/null 2>&1; then  # curl 返回非零 = 连不上
    echo "  [警告] Milvus 未响应，请确认已启动"  # 警告但不退出（让用户自行决定）
fi  # 结束 if

# 检查 Ollama：访问 /api/tags 接口（返回已安装模型列表）
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then  # 连不上
    echo "  [警告] Ollama 未响应，请确认已启动"  # 警告
fi  # 结束 if

# 检查 Redis：用 redis-cli ping，返回 PONG 表示在线
if ! redis-cli ping > /dev/null 2>&1; then  # ping 返回非零 = 连不上
    echo "  [警告] Redis 未响应，请确认已启动"  # 警告
fi  # 结束 if

# 检查 MySQL：用 mysqladmin ping 探活，-h 指定地址
if ! mysqladmin ping -h localhost 2>/dev/null; then  # ping 返回非零 = 连不上，2>/dev/null 丢弃错误信息
    echo "  [警告] MySQL 未响应，请确认已启动"  # 警告
fi  # 结束 if

# ---------- 4. 启动 FastAPI ----------
echo "  启动 FastAPI（端口 $API_PORT）..."  # 打印端口（$API_PORT 从 .env 加载，默认 8066）
exec python main.py  # exec 替换当前进程为 python，Ctrl+C 时信号直接传给 python（不是 shell）
