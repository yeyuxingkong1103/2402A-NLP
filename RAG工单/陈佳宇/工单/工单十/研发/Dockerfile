# Dockerfile 工单10 金融问答系统部署
# 工单编号：人工智能NLP‑RAG 项目‑金融问答系统部署
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制全部业务代码
COPY . .

# 对外暴露容器内部端口7862
EXPOSE 7862

# 启动服务
CMD ["python3","main.py"]
