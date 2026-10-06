# 工单编号：人工智能NLP-RAG-金融问答系统部署
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY app.py rag_engine.py graph_rag_engine.py kg_graph.json /app/

VOLUME ["/app/models", "/app/2_vector_db", "/app/ccf_vector_db"]
EXPOSE 6006
CMD ["python", "app.py"]
