import json
import os
import ollama
import redis
from pymilvus import MilvusClient
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from offline_ingest import data_path,collection_name
from online_chat import chat_events
STATIC_DIR = r"C:\Users\ZhuanZ\Desktop\a\static"#保存静态文件夹路径
app = FastAPI(title="律知 · 法律问答")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
#让浏览器能访问static文件夹里的网页文件,当浏览器访问/static，就会去指向的文件夹里面找文件
class Question(BaseModel):#定义请求数据格式,用户提问时必须遵守这个格式
    question: str
    user_id: str
    role: str
    session_id: str
@app.get("/")#定义接口
def home():
    return FileResponse(STATIC_DIR + r"\index.html")
@app.get("/api/document")
def document():
    if not os.path.isfile(data_path):
        raise HTTPException(404, "没有找到资料")
    return FileResponse(data_path)
def check_service(name):
    try:
        if name == "Redis":
            with redis.Redis(host="127.0.0.1", port=6379, socket_connect_timeout=2) as cache: cache.ping()
        elif name == "Milvus":
            client = MilvusClient(uri="http://127.0.0.1:19530", timeout=3)
            rows = client.query(collection_name=collection_name,
                                filter="id >= 0", output_fields=["id"], limit=2000)
            count = len(rows)
            return {"name": name, "ok": count > 0, "count": count}
        else:
            models = {model.model for model in ollama.list().models}
            missing = {"bge-m3:567m", "deepseek-r1:7b"} - models
            return {"name": name, "ok": not missing,
                    "detail": "缺少模型：" + "、".join(sorted(missing)) if missing else "模型已就绪"}
        return {"name": name, "ok": True}
    except Exception:
        return {"name": name, "ok": False, "detail": "暂未连接"}
@app.get("/api/status")
def status():
    services = [check_service(name) for name in ("Milvus", "Redis", "Ollama")]
    exists = os.path.isfile(data_path)
    filename = os.path.basename(data_path)
    name, extension = os.path.splitext(filename)
    return {"ready": all(item["ok"] for item in services), "services": services,
            "document": exists, "document_name": name,
            "document_count": int(exists), "document_source": filename if exists else "",
            "document_type": extension[1:].upper()}
@app.post("/api/chat")
def chat(body: Question):
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "问题不能为空")
    def events():
        try:
            for event in chat_events(question, body.user_id, body.role, body.session_id):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception:
            yield json.dumps({"type": "error", "text": "回答未完成，请检查 Docker 和 Ollama 后重试。"}, ensure_ascii=False) + "\n"
    return StreamingResponse(events(), media_type="application/x-ndjson")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
