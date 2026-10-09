import sys
sys.path.insert(0, "/root/autodl-tmp/rag_project")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ragflow.api.document_quality import router

app = FastAPI(title="RAGFlow Document Quality API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
