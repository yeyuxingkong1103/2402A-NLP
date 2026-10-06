"""FastAPI application assembly for MedRAG."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from . import auth, chat, health, history, pages, vision
from .dependencies import STATIC_DIR
from .lifecycle import lifespan

API_ROUTERS = (pages.router, auth.router, history.router, health.router, chat.router, vision.router)

def create_app() -> FastAPI:
    application = FastAPI(title="MedRAG 智能医疗助手 API", description="基于 LangGraph + Neo4j 的知识图谱问答系统", version="0.2.0", lifespan=lifespan)
    application.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
    for router in API_ROUTERS:
        application.include_router(router)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return application

app = create_app()
