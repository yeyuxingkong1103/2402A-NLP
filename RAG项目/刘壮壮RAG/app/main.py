from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import auth, characters, chat, conversations, health, knowledge
from app.database import init_db

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with init_db():
        yield


app = FastAPI(title="RAG Assistant", lifespan=lifespan)
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(characters.router, prefix="/api", tags=["characters"])
app.include_router(knowledge.router, prefix="/api", tags=["knowledge"])
app.include_router(conversations.router, prefix="/api", tags=["conversations"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
