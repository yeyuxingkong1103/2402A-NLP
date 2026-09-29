"""HTML page routes."""
import os
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, RedirectResponse
from src import config
from .auth import _extract_token, _session_user
from .dependencies import STATIC_DIR, _AUTH_COOKIE

router = APIRouter(tags=["pages"])

def _request_token(request: Request) -> str:
    return _extract_token(request.headers.get("Authorization", "")) or request.cookies.get(_AUTH_COOKIE, "")

@router.get("/", include_in_schema=False)
def index(request: Request):
    if config.AUTH_ENABLED and _session_user(_request_token(request)) is None:
        return RedirectResponse("/login")
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@router.get("/login", include_in_schema=False)
def login_page(request: Request):
    if config.AUTH_ENABLED and _session_user(_request_token(request)) is not None:
        return RedirectResponse("/")
    return FileResponse(os.path.join(STATIC_DIR, "login.html"))
