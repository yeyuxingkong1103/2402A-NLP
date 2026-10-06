from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Request, UploadFile

from ..auth import current_user
router = APIRouter(prefix="/api/v1/legal/files", tags=["workspace"])


@router.post("")
async def upload_file(request: Request, tasks: BackgroundTasks, file: UploadFile = File(...), session_id: str | None = None, user: dict = Depends(current_user)):
    return {"success": True, "data": await request.app.state.services["workspace"].upload(user, file, session_id, tasks=tasks)}


@router.post("/batch")
async def upload_files(request: Request, tasks: BackgroundTasks, files: list[UploadFile] = File(...), session_id: str | None = None, upload_password: str | None = Form(None), user: dict = Depends(current_user)):
    return {"success": True, "data": await request.app.state.services["workspace"].upload_batch(user, files, session_id, upload_password, tasks=tasks)}


@router.get("")
def list_files(request: Request, session_id: str | None = None, user: dict = Depends(current_user)):
    return {"success": True, "data": request.app.state.services["workspace"].list_files(user, session_id)}


@router.delete("/{document_id}")
def delete_file(document_id: str, request: Request, session_id: str | None = None, user: dict = Depends(current_user)):
    return {"success": True, "data": request.app.state.services["workspace"].delete(user, document_id, session_id)}
