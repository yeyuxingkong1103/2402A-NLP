"""src/api/routers/knowledge.py —— 知识库管理路由（上传 / 构建 / 删除 / 列表）。

在链路中的位置：
    HTTP → 【本文件】 → src/offline/pipeline.py（build_role 批处理构建）
                      → src/offline/milvus_store.py（删向量）
                      → src/offline/metadata_store.py（文档登记与分块元数据）

路由前缀 /api/v1/kb：
    POST   /kb/upload          上传文件到该租户该角色的数据目录
    POST   /kb/build           触发后台构建（把该目录全部文档灌进知识库）
    DELETE /kb/{doc_id}        删除一份文档（向量 + 元数据一起删）
    GET    /kb/docs            列出文档

数据目录布局（按租户、角色分层）：
    data/raw/{tenant_id}/{role_id}/    —— 上传的原始文件

与 backend/server.py 的知识库接口相比，本文件的差异在于"按租户+角色分目录"，
而 backend 那条主线是单角色、单目录（data/pdfs/）。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from configs.settings import get_settings
from src.api.deps import current_user
from src.models.database import User
from src.offline.milvus_store import store
from src.offline.metadata_store import delete_document_metadata, get_document, list_documents
from src.offline.pipeline import build_role

router = APIRouter(prefix="/api/v1/kb", tags=["knowledge"])


@router.post("/upload")
async def upload(role_id: str, file: UploadFile = File(...), user: User = Depends(current_user)):
    """把上传的文件存进该租户该角色的数据目录。

    参数：
        role_id: 角色标识（查询参数）
        file: 上传的文件
        user: 当前用户
    返回：
        {"ok": True, "path": 落盘路径, "role_id": role_id}

    只负责存盘，不触发构建：
        用户可能一次传多份文件，传一份建一次既慢又浪费。
        设计成"先都传完、再调一次 /kb/build 统一构建"。

    Path(file.filename or "upload.txt").name 是安全关键：
        取 .name 只保留路径最后一段，把 "../../etc/passwd" 这类路径穿越
        削成 "passwd" —— 上传接口的文件名直接参与拼路径，
        不这样处理就等于把"文件写到哪"的决定权交给客户端。
        `or "upload.txt"` 兜底处理客户端没给文件名的情况。

    shutil.copyfileobj 而不是 file.read() 一次性读出：
        大文件一次性读进内存可能直接把服务撑爆（OOM）。
        流式拷贝是分块搬运，内存占用与文件大小无关。

    目录按 user.tenant_id 分层（而不是取请求里的值）：
        租户由服务端按登录身份决定 —— 拿客户端传的值会让用户
        把文件写进别人的租户目录。

    mkdir(parents=True, exist_ok=True)：
        目录不存在时自动建；存在时不报错（幂等）。
    """
    target_dir = get_settings().data_dir / "raw" / user.tenant_id / role_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(file.filename or "upload.txt").name
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"ok": True, "path": str(target), "role_id": role_id}


@router.post("/build")
def build(role_id: str, background: BackgroundTasks, chunk: str = "semantic", user: User = Depends(current_user)):
    """把该角色目录下的全部文档构建进知识库（后台执行，立即返回）。

    参数：
        role_id: 角色标识
        background: FastAPI 的后台任务收集器
        chunk: 分块策略，默认 semantic
        user: 当前用户
    返回：
        {"ok": True, "message": "构建任务已提交", "role_id": role_id}
    异常：
        该角色的数据目录不存在 -> 404。

    用 BackgroundTasks 而不是 threading：
        FastAPI 原生的后台任务机制，会在响应发送后执行，
        不需要自己管线程生命周期（对比 backend/server.py 里手工起 threading.Thread 的写法）。

    传 rebuild=True 固定开启重建：
        构建接口的语义是"让知识库反映当前目录的内容"，
        所以每次都对已有文档先删后建 —— 否则重复构建会堆出重复 chunk。
        这也让"改了文件重新构建"成为符合直觉的行为。

    为什么要把构建放后台：
        一份标准文档的解析 + 向量化要几分钟，同步返回会让请求超时。
        这里只回一个"已提交"，构建结果由 /kb/docs 的 status 字段体现
        （building -> ready/failed，见 src/offline/metadata_store.py 的状态约定）。

    注意本接口不校验 role_id 对应的角色是否存在：
        只校验数据目录存在。这是当前实现的宽松点 ——
        传一个不存在的 role_id 时，若恰好有同名目录仍会构建出库
        （会得到一批没人能访问到的数据，因为对话侧取角色配置时会 404）。
    """
    input_dir = get_settings().data_dir / "raw" / user.tenant_id / role_id
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail="角色数据目录不存在，请先上传文档")
    background.add_task(build_role, role_id, input_dir, chunk, True, user.tenant_id)
    return {"ok": True, "message": "构建任务已提交", "role_id": role_id}


@router.delete("/{doc_id}")
def delete_doc(doc_id: int, user: User = Depends(current_user)):
    """删除一份文档（Milvus 向量 + 关系库元数据一起删）。

    参数：
        doc_id: 文档 id（路径参数）
        user: 当前用户
    返回：
        {"ok": True, "doc_id": doc_id}
    异常：
        文档不存在或不属于该租户 -> 404。

    两处删除缺一不可（这是本项目最容易漏的一步）：
        只删元数据 -> Milvus 里留下孤儿向量，检索仍能命中但引用打不开
        只删向量   -> 文档列表还显示这份文档，点进去是空的
        接口层的删除职责就是把两边都做到。

    get_document 内部已做租户校验（不属于本租户时返回 None）：
        所以这里的 404 同时覆盖了"不存在"和"不是你的"两种情况。
        顺序上先取文档再删 —— 必须如此，因为删 Milvus 需要 doc.role_id。
    """
    doc = get_document(doc_id, user.tenant_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    # 先删向量（需要 doc.role_id 做过滤条件），再删元数据
    store.delete_document(doc.id, doc.role_id, user.tenant_id)
    delete_document_metadata(doc.id, user.tenant_id)
    return {"ok": True, "doc_id": doc_id}


@router.get("/docs")
def docs(role_id: str | None = None, user: User = Depends(current_user)):
    """列出文档。

    参数：
        role_id: 可选，只看某个角色的文档
        user: 当前用户
    返回：
        {"documents": [{"id", "role_id", "source", "status", "metadata", "updated_at"}, ...]}

    role_id 传 None 时列出该租户的全部文档：
        由 metadata_store.list_documents 内部处理成可选条件，
        但 tenant_id 过滤始终存在 —— 无论调用方传什么，都不会跨租户。

    updated_at.isoformat()：
        同 session.py 的说明 —— datetime 不能直接进 JSON。

    status 字段是前端轮询构建进度的依据（building / ready / failed）：
        这个接口因此同时充当了"构建状态查询"的角色，
        不需要像 backend 那条主线那样单独提供 /api/build/status。
    """
    rows = list_documents(role_id, user.tenant_id)
    return {"documents": [{"id": doc.id, "role_id": doc.role_id, "source": doc.doc_source, "status": doc.status, "metadata": doc.metadata_json, "updated_at": doc.updated_at.isoformat()} for doc in rows]}
