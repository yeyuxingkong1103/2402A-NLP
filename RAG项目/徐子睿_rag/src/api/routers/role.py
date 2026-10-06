"""src/api/routers/role.py —— 角色管理路由，以及角色配置的读取/同步。

在链路中的位置：
    HTTP                  → 【本文件】 → src/models/database.py（role 表，config 存 JSON）
    src/api/routers/chat.py → get_role_config()（对话前取角色人设）
    src/api/main.py 启动钩子 → sync_roles()（把角色卡文件同步进库）

路由前缀 /api/v1/roles：
    GET  /roles   列出当前租户的全部角色
    POST /roles   新建或覆盖一个角色

配置来源有两条，最终都落进 role 表：
    1. configs/roles/*.json 里的角色卡文件 —— 启动时由 sync_roles 同步入库
    2. 接口 POST /roles 提交的自定义角色
    role.config 是 JSON 列（见 src/models/database.py 的说明），
    所以角色卡加字段不需要改表结构。
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from configs.settings import get_settings
from src.api.deps import current_user
from src.models.database import Role, User, db_session
from src.schemas.roleplay import RoleCard

router = APIRouter(prefix="/api/v1/roles", tags=["roles"])


def sync_roles() -> None:
    """把 configs/roles 下的角色卡文件同步进数据库（应用启动时调用）。

    行为：
        遍历配置目录里的 *.json，逐个按 role_id 查库 ——
        已存在就更新 config，不存在就插入一条。

    为什么是"同步"而不是"只插入一次"：
        角色卡的文案会迭代（改了 persona 或 safety_notice 之后），
        如果只在首次插入，线上跑的服务永远用着第一次启动时的旧文案，
        改文件不生效这件事会非常难排查。每次都覆盖就避免了这个问题。

    跳过 role_id == "template" 的文件：
        template.json 是给使用者抄写新角色用的模板，
        里面是占位内容而不是真角色，同步进库会让用户看到一堆"某某助手"。
        但注意它仍会被列在角色目录里 —— 这是配置约定，不是 bug。

    覆盖写入的是 data（整个 JSON 对象）：
        所以角色卡文件的字段设计就是 API 的字段设计，两边天然一致。

    注意本函数没有删除逻辑：
        配置文件里删掉的角色，库里的记录仍会保留。
        这是有意的保守取舍 —— 避免"配置文件一时手滑"导致线上角色消失。
    """
    settings = get_settings()
    settings.role_dir.mkdir(parents=True, exist_ok=True)  # 目录不存在时先建，避免 glob 报错
    with db_session() as db:
        for path in settings.role_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("role_id") == "template":
                continue  # 模板文件不入库
            role = db.scalar(select(Role).where(Role.role_id == data["role_id"], Role.tenant_id == settings.tenant_id))
            if role:
                role.config = data  # 已存在则用文件内容覆盖（让角色文案的修改能生效）
            else:
                db.add(Role(role_id=data["role_id"], tenant_id=settings.tenant_id, config=data))
        db.commit()


@router.get("")
def list_roles(user: User = Depends(current_user)):
    """列出当前租户下的全部角色。

    参数：
        user: 当前用户（由 Depends 注入）
    返回：
        {"roles": [角色配置 dict, ...]}

    按 role_id 排序：
        让列表顺序稳定可预期，而不是随数据库的物理存储顺序变化。
        用户界面和测试断言的稳定性都依赖这一点。

    只返回 config、不返回 Role 表的其他列：
        调用方只关心角色卡内容；id/tenant_id/时间戳属内部字段，不必暴露。

    按 user.tenant_id 过滤（不是取全局）：
        这是多租户隔离的落点 —— 每个租户看到自己的角色集。
    """
    with db_session() as db:
        roles = db.scalars(select(Role).where(Role.tenant_id == user.tenant_id).order_by(Role.role_id)).all()
        return {"roles": [role.config for role in roles]}


@router.post("")
def create_role(request: RoleCard, user: User = Depends(current_user)):
    """新建或覆盖当前租户下的一个角色。

    参数：
        request: RoleCard（由 pydantic 做字段校验）
        user: 当前用户
    返回：
        {"ok": True, "role": 角色配置}

    为什么不做"已存在就报 409"：
        角色编辑是高频操作（改个人设、调下语气），
        让同一个接口既能建也能改，前端不必分两种情况处理。
        这是与 auth 注册接口有意不同的取舍 ——
        用户名重复通常意味着"你想注册的是别人"，
        而角色重复通常意味着"我要更新这个角色"。

    按 (role_id, tenant_id) 定位而不是只按 role_id：
        两个租户可以各自拥有同名的 "lawyer" 角色，互不干扰。

    返回 request.model_dump() 而不是从库里回读：
        接口写入的就是这份内容，直接回显即可，省一次查询。
        （与 backend/roleplay.py 的 save_role 不同 —— 那边会回读以确认落库结果。）
    """
    with db_session() as db:
        exists = db.scalar(select(Role).where(Role.role_id == request.role_id, Role.tenant_id == user.tenant_id))
        if exists:
            exists.config = request.model_dump()
        else:
            db.add(Role(role_id=request.role_id, tenant_id=user.tenant_id, config=request.model_dump()))
        db.commit()
        return {"ok": True, "role": request.model_dump()}


def get_role_config(role_id: str, tenant_id: str) -> dict:
    """取角色配置（供对话链路调用，不是 HTTP 接口）。

    参数：
        role_id: 角色标识
        tenant_id: 租户 id
    返回：
        角色配置 dict。
    异常：
        角色不存在 -> HTTPException 404。

    在路由函数里抛 HTTPException 是可行的：
        虽然本函数被 src/online/chain.py 间接调用，但它始终在请求处理链内执行，
        异常最终会被 FastAPI 转成 404 响应。

    为什么这里没有像 backend/roleplay.py 那样把配置复制出来：
        role.config 是 JSON 列（普通 dict），不存在 ORM 懒加载的隐患，
        db 会话关闭后仍能安全使用 —— 与 ChatSession 那种需要复制的情况不同。
    """
    with db_session() as db:
        role = db.scalar(select(Role).where(Role.role_id == role_id, Role.tenant_id == tenant_id))
        if not role:
            raise HTTPException(status_code=404, detail="角色不存在")
        return role.config
