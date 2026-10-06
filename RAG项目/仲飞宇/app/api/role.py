"""角色管理接口。

三个接口：列表 / 查单个 / 新建或覆盖。

角色数据的来源是关系库的 roles 表，不是 role_presets.py 那份内存字典：内置的 12 个预设
角色由 main.py 在启动时逐个 ensure_role() 登记进去（所以正常情况下启动完就是齐的），
运行时新建的角色则来自本文件的 POST /role。这样前端只认一个数据源，新建角色不必改代码。

本模块只读/写角色记录，不碰向量库；role_id 的字符集由 schemas.ROLE_ID_PATTERN 校验，
请求体里带引号之类的字符在进到路由函数之前就被 422 挡掉了（id 会被拼进 Milvus 过滤表达式）。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..schemas import RoleCreateRequest, RoleInfo

router = APIRouter(tags=["role"])


@router.get("/role/list", response_model=list[RoleInfo])
def list_roles(request: Request):
    # 直接读库而不是返回 ROLE_PRESETS：这样 POST /role 新建的角色才会出现在下拉框里。
    # 代价是列表内容取决于库——启动时关系库不可用的话 main.py 那轮预设注册会被跳过
    # （它只记 error 不拦启动），此时这里返回空列表，前端会提示「暂无角色」。
    # 另外 sql.list_roles() 没有 order_by，顺序由数据库决定，前端不做二次排序。
    p = request.app.state.pipeline
    # avatar 三处都写成 `r.avatar or ""`：列上虽有默认空串，但迁移前的老行、被手工改过的行
    # 仍可能是 NULL，而 RoleInfo.avatar 声明的是 str，塞 None 会在响应校验处炸成 500。
    return [
        RoleInfo(role_id=r.id, name=r.name, avatar=r.avatar or "", description=r.description)
        for r in p.sql.list_roles()
    ]


@router.get("/role/{role_id}", response_model=RoleInfo)
def get_role(role_id: str, request: Request):
    p = request.app.state.pipeline
    # ensure_role 不只是「查」：内置预设角色首次被访问时会顺手登记入库（见 pipeline），
    # 所以拿一个从没被访问过的预设 role_id 也能成功，但这次 GET 会写库。
    try:
        r = p.ensure_role(role_id)
    except ValueError as exc:
        # 未知角色是"没这条记录"，不是服务器错误：/chat 在同一条件下返回 404，
        # 接口文档也承诺 404。以前它 500 逸出，前端"角色不存在"的分支永远走不到。
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RoleInfo(role_id=r.id, name=r.name, avatar=r.avatar or "", description=r.description)


@router.post("/role", response_model=RoleInfo)
def create_role(req: RoleCreateRequest, request: Request):
    # 是 upsert，不是只插：同一个 role_id 再 POST 一次会**覆盖**已有角色，包括内置预设角色
    # （没有「预设不可改」的保护）——要改就整个 system_prompt 一起给，不会做字段合并。
    # 唯一不覆盖的是头像：req.avatar 为空时 sql_store 会保留库里已有的值，避免「只想改提示词
    # 却把头像清没了」。上传知识库走 /knowledge/upload，与本接口无关，所以新建的角色是空知识库，
    # 一上来就是零召回（提示词里那段「没有检索到资料」的措辞就是给这种情况准备的）。
    p = request.app.state.pipeline
    r = p.sql.upsert_role(req.role_id, req.name, req.description, req.system_prompt, req.avatar)
    return RoleInfo(role_id=r.id, name=r.name, avatar=r.avatar or "", description=r.description)
