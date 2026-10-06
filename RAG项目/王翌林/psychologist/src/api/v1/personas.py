"""心理医生角色接口：列表、详情、切换、启停。

角色（persona）是系统的核心概念：每个角色对应一种"心理咨询流派/风格"。
本模块的读取接口对游客开放（如列表、详情），
而增删改等写操作需要管理员权限（get_current_admin），体现"读开放、写受控"。
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.deps import get_current_admin, get_current_user
from src.core.exceptions import ok
from src.db.mysql import get_db
from src.models import User
from src.schemas import PersonaCreateRequest, PersonaUpdateRequest
from src.services import persona_service

router = APIRouter(prefix="/personas", tags=["心理医生角色"])


@router.get("", summary="心理医生列表（P-01）")
def list_personas(include_inactive: bool = False, db: Session = Depends(get_db)):
    # 该接口无需登录即可浏览，是面向游客的"可公开内容"，
    # 因此没有 get_current_user 依赖，只有数据库依赖。
    personas = persona_service.list_personas(db, only_active=not include_inactive)
    items = []
    for p in personas:
        data = persona_service._to_dict(p)
        data.pop("system_prompt", None)  # 列表不返回完整提示词
        # 提示词是角色的"内核机密"，列表场景无需暴露，也避免被前端误用。
        items.append(data)
    return ok({"items": items, "total": len(items)})


@router.get("/{persona_id}", summary="角色详情（P-02：流派、风格、开场白）")
def get_persona(persona_id: int, db: Session = Depends(get_db)):
    return ok(persona_service.get_persona_dict(db, persona_id))


@router.post("", summary="新增角色（P-08 扩展，管理员）")
def create_persona(payload: PersonaCreateRequest,
                   admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    # 创建角色是全局配置变更，必须由管理员执行；get_current_admin 会先校验身份。
    persona = persona_service.create_persona(db, payload.model_dump())
    return ok({"id": persona.id, "persona_code": persona.persona_code}, "角色创建成功")


@router.put("/{persona_id}", summary="编辑角色（管理员）")
def update_persona(persona_id: int, payload: PersonaUpdateRequest,
                   admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    # exclude_none=True：允许只提交需要修改的字段，其余字段保持原样。
    persona = persona_service.update_persona(db, persona_id, payload.model_dump(exclude_none=True))
    data = persona_service._to_dict(persona)
    data.pop("system_prompt", None)
    return ok(data, "角色更新成功")


@router.post("/{persona_id}/status", summary="角色上下架（P-07，管理员）")
def set_status(persona_id: int, status: int,
               admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    # status 用于控制角色是否对用户可见（上架/下架），而非物理删除。
    persona_service.set_status(db, persona_id, status)
    return ok({"persona_id": persona_id, "status": status}, "角色状态已更新")