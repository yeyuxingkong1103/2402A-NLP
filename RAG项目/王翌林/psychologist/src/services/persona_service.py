"""心理医生角色服务：列表、详情、切换、启停、扩展、缓存、初始化。"""
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.exceptions import ConflictError, NotFoundError
from src.core.logging import get_logger
from src.db import redis as redis_db
from src.models import CounselorPersona, SysRole, User, UserPersonaPreference, UserSysRole
from src.services.persona_seed import PERSONAS, SYS_ROLES

logger = get_logger("service.persona")


def _to_dict(persona: CounselorPersona) -> Dict:
    """把角色 ORM 对象转成字典；model_params 是 JSON 字段，空时给 {} 以免下游 .get() 报错。"""
    return {
        "id": persona.id,
        "persona_code": persona.persona_code,
        "name": persona.name,
        "title": persona.title,
        "therapy_type": persona.therapy_type,
        "style": persona.style,
        "methods": persona.methods,
        "greeting": persona.greeting,
        "system_prompt": persona.system_prompt,
        "knowledge_scope": persona.knowledge_scope,
        "avatar": persona.avatar,
        "safety_boundary": persona.safety_boundary,
        "model_params": persona.model_params or {},
        "status": persona.status,
    }


def list_personas(db: Session, only_active: bool = True) -> List[CounselorPersona]:
    stmt = select(CounselorPersona).order_by(CounselorPersona.id)
    if only_active:
        stmt = stmt.where(CounselorPersona.status == 1)
    return list(db.execute(stmt).scalars().all())


def get_persona(db: Session, persona_id: int) -> CounselorPersona:
    persona = db.get(CounselorPersona, persona_id)
    if not persona:
        raise NotFoundError(f"心理医生角色不存在：{persona_id}")
    return persona


def get_persona_by_code(db: Session, code: str) -> Optional[CounselorPersona]:
    return db.execute(
        select(CounselorPersona).where(CounselorPersona.persona_code == code)
    ).scalars().first()


def get_persona_dict(db: Session, persona_id: int, use_cache: bool = True) -> Dict:
    """获取角色字典（含 system_prompt），优先读 Redis 缓存。角色提示词较长且频繁被问答流程读取，缓存能显著降低 MySQL 压力。"""
    if use_cache:
        cached = redis_db.get_cached_persona(persona_id)
        if cached:
            return cached
    persona = get_persona(db, persona_id)
    data = _to_dict(persona)
    redis_db.cache_persona(persona_id, data)
    return data


def create_persona(db: Session, payload: Dict) -> CounselorPersona:
    # persona_code 是业务主键（如 humanistic_lin），先查重保证唯一，避免重复角色。
    if get_persona_by_code(db, payload["persona_code"]):
        raise ConflictError(f"角色编码已存在：{payload['persona_code']}")
    persona = CounselorPersona(**payload)
    db.add(persona)
    db.commit()
    db.refresh(persona)
    logger.info("新增角色：%s(%s)", persona.name, persona.persona_code)
    return persona


def update_persona(db: Session, persona_id: int, payload: Dict) -> CounselorPersona:
    persona = get_persona(db, persona_id)
    for key, value in payload.items():
        if value is not None and hasattr(persona, key):
            setattr(persona, key, value)
    db.commit()
    db.refresh(persona)
    # 角色内容变化后失效缓存，下次问答流程重新读取到最新提示词/开场白。
    redis_db.invalidate_persona(persona_id)
    logger.info("更新角色：%s", persona_id)
    return persona


def set_status(db: Session, persona_id: int, status: int) -> CounselorPersona:
    persona = get_persona(db, persona_id)
    persona.status = status
    db.commit()
    db.refresh(persona)
    # 上下架会改变角色可见性与可对话性，失效缓存让变更即时生效。
    redis_db.invalidate_persona(persona_id)
    logger.info("角色上下架：%s status=%s", persona_id, status)
    return persona


def set_default_persona(db: Session, user_id: int, persona_id: int) -> None:
    """保存用户默认心理医生（U-08 / P-03）。一个用户同时只能有一个默认角色。"""
    get_persona(db, persona_id)
    existing = db.execute(
        select(UserPersonaPreference).where(UserPersonaPreference.user_id == user_id)
    ).scalars().all()
    # 事务内先把该用户所有偏好置为非默认，再把目标角色置默认，保证"唯一默认"的约束。
    for pref in existing:
        pref.is_default = 0
    target = next((p for p in existing if p.persona_id == persona_id), None)
    if target:
        target.is_default = 1
    else:
        # 用户还没有该角色的偏好记录，则新增一条默认记录。
        db.add(UserPersonaPreference(user_id=user_id, persona_id=persona_id, is_default=1))
    db.commit()
    # 默认角色写进用户缓存，故也要失效用户缓存。
    redis_db.invalidate_user_profile(user_id)
    logger.info("用户 %s 默认角色切换为 %s", user_id, persona_id)


def list_user_preferences(db: Session, user_id: int) -> List[Dict]:
    prefs = db.execute(
        select(UserPersonaPreference).where(UserPersonaPreference.user_id == user_id)
    ).scalars().all()
    result = []
    for pref in prefs:
        persona = db.get(CounselorPersona, pref.persona_id)
        result.append({
            "persona_id": pref.persona_id,
            "persona_code": persona.persona_code if persona else None,
            "persona_name": persona.name if persona else None,
            "is_default": bool(pref.is_default),
        })
    return result


def get_default_persona_id(db: Session, user_id: int) -> Optional[int]:
    pref = db.execute(
        select(UserPersonaPreference).where(
            UserPersonaPreference.user_id == user_id,
            UserPersonaPreference.is_default == 1,
        )
    ).scalars().first()
    return pref.persona_id if pref else None


def seed_roles_and_personas(db: Session) -> Dict[str, int]:
    """幂等初始化系统角色与三个心理医生角色，返回角色 code -> id 映射。可重复执行，不会产生重复数据。"""
    # 系统角色先落库：不存在才插入，保证幂等。
    for role_data in SYS_ROLES:
        exists = db.execute(
            select(SysRole).where(SysRole.role_code == role_data["role_code"])
        ).scalars().first()
        if not exists:
            db.add(SysRole(**role_data))
    db.commit()

    mapping: Dict[str, int] = {}
    for data in PERSONAS:
        persona = get_persona_by_code(db, data["persona_code"])
        if persona is None:
            persona = CounselorPersona(**data)
            db.add(persona)
            db.commit()
            db.refresh(persona)
            logger.info("初始化角色：%s(%s)", persona.name, persona.persona_code)
        else:
            # 已存在时同步提示词与开场白，保证与需求文档一致（种子数据是"标准答案"，升级后也要覆盖旧值）。
            persona.name = data["name"]
            persona.title = data["title"]
            persona.therapy_type = data["therapy_type"]
            persona.style = data["style"]
            persona.methods = data["methods"]
            persona.greeting = data["greeting"]
            persona.system_prompt = data["system_prompt"]
            persona.knowledge_scope = data["knowledge_scope"]
            persona.avatar = data["avatar"]
            persona.safety_boundary = data["safety_boundary"]
            persona.model_params = data["model_params"]
            db.commit()
        mapping[persona.persona_code] = persona.id
        # 种子写入后失效该角色缓存，确保后续读取的是最新种子内容。
        redis_db.invalidate_persona(persona.id)
    return mapping


def assign_role(db: Session, user_id: int, role_code: str) -> None:
    """给用户分配系统角色（幂等：已存在则不重复插入）。"""
    role = db.execute(select(SysRole).where(SysRole.role_code == role_code)).scalars().first()
    if not role:
        return
    exists = db.execute(
        select(UserSysRole).where(UserSysRole.user_id == user_id, UserSysRole.role_id == role.id)
    ).scalars().first()
    if not exists:
        db.add(UserSysRole(user_id=user_id, role_id=role.id))
        db.commit()


def get_user_roles(db: Session, user_id: int) -> List[str]:
    # 通过 UserSysRole 关联表 JOIN SysRole，取出该用户拥有的角色编码列表（如 ["user"] 或 ["admin"]）。
    rows = db.execute(
        select(SysRole.role_code)
        .join(UserSysRole, UserSysRole.role_id == SysRole.id)
        .where(UserSysRole.user_id == user_id)
    ).all()
    return [r[0] for r in rows]


def is_admin(db: Session, user: User) -> bool:
    return "admin" in get_user_roles(db, user.id)