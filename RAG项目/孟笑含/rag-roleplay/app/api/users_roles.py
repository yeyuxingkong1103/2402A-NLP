# -*- coding: utf-8 -*-
"""用户（注册/登录）与角色管理接口。"""
import uuid
# 解析：UUID 模块（生成登录 token）

from fastapi import APIRouter, Depends, HTTPException
# 解析：路由、依赖注入、HTTP 异常
from sqlalchemy.orm import Session
# 解析：数据库会话类型

from app.api.deps import get_current_user, get_db, get_redis
# 解析：依赖（鉴权、会话、Redis）
from app.config import settings
# 解析：全局配置
from app.core.prompts import DEFAULT_PROMPT_TEMPLATE
# 解析：默认提示词模板（角色创建兜底）
from app.models.tables import Role
# 解析：角色表
from app.models.tables import User
# 解析：用户表
from app.schemas import LoginIn, RegisterIn, RoleCreate, RoleOut, RoleUpdate, UserOut
# 解析：请求/响应模型

# ================= 用户 =================

user_router = APIRouter(prefix="/api/users", tags=["用户"])
# 解析：用户路由组（前缀 /api/users）


def _issue_token(redis, user: User) -> str:
    # 解析：签发登录 token
    token = uuid.uuid4().hex
    # 解析：生成随机 token（32 位十六进制）
    redis.set(f"session:token:{token}", user.id, ex=settings.token_ttl_seconds)
    # 解析：Redis 存 token→用户ID（带 7 天过期）
    return token
    # 解析：返回 token
'''
uuid4().hex：32 位十六进制随机串，碰撞概率极低。
redis.set(key, value, ex=秒)：写入并设置过期时间。
key 设计：session:token:{token} → 便于按前缀清理。
value 是 user.id：反向查找用，避免把整个 User 序列化进 Redis。
'''

@user_router.post("/register", response_model=UserOut)
# 解析：注册接口（响应模型 UserOut）
# 注册：用户名唯一校验 + 密码哈希存储 + 签发 token
def register(body: RegisterIn, db: Session = Depends(get_db), redis=Depends(get_redis)):
    # 解析：请求体 + 会话 + Redis 依赖注入
    exists = db.query(User).filter(User.username == body.username).first()
    # 解析：查用户名是否已存在
    if exists:
        # 解析：已存在
        raise HTTPException(status_code=400, detail="用户名已存在")
        # 解析：400 报错
    user = User(username=body.username)
    # 解析：创建用户对象
    user.set_password(body.password)
    # 解析：密码哈希存储（不存明文）
    db.add(user)
    # 解析：加入会话
    db.commit()
    # 解析：提交入库
    db.refresh(user)
    # 解析：刷新拿到自增 ID
    return UserOut(id=user.id, username=user.username, token=_issue_token(redis, user))
    # 解析：返回用户信息与 token
'''
response_model=UserOut：响应自动序列化，只暴露 id/username/token。
body: RegisterIn：请求体自动校验（Pydantic）。
唯一性检查：query().filter().first()。
set_password()：哈希存储（bcrypt/passlib），不存明文。
db.add → db.commit → db.refresh：插入后拿自增 ID。
返回时顺带签发 token，注册即登录，省一次请求。
'''

@user_router.post("/login", response_model=UserOut)
# 解析：登录接口
# 登录：校验用户名密码，签发新 token
def login(body: LoginIn, db: Session = Depends(get_db), redis=Depends(get_redis)):
    # 解析：请求体 + 会话 + Redis
    user = db.query(User).filter(User.username == body.username).first()
    # 解析：按用户名查用户
    if user is None or not user.verify_password(body.password):
        # 解析：用户不存在或密码错
        raise HTTPException(status_code=401, detail="用户名或密码错误")
        # 解析：401（不区分"用户不存在"防枚举）
    return UserOut(id=user.id, username=user.username, token=_issue_token(redis, user))
    # 解析：返回用户信息与新 token
'''
user is None or not verify_password(...)：合并判断，两种失败都返回同一提示。
防用户枚举：不告诉攻击者"用户名不存在"还是"密码错"。
登录成功签发新 token（旧 token 仍有效直到过期，如需单点登录要额外踢旧 token）。
'''

# ================= 角色 =================

role_router = APIRouter(prefix="/api/roles", tags=["角色"])
# 解析：角色路由组（前缀 /api/roles）


@role_router.get("", response_model=list[RoleOut])
# 解析：角色列表接口
# 角色列表（全部角色，含人设与提示词模板）
def list_roles(db: Session = Depends(get_db), user=Depends(get_current_user)):
    # 解析：需登录（get_current_user 校验 X-Token）
    return db.query(Role).order_by(Role.id).all()
    # 解析：查询全部角色按 ID 排序返回
'''
"" 空路径：配合 prefix="/api/roles"，实际是 GET /api/roles。
Depends(get_current_user)：强制登录。虽然没用到 user，但依赖会执行鉴权。
返回全部角色（不区分用户，是公共角色库）。
'''

@role_router.post("", response_model=RoleOut)
# 解析：创建角色接口
# 创建角色：名称唯一，模板缺省用默认模板
def create_role(
    # 解析：创建角色
    body: RoleCreate, db: Session = Depends(get_db), user=Depends(get_current_user)
    # 解析：请求体 + 会话 + 鉴权
):
    if db.query(Role).filter(Role.name == body.name).first():
        # 解析：角色名已存在
        raise HTTPException(status_code=400, detail="角色名已存在")
        # 解析：400 报错
    role = Role(
        # 解析：构造角色对象
        name=body.name,
        # 解析：角色名
        category=body.category,
        # 解析：分类
        persona=body.persona,
        # 解析：人设
        prompt_template=body.prompt_template or DEFAULT_PROMPT_TEMPLATE,
        # 解析：模板（未提供时用默认模板兜底）
    )
    db.add(role)
    # 解析：加入会话
    db.commit()
    # 解析：提交
    db.refresh(role)
    # 解析：刷新拿 ID
    return role
    # 解析：返回角色对象
'''
名称唯一校验。
body.prompt_template or DEFAULT_PROMPT_TEMPLATE：空字符串/None 都兜底（Python 的 or 短路）。
返回 ORM 对象，response_model=RoleOut 自动过滤字段。
'''

@role_router.put("/{role_id}", response_model=RoleOut)
# 解析：更新角色接口（路径参数 role_id）
# 部分更新角色（只改传入字段）
def update_role(
    # 解析：更新角色
    role_id: int,
    # 解析：角色 ID
    body: RoleUpdate,
    # 解析：更新请求（全字段可选）
    db: Session = Depends(get_db),
    # 解析：会话
    user=Depends(get_current_user),
    # 解析：鉴权
):
    role = db.get(Role, role_id)
    # 解析：按主键查角色
    if role is None:
        # 解析：不存在
        raise HTTPException(status_code=404, detail="角色不存在")
        # 解析：404
    for field, value in body.model_dump(exclude_unset=True).items():
        # 解析：只遍历请求中实际传入的字段（部分更新语义）
        setattr(role, field, value)
        # 解析：动态设置属性
    db.commit()
    # 解析：提交
    db.refresh(role)
    # 解析：刷新
    return role
    # 解析：返回更新后的角色
'''
RoleUpdate 所有字段都是 Optional。
若客户端只传 {"persona": "新的人设"}，model_dump(exclude_unset=True) 只返回 {"persona": "新的人设"}。
未传的字段不在 dict 里，setattr 就不会覆盖——实现 PATCH 语义（只改传入的）。
若不加 exclude_unset，未传字段会以 None 覆盖原值，导致数据丢失。
'''

@role_router.delete("/{role_id}")
# 解析：删除角色接口
# 删除角色（知识库 Collection 保留）
def delete_role(
    # 解析：删除角色
    role_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)
    # 解析：角色 ID + 会话 + 鉴权
):
    role = db.get(Role, role_id)
    # 解析：查角色
    if role is None:
        # 解析：不存在
        raise HTTPException(status_code=404, detail="角色不存在")
        # 解析：404
    db.delete(role)
    # 解析：删除角色记录（知识库 Collection 保留——重新创建同名角色可复用数据）
    db.commit()
    # 解析：提交
    return {"ok": True}
    # 解析：返回成功标记
'''
只删 MySQL 记录，注释明确"知识库 Collection 保留"
好处：重新创建同名角色时，Milvus 里的向量数据可复用（前提是 Collection 按角色名命名）。
风险：孤儿数据（角色删了但 Collection 还在），需定期清理。
'''
