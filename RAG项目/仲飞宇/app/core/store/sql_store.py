"""关系库封装：用户 / 角色 / 文档登记。

MVP 用 SQLite（零依赖、免服务），生产改 SQL_URL 为 MySQL 连接串即可。

三张表都只存"结构化的小数据"，文档正文和向量一概不在这里——`documents` 是一份
**登记表**：它只说明"某角色下有过 source 这么一份文档、当时切了多少块"，用来给
/knowledge/list 展示。检索实际用到的正文在 Milvus（见 milvus_store），所以：
    · 登记与向量是两份数据，同一份文档重灌必须两边一起处理（dedup_document + 先删向量），
      只清一边就会得到"列表里显示已入库、检索却全空"或者"列表里两条一模一样的记录"；
    · 换成 MySQL（改 SQL_URL）时也**必须把 documents 表一起搬过去**，否则新库里
      列表是空的、而向量其实还在，看起来像数据丢了。
"""
from __future__ import annotations

import os
import threading
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker

from ..config import Settings
from ..logging_config import get_logger

log = get_logger("sql")

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(128), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Role(Base):
    __tablename__ = "roles"
    id = Column(String(64), primary_key=True)  # role_id
    name = Column(String(128), nullable=False)
    avatar = Column(String(16), default="")  # 单个 emoji，前端渲染用
    description = Column(Text, default="")
    system_prompt = Column(Text, default="")
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Document(Base):
    # 这里**故意没有** (role_id, source) 的唯一约束：约束加在表上会让重灌时的
    # INSERT 直接抛错，而现在重灌是"先删旧登记再插新的"。代价是漏删就会静默出现
    # 重复记录（历史 bug，见 dedup_document），去重全靠调用方自觉。
    # chunk_count 记的是**入库那一刻**的块数，不是向量库里现在的条数，两者可以不一致。
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    role_id = Column(String(64), nullable=False)
    source = Column(String(512), nullable=False)
    title = Column(String(512), default="")
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class SQLStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = None
        self.Session = None
        # "全部就绪"标志，**必须在初始化的最后一步才置位**。
        # 以前快速路径检查的是 `self.engine is not None`，而 engine 是先设的、Session 是
        # 最后设的：并发时第二个线程看到 engine 已设就直接返回，接着用 self.Session ——
        # 那时它还是 None，报 `TypeError: 'NoneType' object is not callable`（实测必现）。
        self._ready = False
        # 首次连接要串行：同步路由跑在线程池里，两个并发请求会同时走进 connect()，
        # 一起跑 create_all —— 实测（sqlite）一个线程抛 `table users already exists`。
        # MilvusStore 那边早有这把锁，SQL 这边漏了。
        self._connect_lock = threading.Lock()

    def connect(self) -> None:
        if self._ready:
            return
        with self._connect_lock:
            if self._ready:  # 等锁期间可能已被别的线程连上
                return
            self._connect()
            self._ready = True

    def _connect(self) -> None:
        connect_args = {}
        if self.settings.sql_url.startswith("sqlite"):
            # 确保 sqlite 文件父目录存在
            db_path = self.settings.sql_url.split("///", 1)[-1]
            if db_path and not db_path.startswith(":memory:"):
                parent = os.path.dirname(os.path.abspath(db_path))
                if parent:
                    os.makedirs(parent, exist_ok=True)
            connect_args["check_same_thread"] = False  # FastAPI 多线程
            # SQLite 同一时刻只允许一个写者，默认只等 5 秒就抛 "database is locked"。
            # 并发首次请求（多线程 connect/create_all + 注册预设角色）会撞上这个等待窗口，
            # 表现出来是随机一条请求失败。放到 30 秒，够躲开启动期的写竞争。
            connect_args["timeout"] = 30

        self.engine = create_engine(self.settings.sql_url, future=True, connect_args=connect_args)
        Base.metadata.create_all(self.engine)
        self._migrate()
        # expire_on_commit=False 是必须的：本类每个方法都是 `with self.Session() as s`
        # 出去之前 return ORM 对象，session 一关就 detach 了。默认 True 时 commit 会把
        # 属性标记为过期，调用方在 session 关闭后再读 role.name 就抛 DetachedInstanceError。
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    # create_all 只建新表，不会给已存在的表补列；这里手工补上后加的字段。
    # 项目没上 alembic（单机演示，迁移脚本本身就是一笔额外维护成本），
    # 于是用这张"表 -> 缺的列 -> DDL"的小字典代替：加字段时往这里加一条即可。
    _ADDED_COLUMNS = {"roles": {"avatar": "VARCHAR(16) DEFAULT ''"}}

    def _migrate(self) -> None:
        insp = inspect(self.engine)
        existing = set(insp.get_table_names())
        for table, cols in self._ADDED_COLUMNS.items():
            if table not in existing:
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            missing = {k: v for k, v in cols.items() if k not in have}
            if not missing:
                continue
            with self.engine.begin() as conn:
                for name, ddl in missing.items():
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                    log.info("补列 %s.%s", table, name)

    def ping(self) -> bool:
        try:
            self.connect()
            with self.engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("关系库不可达: %s", exc)
            return False

    # ---- 用户 ----
    # 下面是"查空就插"，**没有** upsert_role 那样的 IntegrityError 兜底：upsert_role 是因为
    # 多 worker 启动时都会注册同一批角色才必须兜。本方法目前只被 scripts/seed.py 串行调用，
    # 所以没加；万一以后挪到请求路径上，同一 username 并发首次进入会有一个请求抛 IntegrityError。
    def get_or_create_user(self, username: str) -> User:
        self.connect()
        with self.Session() as s:
            user = s.query(User).filter(User.username == username).first()
            if user is None:
                user = User(username=username)
                s.add(user)
                s.commit()
            return user

    # ---- 角色 ----
    def list_roles(self) -> list[Role]:
        self.connect()
        with self.Session() as s:
            return s.query(Role).all()

    def get_role(self, role_id: str) -> Role | None:
        self.connect()
        with self.Session() as s:
            return s.query(Role).filter(Role.id == role_id).first()

    def upsert_role(
        self, role_id: str, name: str, description: str, system_prompt: str, avatar: str = ""
    ) -> Role:
        self.connect()
        try:
            return self._upsert_role(role_id, name, description, system_prompt, avatar)
        except IntegrityError:
            # 「先查再插」在并发下必然有窗口：两个请求同时查到 None、都去 INSERT，
            # 一个成功、一个撞 UNIQUE 约束。本项目是真实场景——nginx 后面 3 个 worker
            # 启动时都会跑一遍预设角色注册，全新库上同时插入同一个 role_id。
            # 撞了就说明别人刚插进去了，回滚后改成更新即可（upsert 的标准做法）。
            log.info("角色 %s 已被并发写入，改为更新", role_id)
            return self._upsert_role(role_id, name, description, system_prompt, avatar)

    def _upsert_role(
        self,
        role_id: str,
        name: str,
        description: str,
        system_prompt: str,
        avatar: str = "",
    ) -> Role:
        with self.Session() as s:
            role = s.query(Role).filter(Role.id == role_id).first()
            if role is None:
                role = Role(
                    id=role_id,
                    name=name,
                    avatar=avatar,
                    description=description,
                    system_prompt=system_prompt,
                )
                s.add(role)
            else:
                role.name = name
                role.avatar = avatar or role.avatar  # 未传头像时保留已有，避免被清空
                role.description = description
                role.system_prompt = system_prompt
            s.commit()
            s.refresh(role)
            return role

    # ---- 文档登记 ----
    # 纯追加、不查重：调用方必须先 dedup_document 再 register，顺序反了就会留下重复登记。
    def register_document(self, role_id: str, source: str, title: str, chunk_count: int) -> Document:
        self.connect()
        with self.Session() as s:
            doc = Document(role_id=role_id, source=source, title=title, chunk_count=chunk_count)
            s.add(doc)
            s.commit()
            s.refresh(doc)
            return doc

    def dedup_document(self, role_id: str, source: str) -> int:
        """删除同 (role_id, source) 的旧登记，返回删掉的条数。

        重灌同一份文档前必须调它。历史 bug（2026-09-21 审计时实测）：
        `scripts/ingest.py --re-ingest` 只删了 Milvus 旧向量，关系库这边一直在累加，
        同一份文档重灌两次就会在 /knowledge/list 里出现两条一模一样的记录。
        """
        self.connect()  # 别的方法都靠调用方先 connect，这里自己补上：少了它先调就 TypeError
        with self.Session() as s:
            n = (
                s.query(Document)
                .filter(Document.role_id == role_id, Document.source == source)
                .delete()
            )
            s.commit()
        return n

    def list_documents(self, role_id: str | None = None) -> list[Document]:
        """role_id 为 None 时返回**所有角色**的登记（/knowledge/list 不带参数就是这个语义）。

        拿到的只是登记表的内容，不代表向量库里现在真有这些 chunk；两边对不上时以
        Milvus 的检索结果为准（见模块 docstring）。
        """
        self.connect()
        with self.Session() as s:
            q = s.query(Document)
            if role_id:
                q = q.filter(Document.role_id == role_id)
            return q.all()
