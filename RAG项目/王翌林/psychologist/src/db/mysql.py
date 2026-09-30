"""MySQL 连接与会话管理（SQLAlchemy 2.0 + PyMySQL）。

本模块有两个核心概念，初学者务必分清：
- Engine（引擎）：进程级唯一的「连接池」。它自己不执行 SQL，只是管理一堆
  可复用的 TCP 连接。创建成本高，所以全进程只应有一个（见 get_engine 单例）。
- Session（会话）：一次业务操作的工作单元，从连接池借出连接、跟踪 ORM 对象
  的变更，最后由 commit/rollback 决定这些变更是落库还是丢弃。

整个文件的两条铁律：
1. 任何 Session 都必须保证 close()（否则连接泄漏，池会被耗尽）；
2. 写操作要么 commit 要么 rollback，绝不能两者都不做（否则事务悬空、
   后续请求可能读到未提交数据的脏快照）。
"""
from contextlib import contextmanager
from typing import Generator, Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings
from src.core.logging import get_logger
# Base 是所有 ORM 模型的公共基类，元数据（Base.metadata）里登记了全部表定义，
# 建表时只要对它调 create_all 就能一次性创建所有表。
from src.models import Base

logger = get_logger("db.mysql")

# 模块级私有变量，充当“进程内单例”的容器。
# 用全局变量而不是每次 new 一个 engine，是因为连接池才是真正昂贵的资源：
# 一个池子能服务成百上千个请求，重复创建会导致连接数暴涨直至数据库拒绝服务。
_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def create_database_if_not_exists() -> None:
    """连接 MySQL 服务端（不指定库）并自动建库。

    为什么要单独建一个「不指定数据库」的引擎？
    因为目标库可能还不存在，此时连接串里带 /db_name 会直接连接失败。
    所以先用 server_url（只到 host:port）连上去执行 CREATE DATABASE，
    建好库之后再切换到带库名的 database_url 引擎。
    """
    # 一次性引擎：只用来建库，用完立刻 dispose，不进入全局单例。
    # pool_pre_ping=True 表示从池中取连接前先 ping 一次，能自动剔除
    # 被 MySQL 服务器 wait_timeout 掐掉的“死连接”，避免偶发 OperationalError。
    server_engine = create_engine(settings.server_url, pool_pre_ping=True, future=True)
    try:
        with server_engine.connect() as conn:
            conn.execute(
                text(
                    # IF NOT EXISTS 保证幂等：库已存在时不会报错，服务可反复重启。
                    # f-string 拼接库名是安全的，因为 db_name 来自本地配置而非用户输入；
                    # 反过来，表名/库名这种标识符无法用绑定参数占位，只能字符串拼接。
                    # utf8mb4 而不是 utf8：前者才是真正的 4 字节 UTF-8，
                    # 能存 emoji（心理陪伴场景用户常发）和生僻字，utf8(3字节) 会截断报错。
                    f"CREATE DATABASE IF NOT EXISTS `{settings.db_name}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
            # SQLAlchemy 2.0 的连接默认在一个隐式事务里，DDL 也需要 commit 才生效。
            conn.commit()
        logger.info("数据库检查完成：%s", settings.db_name)
    finally:
        # 无论建库成功与否都释放一次性引擎，防止连接泄漏。
        # 注意这里没用 except：建库失败要让异常上抛，由 init_db 决定是否容忍。
        server_engine.dispose()


def get_engine() -> Engine:
    """返回全局唯一的 Engine（懒加载单例）。"""
    global _engine
    # 双重判断（这里只判断一次）是为了“用到才创建”：导入模块时不会连数据库，
    # 只有第一个真正执行 SQL 的请求会承担建池成本。
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            # 取连接前先探活，解决 MySQL 8 小时空闲断连的经典问题。
            pool_pre_ping=True,
            # 池中常驻连接数。太小会在并发时排队，太大则白白占用数据库连接名额。
            pool_size=settings.db_pool_size,
            # 允许在高峰时额外临时创建的连接数（超出部分用完即关，不常驻）。
            max_overflow=settings.db_max_overflow,
            # 连接存活超过 1 小时就回收重建。这个值必须小于 MySQL 的
            # wait_timeout，否则会拿到服务器已单方面关闭的连接。
            pool_recycle=3600,
            # 打开后会把每条 SQL 打到日志，仅用于本地排错，生产务必关闭。
            echo=settings.db_echo,
            future=True,
            # 自定义 JSON 序列化：默认 ensure_ascii=True 会把中文转成 \uXXXX，
            # 导致库里存的是转义串、肉眼不可读也不便用 SQL 直接检索。
            json_serializer=lambda obj: __import__("json").dumps(obj, ensure_ascii=False),
        )
        logger.info("MySQL 引擎已创建：%s:%s/%s", settings.db_host, settings.db_port, settings.db_name)
    return _engine


def get_session_factory() -> sessionmaker:
    """返回全局唯一的 sessionmaker（生成 Session 的工厂，懒加载单例）。"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            # autoflush=False：关闭“查询前自动 flush 未提交变更”。
            # 打开它会让一条看似只读的查询偷偷发出 UPDATE，事务边界变得不可预测，
            # 排查问题时非常痛苦，因此显式关闭。
            autoflush=False,
            # expire_on_commit=False：commit 后不让对象属性过期。
            # 默认行为是 commit 后访问属性会再发一条 SELECT（因为 expire 了）；
            # 接口里常在 commit 后直接读对象字段拼响应，若连接已归还就会报错，
            # 设为 False 可直接用内存中的值，省一次往返也避免该类报错。
            expire_on_commit=False,
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖注入使用的数据库会话。

    典型用法：`def api(db: Session = Depends(get_db))`。
    FastAPI 会在请求开始时进入本生成器、请求结束时继续执行 finally。
    """
    session = get_session_factory()()
    try:
        # 把控制权交给路由函数；路由内部自行决定何时 commit。
        yield session
    except Exception:
        # 路由抛异常时必须 rollback：否则这个连接会带着一个未结束的事务
        # 被归还到池里，后续借到它的请求可能看到脏数据或触发锁等待。
        session.rollback()
        raise
    finally:
        # 无论成功、失败还是客户端断开，都一定要 close（归还连接到池）。
        # 不 close 会连接泄漏，高并发下池耗尽后所有请求都会卡住。
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """脚本/后台任务使用的会话上下文。

    与 get_db 的区别：这里**自动 commit**。适合“一件事做完就提交”的场景，
    例如启动时播种角色。需要精细控制多条 commit 边界的业务代码应使用 get_db。
    """
    session = get_session_factory()()
    try:
        # 正常路径：业务代码块执行完毕 => 提交事务。
        yield session
        session.commit()
    except Exception:
        # 异常路径：回滚本次所有未提交变更，保证“全做或全不做”的原子性。
        # rollback 之后才 raise，让调用方仍能感知并处理错误。
        session.rollback()
        raise
    finally:
        session.close()


def init_db(create_all: bool = True) -> None:
    """初始化数据库：建库 + 建表。

    :param create_all: 是否执行建表（测试或只读副本场景可传 False）
    """
    try:
        create_database_if_not_exists()
    except Exception as exc:  # pragma: no cover - 依赖外部服务
        # 降级分支：托管数据库（云 RDS）通常只给业务账号、不允许 CREATE DATABASE，
        # 会返回 "Access denied"。这种情况并不代表不可用——库往往已由 DBA 建好，
        # 所以我们只放行这一种错误并记警告，其他错误（网络不通/密码错）继续上抛。
        if "Access denied" not in str(exc):
            raise
        logger.warning("无建库权限，假定数据库已存在：%s", exc)
    if create_all:
        # create_all 是幂等的：只创建不存在的表，已存在的表不会被改动/清空，
        # 因此可以放心地每次启动都调用。注意它**不会**做字段迁移，
        # 表结构变更仍需 Alembic 之类的迁移工具。
        Base.metadata.create_all(bind=get_engine())
        logger.info("数据表检查/创建完成，共 %d 张表", len(Base.metadata.tables))


def health_check() -> bool:
    """健康检查：只发一条最轻的 SELECT 1 验证连通性。"""
    try:
        # with 语法保证连接用完立刻归还池中，而不是等 GC。
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        # 健康检查必须永远返回 bool 而不是抛异常：
        # 调用方（/health 接口）需要的是“能用/不能用”的状态，
        # 由它统一聚合，而不是让一个探测失败把整个健康接口变成 500。
        logger.error("MySQL 健康检查失败：%s", exc)
        return False