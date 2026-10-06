"""SQLAlchemy 声明式基类与公共列。

本模块是整个 ORM 体系的“地基”：
- Base 是所有数据表模型（users、messages 等）的公共父类；
- TimestampMixin 提供可复用的 created_at / updated_at 两个时间列。
其它 model 文件都从这里导入，从而保证所有表使用同一套声明式风格。
"""
import datetime as dt

# DateTime：SQLAlchemy 的日期时间列类型；func.now()：交给数据库生成当前时间
from sqlalchemy import DateTime, func
# DeclarativeBase：声明式基类；Mapped/mapped_column：带类型标注的现代声明式写法
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有 ORM 模型的公共基类。

    只有继承 Base，SQLAlchemy 才会把这个类识别成一张数据表。
    这里不定义任何表，仅作为统一的声明式入口。
    """

    pass


class TimestampMixin:
    """时间戳混入类（Mixin）。

    “Mixin”是一种只提供字段、不单独建表的辅助类：
    任何模型只要继承它，就自动获得创建时间与更新时间两个字段，
    避免在每个表里重复写一遍时间列。
    """

    # 创建时间：server_default=func.now() 让数据库在插入时自动填当前时间，应用代码无需手动赋值
    # nullable=True 允许为空，是为了兼容早期历史数据或手工导入场景，避免插入报错
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=True
    )
    # 更新时间：onupdate=func.now() 表示每次 UPDATE 时数据库自动刷新该字段，方便做数据审计
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=True
    )