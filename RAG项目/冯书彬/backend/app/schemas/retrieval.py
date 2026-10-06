from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    # 引用结构只保存定位元数据和必要摘录，不承载完整正文。
    kind: str
    title: str
    article: str | None
    paragraph: str | None
    excerpt: str
    version: str
    official_url: str
    material_id: str
