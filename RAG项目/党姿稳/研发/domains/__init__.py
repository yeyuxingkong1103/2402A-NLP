"""
domains/__init__.py — 领域注册表

对外只暴露这一组函数。chat.py / server.py / intent.py 都从这里取领域，
不再直接依赖具体类，加新领域只要往 DOMAIN_CLASSES 追加一行。
"""

from __future__ import annotations

from domains.base_domain import BaseDomain, ChatDomain
from domains.english import EnglishDomain
from domains.legal import LegalDomain
from domains.medical import MedicalDomain

# 三个专业领域的注册顺序即路由兜底时的比较顺序
DOMAIN_CLASSES = (LegalDomain, MedicalDomain, EnglishDomain)

_DOMAINS: tuple[BaseDomain, ...] = tuple(cls() for cls in DOMAIN_CLASSES)
_CHAT: BaseDomain = ChatDomain()

# 闲聊用的角色名，service 层展示"当前角色"时会用到
CHAT_ROLE_NAME = _CHAT.role_name

# 关键词打分至少到这个值（命中 1 个词）才认账，避免个别字面重合就抢走路由
ROUTE_MIN_SCORE = 0.5


def all_domains() -> list[BaseDomain]:
    """返回三个专业领域的实例。"""
    return list(_DOMAINS)


def get_domain(name: str) -> BaseDomain:
    """按领域名取实例，未知领域（含 chat）一律回落到闲聊。"""
    for dom in _DOMAINS:
        if dom.domain == name:
            return dom
    return _CHAT


def chat_domain() -> BaseDomain:
    """闲聊兜底实例。"""
    return _CHAT


def route(user_input: str, history: list[dict] | None = None) -> BaseDomain:
    """判断这条输入该交给哪个领域。

    主路径是 intent.detect_intent（模型 few-shot + 关键词双路），
    模型判成闲聊时再用各类的关键词打分复核一次——短问题（比如
    "试用期一般多久"）很容易被模型归到闲聊，复核一次能救回来。
    """
    import intent

    name = intent.detect_intent(user_input, history)
    if name and name != "chat":
        return get_domain(name)

    best: BaseDomain | None = None
    best_score = 0.0
    for dom in _DOMAINS:
        score = dom.route(user_input, history)
        if score > best_score:
            best, best_score = dom, score
    return best if best is not None and best_score >= ROUTE_MIN_SCORE else _CHAT


def role_for(domain_name: str) -> str:
    """领域 -> 中文角色名。"""
    return get_domain(domain_name).role_name


def list_roles() -> list[dict]:
    """所有角色的概览，供 /roles 接口使用。"""
    return [
        {
            "name": dom.role_name,
            "domain": dom.domain,
            "identity": dom.identity,
            "greeting": dom.greeting,
        }
        for dom in _DOMAINS
    ]
