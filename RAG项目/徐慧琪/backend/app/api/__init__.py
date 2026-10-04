"""HTTP 接口层：路由、请求/响应模型、鉴权依赖。

存在的理由（设计 §三）：本层只做 HTTP 关注点（参数校验、鉴权、序列化、
异常→状态码），不写业务 —— 业务已在 Answerer 里编排好（定调第 1 条）。
它也是**唯一**可以 import core/ 与 db/ 的层（`main.py` 之外），因此
`db`/`retrieval`/`generation`/`recommend`/`ingest` 那些下层包不得 import core/
这条分层规则在这里是**允许**方向的反面。

`install_routes` 存在的理由是本项目的行数闸门：`main.py` 已到 300 行上限的
边缘（每加一行都要挤别处），而「有哪些路由」这件事需要一个显式清单 ——
把清单一处写在装配函数里，main.py 只需一行调用，路由增删也就有了单一落点。
它不做任何业务判断，只 include_router，注册顺序与异常处理器无关。
"""
from __future__ import annotations

from fastapi import FastAPI

from app.api import admin, auth, law, lawyer, public


def install_routes(app: FastAPI) -> None:
    """把各路由模块挂到应用上。

    逐条 include_router 而不是遍历一个模块列表：列表写法要靠 getattr 取 router，
    静态检查与「谁注册了什么」都会变得不可见；而这里只是三五行。
    前缀写在各自的 router 上（/api/v1/...），不在这一层拼 —— 前缀与端点是一体的，
    分开写会让「路径长什么样」散在两处。
    """
    app.include_router(auth.router)
    app.include_router(public.router)
    # 律师侧两条与公众侧两条**并列注册**（不是同一路径里的分支）：设计 §六 的
    # 「公众侧根本没有律师侧路由」= 这个清单里有的才是有的，/qa 与 /public/qa
    # 是两条独立路径，谁的鉴权在谁的 router 上，不在处理函数里按身份分支
    app.include_router(lawyer.router)
    app.include_router(law.router)
    # 管理侧一条（任务 7）：/admin/audit/export。它属「律师侧专用路径」那一组
    # （未认证一律 404），角色门在 router 上（partner），这里只注册
    app.include_router(admin.router)
