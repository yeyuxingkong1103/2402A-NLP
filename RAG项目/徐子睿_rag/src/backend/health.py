# -*- coding: utf-8 -*-
"""旧版生产组件健康检查已从精简版主链路移除。

精简版的健康接口在 backend/server.py 中，只报告 RAG 主链路状态。

在链路中的位置：
    空壳模块 —— 保留它是为了兼容仍按老路径 import 它的旧代码。
    新代码请直接用 backend/server.py 的 GET /api/health。

为什么保留一个"什么都不做"的模块：
    直接删掉会让仍在 import 它的旧脚本以 ImportError 形式失败，
    报错信息还指向一个"找不到的模块"，反而更难排查。
    留一个明确说"已移除、请改用 X"的实现，调用方一看就懂。
"""


def health():
    """返回"该功能已迁移"的提示。

    返回：
        {"status": "removed", "message": "请使用 backend.server.health() 或 GET /api/health"}

    注意它故意不返回 "ok"：
        调用方若把 status 当成健康判定，返回 removed 会让它立刻暴露出来、
        而不是误以为检查通过 —— 静默通过比报错更危险。
    """
    return {"status": "removed", "message": "请使用 backend.server.health() 或 GET /api/health"}
