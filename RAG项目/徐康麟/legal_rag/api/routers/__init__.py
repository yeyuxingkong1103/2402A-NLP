# -*- coding: utf-8 -*-
"""路由模块：每个文件只放**一个业务域**的路由，控制在 300~400 行以内。

拆分约定（2026-09-29 起）
------------------------
1. 路由模块只做两件事：**声明路由**（路径/方法/响应模型）与**组装响应**；
   处理逻辑放 ``services/``，可复用的依赖放 ``dependencies.py``。
2. 每个模块导出一个 ``router = APIRouter()``，由 ``app.create_app`` 用
   ``app.include_router(...)`` 挂载 —— 路径与状态码**一字不变**。
3. 需要 ``AppState`` 时用 ``Depends(get_app_state)``；需要身份校验时用
   ``Depends(require_identity)`` 等，**不要**再依赖闭包。

为什么要拆：原先 34 条路由全在 ``create_app()`` 内部（1,882 行的单函数），
改 auth 的改动会淹没在整体里，review 与定位成本都很高。
"""
