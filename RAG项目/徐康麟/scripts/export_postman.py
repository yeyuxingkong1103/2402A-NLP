#!/usr/bin/env python3
"""从**真实路由**生成 Postman 集合（v2.1）—— 交付物之一，可随时重生成。

为什么要生成而不是手写：手写的集合会**悄悄过期**（加了个端点、改了个字段，
集合里还是旧的），而"从路由反射"能保证**端点数与路径永远与代码一致**；
配套的 ``tests/test_export_postman.py`` 把这条约束锁住（少一个端点就红）。

判定"要不要登录"也**不看手写清单**，而是读端点函数源码里有没有调用
``require_identity`` / ``auth_required_only`` —— 手写清单同样会过期。

用法::

    python scripts/export_postman.py --out docs/postman/legal-rag.postman_collection.json

生成后导入 Postman：先跑 **auth / 注册**（会自动把 cookie 存进集合变量），
再跑其它请求即可（集合变量 ``userId`` 也会被自动填上）。
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 页面端点（给浏览器用的 HTML），单独归一个文件夹，不算"接口"
PAGE_PATHS = {"/", "/ui", "/login", "/register", "/reset"}
#: 需要登录的判定关键字（读端点源码，不维护手写清单）
AUTH_MARKERS = ("require_identity(", "auth_required_only(")

#: 示例取值：按字段名给"能直接跑"的值（{{...}} 是集合变量）
EXAMPLE_BY_NAME: dict[str, Any] = {
    "user_id": "{{userId}}",
    "username": "{{username}}",
    "password": "{{password}}",
    "role_id": "{{roleId}}",
    "session_id": "{{sessionId}}",
    "message": "民间借贷的利率司法保护上限是多少？",
    "question": "民间借贷的利率司法保护上限是多少？",
    "stream": False,
    "top_k": 3,
    "title": "我的第一个会话",
    "theme": "light",
    "q": "合同",
    "limit": 20,
    "recovery_code": "{{recoveryCode}}",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "item"


def _folder_of(path: str) -> str:
    if path in PAGE_PATHS:
        return "0. 页面（浏览器用）"
    if path in ("/health", "/metrics", "/roles", "/livez"):
        return "1. 观测与元信息"
    if path.startswith("/auth"):
        return "2. 账号与登录"
    if path.startswith("/prefs"):
        return "3. 账号偏好"
    if path.startswith("/sessions"):
        return "4. 会话"
    if path == "/chat":
        return "5. 问答（核心）"
    if path.startswith("/documents"):
        return "6. 文档与知识库"
    if path == "/ingest":
        return "6. 文档与知识库"
    return "9. 其它"


def _auth_helpers() -> set[str]:
    """找出 ``create_app`` 里\"会做身份校验\"的嵌套函数名（含调用链闭包）。

    为什么不维护手写清单：手写清单会**悄悄过期**（新加个端点忘了登记 ⇒ 集合里少了
    ``Cookie`` 头 ⇒ 交付配置下全 401，还以为是服务坏了）。这里改成读源码：

    1. 源码里出现 ``require_identity(`` / ``auth_required_only(``，或**直接抛未登录**的函数
       （``ERROR_UNAUTHENTICATED`` / ``status.HTTP_401_UNAUTHORIZED``）算\"校验者\"；
    2. 再沿\"谁调用了校验者\"一层层往上传播（``/prefs`` 就是经由 ``_account_user`` 校验的）。

    宁可**多标**（多带一个 Cookie 头无害）也不漏标（漏标就是一片 401）。
    """
    import ast
    import textwrap

    from legal_rag.api.app import create_app

    source = textwrap.dedent(inspect.getsource(create_app))
    tree = ast.parse(source)
    funcs: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(node.name, node)

    def called_names(node: ast.AST) -> set[str]:
        out: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                target = sub.func
                if isinstance(target, ast.Name):
                    out.add(target.id)
                elif isinstance(target, ast.Attribute):
                    out.add(target.attr)
        return out

    helpers: set[str] = set()
    for name, node in funcs.items():
        segment = ast.get_source_segment(source, node) or ""
        if (any(marker in segment for marker in AUTH_MARKERS)
                or "ERROR_UNAUTHENTICATED" in segment
                or "HTTP_401_UNAUTHORIZED" in segment):
            helpers.add(name)
    changed = True
    while changed:                      # 传递闭包：调用校验者的函数也算校验者
        changed = False
        for name, node in funcs.items():
            if name not in helpers and (called_names(node) & helpers):
                helpers.add(name)
                changed = True
    return helpers


def _is_auth_required(endpoint: Any, helpers: set[str]) -> bool:
    try:
        source = inspect.getsource(endpoint)
    except (OSError, TypeError):
        return False
    if any(marker in source for marker in AUTH_MARKERS):
        return True
    return any(re.search(rf"\b{re.escape(name)}\s*\(", source) for name in helpers)


def _example_value(name: str, annotation: Any) -> Any:
    if name in EXAMPLE_BY_NAME:
        return EXAMPLE_BY_NAME[name]
    if annotation in (int, "int"):
        return 0
    if annotation in (bool, "bool"):
        return False
    if annotation in (float, "float"):
        return 0.0
    return ""


def _body_from_model(model: Any) -> dict:
    fields = getattr(model, "model_fields", None)
    if not fields:
        return {}
    body: dict = {}
    for name, field in fields.items():
        annotation = getattr(field, "annotation", str)
        body[name] = _example_value(name, annotation)
    return body


def _unwrap_optional(annotation: Any) -> Any:
    """剥掉 ``X | None`` / ``Optional[X]`` 外壳（``/auth/reset`` 就是这种签名）。"""
    import types
    import typing

    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if args:
            return args[0]
    return annotation


def _model_of(route: Any) -> Any:
    """取请求体模型：优先 ``dependant.body_params``（新版本 FastAPI 上 ``body_field.type_`` 是 None）。"""
    for param in getattr(route.dependant, "body_params", []) or []:
        for candidate in (getattr(getattr(param, "field_info", None), "annotation", None),
                          getattr(param, "type_", None),
                          getattr(param, "outer_type_", None)):
            candidate = _unwrap_optional(candidate)
            if candidate is not None and hasattr(candidate, "model_fields"):
                return candidate
    field = getattr(route, "body_field", None)
    return _unwrap_optional(getattr(field, "type_", None)) if field is not None else None


def _has_file_param(route: Any) -> bool:
    for param in getattr(route.dependant, "body_params", []) or []:
        text = str(getattr(getattr(param, "field_info", None), "annotation", "")) + \
            str(getattr(param, "type_", ""))
        if "UploadFile" in text or "UploadFile" in str(getattr(param, "name", "")):
            return True
    return False


def _body_for(route: Any, method: str) -> tuple[str, dict | None]:
    """返回 ``(mode, body)``；``mode`` 取 ``raw`` / ``formdata`` / ``none``。"""
    if method in ("GET", "HEAD"):
        return "none", None            # GET 不带请求体（否则 Postman 里会出现莫名其妙的表单）
    try:
        model = _model_of(route)
    except Exception:  # noqa: BLE001 - 反射失败不该让整份集合生成失败
        return "none", None
    if model is not None and hasattr(model, "model_fields"):
        return "raw", _body_from_model(model)
    if _has_file_param(route):
        return "formdata", {"file": "<选择本地文件>"}
    if getattr(route.dependant, "body_params", None):
        # 有 body 参数但不是 pydantic 模型（少见）：给一份通用 JSON 骨架，别猜字段
        return "raw", {}
    return "none", None


def _request_item(route: Any, helpers: set[str]) -> dict:
    method = sorted(m for m in route.methods if m not in ("HEAD", "OPTIONS"))[0]
    # 注册/登录是**建立**登录态的端点：它们本身不该带 Cookie（带了只会带进旧身份）
    auth_required = _is_auth_required(route.endpoint, helpers) and \
        route.path not in ("/auth/register", "/auth/login")
    mode, body = _body_for(route, method)
    headers: list[dict] = []
    if auth_required:
        headers.append({"key": "Cookie", "value": "{{cookie}}",
                        "description": "登录态；由「账号与登录 / 注册」或「登录」自动写入集合变量"})
    request: dict = {
        "method": method,
        "header": headers,
        "url": {
            "raw": "{{baseUrl}}" + route.path,
            "host": ["{{baseUrl}}"],
            "path": [p for p in route.path.split("/") if p],
        },
        "description": (route.description or route.summary or "").strip()[:1000],
    }
    if mode == "raw" and body is not None:
        request["header"].append({"key": "Content-Type", "value": "application/json"})
        request["body"] = {"mode": "raw",
                           "raw": json.dumps(body, ensure_ascii=False, indent=2),
                           "options": {"raw": {"language": "json"}}}
    elif mode == "formdata":
        request["body"] = {"mode": "formdata",
                           "formdata": [{"key": key, "type": "file" if key == "file" else "text",
                                         "src": [] if key == "file" else value}
                                        for key, value in (body or {}).items()]}
    item = {"name": f"{route.path}  [{method}]", "request": request, "response": []}
    if auth_required:
        item["request"]["description"] = (
            request["description"] +
            "\n\n[需登录] AUTH_REQUIRED=true 时无凭据会 401；先跑注册/登录。")
    return item


def _auth_capture_script() -> list[dict]:
    """注册/登录后把 cookie 与 user_id 存进集合变量，后续请求自动带上。"""
    return [{
        "listen": "test",
        "script": {
            "type": "text/javascript",
            "exec": [
                "const setCookie = pm.response.headers.get('Set-Cookie');",
                "if (setCookie) { pm.collectionVariables.set('cookie', setCookie.split(';')[0]); }",
                "let body = {};",
                "try { body = pm.response.json(); } catch (e) { body = {}; }",
                "if (body.user_id) { pm.collectionVariables.set('userId', body.user_id); }",
                "if (body.recovery_code) { pm.collectionVariables.set('recoveryCode', body.recovery_code); }",
                "pm.test('HTTP 2xx', () => pm.response.to.be.success);",
            ],
        },
    }]


def build_collection(title: str) -> dict:
    import os

    os.environ.setdefault("LLM_PROVIDER", "mock")
    from fastapi.routing import APIRoute

    from legal_rag.api.app import create_app, iter_api_routes
    from legal_rag.config import RagConfig

    config = RagConfig()
    config.llm_provider = config.llm_provider or "mock"
    app = create_app(config)

    folders: dict[str, list[dict]] = {}
    total = 0
    helpers = _auth_helpers()
    # 用 app.iter_api_routes 而**不是** `for route in app.routes`：
    # FastAPI ≥0.141 的 include_router 不把子路由拍平进 app.routes，而是放一个
    # `_IncludedRouter` 容器 —— 直接遍历会**静默漏掉**所有经路由模块挂载的端点
    # （2026-09-30 踩过：集合少了 /health /livez /metrics /roles 四条）。
    for route in iter_api_routes(app):
        if not isinstance(route, APIRoute):
            continue
        total += 1
        try:
            item = _request_item(route, helpers)
        except Exception as exc:  # noqa: BLE001 - 单个端点反射失败不能废掉整份集合
            print(f"[注意] {route.path} 生成请求时出错，退化成最简条目："
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            method = sorted(m for m in route.methods if m not in ("HEAD", "OPTIONS"))[0]
            item = {"name": f"{route.path}  [{method}]",
                    "request": {"method": method, "header": [],
                                "url": {"raw": "{{baseUrl}}" + route.path,
                                        "host": ["{{baseUrl}}"],
                                        "path": [p for p in route.path.split("/") if p]},
                                "description": ""},
                    "response": []}
        if route.path.startswith("/auth"):
            item["event"] = _auth_capture_script()
        folders.setdefault(_folder_of(route.path), []).append(item)

    return {
        "info": {
            "name": title,
            "description": (
                "法律 RAG 助手 —— 由 scripts/export_postman.py 从**真实路由**生成，"
                "不要手改（重生成会覆盖）。\n\n"
                "使用顺序：\n"
                "1) 先跑「账号与登录 / 注册」（Cookie 与 userId 会自动写入集合变量）；\n"
                "2) 再跑其它请求（需登录的端点会自动带 Cookie）；\n"
                "3) 交付配置 AUTH_REQUIRED=true 时，不带登录态一律 401 —— 那是预期行为。\n\n"
                f"本集合覆盖 {total} 个路由。",
            ),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "variable": [
            {"key": "baseUrl", "value": "http://127.0.0.1:18080"},
            {"key": "username", "value": "postman-user"},
            {"key": "password", "value": "Postman#2026aa"},
            {"key": "cookie", "value": ""},
            {"key": "userId", "value": ""},
            {"key": "roleId", "value": "lawyer"},
            {"key": "sessionId", "value": "postman-session"},
            {"key": "recoveryCode", "value": ""},
        ],
        "item": [{"name": name, "item": items} for name, items in sorted(folders.items())],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从真实路由导出 Postman 集合")
    parser.add_argument("--out", default="docs/postman/legal-rag.postman_collection.json")
    parser.add_argument("--title", default="法律 RAG 助手（自动生成）")
    args = parser.parse_args()

    collection = build_collection(args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(collection, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    folders = {f["name"]: len(f["item"]) for f in collection["item"]}
    print(f"已写出：{out}")
    for name, count in folders.items():
        print(f"  {name}: {count} 个请求")
    print(f"  合计 {sum(folders.values())} 个请求")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
