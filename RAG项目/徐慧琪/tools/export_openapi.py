"""离线导出 FastAPI 的 OpenAPI 契约（供前端生成 TypeScript 类型）。

不启服务、不进运行时：create_app().openapi() 是纯 schema 计算（lifespan 不跑，
因此不需要 JWT 密钥 / 模型 / 数据库）。前端类型由它生成——**契约的唯一真相源
仍是 backend/app/api/schemas.py**，这里只把它序列化出来。
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.main import create_app  # noqa: E402


def main() -> int:
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "frontend" / "openapi.json"
    schema = create_app().openapi()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已导出 OpenAPI：{out}（{len(schema.get('paths', {}))} 条路径）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
