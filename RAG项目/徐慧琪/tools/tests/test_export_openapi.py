"""契约漂移护栏：导出脚本必须覆盖 11 条路径，且两侧问答键集的红线原样存在。

没有这条：后端偷偷改了键，前端类型生成会照单全收（生成物跟着变），而前端
代码仍在按旧字段名读——编译期看不出来的那种漂移（本项目最防的一类）。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_PATHS = {
    "/api/v1/auth/login", "/api/v1/public/qa", "/api/v1/qa", "/api/v1/search",
    "/api/v1/cases/search", "/api/v1/law/nav",
    "/api/v1/law/{law_id}/articles/{article_no}",
    "/api/v1/lawyers/recommend", "/api/v1/admin/audit/export",
    "/healthz", "/metrics",
}


def test_export_openapi_covers_all_routes_and_key_red_lines(tmp_path):
    out = tmp_path / "openapi.json"
    run = subprocess.run([sys.executable, str(ROOT / "tools" / "export_openapi.py"), str(out)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    schema = json.loads(out.read_text(encoding="utf-8"))
    assert set(schema["paths"]) == EXPECTED_PATHS
    props = schema["components"]["schemas"]["PublicQAAnswer"]["properties"]
    assert {"lawyers", "fee_range"} <= set(props)
    lawyer_props = schema["components"]["schemas"]["QAAnswer"]["properties"]
    assert not ({"lawyers", "fee_range"} & set(lawyer_props))
