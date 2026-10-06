# -*- coding: utf-8 -*-
"""
工单01 Web 服务启动脚本（问答界面 + REST 接口）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

底层为 rag_core/api.py（FastAPI）：内置零构建的网页问答界面，
支持文字提问、多轮对话、答案展示、引用来源与点赞/点踩反馈，并提供 PDF 上传接口。

访问地址（默认端口 8000）：
    问答界面      http://127.0.0.1:8000/
    接口文档      http://127.0.0.1:8000/docs      （Swagger UI，可在此上传 PDF）
    健康检查      http://127.0.0.1:8000/api/health
    性能指标      http://127.0.0.1:8000/api/metrics
    已入库文档    http://127.0.0.1:8000/api/documents

用法：
    python "工单01-基于PDF文档的问答系统/src/serve.py"
    python .../src/serve.py --port 8080 --reload
    python .../src/serve.py --host 0.0.0.0 --port 80      # 局域网/容器对外服务

提示：首次使用请先执行 build_index.py 建立索引；Web 端默认按 wo06_hybrid
      预设检索（无该预设的索引也能工作，但建议保持一致）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- 让脚本可以独立运行：把项目根目录（工单作业/）加入模块搜索路径 ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core import config                        # noqa: E402


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="工单01：启动招股说明书问答 Web 服务（uvicorn + rag_core.api）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1",
                   help="监听地址，默认 127.0.0.1（对外服务用 0.0.0.0）")
    p.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    p.add_argument("--reload", action="store_true",
                   help="开发模式：代码改动自动重启（会额外启动监视进程）")
    p.add_argument("--workers", type=int, default=1, help="工作进程数，默认 1")
    p.add_argument("--log-level", default="info",
                   choices=["critical", "error", "warning", "info", "debug", "trace"],
                   help="日志级别，默认 info")
    p.add_argument("--skip-index-check", action="store_true",
                   help="跳过索引就绪检查（容器编排时可用 /api/health 探活）")
    return p


# ---------------------------------------------------------------------------
# 索引就绪检查
# ---------------------------------------------------------------------------
def check_index() -> bool:
    """检查索引是否已构建，未就绪时打印修复命令（不阻塞启动）。"""
    bm25_path = config.INDEX_DIR / "bm25.pkl"
    meta_path = config.INDEX_DIR / "prospectus.meta.json"
    ok = bm25_path.exists() or meta_path.exists()
    if not ok:
        print("[警告] 未检测到索引文件，问答接口将返回空检索结果。")
        print('       请先执行：python "工单01-基于PDF文档的问答系统/src/build_index.py"')
    else:
        print(f"[就绪] 索引目录：{config.INDEX_DIR}")
    return ok


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()

    try:
        import uvicorn
    except ImportError:
        print("[错误] 未安装 uvicorn，请执行：pip install 'uvicorn[standard]' fastapi")
        return 2

    if not args.skip_index_check:
        check_index()

    if not config.DEEPSEEK_API_KEY:
        print("[警告] 未配置 DEEPSEEK_API_KEY，问答接口将无法调用生成模型。")
        print('       设置示例：set DEEPSEEK_API_KEY=sk-xxxx')

    if args.workers > 1 and args.reload:
        print("[错误] --reload 与 --workers>1 不能同时使用。")
        return 2

    banner = f"""
============================================================
  招股说明书智能问答系统 已启动
  工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
------------------------------------------------------------
  问答界面   : http://{args.host}:{args.port}/
  接口文档   : http://{args.host}:{args.port}/docs
  健康检查   : http://{args.host}:{args.port}/api/health
  停止服务   : Ctrl+C
============================================================
"""
    print(banner)

    # 用导入字符串启动：--reload 时 uvicorn 的子进程同样继承 sys.path，
    # 因此上面的 sys.path.insert 对热重载依然有效。
    uvicorn.run(
        "rag_core.api:app",
        host=args.host, port=args.port,
        reload=args.reload, workers=None if args.reload else args.workers,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
