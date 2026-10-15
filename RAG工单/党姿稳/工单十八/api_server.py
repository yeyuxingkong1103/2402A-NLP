# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：质检 API 服务（Flask，与 RAGFlow 的 API 服务技术栈一致）。新增端点
#           POST /v1/document/quality-inspection：接收文件夹路径或文件列表，触发质检 Skill，
#           返回结构化 JSON 报告，并能以 HTML 简报形式查看待确认/待审核列表。

import os                                                          # 路径
import sys                                                         # 输出编码
import json                                                        # 请求/响应
import socket                                                      # 空闲端口探测
import threading                                                   # 串行化重负载任务

from flask import Flask, request, jsonify, Response                # Web 框架
from werkzeug.serving import make_server                           # 可编程启动（便于测试）

import document_quality_assessment as dqa                          # 质检 Skill
from document_ingestion_workflow import DocumentIngestionWorkflow, register_skill

BASE_DIR = os.path.dirname(os.path.abspath(__file__))              # 工单十八根目录
app = Flask(__name__)                                              # 应用实例
_RUN_LOCK = threading.Lock()                                       # 质检任务串行锁
_LAST = {"report": None, "html": "", "workflow": None}             # 最近一次结果缓存


def free_port(start=8800, end=8899):
    """从给定区间里找一个空闲端口，避免端口占用导致启动失败。"""
    for port in range(start, end + 1):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)      # TCP 探测
        try:
            s.bind(("127.0.0.1", port))                            # 能绑定即空闲
            return port
        except OSError:
            continue                                               # 被占用则试下一个
        finally:
            s.close()
    return start                                                   # 兜底返回起始端口


def _target_from_payload(payload):
    """从请求体解析评估目标：folder 优先，其次 files 列表。"""
    folder = payload.get("folder") or payload.get("path") or payload.get("target")
    files = payload.get("files")                                   # 文件列表
    if isinstance(files, list) and files:                          # 优先文件列表可精确控制
        return files
    if folder:
        return folder
    return None


@app.route("/health", methods=["GET"])
def health():
    """健康检查。"""
    return jsonify({"status": "ok", "skill": dqa.SKILL_META["name"],
                    "version": dqa.SKILL_META["version"]})


@app.route("/v1/document/quality-inspection", methods=["POST"])
def quality_inspection():
    """质检端点：接收文件夹路径或文件列表，返回结构化 JSON 报告。

    请求体（JSON）：
      {"folder": "D:/docs", "files": ["D:/a.pdf"], "output_dir": "data/api_reports",
       "resume": true}
    查询参数：?format=html 时返回 HTML 简报而非 JSON。
    """
    payload = request.get_json(silent=True) or {}                  # 请求体
    target = _target_from_payload(payload)                         # 评估目标
    if not target:
        return jsonify({"error": "请求体缺少 folder 或 files 字段"}), 400
    output_dir = payload.get("output_dir") or os.path.join(BASE_DIR, "data", "api_reports")
    resume = payload.get("resume")                                 # 是否断点续跑
    if not _RUN_LOCK.acquire(blocking=False):                      # 已有任务在跑
        return jsonify({"error": "已有质检任务在执行，请稍后重试"}), 429
    try:
        report = dqa.run_assessment(target=target, output_dir=output_dir,
                                    resume=resume if resume is not None else None)
    except Exception as exc:                                       # 参数/路径类错误
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 400
    finally:
        _RUN_LOCK.release()                                        # 释放锁
    _LAST["report"] = report                                       # 缓存最近结果
    _LAST["html"] = report["meta"]["report_files"].get("html", "")
    if request.args.get("format", "").lower() == "html":           # 直接要 HTML 简报
        with open(_LAST["html"], "r", encoding="utf-8") as fh:
            return Response(fh.read(), mimetype="text/html; charset=utf-8")
    return jsonify(_last_summary(report))                          # 默认返回 JSON 报告


def _last_summary(report):
    """压缩响应体积：长明细列表只回传条数与前若干条，完整内容以文件形式提供。"""
    tri = report["pdf_triage"]                                     # PDF 分诊
    return {"meta": report["meta"],                                # 元信息
            "format_stats": report["format_stats"],                # 格式分布
            "pdf_triage_summary": {"total": tri["total"], "text": tri["text"],
                                   "scan": tri["scan"], "mixed": tri["mixed"],
                                   "failed": tri["failed"], "scan_ratio": tri["scan_ratio"],
                                   "to_confirm_count": len(tri["to_confirm"]),
                                   "to_confirm_sample": tri["to_confirm"][:20]},
            "length_stats": report["length_stats"]["overall"],     # 长度分布
            "dedup_summary": {"exact_group_count": len(report["dedup"]["exact_groups"]),
                              "near_pair_count": len(report["dedup"]["near_pairs"]),
                              "exact_groups": report["dedup"]["exact_groups"][:20],
                              "near_pairs": report["dedup"]["near_pairs"][:20]},
            "classify_summary": {"label_distribution": report["classification"]["label_distribution"],
                                 "tag_distribution": report["classification"]["tag_distribution"],
                                 "route_distribution": report["classification"]["route_distribution"]},
            "to_review": {"pdf_type_to_confirm": report["to_review"]["pdf_type_to_confirm"][:20],
                          "version_conflicts": report["to_review"]["version_conflicts"][:20],
                          "sensitive_items": report["to_review"]["sensitive_items"][:50]},
            "errors": report["errors"][:20],                       # 解析失败
            "report_files": report["meta"]["report_files"]}        # 完整报告文件路径


@app.route("/v1/document/quality-inspection/report", methods=["GET"])
def last_report():
    """返回最近一次质检的完整 JSON 报告。"""
    if not _LAST["report"]:
        return jsonify({"error": "尚无质检结果，请先调用 POST /v1/document/quality-inspection"}), 404
    return Response(json.dumps(_LAST["report"], ensure_ascii=False, indent=2),
                    mimetype="application/json; charset=utf-8")


@app.route("/v1/document/quality-inspection/report.html", methods=["GET"])
def last_report_html():
    """返回最近一次质检的 HTML 简报。"""
    path = _LAST["html"]                                           # HTML 文件
    if not path or not os.path.exists(path):
        return Response("<h3>尚无质检结果，请先调用质检端点</h3>",
                        mimetype="text/html; charset=utf-8"), 404
    with open(path, "r", encoding="utf-8") as fh:
        return Response(fh.read(), mimetype="text/html; charset=utf-8")


@app.route("/v1/document/ingestion-workflow", methods=["POST"])
def ingestion_workflow():
    """工作流端点：触发 document_ingestion_workflow（质检 → 决策 → 路由）。"""
    payload = request.get_json(silent=True) or {}                  # 请求体
    target = _target_from_payload(payload)                         # 目标
    if not target:
        return jsonify({"error": "请求体缺少 folder 或 files 字段"}), 400
    if not _RUN_LOCK.acquire(blocking=False):                      # 防并发
        return jsonify({"error": "已有任务在执行，请稍后重试"}), 429
    try:
        wf = DocumentIngestionWorkflow()                           # 工作流实例
        out = wf.run(target, limit=payload.get("limit"),
                      max_pages=payload.get("max_pages", 1))
    except Exception as exc:
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 400
    finally:
        _RUN_LOCK.release()
    _LAST["workflow"] = out                                        # 缓存
    return jsonify({k: out[k] for k in
                    ("workflow", "route_distribution", "human_review_count",
                     "executed", "succeeded", "elapsed_seconds", "saved_to")
                    if k in out})


@app.route("/v1/skills", methods=["GET"])
def list_skills():
    """列出已注册技能（元数据层）。"""
    path = register_skill()                                        # 刷新清单
    with open(path, "r", encoding="utf-8") as fh:
        return jsonify(json.load(fh))


def main(port=None, host="127.0.0.1"):
    """启动 API 服务；未指定端口时自动挑选空闲端口。"""
    for name in ("stdout", "stderr"):                              # 控制台输出改 UTF-8
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    port = port or free_port()                                     # 空闲端口
    register_skill()                                               # 启动时注册技能
    print("[质检API] 服务地址： http://%s:%d" % (host, port), flush=True)
    print("[质检API] 质检端点： POST http://%s:%d/v1/document/quality-inspection"
          % (host, port), flush=True)
    srv = make_server(host, port, app, threaded=True)              # 多线程 WSGI
    srv.serve_forever()                                            # 前台运行
    return srv


if __name__ == "__main__":                                         # 命令行入口
    main(port=int(sys.argv[1]) if len(sys.argv) > 1 else None)
