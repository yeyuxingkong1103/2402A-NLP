# -*- coding: utf-8 -*-
"""
V9 Flask 启动封装 - 用于 Docker 入口
工单编号: 人工智能 NLP-RAG-金融问答系统部署
"""
import os, sys, logging, json
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v10
import health_check

# 引入 V9
V9_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "..", "工单9", "研发"))
if V9_DIR not in sys.path:
    sys.path.insert(0, V9_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config_v10.LOG_DIR, "app.log"), encoding="utf-8"),
    ]
)

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({
        "service": "Financial RAG QA System",
        "version": "v10.0 (V9 Graph RAG)",
        "health": "ok",
        "docs": "/api/health, /api/ask, /api/data/export, /api/data/dirs",
    })


@app.route("/api/health")
def api_health():
    """Docker HEALTHCHECK 用"""
    result = health_check.check_system()
    code = 200 if result["status"] == "healthy" else 503
    return jsonify(result), code


@app.route("/api/status")
def api_status():
    """系统详细状态"""
    return jsonify({
        "health": health_check.check_system(),
        "dirs": health_check.get_system_dirs(),
        "llm_configured": bool(config_v10.LLM_API_KEY),
        "neo4j_configured": bool(config_v10.NEO4J_URI),
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """V9 Graph RAG 问答"""
    data = request.get_json() or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "question required"}), 400
    try:
        import qa_engine_v9
        result = qa_engine_v9.answer_question(q)
        return jsonify(result)
    except Exception as e:
        logging.exception("问答失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """V8 vs V9 对比"""
    try:
        import qa_engine_v9
        result = qa_engine_v9.run_v8_v9_comparison([
            {"question": "武汉力源的控股股东是谁?", "ref_keywords": ["武汉力源科技", "35", "控股股东"]},
            {"question": "销售部有几个下属?", "ref_keywords": ["销售部", "下属"]},
        ])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data/export")
def api_export():
    """数据导出"""
    return jsonify(health_check.export_data())


@app.route("/api/data/dirs")
def api_dirs():
    """目录信息"""
    return jsonify(health_check.get_system_dirs())


@app.route("/api/data/shared", methods=["GET", "POST"])
def api_shared():
    """容器间数据共享"""
    shared_file = os.path.join(config_v10.SHARED_DIR, "shared_data.json")
    if request.method == "GET":
        if os.path.exists(shared_file):
            with open(shared_file, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        return jsonify({})
    else:
        data = request.get_json() or {}
        with open(shared_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return jsonify({"ok": True, "path": shared_file})


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Financial RAG QA on {config_v10.HOST}:{config_v10.PORT}")
    logger.info(f"Data dir: {config_v10.DATA_DIR}")
    logger.info(f"Cache dir: {config_v10.CACHE_DIR}")
    logger.info(f"LLM: {'configured' if config_v10.LLM_API_KEY else 'not configured (降级模式)'}")
    app.run(host=config_v10.HOST, port=config_v10.PORT, debug=False)
