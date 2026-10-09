# -*- coding: utf-8 -*-
"""
LightRAG vs 传统 RAG Flask 应用
工单编号: 人工智能 NLP-RAG 项目-LightRAG 优化
端口: 5009
"""
import os, sys, json, logging
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v12 as config
from qa_engine_v12 import RAGvsLightRAGEngine, GROUND_TRUTHS

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates_v12")

engine = RAGvsLightRAGEngine()


@app.route("/")
def index():
    return render_template("index_v12.html")


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    mode = data.get("mode", "both")  # rag / lightrag / both

    if not question:
        return jsonify({"error": "question required"}), 400

    result = {"question": question}

    if mode in ("rag", "both"):
        rag = engine.query_rag(question)
        result["rag"] = {
            "answer": rag["answer"],
            "contexts": rag["contexts"][:5],
            "tier": rag["tier"],
        }

    if mode in ("lightrag", "both"):
        light = engine.query_lightrag(question)
        result["lightrag"] = {
            "answer": light["answer"],
            "contexts": light["contexts"][:5],
            "local_kws": light["local_kws"],
            "global_kws": light["global_kws"],
            "num_local": light["num_local"],
            "num_global": light["num_global"],
            "tier": light["tier"],
        }

    return jsonify(result)


@app.route("/api/eval")
def api_eval():
    qids_str = request.args.get("ids", "")
    qids = [int(x.strip()) for x in qids_str.split(",") if x.strip()] if qids_str else None

    results = engine.compare_all(qids)

    # 汇总
    rag_scores = [r["rag"]["score"]["score"] for r in results]
    light_scores = [r["light_rag"]["score"]["score"] for r in results]
    summary = {
        "num_questions": len(results),
        "rag_avg": round(sum(rag_scores) / max(len(rag_scores), 1), 4),
        "lightrag_avg": round(sum(light_scores) / max(len(light_scores), 1), 4),
        "improvement_avg": round(
            sum(r["improvement"] for r in results) / max(len(results), 1), 4
        ),
    }

    return jsonify({"summary": summary, "results": results})


@app.route("/api/ground_truths")
def api_ground_truths():
    return jsonify([
        {"id": qid, "question": gt["question"], "keywords": gt["keywords"]}
        for qid, gt in sorted(GROUND_TRUTHS.items())
    ])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=config.PORT)
    args = parser.parse_args()
    print(f"\nLightRAG V12 启动: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
