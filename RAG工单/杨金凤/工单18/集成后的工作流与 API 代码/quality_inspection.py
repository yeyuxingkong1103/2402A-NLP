from flask import request, jsonify
from api.apps import login_required
import sys, os

SKILL_ROOT = "/ragflow"
if SKILL_ROOT not in sys.path:
    sys.path.insert(0, SKILL_ROOT)


@login_required
def quality_inspection():
    data = request.get_json(force=True)
    folder_path = data.get("folder_path")
    output_dir = data.get("output_dir", "/tmp/dqa_output")

    if not folder_path:
        return jsonify({"code": 400, "message": "必须提供 folder_path"}), 400

    try:
        from document_quality_assessment.assess import assess_folder
        report = assess_folder(
            folder_path=folder_path,
            config_path="/ragflow/document_quality_assessment/assessment_config.yaml",
            output_dir=output_dir
        )
        return jsonify({"code": 0, "data": report, "output_dir": output_dir})
    except Exception as e:
        return jsonify({"code": 500, "message": str(e)}), 500