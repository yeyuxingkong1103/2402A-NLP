import sys
sys.path.insert(0, "/root/autodl-tmp/rag_project/document_quality_assessment")
from main import assess

class DocumentIngestionWorkflow:
    name = "document_ingestion_workflow"

    def run(self, folder_path, config_path="/root/autodl-tmp/rag_project/document_quality_assessment/assessment_config.yaml"):
        report = assess(folder_path, config_path)
        routes = {"ocr": [], "direct": [], "table": [], "multimodal": [], "manual": []}
        for f, label in report["classification_labels"].items():
            if label == "Scan_PDF":
                routes["ocr"].append(f)
            elif label == "Clean_Markdown":
                routes["direct"].append(f)
            elif label == "Table_Heavy":
                routes["table"].append(f)
            elif label == "Image_Heavy":
                routes["multimodal"].append(f)
            else:
                routes["manual"].append(f)
        return {"report": report, "routes": routes, "counts": {k: len(v) for k, v in routes.items()}}
