from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from document_quality_assessment.assessment import DocumentQualityAssessment
from document_quality_assessment.workflow import DocumentIngestionWorkflow
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",default="quality_report.json");p.add_argument("--html",default="quality_report.html");a=p.parse_args();r=DocumentIngestionWorkflow().run([a.input]);DocumentQualityAssessment.write_json(r,a.output);DocumentQualityAssessment.write_html(r,a.html);print(f"assessed={r['summary']['files']} json={a.output} html={a.html}")
if __name__=="__main__":main()
