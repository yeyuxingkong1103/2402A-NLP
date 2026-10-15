# -*- coding: utf-8 -*-
# api_endpoint.py 工单18 新增质检API POST /v1/document/quality‑inspection
from fastapi import FastAPI
from pydantic import BaseModel
from agent_workflow import document_ingestion_workflow

app = FastAPI()

class QualityInspectReq(BaseModel):
    doc_dir:str

@app.post("/v1/document/quality‑inspection")
def document_quality_inspect(req:QualityInspectReq):
    """对外质检API接口"""
    workflow_ret = document_ingestion_workflow(req.doc_dir)
    return {"code":0,"msg":"ok","data":workflow_ret}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8010)
