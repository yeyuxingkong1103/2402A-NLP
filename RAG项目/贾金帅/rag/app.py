# -*- coding: utf-8 -*-
"""MedRAG service entry point; assembly lives in ``src.api.application``."""

import os

# Avoid large per-thread native math buffers on memory-constrained Windows hosts.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import uvicorn

from src.api.application import app
from src.api.dependencies import SERVER_HOST, SERVER_PORT

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=os.environ.get("MEDRAG_RELOAD", "0") == "1",
    )
