# -*- coding: utf-8 -*-
"""Small OpenAI-compatible server for a local Hugging Face causal model.

It is intentionally CPU-safe and lazy-loads the model on the first generation
request.  The project can therefore use an already downloaded Qwen directory
without pulling an Ollama image or downloading another model.
"""

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import BASE, load_env

load_env()
logger = logging.getLogger("local-llm")
app = FastAPI(title="Local Qwen OpenAI API", version="0.1.0")

DEFAULT_MODEL_PATH = "/mnt/d/AI_Friend/models/Qwen/Qwen3-0.6B"
MODEL_PATH = Path(os.environ.get(
    "LOCAL_LLM_MODEL_PATH", DEFAULT_MODEL_PATH
)).expanduser()
MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen3-0.6B")
MAX_INPUT_TOKENS = int(os.environ.get("LOCAL_LLM_MAX_INPUT_TOKENS", "4096"))
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))

_tokenizer = None
_model = None
_load_lock = threading.Lock()
_generation_lock = threading.Lock()


def _model_files_present() -> bool:
    return (MODEL_PATH / "config.json").is_file() and (
        (MODEL_PATH / "model.safetensors").is_file()
        or (MODEL_PATH / "model.safetensors.index.json").is_file()
    ) and (MODEL_PATH / "tokenizer.json").is_file()


def _ensure_model():
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model
    if not _model_files_present():
        raise RuntimeError(
            f"本地模型文件不完整：{MODEL_PATH}；请设置 LOCAL_LLM_MODEL_PATH"
        )
    with _load_lock:
        if _model is None or _tokenizer is None:
            logger.info("loading local model from %s", MODEL_PATH)
            _tokenizer = AutoTokenizer.from_pretrained(
                str(MODEL_PATH), local_files_only=True, use_fast=True
            )
            _model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_PATH),
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )
            _model.to("cpu")
            _model.eval()
            logger.info("local model loaded on CPU")
    return _tokenizer, _model


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(value or "")


def _messages(payload: dict) -> list[dict[str, str]]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")
    result = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        content = _text(item.get("content", "")).strip()
        if content:
            result.append({
                "role": str(item.get("role", "user")),
                "content": content,
            })
    if not result:
        raise HTTPException(status_code=400, detail="messages 中没有可用文本")
    return result


def _prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _clean_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*<think>.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def _generate(payload: dict) -> tuple[str, dict[str, int]]:
    tokenizer, model = _ensure_model()
    prompt = _prompt(tokenizer, _messages(payload))
    encoded = tokenizer(prompt, return_tensors="pt")
    input_length = int(encoded["input_ids"].shape[-1])
    if input_length > MAX_INPUT_TOKENS:
        encoded = {key: value[:, -MAX_INPUT_TOKENS:] for key, value in encoded.items()}
        input_length = MAX_INPUT_TOKENS
    requested = payload.get("max_tokens", DEFAULT_MAX_TOKENS)
    try:
        max_new_tokens = max(1, min(int(requested), 2048))
    except (TypeError, ValueError):
        max_new_tokens = DEFAULT_MAX_TOKENS
    try:
        temperature = float(payload.get("temperature", 0.3))
    except (TypeError, ValueError):
        temperature = 0.3
    do_sample = temperature > 0
    options = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        options["temperature"] = max(0.05, min(temperature, 2.0))
        options["top_p"] = max(0.1, min(float(payload.get("top_p", 0.9)), 1.0))
    with _generation_lock, torch.inference_mode():
        output = model.generate(**encoded, **options)
    answer = _clean_answer(tokenizer.decode(output[0][input_length:], skip_special_tokens=True))
    if not answer:
        raise RuntimeError("本地模型没有生成文本")
    usage = {
        "prompt_tokens": input_length,
        "completion_tokens": int(output.shape[-1] - input_length),
        "total_tokens": int(output.shape[-1]),
    }
    return answer, usage


def _response(answer: str, usage: dict[str, int], request_model: str) -> dict:
    return {
        "id": f"chatcmpl-local-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_model or MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": usage,
    }


@app.get("/v1/models")
def models():
    if not _model_files_present():
        return {"object": "list", "data": []}
    return {"object": "list", "data": [{
        "id": MODEL_NAME, "object": "model", "owned_by": "local",
    }]}


@app.get("/health")
def health():
    return {"ready": _model_files_present(), "loaded": _model is not None,
            "model": MODEL_NAME, "path": str(MODEL_PATH)}


@app.post("/v1/chat/completions")
def chat_completions(payload: dict):
    answer, usage = _generate(payload)
    result = _response(answer, usage, str(payload.get("model", MODEL_NAME)))
    if not payload.get("stream", False):
        return result

    chunk = {
        "id": result["id"], "object": "chat.completion.chunk",
        "created": result["created"], "model": result["model"],
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer},
                     "finish_reason": "stop"}],
    }

    def events():
        yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("LOCAL_LLM_HOST", "127.0.0.1"),
                port=int(os.environ.get("LOCAL_LLM_PORT", "8001")))
