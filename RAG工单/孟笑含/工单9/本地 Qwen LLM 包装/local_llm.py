# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Graph RAG 优化任务
模块：本地 LLM 包装（用于 RAGAS 评估）
功能：把 Qwen2.5-VL-3B 包装为 LangChain LLM
"""

import os
import torch
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

os.environ['HF_HUB_DISABLE_XET'] = '1'

# 模块级全局变量（绕过 Pydantic）
_GLOBAL_MODEL = None
_GLOBAL_PROCESSOR = None


def _ensure_loaded(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """懒加载：只加载一次"""
    global _GLOBAL_MODEL, _GLOBAL_PROCESSOR
    if _GLOBAL_MODEL is None:
        print(f"正在加载 {model_name}...")
        _GLOBAL_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _GLOBAL_PROCESSOR = AutoProcessor.from_pretrained(model_name)
        print("✅ 本地 Qwen2.5-VL-3B 加载完成")


class LocalQwenLLM(LLM):
    """本地 Qwen2.5-VL-3B 包装为 LangChain LLM"""

    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_new_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "local_qwen_vl"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        _ensure_loaded(self.model_name)
        messages = [{"role": "user", "content": prompt}]
        text = _GLOBAL_PROCESSOR.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _GLOBAL_PROCESSOR(text=[text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = _GLOBAL_MODEL.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        output = _GLOBAL_PROCESSOR.batch_decode(
            out[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]
        return output.strip()


if __name__ == "__main__":
    llm = LocalQwenLLM()
    print("=" * 60)
    print("测试 LLM 调用")
    print("=" * 60)
    result = llm.invoke("请用一句话回答：什么是Graph RAG？")
    print(f"回答：{result}")
