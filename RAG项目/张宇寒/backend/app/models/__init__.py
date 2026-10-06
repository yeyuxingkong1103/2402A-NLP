from __future__ import annotations

import sys

from .embedding import EmbeddingCache, EmbeddingClient
from .llm import DeepSeekProvider, ModelGateway, estimate_messages_tokens, estimate_tokens, require_siliconflow_key
from .multimodal import MEDIA_PROMPT, MultimodalAnalyzer
from .prompt import (ANSWER_SYSTEM_PROMPT, SOLUTION_SYSTEM_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_COMPACT,
                     SOLUTION_SYSTEM_PROMPT_COMPACT, build_answer_messages, build_no_evidence_answer_messages, build_solution_messages,
                     extract_final_answer, validate_final_answer)
from .rerank import RerankClient

for alias, target in {"cache": "embedding", "chat": "llm", "embedding": "embedding", "model": "llm", "prompt": "prompt", "rerank": "rerank", "token": "llm", "vision": "multimodal"}.items():
    sys.modules.setdefault(f"{__name__}.{alias}", sys.modules[f"{__name__}.{target}"])

__all__ = ["DeepSeekProvider", "EmbeddingCache", "EmbeddingClient", "ModelGateway", "MultimodalAnalyzer", "RerankClient", "MEDIA_PROMPT", "SYSTEM_PROMPT", "SYSTEM_PROMPT_COMPACT", "ANSWER_SYSTEM_PROMPT", "SOLUTION_SYSTEM_PROMPT", "SOLUTION_SYSTEM_PROMPT_COMPACT", "build_answer_messages", "build_no_evidence_answer_messages", "build_solution_messages", "extract_final_answer", "validate_final_answer", "estimate_messages_tokens", "estimate_tokens", "require_siliconflow_key"]
