"""Work order 16: debug, benchmark and fine-tune a domain vision-language model."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class VLMCase:
    image: str
    prompt: str
    expected: str
    category: str = "general"


def load_cases(path: str) -> list[VLMCase]:
    return [VLMCase(**row) for row in json.loads(Path(path).read_text(encoding="utf-8"))]


def normalize_answer(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text.casefold()).strip()


def evaluate_vlm(cases: list[VLMCase], predict: Callable[[str, str], str]) -> dict[str, Any]:
    exact = 0
    latencies = []
    details = []
    for case in cases:
        started = time.perf_counter()
        output = predict(case.image, case.prompt)
        latencies.append(time.perf_counter() - started)
        matched = normalize_answer(case.expected) in normalize_answer(output)
        exact += int(matched)
        details.append({"category": case.category, "matched": matched, "output": output})
    return {
        "accuracy": exact / max(1, len(cases)),
        "latency_seconds_mean": sum(latencies) / max(1, len(latencies)),
        "details": details,
    }


def create_lora_training_args(output_dir: str = "./vlm-lora"):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        remove_unused_columns=False,
    )


def prepare_lora_model(model_name: str, target_modules: list[str] | None = None):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForVision2Seq

    model = AutoModelForVision2Seq.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target_modules or ["q_proj", "v_proj"])
    return get_peft_model(model, config)
