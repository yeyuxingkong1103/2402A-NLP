from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.rag.evaluation import EvaluationSample, evaluate_samples


@pytest.mark.asyncio
async def test_evaluation_defaults_to_offline_proxy():
    service = SimpleNamespace(settings=Settings(ragas_enabled=False))
    result = await evaluate_samples(
        service,
        [
            EvaluationSample(
                question="What should be monitored?",
                answer="Blood pressure should be monitored regularly.",
                contexts=["Regular blood pressure monitoring is recommended."],
                ground_truth="Regular blood pressure monitoring.",
            )
        ],
    )

    assert result["provider"] == "local_proxy"
    assert result["sample_count"] == 1
    assert "faithfulness_proxy" in result["metrics"]
