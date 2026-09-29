"""使用真实RAGAS/LangChain/OpenAI代码，HTTP由本地固定响应替代；不代表实际质量得分。"""
import asyncio
import importlib.util
import json
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

SCRIPT = Path(__file__).resolve().parents[1] / "app/evaluate.py"
spec = importlib.util.spec_from_file_location("evaluation_integration", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

HAS_RAGAS = importlib.util.find_spec("ragas") is not None


@unittest.skipUnless(HAS_RAGAS, "需先安装固定评测依赖；纯离线单元测试不需要RAGAS")
class RealRagasOfflineIntegration(unittest.TestCase):
    def test_malformed_json_does_not_trigger_hidden_paid_repairs(self):
        import httpx
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json={"id": "invalid-fixture", "object": "chat.completion", "created": 1,
                "model": "offline-fixture", "choices": [{"index": 0, "message": {"role": "assistant", "content": "not JSON"}, "finish_reason": "stop"}]})
        transport = httpx.MockTransport(handler)
        sync_client, async_client = httpx.Client(transport=transport), httpx.AsyncClient(transport=transport)
        def offline_judge(**kwargs):
            return ChatOpenAI(**kwargs, http_client=sync_client, http_async_client=async_client)
        engine = types.SimpleNamespace(model="offline-fixture", llm=types.SimpleNamespace(base_url="https://offline.invalid/v1"),
                                       embedder=object(), options={})
        usage = {"completed_evaluator_calls": 0, "calls_without_token_usage": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        args = evaluation.parse_ragas_args([])
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}), patch("langchain_openai.ChatOpenAI", side_effect=offline_judge):
            metrics = evaluation.build_metrics(engine, args, usage)
        row = evaluation.new_row({"id": "bad", "question": "q", "reference": "r"}, "hybrid")
        row.update(answer="a", contexts=["c"])
        async def exercise():
            try:
                await evaluation.score_row(row, {"faithfulness": metrics["faithfulness"]}, args)
            finally:
                await async_client.aclose()
        asyncio.run(exercise())
        sync_client.close()
        self.assertEqual(len(calls), 1, "JSON修复不应额外调用模型")
        self.assertEqual(row["metrics"]["faithfulness"]["status"], "failed")

    def test_real_metrics_via_mock_http_and_local_embedding(self):
        import httpx
        import numpy as np
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper  # 先导入，使wrapper中的isinstance保留真实类。

        requests = []
        def handler(request):
            body = json.loads(request.content)
            prompt = "\n".join(message["content"] for message in body["messages"])
            requests.append(body)
            if '"classifications"' in prompt:
                answer = {"classifications": [{"statement": "示例事实", "reason": "本地固定响应", "attributed": 1}]}
            elif '"noncommittal"' in prompt:
                answer = {"question": "高血压怎么诊断？", "noncommittal": 0}
            elif '"statements"' in prompt and '"verdict"' in prompt:
                answer = {"statements": [{"statement": "示例事实", "reason": "本地固定响应", "verdict": 1}]}
            elif '"statements"' in prompt:
                answer = {"statements": ["示例事实"]}
            else:
                answer = {"reason": "本地固定响应", "verdict": 1}
            data = {"id": "offline-fixed-response", "object": "chat.completion", "created": 1, "model": "offline-fixture",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(answer)}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
            return httpx.Response(200, json=data)

        transport = httpx.MockTransport(handler)
        sync_client, async_client = httpx.Client(transport=transport), httpx.AsyncClient(transport=transport)
        def offline_judge(**kwargs):
            return ChatOpenAI(**kwargs, http_client=sync_client, http_async_client=async_client)

        class LocalEmbedder:
            def encode(self, texts, normalize_embeddings):
                self.normalize = normalize_embeddings
                return np.array([[1.0, 0.0] for _ in texts])

        engine = types.SimpleNamespace(model="offline-fixture", llm=types.SimpleNamespace(base_url="https://offline.invalid/v1"),
                                       embedder=LocalEmbedder(), options={"extra_body": {"thinking": {"type": "disabled"}}})
        usage = {"completed_evaluator_calls": 0, "calls_without_token_usage": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        args = evaluation.parse_ragas_args([])
        with patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-key", "RAGAS_DO_NOT_TRACK": "true"}), patch("langchain_openai.ChatOpenAI", side_effect=offline_judge):
            metrics = evaluation.build_metrics(engine, args, usage)
        row = evaluation.new_row({"id": "fixture", "question": "高血压怎么诊断？", "reference": "示例参考答案"}, "hybrid")
        row.update(answer="示例答案", contexts=["示例资料"])

        async def exercise():
            try:
                await evaluation.score_row(row, metrics, args)
            finally:
                await async_client.aclose()
        asyncio.run(exercise())
        sync_client.close()
        self.assertEqual(row["status"], "evaluated", row)
        self.assertTrue(all(item["status"] == "ok" for item in row["metrics"].values()), row)
        self.assertEqual(len(requests), 5)  # faithful两次、相关性一次、precision一次、recall一次。
        self.assertEqual(usage["completed_evaluator_calls"], 5)
        self.assertTrue(engine.embedder.normalize)
        self.assertTrue(all(body["thinking"] == {"type": "disabled"} for body in requests))


if __name__ == "__main__":
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    unittest.main()
