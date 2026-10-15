"""工单17：正式RAGFlow流式限流器生命周期单测；轻量Quart接口桩只负责JSON，不伪造模型推理。"""
import importlib.util,sys,types,asyncio
from pathlib import Path
def load_module():
    path=Path(__file__).resolve().parents[1]/'研发/ragflow-main/common/bounded_chat.py'
    spec=importlib.util.spec_from_file_location('bounded_chat_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def test_stream_capacity_released_after_iteration(monkeypatch):
    monkeypatch.setitem(sys.modules,'quart',types.SimpleNamespace(jsonify=lambda x:x))
    async def run():
        module=load_module();initial=module._semaphore._value
        class Response:
            def __init__(self,body):self.response=body
        @module.bounded_chat
        async def handler():
            async def body():yield 'a';yield 'b'
            return Response(body())
        response=await handler();assert module._semaphore._value==initial-1
        assert [x async for x in response.response]==['a','b'];assert module._semaphore._value==initial
    asyncio.run(run())
def test_exception_releases_capacity(monkeypatch):
    monkeypatch.setitem(sys.modules,'quart',types.SimpleNamespace(jsonify=lambda x:x))
    async def run():
        module=load_module();initial=module._semaphore._value
        @module.bounded_chat
        async def handler():raise ValueError('fail')
        try:await handler()
        except ValueError:pass
        assert module._semaphore._value==initial
    asyncio.run(run())
