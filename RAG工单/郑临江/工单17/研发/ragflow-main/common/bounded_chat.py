"""工单17：对真正的流式会话生命周期限流；不在返回Response时过早释放。"""
import asyncio,os
from functools import wraps
ACTIVE=int(os.getenv('CHAT_MAX_ACTIVE','4'));WAITING=int(os.getenv('CHAT_MAX_WAITING','32'));TIMEOUT=float(os.getenv('CHAT_QUEUE_TIMEOUT','5'))
_semaphore=asyncio.Semaphore(ACTIVE);_waiting=0
_metrics={'active':0,'completed':0,'rejected':0}
def snapshot_metrics():return {**_metrics,'waiting':_waiting}
def release_slot():
    _metrics['active']-=1;_metrics['completed']+=1;_semaphore.release()
def bounded_chat(function):
    @wraps(function)
    async def wrapped(*args,**kwargs):
        global _waiting
        from quart import jsonify
        if _waiting>=WAITING:
            _metrics['rejected']+=1
            return jsonify({'error':'chat_queue_full'}),429
        acquired=False;_waiting+=1
        try:
            try:await asyncio.wait_for(_semaphore.acquire(),TIMEOUT);acquired=True;_metrics['active']+=1
            except asyncio.TimeoutError:
                _metrics['rejected']+=1
                return jsonify({'error':'chat_queue_timeout'}),429
        finally:_waiting-=1
        try:
            response=await function(*args,**kwargs)
            if hasattr(response,'response') and hasattr(response.response,'__aiter__'):
                original=response.response
                async def lifetime():
                    try:
                        async for chunk in original:yield chunk
                    finally:release_slot()
                response.response=lifetime();acquired=False
            return response
        finally:
            if acquired:release_slot()
    return wrapped
