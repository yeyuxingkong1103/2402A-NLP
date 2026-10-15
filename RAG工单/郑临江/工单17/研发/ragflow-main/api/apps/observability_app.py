"""工单17：正式Quart蓝图的Prometheus指标，保留现有身份认证。"""
import os
import psutil
from quart import Response
from api.apps import login_required
from common.bounded_chat import snapshot_metrics
@manager.route('/metrics',methods=['GET'])
@login_required
async def metrics():
    values=snapshot_metrics();process=psutil.Process(os.getpid())
    values.update(rss_bytes=process.memory_info().rss,threads=process.num_threads())
    body='\n'.join(f'ragflow_chat_{k} {v}' for k,v in values.items())+'\n'
    return Response(body,content_type='text/plain; version=0.0.4')
