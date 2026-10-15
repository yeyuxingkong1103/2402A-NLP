"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流。独立RAGFlow服务配置。"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for num in range(14,19):
    p=ROOT/f'工单{num}/研发/ragflow-main/docker/.env'
    text=p.read_text(encoding='utf-8')
    changes={'DOC_ENGINE':'infinity','COMPOSE_PROFILES':'infinity,cpu','MEM_LIMIT':'1610612736',
      'SVR_WEB_HTTP_PORT':str(8900+num),'SVR_WEB_HTTPS_PORT':str(9400+num),'SVR_HTTP_PORT':str(9500+num),
      'ADMIN_SVR_HTTP_PORT':str(9600+num),'SVR_MCP_PORT':str(9700+num),'INFINITY_THRIFT_PORT':str(24000+num),
      'INFINITY_HTTP_PORT':str(24100+num),'INFINITY_PSQL_PORT':str(24200+num),'EXPOSE_MYSQL_PORT':str(5500+num),
      'MINIO_PORT':str(9100+num),'MINIO_CONSOLE_PORT':str(9200+num),'REDIS_PORT':str(6400+num),'DEVICE':'cpu'}
    import re
    for key,value in changes.items():text=re.sub(r'(?m)^'+key+r'=.*$',key+'='+value,text)
    p.write_text(text,encoding='utf-8')
    target=ROOT/f'工单{num}/部署/完整RAGFlow启动.sh'
    target.write_text(f'#!/usr/bin/env bash\nset -e\ncd "$(dirname "$0")/../研发/ragflow-main/docker"\ndocker compose -p rag-wo{num} -f docker-compose.yml up -d\n',encoding='utf-8')
print('five separate compose projects configured')
