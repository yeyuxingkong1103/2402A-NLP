#!/bin/bash
# 工单十：采集终端输出，供渲染为截图
cd /home/dabaie/code/工单/工单十
OUT=/tmp/v10_term
mkdir -p $OUT

echo "[1] docker ps" > $OUT/01_docker_ps.txt
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" >> $OUT/01_docker_ps.txt

echo "[2] docker logs rag-v10-api (tail 30)" > $OUT/02_docker_logs_api.txt
docker logs --tail 30 rag-v10-api 2>&1 >> $OUT/02_docker_logs_api.txt

echo "[3] docker logs rag-v10-ui (tail 20)" > $OUT/03_docker_logs_ui.txt
docker logs --tail 20 rag-v10-ui 2>&1 >> $OUT/03_docker_logs_ui.txt

echo "[4] docker volume ls | grep rag" > $OUT/04_docker_volume.txt
docker volume ls | grep rag >> $OUT/04_docker_volume.txt

echo "[5] docker network ls | grep rag" > $OUT/05_docker_network.txt
docker network ls | grep rag >> $OUT/05_docker_network.txt

echo "[6] 验收结果摘要 (deploy_v10_test_results.json)" > $OUT/06_acceptance_result.txt
python3 -c "
import json
d = json.load(open('docs/deploy_v10_test_results.json'))
print('passed:', d['passed'])
for c in d['checks']:
    print(f\"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail'][:90]}\")
" >> $OUT/06_acceptance_result.txt

ls -la $OUT/
