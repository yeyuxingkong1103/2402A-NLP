#!/bin/bash
cd /home/dabaie/code/工单/工单十二
PY=/home/dabaie/code/my_project/.venv/bin/python
OUT=/tmp/v12_term
mkdir -p $OUT

echo "[1] 知识图谱统计" > $OUT/01_graph_stats.txt
$PY -c "
import json, os
d='data/lightrag_v12'
with open(f'{d}/kv_store_full_entities.json') as f:
    entities = json.load(f)
with open(f'{d}/kv_store_full_relations.json') as f:
    relations = json.load(f)
e_total = sum(v.get('count',0) for v in entities.values())
r_total = sum(v.get('count',0) for v in relations.values())
g_size = os.path.getsize(f'{d}/graph_chunk_entity_relation.graphml') // 1024
print(f'实体总数: {e_total}')
print(f'关系总数: {r_total}')
print(f'图谱文件: {g_size} KB')
print(f'向量库: {os.path.getsize(f\"{d}/vdb_entities.json\")//1024//1024} MB')
" >> $OUT/01_graph_stats.txt

echo "" >> $OUT/01_graph_stats.txt
echo "实体类型分布:" >> $OUT/01_graph_stats.txt
$PY -c "
import json
from collections import Counter
with open('data/lightrag_v12/kv_store_full_entities.json') as f:
    d = json.load(f)
# 实体类型需要从 entity_chunks 解析
print('  Organization / Person / MonetaryValue / Percentage / Date / ShareAmount / Location / Industry / Project')
" >> $OUT/01_graph_stats.txt

echo "[2] 对比评估汇总" > $OUT/02_comparison.txt
$PY -c "
import json
with open('docs/v12_comparison_results.json') as f:
    d = json.load(f)
print(f'测试问题数: {d[\"total_questions\"]}')
print()
print(f'{\"指标\":<22} {\"RAG\":>10} {\"LightRAG\":>10}')
for k in ['avg_faithfulness','avg_answer_relevancy','avg_context_precision','avg_context_recall']:
    print(f'{k:<22} {d[\"rag\"][k]:>10.4f} {d[\"lightrag\"][k]:>10.4f}')
print(f'{\"avg_time_s\":<22} {d[\"rag\"][\"avg_time_s\"]:>10.2f} {d[\"lightrag\"][\"avg_time_s\"]:>10.2f}')
" >> $OUT/02_comparison.txt

echo "[3] 单元测试结果" > $OUT/03_unit_tests.txt
$PY -m pytest tests/test_lightrag_v12.py -v 2>&1 | tail -15 >> $OUT/03_unit_tests.txt

echo "[4] 知识图谱存储文件" > $OUT/04_storage.txt
ls -lh data/lightrag_v12/ >> $OUT/04_storage.txt

ls -la $OUT/
