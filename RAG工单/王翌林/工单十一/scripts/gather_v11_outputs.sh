#!/bin/bash
# 工单十一：采集微调任务终端输出，供渲染为截图
cd /home/dabaie/code/工单/工单十一
OUT=/tmp/v11_term
mkdir -p $OUT
PY=/home/dabaie/code/my_project/.venv/bin/python

echo "[1] 微调结果指标对比 (finetune_v11_results.json)" > $OUT/01_finetune_metrics.txt
$PY -c "
import json
d = json.load(open('docs/finetune_v11_results.json'))
print('work_order :', d['work_order'])
print('base_model :', d['base_model'])
print('device     :', d['device'])
print('freeze_bottom_layers:', d['freeze_bottom_layers'])
print('train_params:', d['train_params'])
print('data       :', d['data'])
print()
print('=== 微调前(before) ===')
for k,v in d['before'].items():
    print(f'  {k:14s}: {v}')
print()
print('=== 微调后(after) ===')
for k,v in d['after'].items():
    print(f'  {k:14s}: {v}')
print()
print('=== 对比(delta) ===')
print(f'  {\"metric\":14s} {\"before\":>10s} {\"after\":>10s} {\"delta\":>10s}')
for c in d['compare']:
    print(f'  {c[\"metric\"]:14s} {c[\"before\"]:>10.4f} {c[\"after\"]:>10.4f} {c[\"delta\"]:>+10.4f}')
print()
print('verdict:', d['verdict'])
print('elapsed_min:', d['elapsed_min'])
" >> $OUT/01_finetune_metrics.txt

echo "[2] 训练 loss 曲线 (log_history)" > $OUT/02_train_loss.txt
$PY -c "
import json
d = json.load(open('docs/finetune_v11_results.json'))
print(f'step  epoch   loss       grad_norm   learning_rate')
for h in d['train']['log_history']:
    print(f'{h[\"step\"]:4d}  {h[\"epoch\"]:.3f}  {h[\"loss\"]:.6f}  {h[\"grad_norm\"]:.4f}    {h[\"learning_rate\"]:.2e}')
print(f'\\ntrain.seconds={d[\"train\"][\"seconds\"]}, training_loss={d[\"train\"][\"training_loss\"]}, trainable_params_M={d[\"train\"][\"trainable_params_M\"]}')
" >> $OUT/02_train_loss.txt

echo "[3] 数据集文件 (data/finetune_v11/)" > $OUT/03_dataset.txt
ls -lh data/finetune_v11/ >> $OUT/03_dataset.txt
echo "" >> $OUT/03_dataset.txt
echo "--- qa_pairs_train.jsonl 前 2 条 ---" >> $OUT/03_dataset.txt
head -2 data/finetune_v11/qa_pairs_train.jsonl >> $OUT/03_dataset.txt

echo "[4] 微调后模型目录 (models/bge-m3-ft-v11/)" > $OUT/04_model_dir.txt
ls -lh models/bge-m3-ft-v11/ >> $OUT/04_model_dir.txt

echo "[5] 单元测试运行结果 (tests/test_finetune_v11.py)" > $OUT/05_unit_tests.txt
$PY -m pytest tests/test_finetune_v11.py -v 2>&1 | tail -25 >> $OUT/05_unit_tests.txt

echo "[6] 生成报告 (gen_report.json)" > $OUT/06_gen_report.txt
cat data/finetune_v11/gen_report.json >> $OUT/06_gen_report.txt

ls -la $OUT/
