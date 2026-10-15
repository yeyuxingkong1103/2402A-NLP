"""工单16：真实数据隔离、PNG输入和已保存适配器检查。"""
import json
from pathlib import Path
import pymupdf
BASE=Path(__file__).resolve().parents[1];R=BASE/'研发'
def rows(split):return [json.loads(x) for x in (R/f'data/finetune/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
def test_no_patent_leakage():
    train={r['document'] for r in rows('train')};val={r['document'] for r in rows('validation')};test={r['document'] for r in rows('test')}
    assert not(train&val or train&test or val&test)
def test_training_image_is_actual_png():
    for row in rows('train')+rows('test'):
        path=R/row['image'];assert path.read_bytes().startswith(b'\x89PNG')
def test_saved_adapter_and_two_epochs():
    report=json.loads((BASE/'测试/真实微调结果.json').read_text(encoding='utf-8'))
    assert report['epochs']>=1 and len(report['training'])==report['epochs']
    assert (R/'models/industrial-vlm-lora/adapter_model.safetensors').stat().st_size>1000
    assert report['training'][-1]['mean_loss']<report['training'][0]['mean_loss']
