"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流。"""
import sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'研发'))
import pymupdf,yaml
from document_quality_assessment.assessment import inspect_file,DEFAULT,assess,identity_valid,simhash
def test_corrupt_document(tmp_path):
    p=tmp_path/'broken.pdf';p.write_bytes(b'broken');r=inspect_file(p,yaml.safe_load(DEFAULT.read_text(encoding='utf-8')))
    assert r['route']=='REVIEW' and r['error']
def test_text_scan_and_mixed(tmp_path):
    for name,flags in [('text',[True,True]),('scan',[False,False]),('mixed',[True,False])]:
        p=tmp_path/(name+'.pdf');d=pymupdf.open()
        for flag in flags:
            page=d.new_page()
            if flag:page.insert_text((40,40),'Industrial technical document text. '*5)
        d.save(p);d.close()
        result=inspect_file(p,yaml.safe_load(DEFAULT.read_text(encoding='utf-8')))
        expected={'text':'Text_PDF','scan':'Scan_PDF','mixed':'Mixed_PDF'}[name]
        assert expected in result['labels']
def test_sensitive_checksum_and_bank_default_off(tmp_path):
    cfg=yaml.safe_load(DEFAULT.read_text(encoding='utf-8'));p=tmp_path/'doc.md'
    p.write_text('电话13800138000; 邮箱 demo@example.org; 身份证11010519491231002X; 卡6222020000000000000',encoding='utf-8')
    r=inspect_file(p,cfg);types={x['type'] for x in r['sensitive_matches']}
    assert {'phone','email','identity_card'}<=types and 'bank_card' not in types
def test_near_duplicate_has_actionable_context(tmp_path):
    common='industrial furnace pressure control valve safety technical document. '*40
    (tmp_path/'a.txt').write_text(common+'revision one',encoding='utf-8');(tmp_path/'b.txt').write_text(common+'revision two',encoding='utf-8')
    r=assess([tmp_path],output_dir=tmp_path/'report');assert r['version_conflicts']
    assert all(len(x['paths'])==2 and x['previews'] for x in r['version_conflicts'])
def test_report_output_not_reingested_on_resume(tmp_path):
    (tmp_path/'input.txt').write_text('example text',encoding='utf-8')
    first=assess([tmp_path],output_dir=tmp_path/'out')
    second=assess([tmp_path],output_dir=tmp_path/'out')
    assert first['total_files']==second['total_files']==1
