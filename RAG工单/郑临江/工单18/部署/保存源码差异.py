"""工单14-18：只导出差异，不重复修改源码。"""
from pathlib import Path
import difflib,zipfile
ROOT=Path(__file__).resolve().parents[2]
with zipfile.ZipFile('D:/软件/QQ/data/RAG 工单1/14-17附件/ragflow-main.zip') as z:
    for n in range(14,19):
        repo=ROOT/f'工单{n}/研发/ragflow-main';diff=[];changed=[]
        for p in repo.rglob('*.py'):
            rel=str(p.relative_to(repo)).replace('\\','/');old=z.read('ragflow-main/'+rel).decode('utf-8-sig') if 'ragflow-main/'+rel in z.namelist() else ''
            new=p.read_text(encoding='utf-8-sig')
            if new!=old:
                compile(new,str(p),'exec');changed.append(rel)
                diff+=difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel)
        (ROOT/f'工单{n}/优化/最终源码变更.patch').write_text(''.join(diff),encoding='utf-8')
        print(n,'changed',changed)
