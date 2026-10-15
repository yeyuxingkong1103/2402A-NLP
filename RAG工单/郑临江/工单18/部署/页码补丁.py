"""工单14/15：实际附图页范围边界修复。"""
from pathlib import Path
import shutil,zipfile,difflib
ROOT=Path(__file__).resolve().parents[2]
for num in [14,15]:
    repo=ROOT/f'工单{num}/研发/ragflow-main'
    if num==15:shutil.copy2(ROOT/'工单14/研发/ragflow-main/common/industrial_page_ranges.py',repo/'common/industrial_page_ranges.py')
    p=repo/'api/db/services/task_service.py';text=p.read_text(encoding='utf-8')
    text=text.replace('        for s, e in page_ranges:\n            s -= 1\n            s = max(0, s)\n            e = min(e - 1, pages)',
        '        from common.industrial_page_ranges import normalize_pdf_page_range\n        for s, e in page_ranges:\n            s, e = normalize_pdf_page_range(s, e, pages)')
    p.write_text(text,encoding='utf-8')
    with zipfile.ZipFile('D:/软件/QQ/data/RAG 工单1/14-17附件/ragflow-main.zip') as z:old=z.read('ragflow-main/api/db/services/task_service.py').decode()
    (ROOT/f'工单{num}/优化/页码边界修复.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='a/api/db/services/task_service.py',tofile='b/api/db/services/task_service.py')),encoding='utf-8')
print('single page and last-page off-by-one fixed')
