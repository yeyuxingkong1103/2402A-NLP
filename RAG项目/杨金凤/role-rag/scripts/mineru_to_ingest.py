#!/usr/bin/env python3
"""把 MinerU JSON 输出转成 ingest.py 能吃的格式。
支持 text / paragraph_title / doc_title / table / equation 五种块。
"""
import json
import re
import sys


def clean_eq(text: str) -> str:
    """去掉 <eq>...</eq> 标签，保留公式内容。"""
    return re.sub(r'</?eq>', '', text)


def extract_block_text(block: dict) -> str:
    """从 content 列表里拼接所有 text 内容。"""
    parts = []
    for c in block.get('content', []):
        if isinstance(c, dict):
            t = c.get('content', '')
            if t:
                parts.append(clean_eq(str(t)))
    return '\n'.join(parts)


def extract_table(block: dict) -> str:
    """表格块：caption + HTML body。"""
    caption = ''
    body = ''
    for c in block.get('content', []):
        if not isinstance(c, dict):
            continue
        ctype = c.get('type', '')
        if ctype == 'table_caption':
            for cc in c.get('content', []):
                if isinstance(cc, dict) and cc.get('content'):
                    caption = clean_eq(str(cc['content']))
        elif ctype == 'table_body':
            body = clean_eq(str(c.get('content', '')))
    if not body:
        return ''
    prefix = f'【{caption}】\n' if caption else ''
    return prefix + body


def convert(mineru_json_path, output_path):
    with open(mineru_json_path, encoding='utf-8') as f:
        data = json.load(f)

    results = []
    total_blocks = 0

    for page in data.get('pages', []):
        page_no = page.get('page_idx', 0) + 1
        texts = []

        for block in page.get('blocks', []):
            btype = block.get('type', '')

            if btype == 'doc_title':
                t = extract_block_text(block)
                if t:
                    texts.append(f'# {t}')
                    total_blocks += 1

            elif btype == 'paragraph_title':
                t = extract_block_text(block)
                if t:
                    level = block.get('level', 2)
                    prefix = '#' * min(level + 1, 6)
                    texts.append(f'{prefix} {t}')
                    total_blocks += 1

            elif btype == 'text':
                t = extract_block_text(block)
                if t:
                    texts.append(t)
                    total_blocks += 1

            elif btype == 'table':
                t = extract_table(block)
                if t:
                    texts.append(t)
                    total_blocks += 1

            elif btype == 'equation':
                t = extract_block_text(block)
                if t:
                    texts.append(f'公式：{t}')
                    total_blocks += 1

        if texts:
            results.append({'page': page_no, 'text': '\n\n'.join(texts)})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_chars = sum(len(r['text']) for r in results)
    print('转换完成')
    print(f'  页数: {len(results)}')
    print(f'  块数: {total_blocks}')
    print(f'  字符: {total_chars}')
    print(f'  输出: {output_path}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('用法: python mineru_to_ingest.py <mineru_json> <output_json>')
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
