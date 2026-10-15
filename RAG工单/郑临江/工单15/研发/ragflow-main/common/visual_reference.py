"""工单15：保留显式物理页码与图号，供检索过滤与图片上下文复用。"""
import re
def visual_reference(query):
    page=re.search(r'第\s*(\d+)\s*页',query);figure=re.search(r'图\s*(\d+)',query)
    return {'physical_page':int(page.group(1)) if page else None,'figure':figure.group(1) if figure else None}
