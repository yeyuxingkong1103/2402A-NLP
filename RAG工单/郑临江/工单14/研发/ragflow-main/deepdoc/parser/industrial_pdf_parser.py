"""工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失。
保留每页图像作为RAGFlow表图块；低文本页采用DeepDoc OCR，图号/物理页码写入caption。
"""
from io import BytesIO
import re
import pymupdf
import numpy as np
from PIL import Image
from deepdoc.vision import OCR
from functools import lru_cache
import threading
_ocr_lock=threading.Lock()
@lru_cache(maxsize=1)
def get_ocr():return OCR()
class IndustrialPdfParser:
    def __call__(self,filename,binary=None,from_page=0,to_page=100000,callback=None,**kwargs):
        sections=[];figures=[]
        with pymupdf.open(stream=binary,filetype='pdf') if binary else pymupdf.open(filename) as document:
            for index in range(from_page,min(to_page,len(document))):
                page=document[index];pix=page.get_pixmap(matrix=pymupdf.Matrix(2,2));image=Image.open(BytesIO(pix.tobytes('png'))).convert('RGB')
                text=page.get_text()
                if len(text.strip())<40:
                    with _ocr_lock:result=get_ocr()(np.array(image))
                    text='\n'.join(item[1][0] for item in (result or []) if isinstance(item,(list,tuple)) and len(item)==2)
                position=[(index,0,float(page.rect.width),0,float(page.rect.height))]
                caption=f'文档{filename}，PDF物理第{index+1}页。图号：'+','.join(re.findall(r'图\s*\d+',text))+'\n'+text
                figures.append(((image,caption),position))
                if text.strip():sections.append((text,f'@@{index+1}\t0\t{page.rect.width}\t0\t{page.rect.height}##'))
                if callback:callback((index+1)/len(document),f'工业PDF保留页面图像 {index+1}')
        return sections,figures,None
