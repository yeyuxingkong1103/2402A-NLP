# -*- coding: utf-8 -*-
"""解析高血压指南 PDF，输出清洗后的纯文本到 data/guidelines/ 目录。"""
from pathlib import Path  # pathlib：面向对象路径操作，自动处理分隔符与扩展名替换

from src.config import settings, setup_logging  # 统一配置中心 + 日志初始化；建库脚本先配日志再干活，出问题能查日志
from src.pdf_parser import parse_and_clean  # PDF 解析入口：页眉页脚/水印剥离 + 表格提取 + 连续空行压缩

setup_logging()  # 初始化日志格式与级别：离线脚本的输出既打到控制台也写文件，便于回溯


def main():
    pdf_path = Path(settings.guideline_path)  # 从 .env 读取 PDF 路径，转成 Path 对象
    if not pdf_path.exists():  # 先判存在：友好提示避免 FileNotFoundError 的默认堆栈暴露给用户
        print(f"❌ PDF 不存在: {pdf_path}")  # 错误提示：告诉用户缺了什么文件
        print("   请把高血压指南 PDF 放到 data/guidelines/ 目录下")  # 修复指引：用户按图索骥补全依赖
        print("   或修改 .env 里的 GUIDELINE_PATH")  # 另一条路：改配置指向已有文件
        return  # 退出主流程：文件缺失时没必要继续执行

    print(f"⏳ 解析 PDF: {pdf_path}")  # 进度提示：让用户知道当前在做什么
    text = parse_and_clean(pdf_path)  # 解析+清洗一条龙：输出的是带 [第N页] 标记的纯文本
    print(f"   清洗后字数: {len(text)}")  # 清洗统计：若字数异常少，用户应去检查日志看是否页码标记丢失

    # 输出到同目录的 .txt 文件
    out_path = pdf_path.with_suffix(".txt")  # 保持文件名不变只换扩展名：xxx.pdf → xxx.txt，方便脚本 03 按同一 basename 找文件
    out_path.write_text(text, encoding="utf-8")  # UTF-8 编码：Python 默认文本处理编码，统一读写避免中文乱码
    print(f"✅ 清洗文本已保存: {out_path}")  # 完成提示：告诉用户产物位置


if __name__ == "__main__":
    main()  # 离线建库流程第一步：PDF → 清洗文本，第二步再跑 03_init_kb.py 做切块入库