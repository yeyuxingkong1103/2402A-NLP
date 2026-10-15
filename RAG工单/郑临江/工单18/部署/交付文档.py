"""兼容入口：运行本项目最终交付文档生成器，避免旧草稿覆盖结果。"""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('最终文档.py')),run_name='__main__')
