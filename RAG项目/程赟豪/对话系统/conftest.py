"""pytest 根配置：确保 `import src.*` 可用（无论从哪个目录运行 pytest）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
