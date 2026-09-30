# -*- coding: utf-8 -*-
"""
MinerU 命令行入口（供本项目以指定解释器调用）

存在原因：
    MinerU 安装后提供的 mineru 可执行文件绑定的是它被安装时的那个 Python 解释器。
    本项目运行在项目自带的虚拟环境中，若直接调用全局 mineru 命令，
    会用错误的解释器加载依赖，导致版本冲突。

    因此统一改为「当前解释器 + 本入口脚本」的方式调用，
    保证 MinerU 运行在与本项目完全一致的依赖环境里。

用法（由 scripts/pdf_parse.py 内部调用，无需手工执行）：
    <venv>/python scripts/mineru_entry.py -p <pdf> -o <outdir> -b pipeline -m txt -l ch
"""

import sys

from mineru.cli.client import main

if __name__ == "__main__":
    sys.exit(main())
