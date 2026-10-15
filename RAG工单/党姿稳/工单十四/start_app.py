# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-14-修复低质量工业PDF的解析与信息丢失
# 模块职责：检测可用端口并启动 Streamlit 界面，端口被占用时自动向后寻找空闲端口。

import os                                                        # 路径
import sys                                                       # 解释器信息
import socket                                                    # 端口探测
import subprocess                                                # 启动子进程

BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # 本目录
START_PORT = 8501                                                # 起始端口


def port_listening(host, port):
    """用实际连接判断端口是否已被监听。

    Windows 下 SO_REUSEADDR 允许两个进程绑定同一端口，只靠 bind 探测会把
    已被占用的端口误判为空闲（实测 8501 已被占用时 bind 仍然成功），
    因此这里额外做一次 connect 探测。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)                                        # 短超时，快速返回
        return s.connect_ex((host, port)) == 0                    # 0 表示连接成功=被占用


def free_port(start=START_PORT, tries=20):
    """从 start 开始向后寻找第一个真正空闲的端口。"""
    for port in range(start, start + tries):                     # 逐个尝试
        if port_listening("127.0.0.1", port):                     # 已有服务在监听
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允许快速重用
            try:
                s.bind(("127.0.0.1", port))                      # 能绑定即空闲
                return port
            except OSError:                                      # 被占用
                continue
    raise RuntimeError("未找到可用端口（尝试 %d 次）" % tries)    # 全部占用


def main():
    port = free_port()                                           # 选端口
    app = os.path.join(BASE_DIR, "app.py")                       # 目标脚本
    print(f"启动 Streamlit：http://localhost:{port}")             # 提示地址
    cmd = [sys.executable, "-m", "streamlit", "run", app,        # 启动命令
           "--server.port", str(port),                           # 指定端口
           "--server.headless", "true",                          # 不自动开浏览器
           "--browser.gatherUsageStats", "false"]                # 关闭统计
    subprocess.run(cmd, cwd=BASE_DIR)                            # 运行


if __name__ == "__main__":                                        # 入口
    main()
