# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-RAG 性能瓶颈识别与优化
# 模块职责：以 socket 探测空闲端口并启动 Streamlit 界面，避免端口占用导致启动失败

import os                                                       # 导入 os 用于路径拼接
import sys                                                      # 导入 sys 用于取解释器路径
import socket                                                   # 导入 socket 用于探测空闲端口
import subprocess                                               # 导入 subprocess 用于启动 Streamlit
import config                                                   # 导入配置（同时把标准输出切到 UTF-8，避免中文乱码）

BASE_DIR = config.BASE_DIR                                      # 本工单根目录
APP_PATH = os.path.join(BASE_DIR, "app.py")                     # 待启动的 Streamlit 应用
HOST = "127.0.0.1"                                              # 仅监听本机回环地址
START_PORT = 8513                                               # 起始端口（与其他工单错开）
MAX_TRIES = 20                                                  # 最多向后尝试的端口数


def find_free_port(start=START_PORT, tries=MAX_TRIES):          # 定义空闲端口查找函数
    """从 start 开始逐个尝试绑定端口，返回第一个可用的端口号。"""
    for port in range(start, start + tries):                    # 依次尝试候选端口
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # 创建 TCP 套接字
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # 允许复用地址
            try:                                                # 尝试绑定
                s.bind((HOST, port))                            # 绑定成功说明端口空闲
                return port                                     # 立即返回该端口
            except OSError:                                     # 端口被占用
                continue                                        # 尝试下一个端口
    raise RuntimeError("在 %d~%d 范围内未找到空闲端口" % (start, start + tries - 1))  # 全部失败则报错


def main():                                                     # 定义启动主函数
    """探测空闲端口并以前台方式启动 Streamlit 应用。"""
    if not os.path.exists(APP_PATH):                            # 应用文件不存在
        raise SystemExit("未找到应用文件：%s" % APP_PATH)          # 直接退出并提示
    port = find_free_port()                                     # 探测空闲端口
    url = "http://%s:%d" % (HOST, port)                         # 拼装访问地址
    print("RAG 检索性能跟踪界面已启动，请访问：%s" % url)             # 提示访问地址
    print("按 Ctrl+C 结束服务。")                                 # 提示退出方式
    cmd = [sys.executable, "-m", "streamlit", "run", APP_PATH,  # 组装启动命令
           "--server.port", str(port),                          # 指定端口
           "--server.address", HOST,                            # 指定监听地址
           "--server.headless", "true",                         # 无浏览器环境运行
           "--browser.gatherUsageStats", "false"]               # 关闭使用数据上报
    subprocess.run(cmd, cwd=BASE_DIR)                           # 在工单目录下启动 Streamlit


if __name__ == "__main__":                                      # 仅在直接运行时执行
    main()                                                      # 启动应用
