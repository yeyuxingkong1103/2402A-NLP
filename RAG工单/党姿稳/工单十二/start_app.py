# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：以「自动端口切换」方式启动 Streamlit 界面，避免端口被占用导致启动失败

import os                                                       # 导入 os 用于推导应用路径
import sys                                                      # 导入 sys 用于取当前解释器
import socket                                                   # 导入 socket 用于探测空闲端口
import subprocess                                               # 导入 subprocess 用于拉起 Streamlit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 本文件所在目录（工单十二根目录）
APP_PATH = os.path.join(BASE_DIR, "app.py")                     # 界面入口文件
START_PORT = 8511                                               # 首选端口（避开 Streamlit 默认 8501）
PORT_TRIES = 20                                                 # 端口探测次数上限


def find_free_port(start, tries):                               # 定义空闲端口探测函数
    """从 start 开始逐个尝试绑定 127.0.0.1，返回第一个可绑定的端口号。"""
    for port in range(start, start + tries):                    # 依次尝试候选端口
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 TCP 套接字
        try:                                                    # 尝试绑定端口
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允许复用处于 TIME_WAIT 的地址
            sock.bind(("127.0.0.1", port))                      # 绑定成功说明端口空闲
            return port                                         # 返回可用端口
        except OSError:                                         # 端口被占用时继续尝试下一个
            continue                                            # 处理下一个候选端口
        finally:                                                # 无论成功失败都要释放套接字
            sock.close()                                        # 关闭套接字句柄
    raise RuntimeError("在 %d~%d 范围内未找到可用端口" % (start, start + tries - 1))  # 全部占用则报错


def main():                                                     # 定义启动主函数
    """探测空闲端口后以子进程方式启动 Streamlit，并打印访问地址。"""
    port = find_free_port(START_PORT, PORT_TRIES)               # 探测可用端口
    url = "http://127.0.0.1:%d" % port                          # 拼装访问地址
    print("=" * 58)                                             # 输出分隔线
    print("Streamlit 界面已启动，请在浏览器打开：%s" % url)        # 输出访问地址
    print("按 Ctrl+C 停止服务")                                  # 输出停止提示
    print("=" * 58)                                             # 输出分隔线
    cmd = [sys.executable, "-m", "streamlit", "run", APP_PATH,  # 构造 Streamlit 启动命令
           "--server.port", str(port),                          # 指定端口
           "--server.address", "127.0.0.1",                     # 仅监听本机，避免暴露到局域网
           "--server.headless", "true",                         # 无头模式：不自动弹浏览器，服务照常可用
           "--browser.gatherUsageStats", "false"]               # 关闭使用数据上报
    subprocess.run(cmd, cwd=BASE_DIR)                           # 在工单目录下启动服务并阻塞等待


if __name__ == "__main__":                                      # 仅在直接运行时执行
    main()                                                      # 启动界面
