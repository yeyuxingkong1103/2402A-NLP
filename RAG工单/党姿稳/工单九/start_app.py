# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：自动探测空闲端口并以该端口启动 Streamlit 界面（满足"端口冲突时自动换端口"要求）
# 使用方式：python start_app.py

import os                                                       # 导入 os 用于路径与进程管理
import sys                                                      # 导入 sys 用于解释器路径与退出码
import socket                                                   # 导入 socket 用于探测可用端口
import subprocess                                                # 导入 subprocess 用于拉起 Streamlit 进程

HERE = os.path.dirname(os.path.abspath(__file__))               # 当前脚本所在目录
APP = os.path.join(HERE, "app.py")                              # Streamlit 应用文件路径

# 本模块不导入 config，需自行修正输出编码：Windows 控制台默认 GBK，
# 直接打印中文启动提示会变成乱码（实测日志中输出为「Streamlit �����У�」）
for _stream_name in ("stdout", "stderr"):                       # 依次处理标准输出与标准错误
    _stream = getattr(sys, _stream_name, None)                  # 取出对应流对象
    if _stream is not None and hasattr(_stream, "reconfigure"):  # 流支持重新配置时才处理
        try:                                                    # 尝试重配置编码
            _stream.reconfigure(encoding="utf-8", errors="replace")  # 改为 UTF-8
        except Exception:                                       # 重定向等场景可能失败
            pass                                                # 忽略失败，不影响启动


def find_free_port(start=8501, tries=20):                       # 定义空闲端口探测函数
    """从 start 开始向上探测，返回第一个可绑定的端口。"""
    for port in range(start, start + tries):                    # 依次尝试候选端口
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # 创建 TCP 套接字（自动关闭）
            # 不能设置 SO_REUSEADDR：Windows 下该选项允许两个套接字绑定同一地址端口，
            # 会让「探测」永远成功、失去意义（实测表现为 Streamlit 报 Port already in use）。
            # 探测地址也用 0.0.0.0，与 Streamlit 的 --server.address 0.0.0.0 保持一致。
            try:                                                # 尝试绑定端口
                s.bind(("0.0.0.0", port))                       # 绑定全部网卡地址
                return port                                     # 绑定成功即返回该端口
            except OSError:                                     # 端口被占用
                continue                                        # 尝试下一个端口
    return start                                                # 全部失败时回退默认端口（由 Streamlit 自行处理）


def main():                                                     # 定义主函数
    port = find_free_port()                                     # 探测可用端口
    print("Streamlit 启动中，访问地址：http://localhost:%d" % port)  # 打印访问地址
    cmd = [sys.executable, "-m", "streamlit", "run", APP,       # 组装启动命令
           "--server.port", str(port),                          # 指定端口
           "--server.address", "0.0.0.0",                       # 监听所有网卡便于局域网访问
           "--server.headless", "true",                         # 无头模式，不自动弹浏览器
           "--browser.gatherUsageStats", "false"]               # 关闭使用统计上报
    return subprocess.call(cmd)                                 # 以子进程方式运行并返回其退出码


if __name__ == "__main__":                                      # 仅在直接运行时执行
    sys.exit(main())                                            # 以子进程退出码结束
