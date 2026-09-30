# -*- coding: utf-8 -*-
"""压测准备：预注册 N 个用户，把 token 写入 jmeter/tokens.csv。

用法：python scripts/gen_tokens.py [数量=50]
"""
import csv
# 解析：CSV 写入
import sys
# 解析：命令行参数

import httpx
# 解析：HTTP 客户端

BASE = "http://127.0.0.1:8000"
# 解析：服务地址
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
# 解析：生成数量（默认 50）
OUT = "jmeter/tokens.csv"
# 解析：输出文件

with open(OUT, "w", newline="", encoding="utf-8") as f:
    # 解析：打开输出文件（覆盖写）
    writer = csv.writer(f)
    # 解析：CSV 写入器
    writer.writerow(["token", "username"])
    # 解析：表头（JMeter CSV Data Set 的变量名）
    for i in range(N):
        # 解析：逐个用户
        username = f"load{i:04d}"
        # 解析：压测专用用户名（load0000 格式）
        r = httpx.post(f"{BASE}/api/users/register", json={"username": username, "password": "load123456"}, timeout=30)
        # 解析：注册
        if r.status_code == 400:
            # 解析：已注册（重跑场景）
            r = httpx.post(f"{BASE}/api/users/login", json={"username": username, "password": "load123456"}, timeout=30)
            # 解析：改登录拿 token
        r.raise_for_status()
        # 解析：非 200 抛异常
        writer.writerow([r.json()["token"], username])
        # 解析：写 token 与用户名
        if (i + 1) % 10 == 0:
            # 解析：每 10 个
            print(f"已生成 {i + 1}/{N} 个 token")
            # 解析：进度提示
print(f"完成：{OUT}")
# 解析：完成提示
