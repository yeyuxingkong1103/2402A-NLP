import os, sys, glob
d = "D:/Users/雨子/Desktop/专高六/liu-专高六/RAG 工单/附件/ccf_competition"
for sub in ("pdf","txt"):
    print("===", sub, "===")
    for f in sorted(os.listdir(os.path.join(d,sub))):
        raw = os.path.join(d,sub,f).encode("utf-8")
        # 文件名在盘上是 GBK 字节被当 utf-8 解过，尝试还原
        try: fixed = f.encode("latin-1").decode("gbk")
        except Exception:
            try: fixed = raw.decode("gbk")
            except Exception: fixed = f
        print(f"{os.path.getsize(os.path.join(d,sub,f)):>10}  {fixed}")
