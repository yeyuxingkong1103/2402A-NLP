"""工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统。实际网页帧按时间合成MP4。"""
import json
from pathlib import Path
import cv2
import numpy as np
BASE = Path(__file__).resolve().parents[1]
frames_dir = BASE / "测试/视频帧"
frames = json.loads((frames_dir / "frames.json").read_text())
out = BASE / "部署/完整演示视频.mp4"
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 4, (1366, 900))
if not writer.isOpened():
    raise RuntimeError("MP4 编码器不可用")
index = 0
for tick in np.arange(0, frames[-1]["time"] + .25, .25):
    while index + 1 < len(frames) and frames[index + 1]["time"] <= tick:
        index += 1
    data = np.fromfile(str(frames_dir / frames[index]["name"]), dtype=np.uint8)
    writer.write(cv2.imdecode(data, cv2.IMREAD_COLOR))
writer.release()
cap = cv2.VideoCapture(str(out))
assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
print(json.dumps({"video": str(out), "frames": cap.get(cv2.CAP_PROP_FRAME_COUNT), "fps": cap.get(cv2.CAP_PROP_FPS), "seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)}, ensure_ascii=False))
cap.release()
