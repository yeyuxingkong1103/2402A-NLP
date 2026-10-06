"""工单编号：人工智能NLP-RAG-Query理解优化任务。"""

from pathlib import Path

import imageio.v2 as imageio


ROOT = Path(__file__).parent
SHOTS = ROOT / "问答引擎" / "多轮对话"

with imageio.get_writer(ROOT / "演示视频.mp4", fps=1, codec="libx264", macro_block_size=2) as video:
    for path in sorted(SHOTS.glob("第*.png")) + [SHOTS / "五轮连续对话.png"]:
        frame = imageio.imread(path)
        for _ in range(3):
            video.append_data(frame)

print("演示视频.mp4 已生成")
