"""工单编号：人工智能NLP-RAG-混合检索任务。"""

from pathlib import Path

import imageio.v2 as imageio


ROOT = Path(__file__).parent
SHOTS = ROOT / "问答引擎" / "混合检索测试截图"


def write_video(name, slides):
    with imageio.get_writer(ROOT / name, fps=1, codec="libx264", macro_block_size=2) as video:
        for path in slides:
            frame = imageio.imread(path)
            for _ in range(3):
                video.append_data(frame)


write_video("完整操作演示.mp4", [SHOTS / f"{name}.png" for name in
    ["向量检索", "全文检索", "混合检索", "全文查询功能", "三类重排", "权重与融合配置", "全PDF六题评测"]])
write_video("检索策略讲解.mp4", [SHOTS / f"{name}.png" for name in
    ["向量检索", "全文检索", "混合检索", "三类重排", "全PDF六题评测"]])
print("两段检索演示视频已生成")
