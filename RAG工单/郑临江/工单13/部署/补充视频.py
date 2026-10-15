"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。从实际录屏截取讲解片段。"""
import cv2
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for num,name in [(6,'检索技术讲解视频.mp4'),(8,'知识图谱可视化与创建说明.mp4'),(9,'优化思路与指标对比视频.mp4')]:
    folder=ROOT/f'工单{num}'
    cap=cv2.VideoCapture(str(folder/'部署/完整演示视频.mp4'))
    fps=cap.get(cv2.CAP_PROP_FPS);frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    writer=cv2.VideoWriter(str(folder/'部署'/name),cv2.VideoWriter_fourcc(*'mp4v'),fps,(int(cap.get(3)),int(cap.get(4))))
    start=0 if num==6 else int(frames*.55)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    while True:
        ok,frame=cap.read()
        if not ok:break
        writer.write(frame)
    cap.release();writer.release()
    (folder/'部署/视频说明.md').write_text('# 视频说明\n\n完整视频为实际网页操作，4fps采样；专项视频从该录屏中截取，图谱与技术说明同步显示在页面。无配音、无真人语音输入，具体实体创建和指标说明请配合设计/优化文档阅读。\n',encoding='utf-8')
    print(num,name)
