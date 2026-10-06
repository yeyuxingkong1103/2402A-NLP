# 工单8：金融图谱问答
本工单只有一个Python文件：app.py，复用旁边工单7的run.py与索引。不要只拿走工单8文件夹。

## 启动
安装依赖后双击“启动网页.bat”，浏览器打开 http://127.0.0.1:8508 。
也可以进入本目录运行 `python -m streamlit run app.py --server.port 8508 --server.address 127.0.0.1`。

默认显示检索原文。需要生成回答时先启动Ollama并安装deepseek-r1:7b，然后勾选模型选项。
重跑图谱对比：`python app.py --evaluate`。

## 材料
data/graph.json、eval_questions.md、评估结果.json、模型原始回答.json、7B模型原始回答.json、英文模型回答.json、功能测试.json、网页测试.json、评估报告.md、技术文档.md、用户手册.md、测试结果图/及两段演示视频。
eval_question.md原附件未提供，使用与工单7同组10题，并明确记录替代来源。
语音、真实浏览器上传、真实网页截图和录屏尚未完成验证，详见评估报告。
