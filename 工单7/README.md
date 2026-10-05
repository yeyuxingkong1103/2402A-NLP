# 工单7：功能测试与评估
本工单只有一个Python文件：run.py。9份原始PDF、索引、10道题、真实Top5检索结果和评估报告均已保存。

## 运行
在本目录打开终端，运行：
```powershell
python -m pip install -r requirements.txt
python run.py
```
索引已包含，不需要重新解析。首次运行会下载M3E模型，已有缓存则使用缓存。
需要重建时运行 `python run.py --pdf-dir data/pdf`。

## 提交材料
测试用例.md、questions.json、评估结果.json、评估报告.md、测试结果图/、测试评估演示.mp4。
结果图由真实JSON排版生成，视频是结果图讲解，均不冒充实时网页截图/录屏。
工单8直接复用本目录程序和索引，请将两个文件夹放在同一个父目录。
