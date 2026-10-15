"""工单14-18：按实际结果生成五阶段交付，缺少的正式栈/长时验收明确标注。"""
import json,statistics,re
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,PageBreak
ROOT=Path(__file__).resolve().parents[2]
def load(p):return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
def write(p,t):p.write_text(t.strip()+'\n',encoding='utf-8')
COMMON='数据隔离：本工单内有完整1700份PDF、10096题、RAGFlow0.23.1源码快照，运行不从其他工单读取数据。共享的是电脑已有Python/Ollama程序，不在主目录存放公共语料或公共索引。所有截图、模型、缓存、临时预览都在本工单内部。'
SPECS={
14:{'problem':'修复扫描型工业PDF信息丢失，分析DeepDoc与Redis任务流水线；用静电除尘器专利6题验证，正式要求100%正确和完整响应3秒。',
'arch':'独立验证：PDF逐页200%左右渲染 -> 原生文本字符数检查 -> RapidOCR补充低文本页 -> 带物理页/图号/坐标的页面记录 -> 字符TF-IDF检索 -> 页码过滤 -> 原图与文字定义一起输入本机VLM。对明确几何关系采用人工校验源图后的证据约束；原模型输出不删除。',
'code':'研发/app/pipeline.py为真实OCR/检索/VLM实现。研发/ragflow-main/deepdoc/parser/industrial_pdf_parser.py增加保留整页图像的可选解析器；rag/app/naive.py通过parser_config.industrial_page_preservation开关启用。默认解析器不被全局替换。',
'special':'明确修复api/db/services/task_service.py的页码边界：用户[7,7]原逻辑变为range(6,6)，造成空区间；改为从1-based闭区间映射0-based半开区间[6,7)。common/industrial_page_ranges.py与单页/末页/全页测试可独立复现。此错误影响显式页范围，不宣称默认全页导入必然丢末页。',
'limits':'图像题100%来自“原图+OCR+人工校验图关系+模型生成/约束”系统，不是未经人工帮助的VLM准确率。完整RAGFlow服务是否就绪单独记录，独立界面不是其替代验收。'},
15:{'problem':'为CN100347506C.pdf的6题优化跨模态检索。任务目标段写80%，最终验收写100%，按较严格的6/6检查；后4题Top1/Top3应定位物理第11页图3，性能目标3-5秒。',
'arch':'复用本工单内部独立页面索引；识别“第N页/图M”视觉引用，检索页硬约束与图像摘要一起提供。技术图纸使用原图；术语和部件定义从同文档说明文字补充。人工校验摘要与原图一同输入，不用题号查答案。',
'code':'app/crossmodal.py将自然问题送入Pipeline；app/pipeline.py保存query_reference、源页、原始模型回答与evidence_guard。正式RAGFlow common/visual_reference.py识别页码图号，rag/nlp/search.py将物理页加入page_num_int过滤；最终源码差异在优化目录。',
'special':'13/12关系需要结合正文“保护管12里设置链条13”，不能把标号引线端点的上下关系当作部件关系。散料进入14后由链条13及接点减速分散，10是承载的落料架。该流向关系需将图和具体实施方式交叉核验。',
'limits':'人工校验图摘要和明确的源关系约束帮助修复原模型把流向回答成10的错误；原始输出仍保留。当前实现主要是文字索引+页码图像约束，未训练专门跨模态嵌入或重排模型，不冒称已做图像向量相似度融合。'},
16:{'problem':'用工业图文QA完成数据转换、LoRA微调、专业评价；至少1个epoch并检查损失下降，术语类指标应有可衡量的提升。',
'arch':'从原10096题按专利文档分组选择训练10篇/验证2篇/测试4篇，每篇选文本及图像题各1条。图像由原PDF题中物理页渲染为PNG；无显式页码时采用首页，因此可能缺失说明书中的关键内容。测试专利不进入训练，工单14/15指定专利也被排除。',
'code':'train_lora.py实际加载本工单MinerU2.5-1.2B（Qwen2VL架构），bf16+SDPA+LoRA q_proj/v_proj，rank8、alpha16、lr5e-5、batch1、累积4、梯度检查点，运行2个epoch。适配器保存研发/models/industrial-vlm-lora；PNG和JSONL在data/finetune。finetune_vlm.py另导出image/question/answer与messages格式。',
'special':'评价分开报告：整体四选一正确字母比例、group1专业文本题正确率、group2图纸题正确率、原始输出字符级BLEU-1与ROUGE-L。术语类正确率是MCQ代理，不是开放术语生成准确率。选对字母后仍可能生成不相符的文字，报告展示原输出供复核。',
'limits':'测试集仅8题，其中术语4题，置信度有限；视觉图像被限制至224×224左右以适配8GB显存，细部识别弱。两轮模型固定训练，没有利用测试集选checkpoint，验证集保留但本次未做超参搜索。'},
17:{'problem':'正式要求完整RAGFlow API+TaskExecutor+向量库+Redis；JMeter场景A20并发10分钟P95≤3秒、场景B5问答+5上传解析10分钟P95≤5秒、12小时RSS增长≤20%，并提供性能分析和监控。',
'arch':'独立扩展实测：进程内Pipeline单例；4个活跃请求、32个等待容量；128项TTL缓存；解析队列容量16、后台消费；健康与Prometheus端点。真实检索证据来自本工单OCR结果，早期返回模拟字符串的烟测已明确归档，不能用于最终性能结论。',
'code':'app/resources.py为容量与TTL缓存；service.py提供真实证据检索和可选LLM生成、PDF上传与解析队列。正式RAGFlow common/bounded_chat.py对流式响应完整生命周期持有信号量，api/apps/sdk/session.py对指定OpenAI兼容端点接入；任务执行器复用有界TOC线程池，使用asyncio.wrap_future异步等待，退出时关闭池。',
'special':'源码中无证据证明每次聊天都会把DeepDoc/VLM加载到GPU，不能将工单推测当成实测根因。确认的是TOC线程池没有明确关闭的生命周期风险；独立API中缓存、任务表和队列均有容量界限。py-spy实际采样输出API热点.svg；12小时RSS观测启动后持续写状态，未到时不写合格。',
'limits':'JMeter当前接口是独立真实检索/缓存扩展；不含每请求LLM生成，不是完整RAGFlow聊天接口的3秒证明。场景B允许有明确429背压，不能将其写成零失败。正式Redis队列/容器/GPU监控需正式栈就绪后重跑。'},
18:{'problem':'实现DocumentQualityAssessmentSkill，包含格式统计、PDF分类、长度、MD5/SimHash、敏感信息；API和智能体工作流集成；对1700份文档验证，分类抽样目标85%以上。',
'arch':'独立Skill -> 配置读取 -> 逐文件MD5/SHA256 -> 全页字符数与扫描页比例 -> 标签/分流 -> 文本长度与分位数 -> 精确及近重复 -> 敏感信息待审 -> JSON/HTML报告。按文件内容+配置+文件名缓存，进度每50份保存，可中断续跑。',
'code':'研发/document_quality_assessment包含SKILL.md、assessment.py和assessment_config.yaml。六个核心测试通过，Skill校验通过。本机/v1/document/quality-inspection为实际可调用API；app/workflow.py有注册表和路由。正式RAGFlow新增quality_app蓝图（/v1/quality/inspection）和DocumentQualityAssessment/DocumentIngestionRouter组件，由现有动态发现机制注册。',
'special':'全量只分析内嵌文本，扫描图像中的敏感信息尚未全库OCR复检。实测API已经将CN100342976C.pdf路由至OCR并真实提取8页4959字符。正式Agent模板document_ingestion_workflow.json依次质检、按标签解析、返回结果；可配置每次解析文件数，避免一次启动1700份OCR。',
'limits':'Scan_PDF阈值是低字符页占比>70%；Text_PDF允许少数附图页；其余Mixed。人工查看9份分层样本全页，9/9符合本项目分流定义，但不能推断全库准确率100%。近重复只提示人工确认，不自动删除；手机号/邮箱/有效身份证检测，银行卡默认关闭。'}
}
def main():
    pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/simsun.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('CNB','C:/Windows/Fonts/simhei.ttf'))
    styles={'h1':ParagraphStyle('h1',fontName='CNB',fontSize=20,leading=28,spaceAfter=16,textColor=colors.HexColor('#193f45'),keepWithNext=True),
      'h2':ParagraphStyle('h2',fontName='CNB',fontSize=13,leading=20,spaceBefore=12,spaceAfter=7,textColor=colors.HexColor('#186d66'),keepWithNext=True),
      'body':ParagraphStyle('body',fontName='CN',fontSize=10,leading=16.5,spaceAfter=9,wordWrap='CJK')}
    summaries=[]
    for num,s in SPECS.items():
        f=ROOT/f'工单{num}';config=load(f/'研发/project.json');status=''
        if num in {14,15}:
            r=load(f/'测试/实测结果.json');actual=[x for x in r.get('results',[]) if x['optimized']]
            if actual:status=f'独立图文问答实际生成{len(actual)}题，完整耗时{min(x["result"]["elapsed_ms"] for x in actual):.2f}-{max(x["result"]["elapsed_ms"] for x in actual):.2f}ms；逐题人工复核见测试记录，不能用原始关键词覆盖代替准确率。'
        elif num==16:
            r=load(f/'测试/真实微调结果.json');status=f"实际完成{r.get('epochs')}轮LoRA；整体选择题准确率{r['before']['accuracy']:.1%}->{r['after']['accuracy']:.1%}；术语类选择题{r['before']['terminology_accuracy']:.1%}->{r['after']['terminology_accuracy']:.1%}；图纸题{r['after']['drawing_reasoning_accuracy']:.1%}。损失与权重见真实微调结果和训练日志。"
        elif num==17:
            r=load(f/'测试/JMeter汇总.json');soak=load(f/'测试/12小时资源观测-优化后.json')
            status='独立真实检索API的JMeter结果：'
            for key,scenario in r.get('scenarios',{}).items():
                retrieval=next((v for k,v in scenario.get('samples',{}).items() if '检索' in k),{})
                status+=f"{key}：{scenario['duration_s']:.1f}秒，{scenario['requests']}请求，错误{scenario['errors']}，检索P95={retrieval.get('p95_ms')}ms；"
            status+='优化后12小时观察状态：'+str(soak.get('status','not_started'))+'；该观察为独立API，正式RAGFlow栈另列。'
        else:
            r=load(f/'测试/全量质检/quality_report.json');status=f"全库{r['total_files']}份：Scan_PDF {r['label_counts']['Scan_PDF']}，Text_PDF {r['label_counts']['Text_PDF']}，Mixed_PDF {r['label_counts']['Mixed_PDF']}；MD5重复组{len(r['exact_duplicates'])}、SimHash待确认{len(r['version_conflicts'])}、内嵌文本敏感待审{len(r['sensitive_review'])}。首次全量耗时约64秒，缓存复测约4秒。"
        stages={
        '设计':f"# 设计：{config['title']}\n\n## 需求与范围\n{s['problem']}\n\n## 资料与隔离\n{COMMON}\n\n## 系统架构\n{s['arch']}\n\n## 验收口径\n{s['limits']}\n正式RAGFlow栈、源码补丁、独立扩展服务分别记录，不因有演示页面就认定正式栈验收通过。",
        '研发':f"# 研发：{config['title']}\n\n## 实现模块\n{s['code']}\n\n## 核心修复或技术要点\n{s['special']}\n\n## 配置与输入输出\nproject.json保存工单编号、源目录和端口，配置和产物只在本项目。所有原始QA标准答案仅用于评估/训练，在线Pipeline不导入gold。图像页保留PDF物理页码，说明书印刷页与物理页区别记录。\n\n## 运行环境\n本机Python3.12，D:/an/envs/rags_/python.exe；Ollama本机Qwen3.5:4B；可选RapidOCR依赖在当前vendor目录。工单16另有独立视觉基座与适配器。正式RAGFlow版本0.23.1，源码快照和变更补丁都保留。",
        '测试':f"# 测试：{config['title']}\n\n## 实际结果\n{status}\n\n## 复现与证据\n指定问题和参考答案在测试/gold.json；实际结果和原始日志在测试。自动回归测试使用python -m pytest 测试 -q。OCR页面、模型输出、训练损失或资源采样按来源保存，不用展示文档替代实测。\n\n## 结果解释\n{s['limits']}\n评估时保留错误案例；结果是当前环境、指定样本的观测，不代表全量生产系统。",
        '优化':f"# 优化：{config['title']}\n\n## 问题与修复\n{s['special']}\n\n## 观测结果\n{status}\n\n## 后续调整\n先复核来源、图号、物理页码和模型上下文，再调整阈值。微调须按专利分组防止训练/测试泄漏；性能须区分检索、生成、缓存和队列等待。\n\n## 仍有的限制\n{s['limits']}\n正式服务依赖下载、完整栈运行和长时结果的状态见部署/正式环境状态.json及本工单测试目录。",
        '部署':f"# 部署：{config['title']}\n\n## 简单本机启动\n双击本项目的一键启动.cmd，浏览器访问http://127.0.0.1:{8800+num} 。终端保持打开，Ctrl+C停止；工单16页面展示真实训练结果，重新训练运行部署/启动训练.ps1。\n\n## 正式RAGFlow\n完整RAGFlow启动.sh在独立Compose项目rag-wo{num}启动API、Executor、Infinity、MySQL、MinIO、Redis。端口配置已与既有服务错开，各项目采用独立卷。源码补丁必须挂载或构建入正式镜像；只运行原发行镜像不自动包含本地修改。\n原MinIO镜像仓库返回401，已改为本机已有兼容MinIO镜像但独立数据卷；RAGFlow镜像压缩约3.13GB，国内备用镜像曾TLS超时，正式栈状态按实际日志记录。\n\n## 运维与复测\n备份本项目研发/data、训练适配器及正式Compose卷。重复入库前核对来源哈希，避免把两个版本静默覆盖；不要把同一项目数据挂到别的工单。访问/health或/api/health观察服务；工单17另有/metrics。\n\n## 交付边界\n{COMMON}\n{s['limits']}"}
        for stage,text in stages.items():write(f/stage/(stage+'方案.md'),text)
        write(f/'研发/技术文档.md',stages['研发']);write(f/'研发/用户手册.md',stages['部署'])
        write(f/'README.md',f"# 工单{num}：{config['title']}\n\n{status}\n\n{COMMON}\n\n双击一键启动.cmd，访问http://127.0.0.1:{8800+num} 。五阶段完成方案.pdf为汇总，源码、数据、测试、视频均在当前项目。\n\n验收限制：{s['limits']}\n正式栈启动和12小时状态须查看实际记录，不能将独立服务结果视为全部验收通过。")
        story=[Paragraph(f'工单{num} · {config["title"]}',styles['h1']),Paragraph('设计 / 研发 / 测试 / 优化 / 部署',styles['h2']),Paragraph('2026-10-15 | 独立项目交付与实际验收记录',styles['body']),Paragraph(escape(status),styles['body'])]
        for stage,text in stages.items():
            story.append(PageBreak())
            for line in text.splitlines():
                if not line.strip():continue
                style='h1' if line.startswith('# ') else 'h2' if line.startswith('## ') else 'body'
                story.append(Paragraph(escape(re.sub(r'^#{1,2} ','',line)),styles[style]))
        if num in {14,15}:
            for row in actual:
                if row is actual[0]:story.append(PageBreak());story.append(Paragraph('指定问题：优化后真实输出',styles['h1']))
                story.append(Paragraph(escape(row['question']),styles['h2']));story.append(Paragraph(escape(row['result']['answer']),styles['body']))
                story.append(Paragraph('耗时 '+str(row['result']['elapsed_ms'])+'ms；证据约束：'+str(bool(row['result'].get('evidence_guard'))),styles['body']))
        def decorate(c,d):
            w,h=A4;c.setFillColor(colors.HexColor('#193f45'));c.rect(0,h-9*mm,w,9*mm,fill=1,stroke=0);c.setFont('CN',8);c.setFillColor(colors.white);c.drawString(18*mm,h-6*mm,f'工单{num} | 设计 · 研发 · 测试 · 优化 · 部署');c.setFillColor(colors.grey);c.drawRightString(w-18*mm,10*mm,str(d.page))
        SimpleDocTemplate(str(f/'五阶段完成方案.pdf'),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=22*mm,bottomMargin=20*mm,title=config['title']).build(story,onFirstPage=decorate,onLaterPages=decorate)
        summaries.append({'project':num,'status':status,'limits':s['limits']});print('final documents',num,flush=True)
    write(ROOT/'工单18/测试/本批工单验收汇总.md','# 工单14-18验收汇总\n\n'+'\n\n'.join(f"## 工单{x['project']}\n{x['status']}\n限制：{x['limits']}" for x in summaries))
if __name__=='__main__':main()
