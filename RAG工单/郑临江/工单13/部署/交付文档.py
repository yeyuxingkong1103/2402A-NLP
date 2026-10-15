"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。逐项目生成五阶段方案和PDF。"""
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle
ROOT=Path(__file__).resolve().parents[2]
SPECIAL={
2:{'problem':'原工单1已有PDF问答，需优化解析、分块、检索质量，答案准确率目标90%以上，完整响应不超过3秒。',
   'implementation':'保留原问答引擎的本项目独立快照，对页眉清理、重叠分块、发行人实体过滤、直接披露数值的上下文优先和混合检索进行优化。对baseline/dense/fulltext/hybrid作同样本实测；baseline是当前实现的稀疏策略对照，不冒称原工单1所有历史版本。',
   'tests':'10题兴图新科指定问题；保留逐题答案、来源、完整耗时，检索Hit@5/MRR与回答事实覆盖分别统计。工单1已有结果的独立基线见本项目测试目录。',
   'optimization':'语义模型和字符词法模型互补；已披露的军用收入句优先，防止把利润敏感性表当作收入。说明参数变化和对照条件，检索指标不能充当90%问答准确率。'},
3:{'problem':'新增力源信息招股文件的表格解析，四道新题和十道回归题正确；重点校验发行股数、募投项目及关联方。',
   'implementation':'PyMuPDF find_tables提取真实行列；逐行序列化、保留文档及物理页码；在公司实体范围内索引。存在控制关系与不存在控制关系用表头及列值定位表格，不将已转让历史关联企业混入现有名单。',
   'tests':'14道指定问题。发行1,670万股及25.04%来自物理页22/24，募投五项目来自页22；赵马克42.35%与控股股东、七家非控制关联方来自页157。JSON中保留真实表格。',
   'optimization':'数字、单位和列头同块；表格和文本分别保留以交叉核验；发行人实体过滤减少另一家公司同名字段干扰。复杂跨页表格暂不声称自动无损重建。'},
4:{'problem':'解析组织结构图与2008年IC增长图，并能回答图像题及文本/表格回归题。',
   'implementation':'实际渲染力源物理页39和72，调用本地Qwen3.5:4B多模态识别；原始输出保留，人工对原图逐项校正后构成可检索图像摘要。不能仅凭正文推断图中的负增长项。',
   'tests':'销售部4个子部门、大客户销售部6个销售处；汽车电子14.0%最高、IC卡-2.0%负增长。测试/实测结果包含原文来源，研发/data/images含实际图像。',
   'optimization':'图像文字摘要与正文、表格联合检索。金额与增长率分开，图例名称、连线方向及上下级层次人工核对；摘要校正不是实时模型完全自动无误的声明。'},
5:{'problem':'实现指定五轮对话，正确处理“他”“这个公司”以及从兴图切换到力源的省略问句。',
   'implementation':'每个浏览器session_id独立存储最近12轮；规则规范代词并继承上一轮查询属性。实体明确切换后重写查询，所有轮次保留原问题与rewritten_question。',
   'tests':'依次查询兴图军用收入、获奖工程、法定代表人，问“那武汉力源信息技术股份有限公司呢”，最后查最多销售处。应分别给出程家明和赵马克，并返回大客户销售部6处。',
   'optimization':'查询记忆不直接作为事实证据，重写后重新检索正确公司。用户会话互相隔离，清空浏览器sessionStorage开启新会话。复杂指代和跨会话永久记忆不属于本次初版。'},
6:{'problem':'提供向量召回重排、倒排全文检索及混合策略；支持三种重排、布尔/短语/模糊/多字段、权重和融合配置。',
   'implementation':'BGE语义余弦召回；字符词项倒排索引与BM25全文评分；RRF按weight融合排名。三种重排为TF-IDF分数重排、真实本地LLM序号重排、反馈增益重排；支持body/title/summary字段及AND/OR/NOT、双引号短语、近似词项匹配。',
   'tests':'三策略逐题对照；布尔和短语匹配回归；多轮与反馈入口验证。Hit@5是标注页命中率；工单要求的95%重要事实召回还需用完整相关性标注衡量，不能直接与Hit@5等同。',
   'optimization':'字段过滤后排序，保留精确数字与实体；融合权重0到1可调。本地BGE-small与Ollama bge-m3嵌入均有真实验证，BGE-m3验证为1024维单位范数。EMBEDDING_BACKEND=ollama可选择服务嵌入，更换模型必须重建索引，完整检索对照仍使用BGE-small。'},
7:{'problem':'使用ccf压缩包内年报和sample_questions模板整理10题、真实检索及评估，分析检索问题。',
   'implementation':'九份2019-2021年金融年报全页文本入库，表格抽取限定前30页财务摘要。原压缩包中文文件名按GBK恢复，各来源SHA-256独立记录；输入明确公司后执行文档范围过滤。',
   'tests':'10题涵盖盈利驱动、创新商业模式、财务数字、风险资本、科技提高融资效率及跨银行比较。占位的xx银行具名化为平安/招商等。参考答案必须依真实年报复核，不把模板中的未来判断当作披露事实。',
   'optimization':'董事长致辞标题和主题导引改善长年报召回；去掉整段上下文的泛化推断，区分百万元与亿元。无限制图形页find_tables曾阻塞，改有界抽取避免无效CPU工作，全部正文仍保留。'},
8:{'problem':'基于金融年报实现GraphRAG、实体关系图、问答和研报生成，与工单7对比。',
   'implementation':'完整语料片段进行类型化规则实体抽取：机构、人名、财务指标、风险指标、业务、主题等；关系DISCLOSES_TOPIC及明确法定代表人/董事长。每条关系带文档、页码、chunk_id和抽取来源。实体锚定+图邻接扩展+语义检索用于GraphRAG。',
   'tests':'附件未提供原eval_question.md，本项目在测试目录按年报及示例问题补建并注明。界面/API显示节点与边，10题GraphRAG与常规RAG真实检索对照；“共现主题”不代表因果或持股。',
   'optimization':'正文证据优先于图谱泛化描述；图谱只导引到来源片段，生成必须引用原文。规则型图谱是可追溯初版，不能冒称已使用Microsoft GraphRAG包或覆盖所有隐含关系。'},
9:{'problem':'优化图谱实体关系、检索与生成，并用ContextPrecision/Recall验收，目标0.8/0.9及3秒。',
   'implementation':'在独立GraphRAG快照上做机构范围过滤、主题锚定和声明完整句选择；RAGAS采用本地Qwen3.5:4B评审，保存原始框架指标、异常与模型版本。若阈值未通过明示差距，不补写合格分数。',
   'tests':'同一10题的hybrid、graph和graph_optimized对照；RAGAS ContextPrecision/Recall为真实模型评审，与关键词代理分开。sample中的xx银行按具名年报复核。',
   'optimization':'改进实体别名、关系来源保真、检索去重和上下文预算；图谱全局摘要不能吞掉具体证据。当前图谱技术为自实现GraphRAG，生产完整实体关系识别仍需扩展。'},
10:{'problem':'将金融问答服务以Docker容器在服务器环境成功运行，验证端口、持久化卷和网络。',
    'implementation':'实际使用WSL Ubuntu的Docker29，构建python:3.11-slim镜像rag-workorder10:local；数据卷只挂载本工单研发/data。初次bridge无法访问Windows仅本机绑定Ollama，改为WSL host网络访问本地模型。',
    'tests':'记录真实镜像、容器状态、HTTP健康/问答、反馈卷持久化和模型网络连通。端口8774。host网络不使用-p映射；停止并重启容器后反馈标记应保留。',
    'optimization':'轻量镜像不装CUDA/torch，容器使用BM25+TF-IDF并调用本机LLM；Windows版仍可BGE。容器内向量后端与桌面版指标分别说明，不把Windows测试当容器验收。'},
11:{'problem':'构建专业检索问答数据集、真实微调Embedding，记录训练过程并给出前后测试指标。',
    'implementation':'本地Qwen从真实招股片段生成36条问答，按文档物理页划分训练23/验证5/测试8。BGE-small中文模型在RTX4060上以in-batch InfoNCE训练5轮，lr1e-5、batch4、maxseq256，仅按验证MRR选checkpoint。',
    'tests':'在445个真实候选片段上评估未见8条测试问题，保存before/after/ranks。实际更新后的模型在研发/models/bge-finance-finetuned，可核对模型文件与训练日志；合成答案格式检查不替代人工语义标注。',
    'optimization':'微调前后一致候选池和相同评价算法；训练/验证/测试按来源页无交叉，测试不用于挑参数。中文专业数据选择中文BGE，工单提到英文模型只是示例。结论限小样本，保留单题退步。'},
12:{'problem':'实际使用官方HKUDS LightRAG构建两个招股文件图谱，切换RAG/LightRAG并比较RAGAS指标。',
    'implementation':'官方lightrag-hku1.5.7及依赖安装在本项目vendor内，JsonKV+NanoVectorDB+NetworkX存储。通过ainsert_custom_kg导入带来源的规则抽取实体关系及全部片段，aquery_data mix执行官方局部/全局+向量查询。',
    'tests':'全部工单图像/表格/文本指定题及额外多轮事实共18题；真实LightRAG输出、映射来源和生成回答独立保存。RAGAS分别评估hybrid/light，LLM评审关闭思考并传JSON schema，错误不填0或1冒充分数。',
    'optimization':'初始3500token预算被图谱耗尽、文本块为空；复测改实体1000/关系1200/总10000，减少top_k为4。不是把手写图算法改名LightRAG，也不声称官方自动抽取所有实体；图谱schema范围如实记录。'},
13:{'problem':'定位Query理解、检索、上下文、LLM生成和后处理瓶颈，实际比较优化前后并验证3秒。',
    'implementation':'cProfile及逐请求结构化日志；性能专项逐阶段计时，记录候选数、上下文长度、完整答案和是否缓存。以相同10题对比候选10块与去重3块；区分检索延迟、完整生成延迟、模型驻留和并发排队。',
    'tests':'性能专项结果.json包含五阶段计时及原始答案；profile.prof/profile.txt保留函数耗时；4线程40请求仅做检索引擎负载，不冒充完整RAG生产并发。',
    'optimization':'模型驻留、小嵌入、上下文去重/预算、短答及最多128条缓存；优先解决耗时最大的真实阶段。缓存不能代替无缓存验收，若上下文压缩损失事实需先恢复完整证据。'},
}

def write(path,text):path.write_text(text.strip()+'\n',encoding='utf-8')
def main():
    pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CNB','C:/Windows/Fonts/simhei.ttf'))
    styles={
        'h1':ParagraphStyle('h1',fontName='CNB',fontSize=20,leading=28,textColor=colors.HexColor('#153f43'),spaceAfter=15,keepWithNext=True),
        'h2':ParagraphStyle('h2',fontName='CNB',fontSize=12,leading=19,textColor=colors.HexColor('#186d66'),spaceBefore=12,spaceAfter=6,keepWithNext=True),
        'body':ParagraphStyle('body',fontName='CN',fontSize=10,leading=16,spaceAfter=7,wordWrap='CJK')}
    for num in range(2,14):
        folder=ROOT/f'工单{num}';config=json.loads((folder/'研发/project.json').read_text(encoding='utf-8'))
        manifest=json.loads((folder/'研发/data/index/manifest.json').read_text(encoding='utf-8'))
        report=json.loads((folder/'测试/实测结果.json').read_text(encoding='utf-8')) if (folder/'测试/实测结果.json').exists() else {}
        special=SPECIAL[num];port=config['port']
        answers=report.get('answers',[])
        summary='\n'.join(f"策略{k}：Hit@5={v['hit_at_5']:.1%}，MRR@5={v['mrr_at_5']:.3f}，事实关键词覆盖代理={v['fact_coverage_proxy']:.1%}。" for k,v in report.get('strategies',{}).items())
        if answers:
            times=[x['result']['elapsed_ms'] for x in answers]
            summary+=f"\n真实生成{len(answers)}题，完整稳态耗时{min(times):.2f}-{max(times):.2f}ms；当前样本3秒目标{'通过' if max(times)<=3000 else '未通过'}。不包含模型冷启动。"
        if num==12:
            light=json.loads((folder/'测试/LightRAG生成对比.json').read_text(encoding='utf-8'))
            light_times=[r['result']['elapsed_ms'] for r in light]
            summary+=f"\n官方LightRAG另行生成{len(light)}题，完整耗时{min(light_times):.2f}-{max(light_times):.2f}ms；其3秒目标{'本次通过' if max(light_times)<=3000 else '未全部通过'}。两后端不可混用计时结论。"
        common='所有源PDF、嵌入模型、索引、图谱、图像、测试集与日志均位于本工单目录。本项目运行时不访问其他工单文件。LLM调用本机已有Ollama服务，其环境与模型名称单独记录；主目录无公共数据。'
        stages={
        '设计':f"# 设计：{config['title']}\n\n工单编号：{config['ticket']}\n\n## 工单需求\n{special['problem']}\n\n## 数据与关联\n{common}\n本项目使用{len(manifest['documents'])}个PDF、{sum(x['pages'] for x in manifest['documents'])}个物理页。原工单PDF、提取原文与附件哈希在设计目录。项目间复用的是离线复制的代码/数据快照，无运行时共享。\n\n## 架构\n输入问题 -> 会话/术语规范 -> 本项目检索或图谱扩展 -> 证据上下文 -> 本地模型 -> 页码来源/反馈。离线为PDF解析 -> 文本/表格/必要图像摘要 -> 分块 -> BGE/稀疏索引 -> 持久化。\n\n## 技术选择及边界\n{special['implementation']}\n原PDF只是业务/任务参考资料，不能覆盖用户对目录、环境和部署简化的要求。涉及外部发布或与其他人通信的内容未执行。",
        '研发':f"# 研发：{config['title']}\n\n## 实现与源文件\n{special['implementation']}\n研发/app/base.py保存可追溯PDF解析及LLM核心；core.py实现当前项目扩展；server.py/main.py提供本地HTTP和命令行，static/index.html提供网页。build、serve、ask各入口都有本工单注释。\n\n## 数据结构\n每个片段含id、document、pdf_page、kind、text、sha256；pages.json保留完整原页文字和已识别表格；manifest记录解析范围。图谱节点含name/type，关系含source/target/type及chunk_id、文档页码。反馈为JSONL。\n\n## 实际接口\nGET /api/health、/api/documents、/api/graph；POST /api/ask、/api/settings、/api/upload、/api/feedback；/pdf/文档名查看原文。Ask参数question、mode、strategy、session_id。字段和融合配置可通过settings设置。\n\n## 容错\n空问题拒绝、模糊口径澄清、无证据拒答、同名上传保护、非法PDF拒绝；LLM超时回退证据预览并明确标识。扫描空文本页和未尝试表格解析不冒充全部解析成功。",
        '测试':f"# 测试：{config['title']}\n\n## 验收用例\n{special['tests']}\n\n## 评价口径\nHit@5/MRR是证据页检索指标，事实覆盖是关键词代理；三者均不等于问答准确率。标准答案只在测试脚本中使用，生产问答不读gold.json。完整响应包括生成，无缓存统计另列。\n\n## 本次实测\n{summary or '当前无完成的实测记录，不能认定通过。'}\n自动化日志在测试/自动化测试日志.txt；逐题回答与来源在测试/实测结果.json。\n\n## 复现\n在本目录用指定Python执行“测试/回归测试.py”和“测试/评估.py”。专项微调/性能/LightRAG另有对应脚本与原始日志。\n\n## 结果限制\n单机短时测试不能替代24小时稳定性、生产高并发或真人语音识别。模型评审指标有偏差，RAGAS结果按实际文件读取，失败项单独保留。",
        '优化':f"# 优化：{config['title']}\n\n## 本项目优化\n{special['optimization']}\n\n## 可复现对照\n{summary or '专项对照见测试目录真实记录。'}\n比较保持同问题和同源文档。图谱或检索对照的算法、权重、候选数量及语料范围均需记录，不能用一个代理指标夸大质量。\n\n## 未达标处理\n优先复核单位、页码、实体范围和上下文截断；优化完整证据后复测。未达到阈值或依赖真人/长期负载的要求单独标注，不能因为交付了代码而写成验收合格。",
        '部署':f"# 部署：{config['title']}\n\n## 本机三步\n1. 双击本目录“一键启动.cmd”。\n2. 等待终端显示就绪。\n3. 浏览器访问 http://127.0.0.1:{port} 。\n保持终端打开，Ctrl+C停止。数据与索引已经准备，无需再上传。不同项目使用独立端口，可分别启动。\n\n## 环境\n本次实测Windows、Python3.12，D:/an/envs/rags_/python.exe；各项目自带本地BGE模型目录。所需包在研发/requirements-model.txt；环境版本在部署/环境记录.json。可用- Python参数指定另一环境。Ollama本机服务模型Qwen3.5:4B，地址127.0.0.1:11434；密钥不写文件。\n\n## 简单维护\n备份当前项目研发/data，上传新PDF后后台建库；更新嵌入模型需重建向量。原业务资料不变时无需反复建库。演示视频在部署目录，录制实际网页结果，不伪造语音或模型输出。\n\n## 特殊要求\n{('工单10使用Docker实际部署，详见容器验收记录与容器启动.sh。WSL host网络端口8774，数据卷挂载本工单研发/data。' if num==10 else '本项目无需Docker或独立数据库，优先使用一键本地启动。')}"}
        extras=[]
        for stage,text in stages.items():write(folder/stage/(stage+'方案.md'),text)
        write(folder/'研发/技术文档.md',stages['研发'])
        manual=f"# 用户手册\n\n双击一键启动.cmd，访问 http://127.0.0.1:{port} 。输入完整问题后查询，选择语义/全文/混合或图谱策略。答案注明生成状态和耗时；点击来源跳转原PDF物理页面。反馈按钮写入本项目日志。\n\n上传限制40MB，文本PDF1,200页以内；同名文件不覆盖。语音按钮依赖浏览器及麦克风许可，不支持时可文字输入。新浏览器会话隔离历史，复杂模糊问题要明确公司、年份和单位。\n\n数据仅为所附年份的文件披露，不能作当前经营结论。纯LLM模式不提供来源上下文，可能给错或拒答，属于实际对照。\n\n遇到连接失败检查终端和端口；证据预览说明模型不可用；图谱显示前28实体，完整JSON和来源在本项目索引目录。\n\n{common}"
        write(folder/'研发/用户手册.md',manual)
        for name in ['微调实测结果.json','性能汇总.json','RAGAS指标汇总.json','第二嵌入模型验证.json','容器验收记录.json']:
            path=folder/'测试'/name
            if path.exists():extras.append(f'专项结果 {name}\n'+json.dumps(json.loads(path.read_text(encoding='utf-8')),ensure_ascii=False,indent=2)[:4500])
        for name in ['RAGAS指标汇总.json','微调实测结果.json','性能汇总.json','第二嵌入模型验证.json']:
            path=folder/'测试'/name
            if path.exists():
                special_data=json.loads(path.read_text(encoding='utf-8'))
                if name=='微调实测结果.json':special_data={k:v for k,v in special_data.items() if k in {'before','after','splits','improved','selection','limits'}}
                stages['测试']+='\n\n## '+name+'\n'+json.dumps(special_data,ensure_ascii=False,indent=2)
        for stage,text in stages.items():write(folder/stage/(stage+'方案.md'),text)
        write(folder/'README.md',f"# 工单{num}：{config['title']}\n\n按设计、研发、测试、优化、部署五部分保存。\n\n{summary}\n\n一键启动：双击一键启动.cmd，打开http://127.0.0.1:{port} 。\n\n{common}\n\n各项是否满足验收阈值以测试/实测结果与专项记录为准。未通过不冒称通过。\n\n"+'\n\n'.join(extras))
        # PDF概要方便阅读，原始JSON保持完整且单独交付。
        story=[]
        def para(text,style='body'):story.append(Paragraph(escape(text).replace('`',''),styles[style]))
        para(f"工单{num} · {config['title']}",'h1');para('设计 / 研发 / 测试 / 优化 / 部署','h2');para('交付日期：2026-10-12')
        para('完整代码、源资料、实测记录和演示视频分别保存在当前项目目录。'+common)
        if summary:para(summary)
        for stage,content in stages.items():
            story.append(PageBreak())
            for line in content.splitlines():
                if not line.strip():continue
                para(line[2:] if line.startswith('# ') else line[3:] if line.startswith('## ') else line,
                     'h1' if line.startswith('# ') else 'h2' if line.startswith('## ') else 'body')
        if answers:
            story.append(PageBreak());para('指定问题实测摘录','h1')
            for row in answers:
                para('ID '+str(row['id'])+'：'+row['question'],'h2');para(row['result']['answer']);para('完整耗时 '+str(row['result']['elapsed_ms'])+'ms。')
        def page(canvas,doc):
            w,h=A4;canvas.setFillColor(colors.HexColor('#153f43'));canvas.rect(0,h-9*mm,w,9*mm,fill=1,stroke=0)
            canvas.setFillColor(colors.white);canvas.setFont('CN',8);canvas.drawString(18*mm,h-6*mm,f'工单{num} | 独立项目交付')
            canvas.setFillColor(colors.HexColor('#647986'));canvas.drawString(18*mm,10*mm,'设计 · 研发 · 测试 · 优化 · 部署');canvas.drawRightString(w-18*mm,10*mm,str(doc.page))
        SimpleDocTemplate(str(folder/'五阶段完成方案.pdf'),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=22*mm,bottomMargin=20*mm,
            title=f'工单{num} - {config["title"]}',author='项目交付').build(story,onFirstPage=page,onLaterPages=page)
        print('documents',num,flush=True)
if __name__=='__main__':main()
