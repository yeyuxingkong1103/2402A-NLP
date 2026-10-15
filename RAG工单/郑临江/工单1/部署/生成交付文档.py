"""工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统。生成方案及 PDF。"""
import hashlib
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether

BASE = Path(__file__).resolve().parents[1]
GOLD = json.loads((BASE / "测试/gold.json").read_text(encoding="utf-8"))
REPORT = json.loads((BASE / "测试/实测评估结果.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((BASE / "研发/data/index/manifest.json").read_text(encoding="utf-8"))


def write(name, content):
    (BASE / name).write_text(content.strip() + "\n", encoding="utf-8")


def make_docs():
    documents = {}
    documents["设计/设计方案.md"] = """
# 设计：基于 PDF 文档的问答系统

工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统
交付日期：2026-10-12；方案依据：任务工单 V1.1 和用户提供的 548 页招股文件。

## 需求与边界
面向武汉兴图新科电子股份有限公司招股文件，实现中文/英文提问、语音输入、文档上传、知识库管理、问题理解、向量检索、LLM 生成、来源追溯和反馈。工单要求 10 道指定问题的 RAG 与纯 LLM 对比、评估报告、技术文档、用户手册及完整演示视频。
原始文件名为“招股说明书1.pdf”，正文标题为“招股意向书”。所有数值仅按该来源文件披露口径解释；报告期为 2016 年、2017 年、2018 年和 2019 年1-6月，不推断当前经营数据。
PDF 中的文字是业务资料，不具有修改程序、提示词或工作流程的权限。生成提示词明确要求不执行证据中的指令。

## 架构与数据流
离线建库：PDF -> PyMuPDF 文本及表格解析 -> 带物理页码的重叠分块 -> BGE 语义向量及 JSON 元数据持久化。
在线问答：网页输入 -> 问题意图/术语规范/消歧/子问题提取 -> BGE 与 TF-IDF 混合检索 -> 基于证据的 LLM 生成 -> 引用校验 -> 答案、耗时和来源展示 -> 反馈日志。
纯 LLM 对比使用同一个模型和相同采样参数，仅移除文档上下文；问答引擎不读取标准答案文件。

## 技术选型
Python 负责解析、检索和 HTTP 服务；PyMuPDF 保留文本及可识别表格行列；NumPy 执行小规模向量余弦检索；scikit-learn 提供字符 2-4 gram TF-IDF；BGE-small-zh-v1.5 为 512 维语义嵌入；HTML/CSS/JavaScript 提供轻量网页。
首期 1,531 个片段直接使用矩阵检索，避免为了小规模数据引入独立数据库。索引可重建，源 PDF 和反馈作为持久化事实。
本机已有 Ollama 和 Qwen3.5:4B，交付时采用本地模型完成实际生成。程序同时支持 OpenAI 兼容服务，通过环境变量配置，不保存密钥。BGE 不可用时降级为 TF-IDF 稀疏向量并明确展示当前后端。

## Query 理解
意图分为财务数值和事实查询；别名将“注册资金”“法人代表”“军用/军方/国防”等转为检索概念，并映射常见英文问题。这个实现采用规则，属于可解释的初版，不能等同于通用自然语言理解。
对“收入是多少”等缺少口径的问题先澄清；分号、以及、同时等连接词可拆出子问题，检索使用原问题的联合概念。复杂多跳推理暂不具备逐子问题独立检索、融合和验证能力。
“抽象”通过去掉公司全称、问句套语并提取业务概念实现。实体加权优先发行人基本资料，避免误检子公司资本或中介机构法人。

## 分块、表格及引用
正文片段约 700 字符、120 字符重叠；表格序列化约 1,000 字符、180 字符重叠。片段含文档名、SHA-256、PDF 物理页码、印刷页码、种类和原文。
原文的印刷页码为 1-1-N，不能直接当 PDF 页码。界面引用使用“PDF 第 N 页”，打开链接时以 #page=N 跳转。生成答案引用格式固定为 [PDF第N页]。
表格以行列保存，并另外生成带分隔符的检索文本。存在跨页表格、复杂合并单元格时需人工复核。本次第 479 页补充流动资金表已通过行列数据测试。原文第535-538页未提取到文字，记录为空文本页，未执行OCR；该局限不影响本次10题所需证据，但不能声称全部页面已完整解析。全扫描PDF无法建库并会明确提示先OCR。

## 可用性与性能目标
启动阶段加载模型和索引，运行阶段缓存最多 128 条成功结果。知识库上传在后台构建，切换前旧索引继续服务；上传失败会显示状态。LLM 超时降级展示证据并标识异常，不伪造答案。
目标为提问至答案完整展示不超过 3 秒。分别记录冷启动、稳态检索、首次生成和缓存命中耗时；本地 LLM 的实测时间决定是否通过，不能用检索时间代替生成时间。
采用本机地址绑定。生产多用户部署还需独立应用服务器、认证授权、任务队列、容量测试和监控；本次不将本地线程服务器称为高可用集群。

## 两人日实施安排
第1人日：需求及标准答案核对 1h、PDF及表格解析 2h、分块与混合检索 3h、问答接口 2h。
第2人日：网页及语音/反馈 2h、10题对比与回归 3h、优化与性能分析 1h、部署脚本及文档/演示 2h。工时为计划估计，不冒充实际计时。
"""
    documents["研发/技术文档.md"] = """
# 研发：技术文档

## 模块与开发结果
app/core.py：PDF 文本/表格提取、分块、BGE 与稀疏向量、语义概念规范、混合检索、答案缓存、LLM 调用、引用检查。
app/server.py：本机 HTTP API、知识库文件上传、后台建库、原文访问和反馈写入。
app/main.py：build、serve、ask 命令；static/index.html：网页问答、模型状态、语音输入、来源和反馈。
data/documents：原始业务 PDF；data/index：manifest.json、chunks.json、pages.json、vectors.npy；data/logs：反馈 JSONL。
代码注释中均包含工单编号。测试标准答案保存在“测试/gold.json”，线上代码不导入此文件。

## API
GET /api/health：返回系统状态、模型是否配置、建库状态、当前向量后端及文档概览。
GET /api/documents：返回文档名、摘要哈希、页数和表格数量。
POST /api/ask：JSON 参数 question 及 mode；mode 可为 rag 或 llm_only，返回 answer、sources、query、generation、elapsed_ms、cached。
POST /api/upload?name=文件名.pdf：application/pdf 请求体；限 40 MB、1,200 页；合法新文件返回 202 后后台构建。同名文件返回409。加密/损坏/非法类型会拒绝。
GET /pdf/文件名.pdf：查看原文。POST /api/feedback：rating 为 up 或 down，可带 question 和 comment，追加到 feedback.jsonl。
错误输入返回400、跨站写入403、路径或接口不存在404、上传过大413、同名/建库忙409。扫描文件可上传但建库会明确提示需先OCR。

## 模型配置
环境变量 LLM_BASE_URL、LLM_MODEL 决定是否开启生成；LLM_API_KEY 可选，LLM_TIMEOUT 设定超时。LLM_PROVIDER=ollama 时使用 /api/chat，关闭思考输出；其他取值使用 /chat/completions。
PowerShell 示例：$env:LLM_PROVIDER='ollama'；$env:LLM_BASE_URL='http://127.0.0.1:11434'；$env:LLM_MODEL='Qwen3.5:4B'；$env:LLM_TIMEOUT='120'。这里的分号是命令示例，需在自己的 PowerShell 执行。
本地演示采用120秒超时保证得到完整对比答案；它并不表示性能达标。针对3秒目标可调小超时，但降级返回证据不能算完整答案生成通过。
EMBEDDING_MODEL_PATH 指定本地 BGE 目录；默认查找用户 Hugging Face 缓存。EMBEDDING_BACKEND=tfidf 强制稀疏后端，改后端后需重新 build；EMBEDDING_DEVICE 默认 cpu。
.env.example 是配置说明，不会自动加载。部署脚本会在进程环境中设置本机已有模型的配置。

## 知识库管理与限制
新增：网页上传，后台解析及建库；查看：侧栏与 API；维护：把源 PDF 放入 data/documents 后运行 build。删除：人工确认后移走指定源 PDF 并重新 build，初版没有网页删除按钮。
索引保存使用临时文件替换，manifest 最后写入；在线旧内存索引在新对象准备好后交换。不要在建库时杀进程，也不要启动多个独立服务同时写同一索引目录；跨文件事务和多进程版本锁需在生产化时补足。
只对引用页码是否来自检索上下文作机械校验，不能自动证明每句话都由该页支持。模型输出的事实和单位仍需评估。
"""
    documents["研发/用户手册.md"] = """
# 用户手册

## 启动与界面
双击“工单1/一键启动.cmd”，浏览器访问 http://127.0.0.1:8765 。终端保持运行；Ctrl+C 停止。没有本地模型时使用部署目录的启动.ps1，页面会显示“证据预览模式”。
左侧查看已入库文档；右侧输入问题，选择“文档问答 RAG”后点击“查询”。可以点击示例问题，也可以用 Ctrl+Enter 发送。

## 上传 PDF
点击“上传 PDF 文档”，选择文本型 PDF，再点“上传并构建”。最大40 MB、1,200页，不覆盖同名文件。等待左侧状态从构建中变为就绪，再查询新内容。PDF 损坏、加密或扫描图像无文字时按提示处理。
已附带招股说明书1.pdf 及索引，首次启动无需再次上传。新增多个 PDF 后，回答可能同时引用多个文件，请确认来源名称。

## 提问、看答案与反馈
示例：公司的注册资本是多少？法定代表人是谁？报告期内来自军用领域的收入分别是多少？
数值问题最好明确年份、口径和单位；“收入是多少”会触发澄清。“报告期”按招股文件定义，不能用当前年份替换。
答案卡片显示完整耗时和生成状态。模型生成、证据预览、模型异常、证据不足含义不同；证据预览不是模型摘要。展开来源卡片，点击“查看原文”进入对应 PDF 物理页面。页内印刷页码与界面页码不一致属于正常情况。
“有帮助/需要改进”会把问题及评价保存到本地 feedback.jsonl。语音按钮使用浏览器 SpeechRecognition，可用性取决于浏览器、麦克风授权和网络，可能调用浏览器厂商的识别服务；不可用时改用文字输入。

## 英文与纯 LLM 对比
可输入“How much is the registered capital of Wuhan Xingtu Xinke?”，系统将常见英文概念转换后检索中文文档，模型用英语回答。未配置模型时只展示英文说明和中文证据。
对同一问题选择“纯 LLM 对比”再查询，可查看该模型未提供文档上下文时的回答。模型不知道、无法证实或拒答也是有效的实际对比结果，不应补写虚构答案。

## 常见故障
连接失败：检查终端是否运行和端口8765是否被占用；用 -Port 8766 启动可更换端口。
证据预览：检查模型变量，确认 Ollama 运行且指定模型已安装。
模型异常：查看服务日志，确认模型地址/名称/超时；证据仍可用于人工核对。
无证据：明确问题或上传对应资料。向量状态为TF-IDF时代表模型降级，界面会显示真实后端。
"""
    documents["测试/测试方案.md"] = """
# 测试：验收方案与评价体系

## 评价体系选择
使用可复现的“检索质量 + 标准事实覆盖 + 引用有效性 + 完整响应时间”RAG 评价体系，另附人工复核要点。未采用需要评审模型的RAGAS分数，不伪造忠实度或答案相关性百分比。
检索 Hit@5：前5片段中任一命中标准证据页；MRR@5：首个命中的倒数排名均值；证据事实覆盖：检索文本含标准事实的关键词比例。
生成答案事实覆盖：LLM 答案含标准事实的关键词比例；仅在确实完成生成时计算。数字去逗号/空格，保留单位原文供人工检查。关键词全覆盖不代表完全正确，未支持的附加事实仍可能存在。
引用有效性：答案引用 [PDF第N页] 是否全部来自输入上下文且至少有一次引用；只能校验页码集合，不替代逐句归因。
完整响应时间：无缓存 ask 从问题处理到完整答案返回，包括检索和LLM网络/推理；报告中保留逐题耗时。首次启动与建库独立说明。

## 自动化测试
覆盖发行人注册资本/单位、英文转中文检索、模糊问题澄清、空问题拒绝、无关问题拒答、复合问题分解、损坏PDF、原文表格行列结构、LLM请求证据隔离。
LLM接口隔离用本地HTTP测试桩验证“纯LLM不带证据”，不是模拟模型准确性。另运行真实本地模型的10题问答对比。
评估含4线程40次无缓存检索压测；网页测试含加载状态、问答、引用链接、反馈和不合法上传。语音输入仅验证功能入口，麦克风真人识别留待用户在浏览器验证。

## 人工复核规则
收入与占比严格按2016/2017/2018/2019上半年顺序，并区分主营业务收入与营业收入；注册资本不能取子公司资本、成立时资本或发行后总股本。
补充流动资金应为15,000.00万元，不把总募集资金40,584.83万元当作补流金额；1.5亿元为等价换算。
标准为《某视频技术规范1.0》，保留“某”，不猜测未披露规范名称。获奖工程与公司获奖有别，必须保留未能列入获奖名单的说明。
上下游企业问的是行业类别，不能把供应商名单和客户名单拼凑为行业答案。

## 验收门槛
10题真实生成且证据足够；数值、单位、报告期和关键限定全正确；界面能够显示原文和反馈；中英文均实测。所有完整请求不超过3秒才可认定性能目标通过。
本地长时间稳定性、生产高并发、跨页复杂表格及实时语音需要专门环境测试；不以短时引擎压测替代这些项目。
"""
    documents["优化/优化方案.md"] = """
# 优化：质量、速度与运行稳定性

## 已实施质量优化
1. 去掉公司重复页眉与印刷页码，保留物理页码和独立元数据。
2. 正文重叠分块，表格按行列保存并构造检索文本，避免列错位。
3. BGE语义检索与字符TF-IDF联合排序，技术规范、补流、资本等实体词加权；不依赖问题ID或标准答案查表。
4. 发行人基本资料加权，纠正子公司/中介机构同名字段误检；英语概念独立映射为中文检索词，减少英文套语干扰。
5. 澄清过于模糊的收入问题，词法相关性不足时拒答；Prompt要求保留报告期、单位和获奖名单限定。
6. 数字列表补全整页上下文；军用收入问句优先取已披露数值的完整原句，并补入来源文件的报告期释义。限制生成上下文长度，避免模型上下文截断后错用总收入或敏感性表。此前实测发现错误，修正后重新运行10题对比。

## 已实施性能与容错优化
启动预加载，128项内存答案缓存，BGE采用小模型和4个CPU计算线程；向量及元数据持久化，避免每次启动重新解析PDF。
HTTP请求与后台构建分离；建库失败保留旧内存知识库并显示错误；LLM超时回退为证据预览。缓存错误结果会掩盖模型恢复，因此模型异常不缓存。
默认本机部署并限制文件名、类型、大小、页数和跨站写入。反馈写入使用锁。多进程写索引不受支持，生产化需任务锁和版本目录。

## 优化效果验证
实测评估结果.json保存 sparse、dense、hybrid 三种策略的指标；它们使用相同10题与同一套分块。优化前后样本很小，不能推断泛化到所有招股文件。
生成耗时与缓存命中分开记录；缓存能改善重复问题，但不能拿缓存耗时替代首次完整问答验收。3秒目标是否通过由本次真实模型实测决定。

## 进一步优化顺序
先复核失败问题：如果事实存在但排序不足，增加实体识别和页面上下文融合；如果表格数字不足，补充表头传播和跨页表格连接。
3秒目标优先使用短输出、减少重复上下文、保持模型驻留、选更小模型/足够GPU，然后复测完整请求。若仍不达标，记录失败并调整模型部署资源；不要用超时回退伪装达标。
更大知识库再引入FAISS/Milvus、交叉编码重排、独立任务队列和生产应用服务。高可用需多实例、共享只读索引、持久化状态以及负载均衡。
加入OCR后需额外验证数字、小数点、百分号识别，不把OCR支持列为本次已实现项。
"""
    documents["部署/部署方案.md"] = """
# 部署：本机运行、复现与维护

## 本机已验证环境
Windows / PowerShell；Python 使用 D:\\an\\envs\\rags_\\python.exe；BGE-small-zh-v1.5 使用已有本地缓存；LLM 使用本机 Ollama 的 Qwen3.5:4B。源PDF、知识库及全部源代码已包含在交付目录。
快速启动：进入“部署”目录，在PowerShell执行 `powershell -ExecutionPolicy Bypass -File .\\启动本地大模型.ps1`。页面地址 http://127.0.0.1:8765 。启动普通版 `启动.ps1` 可在无LLM时使用证据预览。
启动脚本优先“研发/.venv”，其次本机rags_环境，否则python命令；可用 -Python 明确指定。-Port 可更换监听端口。脚本不会自动打开浏览器，也不改变系统设置。

## 新机器安装
在“研发”目录执行 `python -m venv .venv`；安装 `requirements.txt` 可运行TF-IDF版；安装 `requirements-model.txt` 可启用BGE。然后 `python -m app.main build`，再 `python -m app.main serve`。
依赖范围已提供，实际环境版本记录于“部署/环境记录.json”。模型权重未复制进交付目录；新机器需准备本地BGE模型，或采用TF-IDF降级并重建索引。
模型选择、模型地址及凭据由环境变量传入；本地Ollama版变量见启动脚本。外部服务是否允许上传招股内容应由使用者自行确定，本次实测仅使用本机服务。

## Docker 可选方案
在“工单1”目录执行 `docker compose -f 部署/compose.yaml up --build -d`。镜像默认TF-IDF，启动时重建索引，服务仅映射127.0.0.1:8765。数据目录挂载以保留源PDF、索引及反馈。
Docker容器访问宿主机Ollama应使用适合当前Docker环境的 host.docker.internal 地址；容器内127.0.0.1不是宿主机。需要设置LLM_PROVIDER、LLM_BASE_URL和LLM_MODEL。
Dockerfile与Compose是部署材料，本次未实际构建镜像，不把它列为已验证部署。镜像构建可能下载依赖；添加BGE时应提供模型卷并重新建库。

## 检查与运维
打开 /api/health 检查就绪、真实检索后端及模型状态。执行复测.ps1重新跑自动化测试及评估；生成模型对比时先设置模型环境变量。
备份 data/documents、data/index、data/logs。移走文档或更新模型后重建索引；不在同目录同时启动多个建库写任务。终端Ctrl+C停止服务，反馈和源文件保留。
生产前增加认证、反向代理、限流、日志轮转、端到端并发与24小时稳定性测试。当前只有本机功能与短时测试证据。

## 演示交付
演示视频由实际网页操作录制，展示知识库状态、数字问答、PDF来源、英文问题、纯LLM模式及反馈。演示脚本与测试工具保留，可复现。浏览器语音入口已提供，但视频不伪造真人麦克风输入。
"""
    # 用户要求简化部署：仅保留双击启动和本机使用说明。
    documents["部署/部署方案.md"] = """
# 部署：一键本机启动

## 三步使用
1. 双击“工单1/一键启动.cmd”。
2. 等待终端显示“系统就绪”，浏览器打开 http://127.0.0.1:8765 。
3. 输入问题，选择RAG或纯LLM，查看答案及原文来源。

## 当前电脑
已使用本机Python环境 D:\\an\\envs\\rags_\\python.exe 和已有Ollama模型 Qwen3.5:4B。源PDF和已建索引都在“研发/data”中，无需重新上传或建库。
运行时保持终端窗口打开，Ctrl+C停止。若模型服务未启动或模型不可用，网页会显示异常并给出检索证据。无需Docker或数据库服务。

## 端口占用或其他环境
若8765被占用，在部署目录运行 `powershell -ExecutionPolicy Bypass -File .\\启动本地大模型.ps1 -Port 8766`，网页地址相应改成8766。
若迁移到其他电脑，先安装研发/requirements-model.txt中的依赖，准备本地模型，并用 -Python 指定Python路径；没有BGE时会明确降级为TF-IDF。
需要复测时执行“复测.ps1”；需要更新PDF时通过网页上传。其他生产部署不在本次简单部署范围。

## 演示
“完整演示视频.mp4”记录每秒4帧采样的实际网页操作；语音按钮需在支持语音识别的浏览器中允许麦克风并自行尝试。
"""
    for name, content in documents.items():
        write(name, content)
    answer_lines = ["# 十道指定问题：标准答案与原文依据", "", "PDF页码为从1开始的物理页码。标准答案经原文核对；问答程序不读取本文件。", ""]
    for s in GOLD:
        answer_lines += [f"## ID {s['id']}：{s['question']}", "", s["answer"], "", f"证据页：{', '.join(map(str,s['pages']))}。", ""]
    write("测试/十题标准答案.md", "\n".join(answer_lines))
    comparison = ["# 真实 RAG 与纯 LLM 对比", "", "模型及环境见部署/环境记录.json；全部原始回答见实测评估结果.json。事实覆盖是关键词代理指标，需配合人工复核，不能称为总体准确率。", ""]
    for item in REPORT["answers"]:
        comparison += [f"## ID {item['id']}：{item['question']}", "", "标准答案：" + item["gold_answer"], "",
                       "RAG回答：", "", item["rag"]["answer"], "", f"RAG耗时：{item['rag']['elapsed_ms']:.2f} ms；状态：{item['rag']['generation']}。", "",
                       "纯LLM回答：", "", item["llm_only"]["answer"], "", f"纯LLM耗时：{item['llm_only']['elapsed_ms']:.2f} ms；状态：{item['llm_only']['generation']}。", ""]
    write("测试/RAG与纯LLM对比.md", "\n".join(comparison))
    rag_times = [a["rag"]["elapsed_ms"] for a in REPORT["answers"]]
    completed = sum(a["status"] == "completed" for a in REPORT["answers"])
    hybrid = REPORT["strategies"]["hybrid"]
    summary = f"""
# 工单1：PDF问答系统完成方案

按图片顺序归档：设计 -> 研发 -> 测试 -> 优化 -> 部署。
工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统

## 交付内容
设计：需求、架构、技术选型、工时安排、PDF与表格方案。
研发：完整Python源代码、网页、业务PDF、已构建索引、技术文档和用户手册。
测试：10题标准答案、真实生成对比、检索/性能评估、自动化与网页测试记录。
优化：已实施优化、实测对照和后续改进。
部署：一键启动文件、简单使用说明、环境记录、实际操作演示视频。

## 本次实测
源文档 {MANIFEST['documents'][0]['pages']} 页，识别表格 {MANIFEST['documents'][0]['tables']} 个，索引片段 {MANIFEST['chunks']} 个。
混合检索 Hit@5={hybrid['hit_at_5']:.1%}，MRR@5={hybrid['mrr_at_5']:.3f}，证据事实覆盖={hybrid['evidence_fact_coverage']:.1%}。这些是检索指标，不是生成问答准确率。
10题中完成真实RAG与纯LLM双向生成对比 {completed} 题；完整RAG无缓存耗时范围 {min(rag_times):.2f}-{max(rag_times):.2f} ms。
3秒目标：{'10条指定问题本次样本均通过；仍需扩大样本及生产负载验证' if completed == 10 and max(rag_times) <= 3000 else '本次未通过；详见逐题完整耗时，不使用检索耗时冒充端到端性能'}。
自动化测试结果见测试/自动化测试日志.txt；浏览器结果见测试/网页验收记录.json。所有实测结论只限当前环境和样本。

## 仍需专门验证
语音识别依赖浏览器和麦克风许可，需真人验证；复杂扫描PDF的OCR暂不支持；24小时稳定性与生产高并发未验收。RAGAS评审未执行，本次选择可复现的检索与事实/引用评价体系。

## 打开顺序
先看本目录的“五阶段完成方案.pdf”或各阶段Markdown；双击“一键启动.cmd”，访问 http://127.0.0.1:8765 。演示在“部署/完整演示视频.mp4”。
"""
    write("README.md", summary)
    return documents, summary


def make_pdf(documents, summary):
    pdfmetrics.registerFont(TTFont("CN", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("CNBold", "C:/Windows/Fonts/simhei.ttf"))
    styles = {
        "title": ParagraphStyle("title", fontName="CNBold", fontSize=26, leading=38, textColor=colors.HexColor("#153f43"), spaceAfter=18),
        "h1": ParagraphStyle("h1", fontName="CNBold", fontSize=20, leading=28, textColor=colors.HexColor("#153f43"), spaceBefore=8, spaceAfter=15),
        "h2": ParagraphStyle("h2", fontName="CNBold", fontSize=13, leading=20, textColor=colors.HexColor("#186d66"), spaceBefore=13, spaceAfter=7, keepWithNext=True),
        "body": ParagraphStyle("body", fontName="CN", fontSize=10, leading=15.8, spaceAfter=6.5, wordWrap="CJK"),
        "small": ParagraphStyle("small", fontName="CN", fontSize=8.5, leading=14, textColor=colors.HexColor("#536d7a"), spaceAfter=7, wordWrap="CJK"),
    }
    doc = SimpleDocTemplate(str(BASE / "五阶段完成方案.pdf"), pagesize=A4,
        rightMargin=19*mm, leftMargin=19*mm, topMargin=24*mm, bottomMargin=21*mm,
        title="工单1 - PDF问答系统五阶段完成方案", author="项目交付")
    story = []
    def para(text, style="body"):
        text = escape(text).replace("`", "")
        story.append(Paragraph(text, styles[style]))
    def add_md(content):
        for line in content.strip().splitlines():
            if not line.strip():
                continue
            if line.startswith("## "):
                para(line[3:], "h2")
            elif line.startswith("# "):
                para(line[2:], "h1")
            else:
                para(line)
    para("PDF 文档问答系统", "title")
    para("工单1 · 五阶段完成方案", "h1")
    para("设计 / 研发 / 测试 / 优化 / 部署", "h2")
    para("交付日期 2026-10-12 | 业务资料：武汉兴图新科招股文件", "small")
    para("交付包含源代码、知识库、技术文档、用户手册、10题对比、实测评估及演示视频。结果以实际日志为依据；未验证项目单独注明。")
    story.append(Spacer(1, 12*mm))
    add_md(summary[summary.index("## 本次实测"):summary.index("## 打开顺序")])
    for name, content in documents.items():
        story.append(PageBreak())
        add_md(content)
    story.append(PageBreak())
    para("附录：十道指定问题与原文依据", "h1")
    for sample in GOLD:
        para(f"ID {sample['id']} · {sample['question']}", "h2")
        para(sample["answer"])
        para("PDF物理页码：" + ", ".join(map(str, sample["pages"])), "small")
    story.append(PageBreak())
    para("附录：实际模型对比摘录", "h1")
    for item in REPORT["answers"]:
        para(f"ID {item['id']} · {item['question']}", "h2")
        para("RAG：" + item["rag"]["answer"])
        para("纯LLM：" + item["llm_only"]["answer"])
        para(f"完整耗时：RAG {item['rag']['elapsed_ms']:.2f} ms；纯LLM {item['llm_only']['elapsed_ms']:.2f} ms。", "small")
    def decorate(canvas, document):
        width, height = A4
        canvas.setFillColor(colors.HexColor("#153f43"))
        canvas.rect(0, height-10*mm, width, 10*mm, fill=1, stroke=0)
        canvas.setFont("CN", 8)
        canvas.setFillColor(colors.white)
        canvas.drawString(19*mm, height-6.8*mm, "工单1 | 基于 PDF 文档的问答系统")
        canvas.setFillColor(colors.HexColor("#536d7a"))
        canvas.drawString(19*mm, 11*mm, "设计 · 研发 · 测试 · 优化 · 部署")
        canvas.drawRightString(width-19*mm, 11*mm, str(document.page))
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)


if __name__ == "__main__":
    docs, summary = make_docs()
    make_pdf(docs, summary)
    print("五阶段文档及 PDF 生成完成")
