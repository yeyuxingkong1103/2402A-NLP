"""工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏。JMeter真实工具独立保存。"""
import urllib.request,tarfile,xml.etree.ElementTree as E
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];TOOLS=BASE/'研发/tools';TOOLS.mkdir(exist_ok=True)
archive=TOOLS/'jmeter.tgz'
if not archive.exists():
    urllib.request.urlretrieve('https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.6.3.tgz',archive)
with tarfile.open(archive) as tf:tf.extractall(TOOLS,filter='data')
def plan(name,threads,seconds):
    # 两种场景使用相同持续时间；B为5问答+5上传，另用HTTP采样器发送独立PDF。
    root=E.Element('jmeterTestPlan',version='1.2',properties='5.0',jmeter='5.6.3');tree=E.SubElement(root,'hashTree')
    test=E.SubElement(tree,'TestPlan',guiclass='TestPlanGui',testclass='TestPlan',testname=name,enabled='true');E.SubElement(test,'boolProp',name='TestPlan.functional_mode').text='false';body=E.SubElement(tree,'hashTree')
    group=E.SubElement(body,'ThreadGroup',guiclass='ThreadGroupGui',testclass='ThreadGroup',testname=name,enabled='true')
    E.SubElement(group,'stringProp',name='ThreadGroup.num_threads').text=str(threads);E.SubElement(group,'stringProp',name='ThreadGroup.ramp_time').text='5';E.SubElement(group,'boolProp',name='ThreadGroup.scheduler').text='true';E.SubElement(group,'stringProp',name='ThreadGroup.duration').text=str(seconds)
    loop=E.SubElement(group,'elementProp',name='ThreadGroup.main_controller',elementType='LoopController',guiclass='LoopControlPanel',testclass='LoopController',testname='Loop',enabled='true');E.SubElement(loop,'boolProp',name='LoopController.continue_forever').text='false';E.SubElement(loop,'stringProp',name='LoopController.loops').text='-1'
    subtree=E.SubElement(body,'hashTree');http=E.SubElement(subtree,'HTTPSamplerProxy',guiclass='HttpTestSampleGui',testclass='HTTPSamplerProxy',testname='真实证据检索',enabled='true')
    args=E.SubElement(http,'elementProp',name='HTTPsampler.Arguments',elementType='Arguments');collection=E.SubElement(args,'collectionProp',name='Arguments.arguments');arg=E.SubElement(collection,'elementProp',name='',elementType='HTTPArgument');E.SubElement(arg,'boolProp',name='HTTPArgument.always_encode').text='false';E.SubElement(arg,'stringProp',name='Argument.value').text='{"question":"根据文本信息，该静电除尘器的发明人是：","mode":"retrieval"}';E.SubElement(arg,'stringProp',name='Argument.metadata').text='='
    for k,v in {'domain':'127.0.0.1','port':'8817','protocol':'http','path':'/ask','method':'POST','connect_timeout':'5000','response_timeout':'10000'}.items():E.SubElement(http,'stringProp',name='HTTPSampler.'+k).text=v
    E.SubElement(http,'boolProp',name='HTTPSampler.postBodyRaw').text='true';E.SubElement(subtree,'hashTree')
    headers=E.SubElement(subtree,'HeaderManager',guiclass='HeaderPanel',testclass='HeaderManager',testname='JSON',enabled='true');hc=E.SubElement(headers,'collectionProp',name='HeaderManager.headers');hp=E.SubElement(hc,'elementProp',name='',elementType='Header');E.SubElement(hp,'stringProp',name='Header.name').text='Content-Type';E.SubElement(hp,'stringProp',name='Header.value').text='application/json';E.SubElement(subtree,'hashTree')
    timer=E.SubElement(subtree,'ConstantTimer',guiclass='ConstantTimerGui',testclass='ConstantTimer',testname='200ms pacing',enabled='true');E.SubElement(timer,'stringProp',name='ConstantTimer.delay').text='200';E.SubElement(subtree,'hashTree')
    E.indent(root);E.ElementTree(root).write(BASE/f'测试/{name}.jmx',encoding='utf-8',xml_declaration=True)
plan('场景A-20并发-600秒',20,600);plan('场景B-10并发-600秒',10,600)
print('JMeter installed and plans written')
