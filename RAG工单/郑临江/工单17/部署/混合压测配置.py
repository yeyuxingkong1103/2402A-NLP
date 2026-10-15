"""工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏。B为真实5问答+5上传解析。"""
import copy,xml.etree.ElementTree as E
from pathlib import Path
import pymupdf
BASE=Path(__file__).resolve().parents[1]
pdf=pymupdf.open(BASE/'研发/original_problems/documents/CN100342976C.pdf');one=pymupdf.open();one.insert_pdf(pdf,from_page=0,to_page=0);fixture=BASE/'测试/load_sample.pdf';one.save(fixture);one.close();pdf.close()
p=BASE/'测试/场景B-10并发-600秒.jmx';tree=E.parse(p);root=tree.getroot();body=root.find('./hashTree/hashTree')
group=body.find('ThreadGroup');group.find("stringProp[@name='ThreadGroup.num_threads']").text='5'
g2=copy.deepcopy(group);g2.set('testname','5用户上传及解析');body.append(g2);sub=E.SubElement(body,'hashTree')
http=E.SubElement(sub,'HTTPSamplerProxy',guiclass='HttpTestSampleGui',testclass='HTTPSamplerProxy',testname='上传真实一页PDF并入队',enabled='true')
args=E.SubElement(http,'elementProp',name='HTTPsampler.Arguments',elementType='Arguments');E.SubElement(args,'collectionProp',name='Arguments.arguments')
for k,v in {'domain':'127.0.0.1','port':'8817','protocol':'http','path':'/api/upload?name=load_sample.pdf','method':'POST','connect_timeout':'5000','response_timeout':'10000'}.items():E.SubElement(http,'stringProp',name='HTTPSampler.'+k).text=v
files=E.SubElement(http,'elementProp',name='HTTPsampler.Files',elementType='HTTPFileArgs');collection=E.SubElement(files,'collectionProp',name='HTTPFileArgs.files');file=E.SubElement(collection,'elementProp',name=str(fixture),elementType='HTTPFileArg')
for k,v in {'File.path':str(fixture),'File.paramname':'','File.mimetype':'application/pdf'}.items():E.SubElement(file,'stringProp',name=k).text=v
E.SubElement(sub,'hashTree');timer=E.SubElement(sub,'ConstantTimer',guiclass='ConstantTimerGui',testclass='ConstantTimer',testname='上传节流',enabled='true');E.SubElement(timer,'stringProp',name='ConstantTimer.delay').text='10000';E.SubElement(sub,'hashTree')
E.indent(root);tree.write(p,encoding='utf-8',xml_declaration=True)
