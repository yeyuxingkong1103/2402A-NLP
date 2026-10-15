"""工单17：显式HTTP keepalive，避免Windows临时端口耗尽被误判为服务故障。"""
from pathlib import Path
import xml.etree.ElementTree as E
BASE=Path(__file__).resolve().parents[1]
for p in (BASE/'测试').glob('*.jmx'):
    tree=E.parse(p)
    for sampler in tree.getroot().iter('HTTPSamplerProxy'):
        prop=sampler.find("boolProp[@name='HTTPSampler.use_keepalive']")
        if prop is None:prop=E.SubElement(sampler,'boolProp',name='HTTPSampler.use_keepalive')
        prop.text='true'
    E.indent(tree.getroot());tree.write(p,encoding='utf-8',xml_declaration=True)
print('JMeter HTTP keepalive enabled')
