"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。结构与关键检索测试。"""
import json
import sys
import unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'研发'))
from app.core import KnowledgeBase,rewrite,CONFIG

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.kb=KnowledgeBase()
    def test_local_paths(self):
        from app.core import INDEX,DOCS
        self.assertTrue(INDEX.resolve().is_relative_to(BASE.resolve()))
        self.assertTrue(DOCS.resolve().is_relative_to(BASE.resolve()))
    def test_empty_input(self):
        with self.assertRaises(ValueError):self.kb.ask(' ')
    def test_pronoun_resolution(self):
        q=rewrite('他的法定代表人是谁？',[{'question':'武汉兴图新科电子股份有限公司收入是多少？'}])
        self.assertIn('武汉兴图新科电子股份有限公司',q)
    def test_company_switch(self):
        q=rewrite('那武汉力源信息技术股份有限公司呢？',[{'question':'武汉兴图新科电子股份有限公司法定代表人是谁？'}])
        self.assertIn('法定代表人',q);self.assertIn('力源',q);self.assertNotIn('兴图',q)
    def test_boolean_fulltext(self):
        ids,_=self.kb.fulltext('"营业收入" AND "净利润"')
        for i in ids[:10]:self.assertIn('营业收入',self.kb.chunks[i]['text']);self.assertIn('净利润',self.kb.chunks[i]['text'])
    def test_graph_provenance(self):
        self.assertTrue(self.kb.graph['edges'])
        self.assertTrue(all('document' in e and 'pdf_page' in e for e in self.kb.graph['edges']))
    def test_tables_exist(self):
        self.assertTrue(any(c['kind']=='table' for c in self.kb.chunks))
    def test_image_data_when_required(self):
        if CONFIG['multimodal']:self.assertTrue(any(c['kind']=='image' for c in self.kb.chunks))
    def test_document_filters(self):
        if CONFIG['dataset']=='prospectus' and CONFIG['number']>2:
            _,hits=self.kb.retrieve('武汉力源信息技术股份有限公司本次发行股数是多少？')
            self.assertTrue(hits);self.assertTrue(all(h['document']=='招股说明书2.pdf' for h in hits))

if __name__=='__main__':unittest.main(verbosity=2)
