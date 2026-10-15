"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。添加各项目可复现专项入口。"""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
for num in range(2,14):
    folder=ROOT/f'工单{num}';server=folder/'研发/app/server.py';text=server.read_text(encoding='utf-8')
    text=text.replace('if path == "/api/settings":','if path == "/api/report":\n                    result = state.kb.ask("根据平安银行和招商银行2019年年报，按经营概况、风险管理、数字化创新三部分生成金融研报摘要。仅依据检索证据，未知内容明确指出。", strategy="graph_optimized", use_cache=False)\n                    return self.send_json(result)\n                if path == "/api/settings":')
    text=text.replace('k in {"strategy","reranker","weight"}','k in {"strategy","reranker","weight","field","fuzzy"}')
    server.write_text(text,encoding='utf-8')
    html=folder/'研发/static/index.html';text=html.read_text(encoding='utf-8')
    if num==6 and 'id="weight"' not in text:
        text=text.replace('<button id="ask" class="primary">','<select id="field"><option value="body">正文</option><option value="title">标题</option><option value="summary">摘要</option></select><label>语义权重 <input id="weight" type="number" min="0" max="1" step="0.1" value="0.55" style="width:55px"></label><label><input id="fuzzy" type="checkbox">模糊匹配</label><button id="ask" class="primary">')
        text=text.replace("reranker:$('reranker').value}","reranker:$('reranker').value,field:$('field').value,weight:Number($('weight').value),fuzzy:$('fuzzy').checked}")
    explanation=''
    if num==6:explanation='语义召回使用BGE向量和余弦相似度；全文使用字符词项倒排索引及BM25；混合以RRF排名融合并支持权重调整。重排可选择TF-IDF、真实LLM、用户反馈。'
    if num in {8,9,10}:explanation='GraphRAG将PDF机构、人名、指标及主题构成有来源的图谱。先定位实体与关系，再检索关联片段生成回答；主题提及不是因果关系。'
    if num==11:
        r=json.loads((folder/'测试/微调实测结果.json').read_text(encoding='utf-8'))
        explanation=f"真实Embedding微调：训练{r['splits']['train']}、验证{r['splits']['validation']}、测试{r['splits']['test']}条，按来源页隔离。测试Recall@1从{r['before']['recall_at_1']:.1%}到{r['after']['recall_at_1']:.1%}；模型权重和完整训练记录保存在本项目。"
    if num==12:explanation='官方LightRAG1.5.7以局部实体、全局关系及向量片段做mix查询，带来源的规则型知识图谱通过官方自定义KG接口入库；JSON与图数据库均保存在本项目。'
    if num==13:explanation='性能追踪分别记录查询理解、检索、上下文组装、LLM生成、后处理。比较10块上下文与去重3块，完整无缓存计时和原始答案均保留；不能把检索耗时当作完整问答耗时。'
    if explanation and '项目实现说明' not in text:text=text.replace('<article>','<article><section class="panel"><h2>项目实现说明</h2><p>'+explanation+'</p></section>')
    if num in {8,9,10} and 'id="reportButton"' not in text:
        text=text.replace('<button id="voice">','<button id="reportButton">金融研报摘要</button><button id="voice">')
        text=text.replace('const Speech=',"$('reportButton').onclick=()=>{$('question').value='根据平安银行和招商银行2019年年报，比较经营概况、风险管理和数字化创新，生成有来源的金融研报摘要。';$('ask').click()};const Speech=")
    html.write_text(text,encoding='utf-8')
print('专项界面已更新')
