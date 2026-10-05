import { useEffect, useState } from "react";
import { Alert, Button, Card, Col, Divider, Empty, Input, Layout, List, Progress, Row, Space, Tag, Typography, Upload, message } from "antd";
import { AudioOutlined, DeleteOutlined, LikeOutlined, DislikeOutlined, SendOutlined, UploadOutlined } from "@ant-design/icons";
import "./style.css";

const { Header, Sider, Content } = Layout;
const API = "/api";
const seedQuestions = [
  "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
  "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
  "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
  "武汉兴图新科电子股份有限公司注册资本是多少？",
];

function App() {
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lang, setLang] = useState("auto");

  const loadDocuments = async () => {
    const response = await fetch(`${API}/documents`);
    if (response.ok) setDocuments(await response.json());
  };
  useEffect(() => { loadDocuments(); }, []);

  const ask = async (text = question) => {
    if (!text.trim()) return;
    setQuestion(text); setLoading(true); setAnswer(null);
    try {
      const response = await fetch(`${API}/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text, document_id: selected?.id, language: lang }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "问答失败");
      setAnswer(data);
    } catch (error) { message.error(error.message); }
    finally { setLoading(false); }
  };

  const upload = async ({ file }) => {
    const token = window.prompt("请输入管理员口令");
    if (!token) return;
    const body = new FormData(); body.append("file", file);
    const response = await fetch(`${API}/documents/upload`, { method: "POST", headers: { "X-Admin-Token": token }, body });
    const data = await response.json();
    if (!response.ok) return message.error(data.detail || "上传失败");
    message.success("上传成功，后台开始解析"); await loadDocuments();
  };

  const feedback = async (helpful) => {
    if (!answer) return;
    await fetch(`${API}/feedback/${answer.request_id}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ helpful }) });
    message.success("感谢反馈");
  };

  return <Layout className="app-shell">
    <Header className="topbar"><div className="brand">招股说明书智能问答</div><Space><Tag color="blue">DeepSeek + Milvus</Tag><Button type={lang === "zh" ? "primary" : "default"} onClick={() => setLang("zh")}>中文</Button><Button type={lang === "en" ? "primary" : "default"} onClick={() => setLang("en")}>English</Button></Space></Header>
    <Layout>
      <Sider width={300} className="sidebar"><Typography.Title level={4}>知识库</Typography.Title><Upload customRequest={upload} showUploadList={false} accept=".pdf"><Button icon={<UploadOutlined />} block>上传 PDF</Button></Upload><Divider /><List dataSource={documents} locale={{ emptyText: "暂无文档" }} renderItem={(doc) => <List.Item className={selected?.id === doc.id ? "selected-doc" : "doc-item"} onClick={() => setSelected(doc)}><List.Item.Meta title={doc.display_name} description={<><Tag color={doc.status === "completed" ? "green" : "orange"}>{doc.status}</Tag><span>{doc.pages} 页 · {doc.chunks} 切片</span></>} /></List.Item>} /><Divider /><Typography.Text type="secondary">快捷问题</Typography.Text>{seedQuestions.map((item) => <Button key={item} type="link" className="question-link" onClick={() => ask(item)}>{item}</Button>)}</Sider>
      <Content className="content"><Row gutter={20}><Col xs={24} lg={15}><Card title={selected ? `当前文档：${selected.display_name}` : "请选择或上传招股说明书"} className="chat-card"><div className="answer-area">{loading ? <><Progress percent={65} status="active" showInfo={false} /><Typography.Text type="secondary">正在进行 Query 理解、混合检索和答案生成…</Typography.Text></> : answer ? <><Typography.Title level={3}>回答</Typography.Title><Typography.Paragraph className="answer-text">{answer.answer}</Typography.Paragraph><Space><Tag color={answer.confidence === "high" ? "green" : "orange"}>{answer.confidence}</Tag><Typography.Text type="secondary">耗时 {Math.round(answer.latency.total_ms)} ms</Typography.Text><Button icon={<LikeOutlined />} onClick={() => feedback(true)} /><Button icon={<DislikeOutlined />} onClick={() => feedback(false)} /></Space>{answer.refusal && <Alert className="notice" type="warning" message={answer.refusal} />}</> : <Empty description="输入问题，答案将基于 PDF 证据生成" />}</div><Input.TextArea value={question} onChange={(event) => setQuestion(event.target.value)} onPressEnter={(event) => { if (!event.shiftKey) { event.preventDefault(); ask(); } }} autoSize={{ minRows: 3, maxRows: 8 }} placeholder="例如：报告期内来自军用领域的收入分别是多少？" /><div className="composer"><Space><Button icon={<AudioOutlined />} onClick={() => message.info("请在支持 Web Speech API 的浏览器中启用麦克风")}>语音输入</Button><Typography.Text type="secondary">Enter 发送，Shift+Enter 换行</Typography.Text></Space><Button type="primary" icon={<SendOutlined />} onClick={() => ask()} loading={loading}>提问</Button></div></Card></Col><Col xs={24} lg={9}><Card title="证据引用" className="citation-card">{answer?.citations?.length ? <List dataSource={answer.citations} renderItem={(citation, index) => <List.Item><div><Space><Tag color="blue">证据 {index + 1}</Tag><Typography.Text strong>PDF 第 {citation.page_number} 页</Typography.Text></Space><Typography.Paragraph className="citation-text">{citation.content}</Typography.Paragraph><Typography.Text type="secondary">{citation.section_title || "未识别章节"}</Typography.Text></div></List.Item>} /> : <Empty description="回答后显示来源页码与原文片段" />}</Card></Col></Row></Content>
    </Layout>
  </Layout>;
}
export default App;
