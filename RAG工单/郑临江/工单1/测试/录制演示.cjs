// 工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..');
(async () => {
  const browser = await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless:true});
  const context = await browser.newContext({viewport:{width:1366,height:900}});
  const page = await context.newPage();
  page.setDefaultTimeout(120000);
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8765');
  await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('服务就绪'));
  // 每秒4帧捕获实际网页画面，无需安装 ffmpeg。
  const frameDir=path.join(root,'测试','视频帧');fs.mkdirSync(frameDir,{recursive:true});
  const frames=[];let capturing=false;const started=Date.now();
  async function capture(){if(capturing)return;capturing=true;try{const name=String(frames.length).padStart(5,'0')+'.jpg';await page.screenshot({path:path.join(frameDir,name),type:'jpeg',quality:80});frames.push({name,time:(Date.now()-started)/1000})}finally{capturing=false}}
  await capture();const timer=setInterval(()=>capture().catch(()=>{}),250);
  await page.waitForTimeout(1800);
  await page.screenshot({path:path.join(root,'测试','网页首页.png'),fullPage:true});
  async function ask(question,mode='rag') {
    await page.locator('#question').fill(question);
    await page.locator('#mode').selectOption(mode);
    const response=page.waitForResponse(r=>r.url().endsWith('/api/ask')&&r.request().method()==='POST');
    await page.locator('#ask').click();
    const result=await (await response).json();
    await page.waitForFunction(()=>!document.querySelector('#ask').disabled);
    await page.waitForTimeout(1600);
    return result;
  }
  const capital=await ask('武汉兴图新科电子股份有限公司注册资本是多少？');
  if(!capital.answer.includes('5,520')&&!capital.answer.includes('5520'))throw Error('注册资本答案不正确');
  await page.locator('#sources details').first().locator('summary').click();
  await page.screenshot({path:path.join(root,'测试','答案与来源.png'),fullPage:true});
  const link=await page.locator('#sources a').first().getAttribute('href');
  const pdf=await context.request.get('http://127.0.0.1:8765'+link.split('#')[0]);
  if(pdf.status()!==200||!pdf.headers()['content-type'].includes('pdf'))throw Error('原文查看失败');
  await page.locator('[data-rating="up"]').click();
  await page.waitForFunction(()=>document.querySelector('#feedbackStatus').textContent.includes('已保存'));
  await page.waitForTimeout(1200);
  const english=await ask('How much is the registered capital of Wuhan Xingtu Xinke?');
  await page.screenshot({path:path.join(root,'测试','英文问答.png'),fullPage:true});
  const baseline=await ask('武汉兴图新科电子股份有限公司注册资本是多少？','llm_only');
  await page.waitForTimeout(1600);
  const revenue=await ask('报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？');
  for(const value of ['6,464.51','14,414.16','18,780.67','4,627.14'])if(!revenue.answer.includes(value))throw Error('收入答案未通过数值校验：'+value);
  await page.screenshot({path:path.join(root,'测试','收入问答.png'),fullPage:true});
  const clarify=await ask('收入是多少？');
  if(clarify.generation!=='clarification')throw Error('消歧失败');
  await page.locator('#file').setInputFiles(path.join(root,'研发','data','documents','招股说明书1.pdf'));
  const duplicate=page.waitForResponse(r=>r.url().includes('/api/upload'));
  await page.locator('#upload').click();
  const dup=await duplicate;
  if(dup.status()!==409)throw Error('同名文件保护失败');
  await page.waitForTimeout(1600);
  const invalid=await context.request.post('http://127.0.0.1:8765/api/upload?name=bad.pdf',{data:Buffer.from('invalid'),headers:{'Content-Type':'application/pdf'}});
  const health=await (await context.request.get('http://127.0.0.1:8765/api/health')).json();
  const record={status:'passed',page_errors:errors,capital,english,baseline,revenue,clarify,
    pdf_link:link,pdf_http_status:pdf.status(),feedback:'persisted',duplicate_upload_http_status:dup.status(),
    invalid_upload_http_status:invalid.status(),health,voice:'仅检查入口；真人麦克风识别未测',recording:'每秒4帧采样的实际网页操作画面，无模拟回答'};
  if(errors.length)throw Error('浏览器脚本错误：'+errors.join(','));
  fs.writeFileSync(path.join(root,'测试','网页验收记录.json'),JSON.stringify(record,null,2));
  clearInterval(timer);while(capturing)await new Promise(r=>setTimeout(r,40));await capture();
  fs.writeFileSync(path.join(frameDir,'frames.json'),JSON.stringify(frames));
  await context.close();
  await browser.close();
  console.log(JSON.stringify({status:'passed',frames:frames.length,capitalMs:capital.elapsed_ms,englishMs:english.elapsed_ms}));
})().catch(e=>{console.error(e);process.exit(1)});
