// 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
// 每个项目复制本文件后，只读本项目配置和测试数据，录制实际网页操作。
const fs=require('fs'),path=require('path'),{spawn}=require('child_process');
const {chromium}=require('C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base=path.resolve(__dirname,'..'),config=JSON.parse(fs.readFileSync(path.join(base,'研发','project.json'),'utf8'));
const url='http://127.0.0.1:'+config.port;
(async()=>{
  let server=null;
  if(config.number!==10){server=spawn('D:/an/envs/rags_/python.exe',['-m','app.main','serve','--port',String(config.port)],{cwd:path.join(base,'研发'),windowsHide:true,env:{...process.env,PYTHONIOENCODING:'utf-8',LLM_PROVIDER:'ollama',LLM_BASE_URL:'http://127.0.0.1:11434',LLM_MODEL:'Qwen3.5:4B',LLM_TIMEOUT:'120',EMBEDDING_MODEL_PATH:path.join(base,'研发','models','bge-small-zh-v1.5')}});server.stdout.pipe(fs.createWriteStream(path.join(base,'测试','网页服务日志.txt')));server.stderr.pipe(fs.createWriteStream(path.join(base,'测试','网页服务错误日志.txt')))}
  for(let i=0;i<80;i++){try{const r=await fetch(url+'/api/health');if(r.ok)break}catch{}if(i===79)throw Error('服务未就绪');await new Promise(r=>setTimeout(r,500))}
  const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
  const context=await browser.newContext({viewport:{width:1366,height:900}}),page=await context.newPage();page.setDefaultTimeout(120000);
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('服务就绪'));
  const frameDir=path.join(base,'测试','视频帧');fs.mkdirSync(frameDir,{recursive:true});let running=false;
  const frames=[],started=Date.now();async function shot(){if(running)return;running=true;try{const name=String(frames.length).padStart(5,'0')+'.jpg';await page.screenshot({path:path.join(frameDir,name),type:'jpeg',quality:75});frames.push({name,time:(Date.now()-started)/1000})}finally{running=false}}
  await shot();const timer=setInterval(()=>shot().catch(()=>{}),250);
  await page.waitForTimeout(1200);await page.screenshot({path:path.join(base,'测试','网页首页.png'),fullPage:true});
  const gold=JSON.parse(fs.readFileSync(path.join(base,'测试','gold.json'),'utf8'));
  const sample=gold.find(g=>g.id===(config.dataset==='annual_reports'?103:config.multimodal?5:543))||gold[0];
  await page.locator('#question').fill(sample.question);
  const response=page.waitForResponse(r=>r.url().endsWith('/api/ask'));
  await page.locator('#ask').click();const answer=await(await response).json();await page.waitForFunction(()=>!document.querySelector('#ask').disabled);
  await page.waitForTimeout(1400);
  if(answer.sources.length)await page.locator('#sources details').first().locator('summary').click();
  await page.screenshot({path:path.join(base,'测试','网页问答.png'),fullPage:true});
  await page.locator('[data-rating="up"]').click();await page.waitForFunction(()=>document.querySelector('#feedbackStatus').textContent.includes('已保存'));
  if(config.graph){await page.locator('#graphButton').click();await page.locator('#graphSvg').scrollIntoViewIfNeeded();await page.waitForTimeout(1600);await page.screenshot({path:path.join(base,'测试','知识图谱.png'),fullPage:true})}
  const follow=[];
  if(config.number===5){await page.locator('#question').fill('这个公司的法定代表人是谁？');const r=page.waitForResponse(x=>x.url().endsWith('/api/ask'));await page.locator('#ask').click();follow.push(await(await r).json());await page.waitForFunction(()=>!document.querySelector('#ask').disabled);await page.waitForTimeout(1200)}
  if(config.number===6){await page.locator('#strategy').selectOption('fulltext');await page.locator('#question').fill('"营业收入" AND "净利润"');const r=page.waitForResponse(x=>x.url().endsWith('/api/ask'));await page.locator('#ask').click();follow.push(await(await r).json());await page.waitForFunction(()=>!document.querySelector('#ask').disabled);await page.waitForTimeout(1000)}
  const englishRequest=await context.request.post(url+'/api/ask',{data:{question:config.dataset==='annual_reports'?'What was the net profit of China Merchants Bank in 2019?':'How much is the registered capital of Wuhan Xingtu Xinke?',mode:'rag',strategy:'hybrid',session_id:'english-check'}});
  const english=await englishRequest.json();
  clearInterval(timer);while(running)await new Promise(r=>setTimeout(r,40));await shot();
  fs.writeFileSync(path.join(frameDir,'frames.json'),JSON.stringify(frames));
  const health=await(await context.request.get(url+'/api/health')).json();
  fs.writeFileSync(path.join(base,'测试','网页验收记录.json'),JSON.stringify({page_errors:errors,answer,follow,english,health,graph_visible:!!config.graph,feedback:'saved',recording:'4fps实际网页画面；无模拟回答',voice:'浏览器入口；真人麦克风未测'},null,2));
  await context.close();await browser.close();if(server)server.kill();
  const child=spawn('D:/an/envs/rags_/python.exe',[path.join(base,'测试','合成视频.py')],{cwd:base,windowsHide:true});
  child.stdout.on('data',d=>process.stdout.write(d));await new Promise((resolve,reject)=>{child.on('exit',c=>c===0?resolve():reject(Error('视频编码失败')))});
  console.log('工单'+config.number+'网页与视频完成');
})().catch(e=>{console.error(e);process.exit(1)});
