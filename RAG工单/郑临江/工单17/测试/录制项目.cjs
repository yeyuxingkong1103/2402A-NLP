// 工单14-18：真实网页和API操作录屏，未模拟模型答案。4fps采样后本机编码。
const fs=require('fs'),path=require('path');
const {chromium}=require('C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base=path.resolve(__dirname,'..'),config=JSON.parse(fs.readFileSync(path.join(base,'研发/project.json'),'utf8')),n=config.number;
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const context=await browser.newContext({viewport:{width:1366,height:900}}),page=await context.newPage();page.setDefaultTimeout(180000);
 const framesDir=path.join(base,'测试/视频帧');fs.mkdirSync(framesDir,{recursive:true});const frames=[],start=Date.now();let pending=false;
 async function shot(){if(pending)return;pending=true;try{const name=String(frames.length).padStart(5,'0')+'.jpg';await page.screenshot({path:path.join(framesDir,name),type:'jpeg',quality:75});frames.push({name,time:(Date.now()-start)/1000})}finally{pending=false}}
 const errors=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('http://127.0.0.1:'+(8800+n));
 await page.waitForTimeout(1500);await shot();const timer=setInterval(()=>shot().catch(()=>{}),250);
 const results=[];
 if(n===14||n===15){
  const gold=JSON.parse(fs.readFileSync(path.join(base,'测试/gold.json'),'utf8'));
  for(const row of gold){await page.locator('#question').fill(row.question);await page.locator('#document').selectOption(row.document);const wait=page.waitForResponse(r=>r.url().endsWith('/api/ask'));await page.locator('#ask').click();const result=await(await wait).json();results.push(result);await page.waitForFunction(()=>!document.querySelector('#ask').disabled);await page.locator('#answer').scrollIntoViewIfNeeded();await page.waitForTimeout(1000);if(result.sources?.length){await page.locator('#sources details').first().locator('summary').click();await page.waitForTimeout(600)}}
 } else if(n===16){await page.locator('#metrics').waitFor();await page.waitForTimeout(2500);await page.locator('#loss').scrollIntoViewIfNeeded();await page.waitForTimeout(2200);await page.locator('#samples').scrollIntoViewIfNeeded();await page.waitForTimeout(2200);results.push(await(await context.request.get('http://127.0.0.1:8816/api/report')).json());}
 else if(n===17){await page.waitForTimeout(2000);results.push(await(await context.request.get('http://127.0.0.1:8817/health')).json());results.push(await(await context.request.get('http://127.0.0.1:8817/benchmark')).json());await page.getByText('真实JMeter前后对照',{exact:true}).scrollIntoViewIfNeeded();await page.waitForTimeout(2200);}
 else if(n===18){const report=await context.request.post('http://127.0.0.1:8818/v1/document/quality-inspection',{data:{paths:[path.join(base,'研发/original_problems/documents/CN100342976C.pdf')]}});results.push(await report.json());await page.goto('http://127.0.0.1:8818/quality-report');await page.waitForTimeout(1800);await page.getByText('label_counts',{exact:true}).scrollIntoViewIfNeeded();await page.waitForTimeout(1200);await page.getByText('version_conflicts',{exact:true}).scrollIntoViewIfNeeded();await page.waitForTimeout(1800);}
 await page.screenshot({path:path.join(base,'测试/网页验收.png'),fullPage:true});clearInterval(timer);while(pending)await new Promise(r=>setTimeout(r,30));await shot();fs.writeFileSync(path.join(framesDir,'frames.json'),JSON.stringify(frames));fs.writeFileSync(path.join(base,'测试/网页验收记录.json'),JSON.stringify({project:n,page_errors:errors,results,recording:'真实网页/API，4fps无配音；与正式RAGFlow栈验收分开'},null,2));await context.close();await browser.close();console.log(n,'recorded',frames.length)
})().catch(e=>{console.error(e);process.exit(1)});
