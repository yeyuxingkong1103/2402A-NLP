// 工单编号：人工智能NLP-RAG-混合检索任务
// 文件说明：演示截图脚本。puppeteer-core 驱动无头 Chromium 打开真实页面，输入问题、切换检索配置并截图，
//           用来证明 Vue 页面确实可用（而不是只交付了一份 HTML）。
// 用法：cd 工单六 && npm install && node docs/screenshot.mjs
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(import.meta.dirname, '..');                     // 项目根
const OUT = path.join(ROOT, 'docs', 'screenshots');                       // 截图输出目录
const URL = process.env.DEMO_URL || 'http://127.0.0.1:8010/';             // 演示地址（FastAPI 挂载的前端）

// 本机可用浏览器：优先环境变量，其次 Playwright 自带 Chromium，最后用系统 Edge
const CANDIDATES = [process.env.CHROME_PATH,
  'C:/Users/雨子/AppData/Local/ms-playwright/chromium-1234/chrome-win64/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe'].filter(Boolean);

const wait = ms => new Promise(r => setTimeout(r, ms));                   // 简易等待
const exe = CANDIDATES.find(p => fs.existsSync(p));                       // 选定浏览器
if (!exe) { console.error('未找到 Chromium/Edge，请设置 CHROME_PATH'); process.exit(1); }
fs.mkdirSync(OUT, { recursive: true });                                   // 建输出目录

// 启动浏览器。
// 坑（实测）：必须加 --no-proxy-server。本机开着系统代理时，连 127.0.0.1 也会被代理走，
// 页面表现为"一直转圈/请求失败"，但用 curl 访问同一个地址却是好的，极易误判成前端 bug。
const browser = await puppeteer.launch({
  executablePath: exe, headless: 'new',
  args: ['--no-proxy-server', '--no-sandbox', '--disable-dev-shm-usage'],
  defaultViewport: { width: 1440, height: 900 },
});
const page = await browser.newPage();
page.on('pageerror', e => console.error('[页面错误]', e.message));         // 页面 JS 异常要能看到

await page.goto(URL, { waitUntil: 'networkidle2', timeout: 60000 });      // 打开并等网络空闲
await wait(2500);                                                         // 等 /health 与 /config 渲染

/** 截一张图。 */
async function shot(name) {
  const f = path.join(OUT, name + '.png');
  await page.screenshot({ path: f, fullPage: false });
  console.log('已保存', f);
}

/** 提问并等待回答渲染完成（轮询机器人消息条数）。 */
async function ask(text) {
  const before = await page.$$eval('.msg.bot', els => els.length);         // 提问前的回答条数
  await page.evaluate(t => {                                              // 填输入框（Vue v-model 需要派发 input 事件）
    const ta = document.querySelector('textarea');
    ta.value = t;
    ta.dispatchEvent(new Event('input', { bubbles: true }));
  }, text);
  await wait(300);
  // 坑（实测）：不能用 'button:not(.ghost)'——回答里每条引用都带 👍/👎 反馈按钮（同样不是 .ghost），
  // 会选中第一条回答的点赞按钮，导致问题根本没发出去、截图却"看起来正常"。
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll('button')].find(b => b.textContent.includes('发送提问'));
    if (btn) btn.click();
  });
  for (let i = 0; i < 90; i++) {                                          // 最多等 90 秒
    await wait(1000);
    const now = await page.$$eval('.msg.bot', els => els.length);         // 当前回答条数
    if (now > before) break;                                              // 出现新回答即完成
  }
  const after = await page.$$eval('.msg.bot', els => els.length);         // 最终回答条数
  if (after <= before) {                                                  // 没出现新回答
    throw new Error(`问题未得到回答，截图不可信：${text}`);                  // 宁可中断，也不留一张"看起来正常"的假截图
  }
  await wait(600);                                                        // 留渲染时间
}

/** 切换左侧配置下拉框。 */
async function setSelect(index, value) {
  await page.evaluate((i, v) => {
    const sels = document.querySelectorAll('select');
    const s = sels[i];
    s.value = v;
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, index, value);
  await wait(400);
}

await shot('01-首屏与配置区');                                              // 首屏

await ask('报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？');
await shot('02-混合检索-军用收入');

await setSelect(0, 'vector');                                             // 检索模式 → 向量检索
await ask('武汉力源信息技术股份有限公司的法定代表人是谁？');
await shot('03-向量检索-力源法定代表人');

await setSelect(0, 'bm25');                                               // 检索模式 → 全文检索
await ask('程加明~');                                                      // 故意写错人名，演示模糊匹配
await shot('04-全文检索-模糊匹配');

await setSelect(0, 'hybrid');                                             // 回到混合检索
await ask('武汉兴图新科电子股份有限公司的法定代表人是谁？');
await ask('那武汉力源信息技术股份有限公司呢？它的法定代表人是谁？');         // 多轮指代
await shot('05-多轮指代消解');

await ask('Who is the legal representative of Wuhan P&S Information Technology Co., Ltd.?');
await shot('06-英文问答');

await browser.close();                                                    // 关闭浏览器
console.log('全部截图完成');
