// 工单编号：人工智能NLP-RAG-Query理解优化
// 文件说明：演示截图脚本。puppeteer-core 驱动无头 Chromium 打开真实页面，依次点击工单五的 5 个多轮问题并截图。
// 用法：cd 工单五 && node docs/screenshot.mjs
// 依赖：puppeteer-core@21（见根目录 package.json 的 devDependencies）
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(import.meta.dirname, '..');                     // 项目根
const OUT = path.join(ROOT, 'docs', 'screenshots');                       // 截图输出目录
const URL = process.env.DEMO_URL || 'http://127.0.0.1:5173/';             // 演示地址（Vite 开发服务器）
// 本机可用的浏览器：优先环境变量，其次 Playwright 自带 Chromium，最后用系统 Edge
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
  executablePath: exe, headless: 'new',                                   // 无头模式
  args: ['--no-proxy-server', '--no-sandbox', '--disable-dev-shm-usage'], // 关键：绕开系统代理
  defaultViewport: { width: 1440, height: 900 },                          // 画布尺寸
});
const page = await browser.newPage();                                     // 新建标签页
page.on('pageerror', e => console.error('[页面错误]', e.message));         // 页面 JS 异常要能看到

await page.goto(URL, { waitUntil: 'networkidle2', timeout: 60000 });      // 打开页面并等网络空闲
await wait(2500);                                                         // 等首屏健康检查与预置问题渲染

/** 截一张图。 */
async function shot(name) {
  const f = path.join(OUT, name + '.png');                                // 文件名
  await page.screenshot({ path: f, fullPage: false });                    // 视口截图
  console.log('已保存', f);
}

/** 点击侧栏第 i 个预置问题并等待回答出现。 */
async function ask(i) {
  const items = await page.$$('aside .qs li');                            // 侧栏问题列表
  await items[i].click();                                                 // 点击第 i 个
  // 等待本轮回答渲染完毕：轮询消息条数，最后一条助手消息出现且不再有"正在检索"占位
  for (let t = 0; t < 90; t++) {                                          // 最多等 90 秒
    await wait(1000);
    const done = await page.evaluate(() => {
      const msgs = document.querySelectorAll('.msg.assistant');           // 助手消息
      const thinking = document.querySelector('.thinking');               // 加载占位
      return msgs.length > 0 && !thinking;                                // 已出结果且不在加载中
    });
    if (done) break;
  }
  await wait(600);                                                        // 留一点渲染时间
}

await shot('01-首屏与依赖状态');                                             // 首屏：健康状态 + 知识库 + 5 个问题

// 依次提问工单五的 5 个多轮问题，并在关键轮次截图
for (let i = 0; i < 5; i++) {
  await ask(i);
  await shot(`1${i + 1}-第${i + 1}轮-${['军用收入', '科技进步奖', '法定代表人', '实体切换', '组织结构图'][i]}`);
}

// 展开最后一条回答的引用来源，单独截一张，展示"带引用"这一验收点
await page.evaluate(() => {
  const d = [...document.querySelectorAll('.refs')].pop();               // 最后一个引用折叠块
  if (d) d.open = true;                                                   // 展开
  if (d) d.scrollIntoView({ block: 'center' });                           // 滚到可见处
});
await wait(800);
await shot('16-引用来源展开');

await browser.close();                                                    // 关闭浏览器
console.log('全部截图完成');
