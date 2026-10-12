// 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
// 文件说明：演示视频录制脚本。puppeteer-core 驱动无头 Chromium 操作真实页面，定时截 JPEG 帧，
//           再用 Playwright 自带的精简 ffmpeg（只认 image2pipe）合成 webm。字幕由页面内 div 实现，与帧天然同步。
// 用法：cd frontend && npm i puppeteer-core@21 && node ../docs/record_demo.mjs
import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(import.meta.dirname, '..');                       // 项目根
const OUT = path.join(ROOT, 'docs', 'video');                               // 视频输出目录
const SHOTS = path.join(ROOT, 'docs', 'screenshots');                       // 截图目录
const URL = process.env.DEMO_URL || 'http://127.0.0.1:8000/';               // 演示地址
// 本机没有独立 ffmpeg，只有 Playwright 附带的那份精简构建（ms-playwright/ffmpeg-1011）
const FFMPEG = process.env.FFMPEG || 'C:/Users/雨子/AppData/Local/ms-playwright/ffmpeg-1011/ffmpeg-win64.exe';
// 浏览器优先用环境变量指定；否则依次尝试本机常见的 Chromium / Edge
const CANDIDATES = [process.env.CHROME_PATH,
  'C:/Users/雨子/AppData/Local/ms-playwright/chromium-1234/chrome-win64/chrome.exe',  // Playwright 自带的 Chromium
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',                    // 本机 Edge 兜底
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe'].filter(Boolean);

const FPS = 5;                                                              // 帧率：5 fps 足够，文件更小
const W = 1280, H = 720;                                                    // 画布尺寸
const wait = ms => new Promise(r => setTimeout(r, ms));                     // 简易等待（后面多处用到，先定义）
for (const d of [OUT, SHOTS]) fs.mkdirSync(d, { recursive: true });         // 建输出目录

const exe = CANDIDATES.find(p => fs.existsSync(p));                         // 选定浏览器
if (!exe) { console.error('未找到 Chromium/Edge，请设置 CHROME_PATH'); process.exit(1); }

// 启动 ffmpeg：从 stdin 顺序读 JPEG 帧，编码为 webm
// 关键：必须显式写 `-vcodec mjpeg`。管道不可寻址，ffmpeg 无法自行探测输入编码，
// 省掉它会报 "Output file does not contain any stream"（`-i -` 则报 Protocol not found），
// 看起来像"管道输入不支持"，其实只是没告诉它该按什么解码。日志也务必留出来——
// 之前把 stderr 设成 'ignore'，ffmpeg 启动即死却完全看不出原因，表现只是"脚本跑完、视频没生成"。
const FFLOG = path.join(OUT, 'ffmpeg.log');                                 // ffmpeg 运行日志
const ffErr = fs.createWriteStream(FFLOG);                                  // 落盘
const ff = spawn(FFMPEG, ['-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-framerate', String(FPS),
  '-i', 'pipe:0', '-c:v', 'libvpx', '-b:v', '1M', '-pix_fmt', 'yuv420p',
  path.join(OUT, '工单四演示视频.webm')], { stdio: ['pipe', 'ignore', 'pipe'] });  // stderr 接日志
ff.stderr.pipe(ffErr);                                                      // 转发

// 提前挂退出监听：若 ffmpeg 早退，最后的 await 会永远等不到 'close' 而挂死进程
let ffClosed = false;                                                       // ffmpeg 是否已退出
const ffDone = new Promise(r => ff.on('close', c => { ffClosed = true; r(c); }));  // 提前建立

let stopped = false, frames = 0;                                            // 避免重复收尾
const stop = () => { if (!stopped) { stopped = true; if (!ffClosed) ff.stdin.end(); } };  // 结束输入 → ffmpeg 收尾
process.on('exit', stop);

const browser = await puppeteer.launch({ executablePath: exe, headless: 'new',  // 无头模式
  // 坑：本机 Windows 系统代理是开启的（注册表 ProxyEnable=1，指向 127.0.0.1:65532）。
  // Chromium 默认继承系统代理，会把 http://127.0.0.1:8000 也送去代理，导航直接挂到超时。
  // 必须显式 --no-proxy-server 绕开；curl 不走系统代理，所以命令行测着是通的，极具迷惑性。
  args: ['--no-sandbox', '--disable-dev-shm-usage', `--window-size=${W},${H}`,   // 容器/无沙箱环境可跑
         '--no-proxy-server', '--force-device-scale-factor=1', '--hide-scrollbars'] });
const page = await browser.newPage();                                       // 新建页面
await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });      // 固定视口，保证每帧同尺寸
page.on('console', m => { if (m.type() === 'error') console.error('[页面错误]', m.text()); });
// 坑：用 networkidle2 等页面会 60 秒超时——前端是 CDN 单文件，外部 CDN 的连接常驻，
// 网络迟迟不进入 idle。改为 domcontentloaded + 显式等 Vue 渲染出输入框，稳定且更快。
await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 90000 });    // 打开页面
await page.waitForSelector('textarea', { timeout: 60000 });                 // 输入框出现（注意：它是静态 HTML，出现≠Vue 已挂载）
// 坑：Vue 走 CDN，偶发 ERR_CONNECTION_RESET 会让页面停在"未渲染"状态——满屏 {{ res.answer }} 之类的
// 原始占位符，但样式正常，肉眼不细看以为是渲染好了。这样截出来的图和视频全是废的，且不报错。
// 判据用"页面里还有没有 {{ 占位符"，因为 textarea 是静态节点，等它出现证明不了 Vue 挂载。
const rendered = async () => page.evaluate(() => !document.body.innerText.includes('{{'));  // 是否已渲染
for (let i = 1; i <= 3 && !(await rendered()); i++) {                       // 最多重载 3 次
  console.log(`  页面未渲染（Vue 未加载），第 ${i} 次重载 …`);                  // 提示
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 90000 });      // 重载
  await page.waitForSelector('textarea', { timeout: 60000 });               // 等输入框
  await wait(1500);                                                          // 给 Vue 挂载留时间
}
if (!(await rendered())) { console.error('页面始终未渲染，请检查 CDN 可达性后重试'); process.exit(1); }  // 仍失败则明确退出，不出废片
const mounted = true;                                                       // 已确认渲染

// 注入字幕层：固定在底部，白字黑底半透明，随步骤切换文案
await page.evaluate(() => {
  const d = document.createElement('div');                                  // 字幕容器
  d.id = '__cap';                                                           // 便于后续按 id 更新
  d.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99999;padding:10px 18px;' +
    'background:rgba(17,24,39,.86);color:#fff;font:15px/1.5 "Microsoft YaHei",sans-serif;';
  document.body.appendChild(d);                                             // 挂到页面
  const b = document.createElement('div');                                  // 品牌角标
  b.style.cssText = 'position:fixed;right:12px;top:10px;z-index:99999;padding:4px 10px;border-radius:99px;' +
    'background:rgba(30,58,95,.9);color:#fff;font:12px "Microsoft YaHei",sans-serif;';
  b.textContent = '人工智能NLP-RAG-图像内容解析及检索优化';                     // 工单编号
  document.body.appendChild(b);
});
const cap = t => page.evaluate(x => { document.getElementById('__cap').textContent = x; }, t);  // 切字幕

let capIdx = 0, shotIdx = 0;                                                // 字幕与截图计数
// 定时截帧：与业务操作并行，保证画面连续
// 闸门保证同一时刻只有一路在写：截图耗时常超过帧间隔，setInterval 的 async 回调会重叠，
// 多路并发写 stdin 会把 JPEG 流写花，ffmpeg 读到损坏数据直接退出。
let writing = false;                                                        // 写帧闸门
const tick = setInterval(async () => {
  if (stopped || writing || ffClosed) return;                               // 上一帧没写完就跳过，宁可丢帧
  writing = true;                                                           // 上闸
  try { const b = await page.screenshot({ type: 'jpeg', quality: 72 }); ff.stdin.write(b); frames++; }
  catch { /* 页面瞬时繁忙或 ffmpeg 已退，跳过该帧 */ }
  finally { writing = false; }                                              // 开闸
}, 1000 / FPS);

// 等答案渲染完成：等待"来源"标题出现，避免截到 loading 中间态
const settle = async (ms = 90000) => {
  await page.waitForFunction(() => {
    const t = document.body.innerText;
    return t.includes('来源（') || t.includes('资料中未找到') || t.includes('系统繁忙') || t.includes('请输入');
  }, { timeout: ms }).catch(() => {});
  await wait(700);                                                          // 再等渲染稳定
};
const ask = async (q, lang = 'auto') => {                                   // 提问：填框、选语言、提交、等结果
  await page.evaluate((q, lang) => {
    const ta = document.querySelector('textarea');                          // 问题输入框
    ta.value = q; ta.dispatchEvent(new Event('input', { bubbles: true }));   // 触发 Vue 双向绑定
    const sel = document.querySelector('select');                           // 语言下拉
    sel.value = lang; sel.dispatchEvent(new Event('change', { bubbles: true }));
  }, q, lang);
  await page.evaluate(() => [...document.querySelectorAll('button')].find(b => b.textContent.includes('提问')).click());
  await settle();                                                           // 等结果
};
const shot = async name => {                                                // 存一张静态截图
  const p = path.join(SHOTS, `${String(++shotIdx).padStart(2, '0')}-${name}.png`);  // 命名带序号便于排序
  await page.screenshot({ path: p });                                       // 截图
  console.log('  截图', path.basename(p));                                   // 打印
};
const step = async (seconds, text) => { await cap(text); await wait(seconds * 1000); };  // 保持某字幕一段时间

// ── 分镜 ──────────────────────────────────────────────────────────────
await step(5, '① 招股说明书图文问答系统 · 图像内容解析及检索优化（工单编号：人工智能NLP-RAG-图像内容解析及检索优化）');
await shot('首屏');

await page.evaluate(() => window.scrollTo(0, 0));                           // 回到顶部
await step(5, '② 左栏内置工单四指定的 16 个问题，点击即可填入并检索');

const Q6 = '武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？';
await cap('③ 提问 id=6：IC市场应用结构与增长图 —— 增长率最快的是汽车(14.0%)，负增长的是IC卡(-2.0%)');
await ask(Q6);                                                              // 提问并等待
await wait(2500);                                                           // 停留展示
await shot('图像独有题-IC市场图');
await step(4, '④ 来源标注「图像 · 饼图」并显示原图缩略图：答案确实由图内文字与语义解析得到');

const Q5 = '武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？';
await cap('⑤ 提问 id=5：组织结构图 —— 销售部 4 个部门，大客户销售部 6 个销售处');
await ask(Q5);
await wait(2500);
await shot('组织结构图层级还原');
await step(4, '⑥ 图像块的【图内层级】由像素几何还原：横总线上挂 6 条下垂竖线 → 大客户销售部 下辖 6 个销售处');

await cap('⑦ 切换 English，用英文提问，系统以英文作答（多语言支持）');
await ask('How many shares will Wuhan P&S issue, and what percentage of the post-issuance total share capital is that?', 'en');
await wait(2000);
await shot('英文提问');
await step(2, '⑦ 英文提问：答案以英文给出，来源与检索精确度同样展示');

await cap('⑧ 继续覆盖工单其余问题：军用收入、占比、一等奖工程、注册资本、法定代表人…');
for (const [q, name] of [
  ['报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？', '军用领域收入'],
  ['武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？', '国家科技进步一等奖'],
  ['武汉兴图新科电子股份有限公司法定代表人是谁？', '法定代表人']]) {
  await ask(q); await wait(1500); await shot(name);                          // 逐题演示
}
await step(3, '⑧ 逐题展示：答案 + 来源 + 检索精确度 + 响应时间');

await cap('⑨ 指标卡：响应时间 / 检索精确度 / 检索通道 / 检索与生成耗时分解（工单要求显示检索精确度）');
await wait(4000);
await shot('指标卡');

await cap('⑩ 相同问题再次提问 → 命中缓存，端到端降至毫秒级');
await ask(Q5);                                                              // 再问同一问题
await wait(2000);
await shot('缓存命中');
await step(3, '⑩ 缓存命中：不再调用大模型，直接返回');

clearInterval(tick);                                                        // 停止截帧
await wait(400);                                                            // 等最后一帧写入
stop();                                                                     // 结束 ffmpeg 输入
const code = await Promise.race([ffDone, wait(25000).then(() => 'timeout')]);  // 等编码完成，最多 25 秒
await browser.close();                                                      // 关浏览器
const v = path.join(OUT, '工单四演示视频.webm');                              // 产物路径
const kb = fs.existsSync(v) ? Math.round(fs.statSync(v).size / 1024) : 0;    // 产物大小
console.log(`完成：${frames} 帧 → docs/video/工单四演示视频.webm（${kb} KB，ffmpeg 退出码 ${code}）`);
if (!kb) { console.log('⚠ 未生成视频，请看 docs/video/ffmpeg.log'); process.exitCode = 1; }  // 失败显式提示且置非零退出码
