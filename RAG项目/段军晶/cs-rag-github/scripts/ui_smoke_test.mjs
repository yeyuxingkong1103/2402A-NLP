// 真实浏览器交互验证（Edge headless + CDP），覆盖：登录 / 提问 / 双身份 / 收藏 / 历史 / 面板收起
import { spawn } from 'node:child_process';
import fs from 'node:fs';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9333;
const SHOTS = 'D:\\桌面\\cs-rag\\_shots';
fs.mkdirSync(SHOTS, { recursive: true });

const browser = spawn(EDGE, [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  '--hide-scrollbars', '--window-size=1680,1020',
  '--remote-debugging-port=' + PORT,
  '--user-data-dir=' + SHOTS + '\\profile',
  'about:blank'
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function findPage() {
  for (let i = 0; i < 60; i++) {
    try {
      const list = await (await fetch('http://127.0.0.1:' + PORT + '/json/list')).json();
      const page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
      if (page) return page;
    } catch (e) { /* 还没起来 */ }
    await sleep(500);
  }
  throw new Error('CDP 未就绪');
}

let msgId = 0;
const pending = new Map();
const consoleErrors = [];
let ws;

function send(method, params = {}) {
  const id = ++msgId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error('CDP 调用超时: ' + method));
    }, 60000);
    pending.set(id, {
      resolve: (v) => { clearTimeout(timer); resolve(v); },
      reject: (e) => { clearTimeout(timer); reject(e); }
    });
    ws.send(JSON.stringify({ id, method, params }));
  });
}

async function evalJs(expression) {
  const res = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (res.exceptionDetails) {
    throw new Error('页面 JS 异常: ' + JSON.stringify(res.exceptionDetails.exception && res.exceptionDetails.exception.description));
  }
  return res.result.value;
}

async function shoot(name) {
  const res = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  fs.writeFileSync(SHOTS + '\\' + name + '.png', Buffer.from(res.data, 'base64'));
  console.log('    📷 ' + name + '.png');
}

async function waitAnswer(timeoutSec) {
  for (let i = 0; i < timeoutSec * 2; i++) {
    await sleep(500);
    const n = await evalJs('document.querySelectorAll(".qa-block .answer-text").length');
    if (n > 0) return true;
  }
  return false;
}

async function main() {
  const page = await findPage();
  ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener('open', r));
  ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    // ★ 关键：把 CDP 的响应按 id 派发回对应的 Promise（上一版漏了这段，导致第一次调用就挂死）
    if (m.id && pending.has(m.id)) {
      const item = pending.get(m.id);
      pending.delete(m.id);
      if (m.error) { item.reject(new Error(JSON.stringify(m.error))); }
      else { item.resolve(m.result); }
      return;
    }
    if (m.method === 'Runtime.exceptionThrown') {
      consoleErrors.push('未捕获异常: ' + (m.params.exceptionDetails.exception && m.params.exceptionDetails.exception.description));
    }
    if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
      consoleErrors.push('console.error: ' + m.params.args.map((a) => a.value || a.description).join(' '));
    }
  });

  await send('Page.enable');
  await send('Runtime.enable');
  await send('Page.navigate', { url: 'http://127.0.0.1:8000/' });
  await sleep(2500);

  console.log('【1】未登录时的登录界面');
  console.log('    登录遮罩可见:', await evalJs('!document.getElementById("loginMask").hidden'));
  console.log('    输入框被禁用:', await evalJs('document.getElementById("questionInput").disabled'));
  await shoot('01-登录界面');

  console.log('【2】输入账号登录（自动建号）');
  await evalJs('document.getElementById("loginInput").value = "演示同学"; document.getElementById("doLoginBtn").click();');
  for (let i = 0; i < 20; i++) {
    await sleep(500);
    if (await evalJs('!document.getElementById("userBox").hidden')) break;
  }
  console.log('    登录后用户名:', await evalJs('document.getElementById("userName").textContent'));
  console.log('    遮罩已关闭:', await evalJs('document.getElementById("loginMask").hidden'));
  await shoot('02-主界面-已登录');

  console.log('【3】学生身份提问');
  await evalJs('document.getElementById("questionInput").value = "数据中心业务连续性分为几个等级？"; document.getElementById("sendBtn").click();');
  const ok1 = await waitAnswer(120);
  await sleep(800);
  console.log('    拿到答案:', ok1, '| 答案字数:', String(await evalJs('document.querySelector(".qa-block .answer-text").textContent')).length);
  console.log('    身份标签:', await evalJs('document.querySelector(".qa-block .role-tag").textContent'));
  console.log('    溯源面板卡片数:', await evalJs('document.querySelectorAll("#sourceBody .src-card").length'));
  console.log('    面板首条页码:', await evalJs('(document.querySelector("#sourceBody .src-page")||{}).textContent'));
  await shoot('03-学生身份-答案与溯源');

  console.log('【4】切到职场身份，问同一句（验证不串缓存）');
  await evalJs('document.querySelector(".seg-btn[data-role=workplace]").click()');
  await sleep(300);
  await evalJs('document.getElementById("questionInput").value = "数据中心业务连续性分为几个等级？"; document.getElementById("sendBtn").click();');
  let ok2 = false;
  for (let i = 0; i < 240; i++) {
    await sleep(500);
    if (await evalJs('document.querySelectorAll(".qa-block .answer-text").length >= 2')) { ok2 = true; break; }
  }
  await sleep(800);
  console.log('    两条答案:', ok2, '| 第二条身份标签:', await evalJs('document.querySelectorAll(".qa-block .role-tag")[1].textContent'));
  const twoAnswers = await evalJs('Array.from(document.querySelectorAll(".qa-block .answer-text")).map(function(e){return e.textContent})');
  console.log('    两条答案是否不同:', twoAnswers[0] !== twoAnswers[1]);
  await shoot('04-职场身份-同问题');

  console.log('【5】收藏第一条答案 → 我的收藏 Tab');
  await evalJs('document.querySelectorAll(".qa-block")[0].querySelector(".msg-tools .tool-btn").click()');
  await sleep(1500);
  console.log('    收藏按钮文案:', await evalJs('document.querySelectorAll(".qa-block")[0].querySelector(".msg-tools .tool-btn").textContent'));
  await evalJs('document.getElementById("tabFavorite").click()');
  await sleep(1200);
  console.log('    收藏列表条数:', await evalJs('document.querySelectorAll("#favoriteList .fav-item").length'));
  await shoot('05-我的收藏');

  console.log('【6】历史记录 Tab');
  await evalJs('document.getElementById("tabHistory").click()');
  await sleep(1200);
  console.log('    历史会话数:', await evalJs('document.querySelectorAll("#historyList .hist-item").length'));
  console.log('    会话标题:', await evalJs('(document.querySelector("#historyList .hist-title")||{}).textContent'));
  console.log('    会话副标题:', await evalJs('(document.querySelector("#historyList .hist-sub")||{}).textContent'));
  await shoot('06-历史记录');

  console.log('【7】溯源面板收起（中间区域应敞开、左栏不动）');
  const leftBefore = await evalJs('document.querySelector(".sidebar").getBoundingClientRect().width');
  await evalJs('document.getElementById("collapsePanelBtn").click()');
  await sleep(700);
  const collapsed = await evalJs('document.getElementById("layout").classList.contains("panel-collapsed")');
  const leftAfter = await evalJs('document.querySelector(".sidebar").getBoundingClientRect().width');
  const mainAfter = await evalJs('document.querySelector(".main").getBoundingClientRect().width');
  const handleShown = await evalJs('!document.getElementById("expandPanelBtn").hidden');
  console.log('    已收起:', collapsed, '| 左栏宽度', leftBefore, '->', leftAfter, '（应不变）');
  console.log('    中间区域宽度:', mainAfter, '| 右侧「参考来源」竖标签出现:', handleShown);
  await shoot('07-溯源面板收起');

  console.log('【8】刷新页面（模拟清缓存后仍能登录、历史仍在）');
  await send('Page.navigate', { url: 'http://127.0.0.1:8000/' });
  await sleep(3000);
  console.log('    刷新后仍登录:', await evalJs('!document.getElementById("userBox").hidden'));
  console.log('    刷新后历史会话数:', await evalJs('document.querySelectorAll("#historyList .hist-item").length'));
  await evalJs('document.querySelectorAll("#historyList .hist-item")[0].click()');
  await sleep(3000);
  console.log('    打开历史会话后消息数:', await evalJs('document.querySelectorAll(".qa-block").length'));
  await shoot('08-打开历史会话');

  console.log('');
  console.log('页面控制台错误数:', consoleErrors.length);
  consoleErrors.slice(0, 10).forEach((e) => console.log('   ✗ ' + e));
}

main()
  .then(() => { browser.kill(); process.exit(0); })
  .catch((e) => { console.error('验证失败:', e.message); browser.kill(); process.exit(1); });