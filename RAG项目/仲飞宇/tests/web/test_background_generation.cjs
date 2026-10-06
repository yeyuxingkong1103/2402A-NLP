/**
 * 前端行为测试：用 jsdom 加载「真实的」app/static/index.html，
 * 用可控的 SSE 流仿真后端，验证流式生成期间切换角色/会话的行为。
 *
 * 为什么要有这个：对话区的流式状态是前端最容易改坏的一块——
 * 生成中途切角色、切会话、删会话、点停止，几条路径互相纠缠。
 * 只靠肉眼看代码很容易漏，所以这里直接驱动真实页面。
 *
 * 运行：
 *     cd tests/web && npm install && npm test
 *
 * 断言只看可观测输出（DOM 文本 + localStorage），不碰页面内部变量——
 * 页面里的 const state 是脚本作用域，本来就取不到，这样也更接近真实用户视角。
 *
 * 为什么跑真实 HTML 而不是把逻辑抽出来做单元测试：被测的正是「哪段代码在什么时候碰哪个
 * DOM 节点」，抽出来就等于把要验的东西替换掉了。而且页面一旦改名改结构，这里会立刻红，
 * 不会出现「测试还绿着、页面早就对不上」的假绿。
 */
const fs = require("fs");
const path = require("path");

let JSDOM;
try {
  ({ JSDOM } = require("jsdom"));
} catch (_) {
  // 以退出码 2 区分「环境没装好」和「断言失败」：前者是使用问题，不该混进测试失败里
  console.error("缺少 jsdom，请先执行：cd tests/web && npm install");
  process.exit(2);
}

// 用 path.resolve(__dirname, ...) 定位而不是相对 cwd：从仓库根目录或 tests/web 里跑结果一致
const INDEX_HTML = path.resolve(__dirname, "../../app/static/index.html");
const HTML = fs.readFileSync(INDEX_HTML, "utf8");

const ROLES = [
  { role_id: "psychologist", name: "心理咨询师", avatar: "🧠", description: "" },
  { role_id: "nutritionist", name: "营养师", avatar: "🥗", description: "" },
];

// 固定时长等待，而不是轮询 DOM 直到出现某个条件：页面里的渲染是同步的、请求是微任务，
// 一个 tick 足够。代价是慢机器上偶发不稳，真遇到了把这里的默认值调大即可。
const tick = (ms = 25) => new Promise((r) => setTimeout(r, ms));
let failures = 0;

// 不 throw 而是累计：一次跑完拿到全部失败项，比第一个断言就中断更容易定位是哪次改动伤的
function check(name, cond, extra = "") {
  // extra 是为失败准备的现场信息（被断言的实际值），通过时不必打印
  console.log(`${cond ? "  ✓" : "  ✗"} ${name}${cond ? "" : "   " + extra}`);
  if (!cond) failures++;
}

// ---- 可控 SSE 流：测试自己决定什么时候吐 delta、什么时候结束 ----
// 必须自己造流、不能用真后端：本项目最关键的几个场景都是时序竞态
//（「用户在生成没结束时切走」「切走之后那一路才 done」），靠真服务端复现不出来。
function makeStream() {
  const enc = new TextEncoder();
  let ctrl;
  const stream = new ReadableStream({ start(c) { ctrl = c; } });
  return {
    stream,
    close: () => ctrl.close(),
    error: (e) => ctrl.error(e),
    sse: (ev, obj) => ctrl.enqueue(enc.encode(`event: ${ev}\ndata: ${JSON.stringify(obj)}\n\n`)),
  };
}

const requests = [];   // 记录每次 /chat/stream 的请求体
// pending 会被下一次请求覆盖，所以测试里拿到手就要立刻存进局部变量（如 psyStream），
// 否则后开的流会把前面那一路冲掉
let pending = null;    // 最近一次 /chat/stream 的句柄

// 手写的 fetch 替身：页面只用 GET/POST 这几个固定地址，够用且完全可控。
// 数据是写死的，测试不依赖任何后端服务是否在跑。
function fetchStub(url, opts = {}) {
  if (url === "/role/list") {
    return Promise.resolve({ ok: true, status: 200, json: async () => ROLES });
  }
  if (url === "/knowledge/list") {
    return Promise.resolve({ ok: true, status: 200, json: async () => ({ documents: [], total: 0 }) });
  }
  if (url === "/health") {
    return Promise.resolve({ ok: true, status: 200, json: async () => ({ status: "ok", components: {} }) });
  }
  if (url === "/chat/stream") {
    const body = JSON.parse(opts.body);
    requests.push(body);
    const s = makeStream();
    // 真实浏览器里 abort 会让响应体读取抛 AbortError，这里必须一并仿真，
    // 否则「中途点停止」那条路径根本走不到（假流不会因为 abort 而中断）。
    if (opts.signal) {
      opts.signal.addEventListener("abort", () => {
        try { s.error(new DOMException("The operation was aborted.", "AbortError")); } catch (_) {}
      });
    }
    pending = { ...s, req: body };
    return Promise.resolve({ ok: true, status: 200, body: s.stream });
  }
  // 未桩的地址直接失败而不是返回 undefined：页面将来新增请求时会当场暴露，
  // 不会静默拿到一个假响应、让断言在别处莫名其妙地红
  return Promise.reject(new Error("测试没有桩这个地址: " + url));
}

(async () => {
  const dom = new JSDOM(HTML, {
    // 必须开：页面逻辑全在内联 <script> 里，不执行就等于什么都没测
    runScripts: "dangerously",
    // 得给一个真实 origin：localStorage 在同源策略下才有，相对路径的 fetch 也才解析得动
    url: "http://localhost:8000/",
    // 这些只能在页面脚本执行「之前」注入——beforeParse 就是最后的机会窗口。
    // jsdom 的 window 上默认没有 fetch / ReadableStream，而 Node 全局有，所以要手工搬进去。
    beforeParse(window) {
      window.fetch = fetchStub;
      window.TextDecoder = TextDecoder;
      window.TextEncoder = TextEncoder;
      window.ReadableStream = ReadableStream;
      // 预置角色，跳过「首次访问自动挑第一个角色」那条分支，让每个用例的起点都确定
      window.localStorage.setItem("rp_role", "psychologist");
    },
  });
  const { window } = dom;
  const $ = (id) => window.document.getElementById(id);
  // 用 textContent 而不是 innerHTML 做断言：只关心「用户看得见什么」，
  // 顺带避免因为多了个 class 或标签就误判
  const msgs = () => $("msgs").textContent;
  const sessions = () => $("sessionList").textContent;
  // 走真实的点击事件而不是直接调页面里的函数——那些函数在脚本作用域里本来就够不着
  const send = async (text) => {
    $("input").value = text;
    $("btnSend").dispatchEvent(new window.Event("click"));
    await tick(40);
  };
  // 同理走 change 事件：这样才会经过页面 select 上挂的 onchange
  const switchRole = async (id) => {
    $("roleSelect").value = id;
    $("roleSelect").dispatchEvent(new window.Event("change"));
    await tick(40);
  };

  // 等 loadRoles().then(loadKb) 这条启动链跑完，否则第一个用例可能在角色还没填充时就断言
  await tick(60);

  // 【1】【2】合起来才是完整的一条回归：生成没结束就切走，被切走的那一路既
  // 不能污染新角色的会话，也不能因此被中断。这两件事过去是同一个 bug 的两面。
  console.log("\n【1】以心理咨询师身份提问，生成中切到营养师");
  check("角色选中心理咨询师", $("roleSelect").value === "psychologist", $("roleSelect").value);

  await send("最近总是失眠怎么办？");
  const psy = requests[0];
  const psyStream = pending;   // 立刻抓住这一路，后面 pending 会被新请求覆盖
  check("发起生成，role_id 正确", psy && psy.role_id === "psychologist", JSON.stringify(psy));
  check("发送后按钮切到停止", $("btnStop").style.display === "" && $("btnSend").disabled === true);
  check("会话列表显示「生成中」", sessions().includes("生成中"), sessions().slice(0, 60));

  psyStream.sse("sources", { sources: [{ source: "sleep_guide.md", text: "x", score: 0.1 }] });
  psyStream.sse("delta", { delta: "失眠可以从作息规律开始调整" });
  await tick(40);
  check("心理医生的文字已上屏", msgs().includes("失眠可以从作息规律开始调整"), msgs().slice(0, 80));

  await switchRole("nutritionist");   // <<< 关键：生成没结束就切走
  check("已切到营养师", $("roleSelect").value === "nutritionist");
  check("切走后输入区解锁，可以继续提问", $("btnSend").disabled === false);
  check("心理医生的内容不再显示", !msgs().includes("失眠可以从作息规律开始调整"), msgs().slice(0, 80));

  console.log("\n【2】切走后心理医生那条流仍在后台跑");
  psyStream.sse("delta", { delta: "，并保持固定起床时间。" });
  await tick(40);
  check("营养师会话未被写入心理医生的内容", !msgs().includes("失眠"), msgs().slice(0, 80));

  console.log("\n【2b】切回一个仍在生成的会话");
  await switchRole("psychologist");
  check("能看到后台已生成的内容", msgs().includes("失眠可以从作息规律开始调整"), msgs().slice(0, 90));
  check("该会话生成中：发送禁用 / 停止可用",
    $("btnSend").disabled === true && $("btnStop").style.display === "");
  check("该会话带「生成中」角标", sessions().includes("生成中"));
  await switchRole("nutritionist");

  console.log("\n【3】在营养师下也发一条，两路并发");
  await send("减脂期怎么吃？");
  const nut = requests[1];
  const nutStream = pending;
  check("第二路 role_id 是营养师", nut && nut.role_id === "nutritionist", JSON.stringify(nut));
  check("两条流分属不同会话", psy.session_id !== nut.session_id);
  nutStream.sse("delta", { delta: "减脂期优先保证蛋白质摄入" });
  await tick(40);
  check("营养师的文字上屏", msgs().includes("减脂期优先保证蛋白质摄入"), msgs().slice(0, 80));

  // 【4】是本文件最核心的一组断言：答案必须落到「发起时那个会话」，
  // 而不是「结束时用户正看着的那个会话」——串会话在界面上很难发现，只会在存储里留下证据
  console.log("\n【4】两路先后结束，各自落盘到自己的会话");
  nutStream.sse("done", { answer: "减脂期优先保证蛋白质摄入。" });
  nutStream.close();
  await tick(60);
  // 注意：心理医生是在用户切走之后才结束的
  psyStream.sse("done", { answer: "失眠可以从作息规律开始调整，并保持固定起床时间。" });
  psyStream.close();
  await tick(80);

  const stored = JSON.parse(window.localStorage.getItem("rp_sessions") || "{}");
  const psyMsgs = JSON.parse(window.localStorage.getItem("rp_msgs_" + psy.session_id) || "[]");
  const nutMsgs = JSON.parse(window.localStorage.getItem("rp_msgs_" + nut.session_id) || "[]");

  check("心理医生会话有 2 条消息", psyMsgs.length === 2, JSON.stringify(psyMsgs).slice(0, 120));
  check("答案落在自己的会话里",
    psyMsgs[1] && psyMsgs[1].text.includes("并保持固定起床时间"), JSON.stringify(psyMsgs[1]).slice(0, 120));
  check("没串进营养师会话",
    !JSON.stringify(nutMsgs).includes("失眠"), JSON.stringify(nutMsgs).slice(0, 160));
  check("营养师会话有 2 条消息", nutMsgs.length === 2, JSON.stringify(nutMsgs).slice(0, 120));
  check("会话分别挂在各自角色下",
    Object.keys(stored).length === 2 && stored.psychologist && stored.nutritionist,
    JSON.stringify(Object.keys(stored)));

  console.log("\n【5】切回心理医生，能看到完整结果");
  await switchRole("psychologist");
  check("能看到完整回答", msgs().includes("并保持固定起床时间"), msgs().slice(0, 100));
  check("「生成中」角标消失", !sessions().includes("生成中"), sessions().slice(0, 60));
  check("引用来源已渲染", $("msgs").innerHTML.includes("src-list"));

  // 【6】停止的语义是「保住已经流出来的部分」，不是「把这条回答丢掉」；
  // 这里要的是 abort 之后落盘路径仍然跑完，且输入区解锁
  console.log("\n【6】中途点停止：保留已生成部分");
  $("btnNew").dispatchEvent(new window.Event("click"));   // 开新会话，断言才干净
  await tick(30);
  await send("还有什么建议？");
  const stopStream = pending;
  stopStream.sse("delta", { delta: "可以试试睡前冥想" });
  await tick(40);
  $("btnStop").dispatchEvent(new window.Event("click"));
  await tick(80);
  const stopMsgs = JSON.parse(window.localStorage.getItem("rp_msgs_" + stopStream.req.session_id) || "[]");
  check("已生成部分被保留",
    JSON.stringify(stopMsgs).includes("可以试试睡前冥想"), JSON.stringify(stopMsgs).slice(0, 140));
  check("输入区恢复", $("btnSend").disabled === false);

  // 【7】防的是「删了会话、但那条流还在跑，跑完又把半截回答写回刚清空的存储」——
  // 界面上完全看不出来，只会在 localStorage 里留下一堆再也访问不到的孤儿键。
  // 这里必须让 delta 先到、再删，才能走到那条竞态路径上。
  console.log("\n【7】删除正在生成的会话，不留孤儿数据");
  await send("再问一个");
  const orphan = pending;
  orphan.sse("delta", { delta: "这段不该被保存" });
  await tick(40);
  // 点的是当前激活条目上的删除按钮，走的是真实的「条目内按钮不能触发切会话」那条路径
  window.document.querySelector(".session-item.active .sdel").dispatchEvent(new window.Event("click"));
  await tick(80);
  check("被删会话的 localStorage 已清空",
    window.localStorage.getItem("rp_msgs_" + orphan.req.session_id) === null,
    String(window.localStorage.getItem("rp_msgs_" + orphan.req.session_id)).slice(0, 80));

  console.log(failures === 0 ? "\n全部通过 ✓" : `\n${failures} 项失败 ✗`);
  // 必须关窗：页面里有一个 30 秒的健康轮询定时器，不关掉 jsdom 会一直吊着事件循环、进程不退出
  window.close();
  // 显式给退出码，CI 才拿得到「有没有失败」这个信号
  process.exit(failures ? 1 : 0);
})();
