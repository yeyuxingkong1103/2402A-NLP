// 开启 JavaScript 严格模式
"use strict";

const $ = (id) => document.getElementById(id);
const pageShell = $("pageShell");
const welcome = $("welcome");
const conversation = $("conversation");
const messageList = $("messageList");
const serviceState = $("serviceState");
const serviceText = $("serviceText");
const webSearchBtn = $("webSearchBtn");

const qaForm = $("qaForm");
const qaInput = $("qaInput");
const qaSendBtn = $("qaSendBtn");
const qaAttachBtn = $("qaAttachBtn");
const qaImageInput = $("qaImageInput");

const disclaimer = $("disclaimer");
const attachPanel = $("attachPanel");
const attachThumbs = $("attachThumbs");
const attachReady = $("attachReady");
const attachHint = $("attachHint");
const attachPanelClose = $("attachPanelClose");
const attachAddTile = $("attachAddTile");
const lightbox = $("lightbox");
const lightboxImg = $("lightboxImg");
const lightboxClose = $("lightboxClose");
const sourceModal = $("sourceModal");
const sourceModalTitle = $("sourceModalTitle");
const sourceModalMeta = $("sourceModalMeta");
const sourceModalText = $("sourceModalText");
const sourceModalClose = $("sourceModalClose");
const vectorState = $("vectorState");
const vectorText = $("vectorText");
const ENGINE_LABELS = {
  // 向量检索（Milvus）
  vector: "向量化数据库（Milvus）",
  // 图谱检索（Neo4j）
  graph: "药品知识图谱（Neo4j）",
  // Tavily 联网搜索
  web: "联网搜索（Tavily）",
};

// 是否正在请求
let requesting = false;
// 问答附带的图片（最多 3 张）
let qaAttachedImages = [];
// 是否在本次问答中追加 Tavily 联网搜索（默认关闭）
let webSearchEnabled = false;
// 当前联网请求控制器
let activeChatController = null;

// 会话 ID 存储键
const CHAT_SESSION_KEY = "medrag_chat_session_id";
// 历史对话 ID 存储键
const CHAT_CONVERSATION_KEY = "medrag_conversation_id";

// 生成新的会话 ID
function createChatSessionId() {
  // 浏览器支持随机 UUID 时
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    // 浏览器支持随机 UUID 时
    return window.crypto.randomUUID().replace(/-/g, "");
  }
  // 否则用时间+随机数拼
  return `chat_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

// 从 localStorage 读会话 ID（同一浏览器多标签页共享）
function loadChatSessionId() {
  try {
    // 会话 ID 存储键
    return localStorage.getItem(CHAT_SESSION_KEY) || createChatSessionId();
  } catch (_) {
    // 读取出错时兜底新建
    return createChatSessionId();
  }
}

// 保存会话 ID
function saveChatSessionId(value) {
  // 空值不保存
  if (!value) return;
  // 更新内存变量
  chatSessionId = value;
  try {
    // 会话 ID 存储键
    localStorage.setItem(CHAT_SESSION_KEY, value);
  } catch (_) {
    // Storage may be disabled; the in-page session still works.
  }
}

// 初始化会话 ID
let chatSessionId = loadChatSessionId();
// 立即保存一次
saveChatSessionId(chatSessionId);

// 读取历史对话 ID
function loadConversationId() {
  try {
    // 历史对话 ID 存储键
    return localStorage.getItem(CHAT_CONVERSATION_KEY) || "";
  } catch (_) {
    // 取不到返回空串
    return "";
  }
}

// 保存历史对话 ID
function saveConversationId(value) {
  // 同步内存变量
  chatConversationId = value || "";
  try {
    if (value) {
      // 历史对话 ID 存储键
      localStorage.setItem(CHAT_CONVERSATION_KEY, String(value));
    } else {
      // 历史对话 ID 存储键
      localStorage.removeItem(CHAT_CONVERSATION_KEY);
    }
  } catch (_) {
    // Storage may be disabled; the in-page conversation still works.
  }
}

// 初始化对话 ID
let chatConversationId = loadConversationId();

// 设置服务状态样式与文字
function setServiceState(kind, text) {
  // 状态类名（online/offline）
  serviceState.className = "service-state " + kind;
  // 状态文字
  serviceText.textContent = text;
}

// 设置向量库状态
function setVectorState(kind, text) {
  // 状态类名
  vectorState.className = "service-state " + kind;
  // 状态文字
  vectorText.textContent = text;
}

// 页面加载时检查服务健康
async function checkHealth() {
  try {
    // 请求健康接口
    const response = await authFetch("/api/health", { cache: "no-store" });
    // 失败直接抛错
    if (!response.ok) throw new Error("服务异常");
    // 解析 JSON
    const data = await response.json();
    // 失败直接抛错
    if (data.status !== "ok") throw new Error("服务异常");
    // 服务在线文案
    setServiceState("online", "图谱问答服务已就绪");
    // 取向量库状态
    const vector = data.vector || {};
    if (vector.available) {
      // 可用则显示已连接
      setVectorState("online", "向量化数据库已连接");
    } else {
      // 不可用显示未连接
      setVectorState("offline", "向量化数据库未连接");
    }
  } catch (_) {
    // 网络异常时显示离线
    setServiceState("offline", "服务暂未连接");
    // 不可用显示未连接
    setVectorState("offline", "向量库状态未知");
  }
}

// 输入框自适应高度
function autoResize() {
  const el = qaInput;
  // 先重置高度
  el.style.height = "auto";
  // 按内容高度设置，最高 140
  el.style.height = Math.min(el.scrollHeight, 140) + "px";
}

// 滚动到页面底部
function scrollToLatest() {
  // 下一帧平滑滚动
  requestAnimationFrame(() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }));
}

// 打开大图预览
function openLightbox(src) {
  // 设置图片来源
  lightboxImg.src = src;
  // 显示预览层
  lightbox.hidden = false;
}

// 关闭大图预览
function closeLightbox() {
  // 隐藏预览层
  lightbox.hidden = true;
  // 清空图片来源
  lightboxImg.src = "";
}

// 点击遮罩空白处关闭
lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) closeLightbox();
});
// 点关闭按钮关闭
lightboxClose.addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => {
  // 按 Esc 键关闭
  if (e.key === "Escape") closeLightbox();
});

// 来源数据查看器：打开伪链接的引用数据
function openSourceViewer(ref) {
  if (!sourceModal) return;
  sourceModalTitle.textContent = ref.name || "来源数据";
  const metaParts = [];
  if (ref.evidence_level) metaParts.push(`证据等级：${ref.evidence_level}`);
  if (ref.section_path) metaParts.push(`章节：${ref.section_path}`);
  if (ref.source_path) metaParts.push(`来源：${ref.source_path}`);
  sourceModalMeta.textContent = metaParts.join(" ｜ ");
  sourceModalMeta.hidden = !metaParts.length;
  sourceModalText.textContent = ref.text || "（该来源暂无正文内容）";
  sourceModal.hidden = false;
}

function closeSourceViewer() {
  if (!sourceModal) return;
  sourceModal.hidden = true;
}

if (sourceModalClose) {
  sourceModalClose.addEventListener("click", closeSourceViewer);
}
if (sourceModal) {
  sourceModal.addEventListener("click", (e) => {
    if (e.target === sourceModal) closeSourceViewer();
  });
}
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeSourceViewer();
});

// 引用来源面板：可折叠，含编号 / 标题 / 来源类型 / 证据等级 / 原文链接
function renderReferencePanel(bubble, details) {
  const panel = document.createElement("div");
  panel.className = "reference-panel";

  const header = document.createElement("div");
  header.className = "reference-header";
  header.innerHTML =
    `<span class="reference-header-label">📚 引用来源（共 ${details.length} 条）</span>` +
    '<span class="toggle-icon">▼</span>';
  panel.appendChild(header);

  const list = document.createElement("div");
  list.className = "reference-list";
  list.hidden = true;  // 默认折叠

  details.forEach((ref) => {
    const item = document.createElement("div");
    item.className = "reference-item";

    const titleRow = document.createElement("div");
    titleRow.className = "ref-title-row";
    const indexBadge = document.createElement("span");
    indexBadge.className = "ref-index";
    indexBadge.textContent = `[${ref.index}]`;
    const titleSpan = document.createElement("span");
    titleSpan.className = "ref-item-title";
    titleSpan.textContent = ref.title || "未知";
    titleRow.append(indexBadge, titleSpan);
    item.appendChild(titleRow);

    const metaRow = document.createElement("div");
    metaRow.className = "ref-meta-row";
    const sourceType = document.createElement("span");
    sourceType.className = "ref-source-type";
    sourceType.textContent = ref.source_type || "source";
    metaRow.appendChild(sourceType);

    const level = document.createElement("span");
    level.className =
      "ref-evidence " + (ref.evidence_level === "高" ? "evidence-high" : "evidence-mid");
    level.textContent = `证据等级：${ref.evidence_level || "中"}`;
    metaRow.appendChild(level);

    // 原文链接：真实外链跳转；本地路径/空链接 → 伪链接弹窗查看数据
    const link = document.createElement("a");
    link.className = "ref-link";
    if (ref.source_link) {
      link.href = ref.source_link;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "查看原文 →";
      link.addEventListener("click", (event) => {
        event.stopPropagation();  // 点链接不触发折叠
      });
    } else {
      link.href = "javascript:void(0)";
      link.textContent = "查看数据 →";
      link.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();  // 点链接不折叠面板
        openSourceViewer({
          name: ref.title,
          evidence_level: ref.evidence_level,
          section_path: "",
          source_path: ref.source_path,
          text: ref.preview || "",
        });
      });
    }
    metaRow.appendChild(link);
    item.appendChild(metaRow);
    list.appendChild(item);
  });

  panel.appendChild(list);

  // 点一下展开，再点一下折叠
  const icon = header.querySelector(".toggle-icon");
  panel.addEventListener("click", () => {
    const isHidden = list.hidden;
    list.hidden = !isHidden;
    icon.textContent = isHidden ? "▲" : "▼";
  });

  bubble.appendChild(panel);
}

// 创建一条消息气泡
function createMessage(role, text, images = []) {
  // 消息行容器
  const row = document.createElement("article");
  // 按角色设置 class
  row.className = "message " + role;

  // 气泡元素
  const bubble = document.createElement("div");
  // 气泡样式
  bubble.className = "bubble";
  // 用文本方式写入内容（防注入）
  bubble.textContent = text;

  // 助手消息带头像
  if (role === "assistant") {
    // 头像元素
    const avatar = document.createElement("div");
    // 头像样式
    avatar.className = "avatar";
    // 头像文字“知”
    avatar.textContent = "知";
    // 把头像和气泡放入行
    row.append(avatar, bubble);
  } else {
    // 附件（图片/PDF）直接放进消息行
    if (images.length) {
      // 分类容器
      const wrap = document.createElement("div");
      // 图片容器
      wrap.className = "message-images";
      // 逐张添加图片/文件
      images.forEach((src) => {
        // 兼容字符串（纯 dataUrl）和对象（{dataUrl,type,name}）
        const item = typeof src === "string" ? { dataUrl: src, type: "image" } : src;
        // PDF：显示可打开的文件标签
        if (item.type === "pdf") {
          const chip = document.createElement("a");
          chip.className = "pdf-message-chip";
          chip.textContent = "📄 " + (item.name || "PDF 文件");
          chip.href = item.dataUrl || "#";
          chip.target = "_blank";
          chip.rel = "noopener";
          wrap.appendChild(chip);
          return;
        }
        // 图片元素
        const img = document.createElement("img");
        // 图片来源
        img.src = item.dataUrl;
        // 图片说明
        img.alt = item.name || "已上传图片";
        // 提示可点击放大
        img.title = "点击查看大图";
        // 点击打开大图
        img.addEventListener("click", () => openLightbox(item.dataUrl));
        // 加入容器
        wrap.appendChild(img);
      });
      // 附件行加入消息行
      row.appendChild(wrap);
    }
    // 有文字才显示文字气泡；纯图片/文件上传不显示气泡
    if (text) {
      // 气泡文字
      bubble.textContent = text;
      // 加入消息行
      row.append(bubble);
    } else if (!images.length) {
      // 既没文字也没附件：保留空气泡占位
      row.append(bubble);
    }
  }

  // 追加到消息列表
  messageList.appendChild(row);
  // 滚动到底部
  scrollToLatest();
  // 返回气泡元素
  return bubble;
}

// 创建加载中的气泡
function createLoadingMessage() {
  // 先建空助手气泡
  const bubble = createMessage("assistant", "");
  // 三个小圆点
  const dots = document.createElement("span");
  // 圆点样式
  dots.className = "loading-dots";
  // 无障碍说明
  dots.setAttribute("aria-label", "正在生成回答");
  // 内嵌三个点
  dots.innerHTML = "<i></i><i></i><i></i>";
  // 放进气泡
  bubble.appendChild(dots);
  // 返回气泡元素
  return bubble;
}

// 添加一个标签
function addChip(container, text, className = "") {
  // 空文字不添加
  if (!text) return;
  // 三个小圆点
  const chip = document.createElement("span");
  // 标签样式
  chip.className = "meta-chip " + className;
  // 标签文字
  chip.textContent = text;
  // 加入容器
  container.appendChild(chip);
}

// 在答案后追加元信息标签
function appendAnswerMeta(bubble, data) {
  // 元信息容器
  const meta = document.createElement("div");
  // 元信息容器
  meta.className = "answer-meta";
  // 有命中数时
  if (data.result_count != null) {
    // 显示命中记录数
    addChip(meta, `命中 ${data.result_count} 条记录`);
  }
  // 实体名逐个加标签
  (data.entities || []).forEach((entity) => {
    addChip(meta, entity.name, "");
  });
  // 有引擎统计时
  if (data.engine_result_counts) {
    // 有引擎统计时
    const counts = data.engine_result_counts || {};
    // 引擎状态
    const status = data.engine_status || {};
    // 遍历检索引擎
    ["vector", "graph"].forEach((name) => {
      const count = counts[name] != null ? counts[name] : 0;
      // 引擎状态对象
      const st = status[name] || {};
      const registered = Object.prototype.hasOwnProperty.call(status, name);
      // 引擎中文名
      const label = ENGINE_LABELS[name] || name;
      // 引擎不可用显示未连接
      if (!registered || st.available === false) {
        addChip(meta, `${label} · 未连接`, "engine-down");
      // 有命中显示命中数
      } else if (count > 0) {
        // 命中数标签
        addChip(meta, `${label} · 命中 ${count} 条`, "engine-hit");
      } else {
        // 否则显示未命中
        addChip(meta, `${label} · 未命中`, "engine-miss");
      }
    });
    // 联网搜索：只在本次真的开启了才显示（命中/未命中/失败）
    if (status && Object.prototype.hasOwnProperty.call(status, "web")) {
      const webStatus = status.web || {};
      const webCount = counts.web != null ? counts.web : 0;
      if (webStatus.available === false) {
        addChip(meta, "联网搜索 · 失败", "engine-down");
      } else if (webCount > 0) {
        addChip(meta, `联网搜索 · 命中 ${webCount} 条`, "engine-hit");
      } else {
        addChip(meta, "联网搜索 · 未命中", "engine-miss");
      }
    }
  } else {
    // 没有引擎统计时
    const metaSources = Array.isArray(data.sources)
      // 单个来源转数组
      ? data.sources
      // 单个来源转数组
      : data.source
        // 单个来源转数组
        ? [data.source]
        : [];
    // 逐个添加来源标签
    metaSources.forEach((source) => {
      addChip(meta, source, "source");
    });
  }
  // 元信息加入气泡
  bubble.appendChild(meta);
  // 参考来源：来源名 + 证据等级 + 溯源链接
  if (Array.isArray(data.references) && data.references.length) {
    const refBox = document.createElement("div");
    refBox.className = "answer-references";
    const refTitle = document.createElement("span");
    refTitle.className = "ref-title";
    refTitle.textContent = "参考来源";
    refBox.appendChild(refTitle);
    data.references.forEach((ref) => {
      const chip = document.createElement("a");
      chip.className = "ref-chip";
      if (ref.url) {
        chip.href = ref.url;
        chip.target = "_blank";
        chip.rel = "noopener";
        chip.title = "打开来源链接";
      } else {
        // 伪链接：点击弹出查看命中的来源数据
        chip.classList.add("pseudo");
        chip.href = "javascript:void(0)";
        chip.title = "查看来源数据：" + (ref.text || ref.name || "").slice(0, 50);
        chip.addEventListener("click", (event) => {
          event.preventDefault();
          openSourceViewer(ref);
        });
      }
      const icon = document.createElement("span");
      icon.textContent = ref.url ? "🔗" : "📄";
      const label = document.createElement("span");
      label.textContent = ref.evidence_level
        ? `${ref.name}（证据等级：${ref.evidence_level}）`
        : ref.name || "来源";
      chip.append(icon, label);
      refBox.appendChild(chip);
    });
    bubble.appendChild(refBox);
  }
  // 有错误时
  if (data.error) {
    // 错误提示元素
    const error = document.createElement("div");
    // 错误样式
    error.className = "error-note";
    // 错误文字
    error.textContent = "检索提示：" + data.error;
    // 加入气泡
    bubble.appendChild(error);
  }
}

// 渲染问答流式响应
async function renderChatStream(response, bubble) {
  // 浏览器不支持流式就报错
  if (!response.body) throw new Error("当前浏览器不支持流式响应");

  // 读取器
  const reader = response.body.getReader();
  // UTF-8 解码器
  const decoder = new TextDecoder("utf-8");
  // 答案文本节点
  const answerNode = document.createTextNode("");
  // 缓冲字符串
  let buffer = "";
  // 答案是否开始输出
  let answerStarted = false;
  // 流错误信息
  let streamError = "";
  // 汇总结果对象
  const result = {
    result_count: null,
    entities: [],
    sources: [],
    references: [],
    referenceDetails: [],
    error: "",
    engine_result_counts: null,
    engine_status: null,
  };

  // 处理单个事件
  const handleEvent = (event) => {
    // 状态事件且还没开始答案
    if (event.type === "status" && !answerStarted) {
      // 显示处理中文字
      bubble.textContent = `${event.message || "正在处理"}…`;
      // 提前结束
      return;
    }
    // 元信息事件
    if (event.type === "meta") {
      // 命中数
      result.result_count = event.result_count;
      // 实体列表
      result.entities = event.entities || [];
      // 错误信息
      result.error = event.error || "";
      // 引擎命中数
      result.engine_result_counts = event.engine_result_counts || null;
      // 引擎状态
      result.engine_status = event.engine_status || null;
      // 保存服务端会话 ID
      if (event.session_id) saveChatSessionId(event.session_id);
      // 保存历史对话 ID
      if (event.conversation_id) saveConversationId(event.conversation_id);
      // 提前结束
      return;
    }
    // 文字片段事件
    if (event.type === "token") {
      // 第一次收到文字
      if (!answerStarted) {
        // 标记已开始
        answerStarted = true;
        // 清空气泡换成文本节点
        bubble.replaceChildren(answerNode);
      }
      // 追加文字
      answerNode.appendData(event.text || "");
      // 滚动到底部
      scrollToLatest();
      // 提前结束
      return;
    }
    // 错误事件
    if (event.type === "error") {
      // 记录错误信息
      streamError = event.message || "流式回答生成失败";
    }
    if (event.type === "done") {
      // 收集参考来源（done 事件的 answer.references）
      if (event.answer && Array.isArray(event.answer.references)) {
        result.references = event.answer.references;
      }
      // 收集引用详情（done 事件的 reference_details，用于引用面板）
      if (Array.isArray(event.reference_details)) {
        result.referenceDetails = event.reference_details;
      }
    }
  };

  // 循环读取流
  while (true) {
    const { value, done } = await reader.read();
    // 解码新数据（stream 模式）
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    // 按换行切分
    const lines = buffer.split("\n");
    // 最后一段留到缓冲区
    buffer = done ? "" : lines.pop();
    // 逐行处理
    for (const line of lines) {
      // 非空行解析 JSON 并处理
      if (line.trim()) handleEvent(JSON.parse(line));
    }
    // 流结束
    if (done) {
      // 处理缓冲残余
      if (buffer.trim()) handleEvent(JSON.parse(buffer));
      break;
    }
  }

  // 第一次收到文字
  if (!answerStarted) {
    // 显示错误或兜底文案
    bubble.textContent = streamError || "当前知识图谱未返回答案。";
  // 还有错误时
  } else if (streamError) {
    // 有错误且已开始
    result.error = streamError;
  }
  // 有引用详情时渲染引用面板（并避免重复渲染旧标签）
  if (result.referenceDetails && result.referenceDetails.length) {
    renderReferencePanel(bubble, result.referenceDetails);
    result.references = [];
  }
  // 追加元信息
  appendAnswerMeta(bubble, result);
}

// Tavily 联网搜索按钮仅控制后续问答请求
webSearchBtn.addEventListener("click", () => {
  webSearchEnabled = !webSearchEnabled;
  webSearchBtn.classList.toggle("active", webSearchEnabled);
  webSearchBtn.setAttribute("aria-pressed", String(webSearchEnabled));
  webSearchBtn.title = webSearchEnabled
    ? "已开启：本次问答将在本地检索后追加 Tavily 联网搜索"
    : "已关闭：本次问答只使用本地检索";
});
/* ---------- 健康问答 ---------- */
// 发送健康问答
async function sendQuestion(rawQuestion) {
  // 问题去空白
  const question = (rawQuestion || "").trim();
  // 已选图片（最多 3 张）
  const images = qaAttachedImages.slice();
  // 是否带图片
  const hasImage = images.length > 0;
  // 提前结束
  if ((!question && !hasImage) || requesting) return;
  // 标记请求中
  requesting = true;
  // 联网纯文本请求可取消
  if (webSearchEnabled && !hasImage) {
    activeChatController = new AbortController();
    qaSendBtn.setAttribute("aria-label", "取消联网搜索");
    qaSendBtn.title = "取消当前联网搜索请求";
  }
  // 请求处理中禁用发送按钮（联网纯文本请求保留为取消按钮）
  qaSendBtn.disabled = !(webSearchEnabled && !hasImage);
  // 切到对话布局
  pageShell.classList.add("chatting");
  // 显示对话区域
  conversation.hidden = false;
  // 先插入用户消息（带图）
  createMessage("user", question, images.map((img) => img.dataUrl));

  // 清空输入框
  qaInput.value = "";
  // 重置高度
  autoResize();

  // 当前答案气泡（用于错误兜底）
  let currentBubble = null;
  try {
    if (!hasImage) {
      // 创建加载气泡
      currentBubble = createLoadingMessage();
      // 带登录凭证请求问答接口
      const response = await authFetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: activeChatController ? activeChatController.signal : undefined,
        // 请求体
        body: JSON.stringify({
          // 问题
          question,
          // 会话 ID
          session_id: chatSessionId,
          conversation_id: chatConversationId || null,
          web_search: webSearchEnabled,
        }),
      });
      // 响应异常时
      if (!response.ok) {
        // 解析 JSON
        const data = await response.json().catch(() => ({}));
        // 登录失效处理
        if (response.status === 401) {
          // 显示登录提示
          handleAuthError(currentBubble, data.detail || "登录状态已失效");
          // 提前结束
          return;
        }
        throw new Error(data.detail || `请求失败（${response.status}）`);
      }
      // 问答走流式渲染
      await renderChatStream(response, currentBubble);
    } else {
      // 多张图片：逐张单独发送
      for (const img of images) {
        // 每张图一个加载气泡
        currentBubble = createLoadingMessage();
        // 带登录凭证请求图片接口
        const response = await authFetch("/api/analyze-image", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          // 请求体
          body: JSON.stringify({
            // 图片接口请求体
            image: img.b64,
            filename: img.name,
            // 问题
            question,
            conversation_id: chatConversationId || null,
          }),
        });
        // 响应异常时
        if (!response.ok) {
          // 解析 JSON
          const data = await response.json().catch(() => ({}));
          // 登录失效处理
          if (response.status === 401) {
            // 显示登录提示
            handleAuthError(currentBubble, data.detail || "登录状态已失效");
            // 提前结束
            return;
          }
          throw new Error(data.detail || `请求失败（${response.status}）`);
        }
        // 解析 JSON
        const data = await response.json();
        // 保存历史会话 ID
        if (data.conversation_id) saveConversationId(data.conversation_id);
        // 显示答案
        currentBubble.textContent = data.answer || "当前服务未返回答案。";
        // 追加元信息
        appendAnswerMeta(currentBubble, data);
      }
    }
  // 捕获异常
  } catch (error) {
    // 用户主动取消联网请求时不显示网络异常
    if (error.name === "AbortError") {
      const bubble = currentBubble || createLoadingMessage();
      bubble.textContent = "已取消本次联网搜索。";
      return;
    }
    // 复用当前气泡，没有则新建
    const bubble = currentBubble || createLoadingMessage();
    // 显示失败文案
    bubble.textContent = "暂时无法完成检索。";
    // 错误提示元素
    const note = document.createElement("div");
    // 错误样式
    note.className = "error-note";
    // 错误内容
    note.textContent = error.message || "请检查后端服务与 Neo4j 连接。";
    bubble.appendChild(note);
  // 无论成败都执行
  } finally {
    // 统一清理已选图片并关闭上传面板
    resetAttachImage();
    activeChatController = null;
    qaSendBtn.setAttribute("aria-label", "发送问题");
    qaSendBtn.title = "发送问题";
    // 取消请求中
    requesting = false;
    // 恢复按钮
    qaSendBtn.disabled = false;
    // 聚焦问答输入框
    qaInput.focus();
    // 滚动到底部
    scrollToLatest();
    // 刷新历史列表
    loadHistory();
  }
}

// 表单提交发送问题
qaForm.addEventListener("submit", (event) => {
  // 阻止默认提交
  event.preventDefault();
  // 联网请求进行中时点击发送按钮即取消
  if (requesting && activeChatController) {
    activeChatController.abort();
    return;
  }
  // 发送
  sendQuestion(qaInput.value);
});

// 输入时自动调整高度
qaInput.addEventListener("input", autoResize);
// 回车发送
qaInput.addEventListener("keydown", (event) => {
  // 非 Shift 回车
  if (event.key === "Enter" && !event.shiftKey) {
    // 阻止默认提交
    event.preventDefault();
    // 发送
    sendQuestion(qaInput.value);
  }
});

// 附件上传：格式化文件大小
function formatFileSize(bytes) {
  if (!bytes) return "";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// 显示上传面板（带弹出动画）
function showAttachPanel() {
  if (!attachPanel) return;
  attachPanel.hidden = false;
  // 重新触发弹出动画
  attachPanel.classList.remove("attach-pop");
  void attachPanel.offsetWidth;
  attachPanel.classList.add("attach-pop");
}

// 关闭上传面板（保留已选图片）
function hideAttachPanel() {
  if (!attachPanel) return;
  attachPanel.hidden = true;
}

// 渲染已选图片缩略图
function renderAttachThumbs() {
  if (!attachThumbs || !attachReady) return;
  attachThumbs.innerHTML = "";
  qaAttachedImages.forEach((img, index) => {
    const tile = document.createElement("div");
    tile.className = "attach-tile";
    const thumb = document.createElement("img");
    thumb.src = img.dataUrl;
    thumb.alt = img.name;
    thumb.title = `${img.name}（${formatFileSize(img.size)}）`;
    thumb.addEventListener("click", () => openLightbox(img.dataUrl));
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "attach-tile-rm";
    rm.textContent = "✕";
    rm.title = "移除这张图片";
    rm.addEventListener("click", (event) => {
      event.stopPropagation();
      removeAttachImage(index);
    });
    tile.append(thumb, rm);
    attachThumbs.appendChild(tile);
  });
  // 已就绪状态
  attachReady.hidden = qaAttachedImages.length === 0;
  attachReady.textContent = `已就绪 ✓（${qaAttachedImages.length} 张）`;
  // 添加按钮：满 3 张隐藏
  if (attachAddTile) attachAddTile.hidden = qaAttachedImages.length >= 3;
}

// 判断是否为图片（按 MIME 或扩展名，兼容部分浏览器不带类型的情况）
function isImageFile(file) {
  const name = (file.name || "").toLowerCase();
  return (
    (file.type && file.type.startsWith("image/")) ||
    /\.(jpe?g|png|webp|gif|bmp)$/.test(name)
  );
}

// 加入选中的图片（最多 3 张）
function addAttachFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const files = Array.from(fileList).filter(isImageFile);
  if (!files.length) return;
  const room = 3 - qaAttachedImages.length;
  if (room <= 0) {
    showAttachHint();
    return;
  }
  const picked = files.slice(0, room);
  if (files.length > room) showAttachHint();
  picked.forEach(readAttachFile);
}

// 短暂提示最多 3 张
let attachHintTimer = null;
function showAttachHint() {
  if (!attachHint) return;
  attachHint.hidden = false;
  clearTimeout(attachHintTimer);
  attachHintTimer = setTimeout(() => {
    attachHint.hidden = true;
  }, 2000);
}

// 读取单个文件加入列表
function readAttachFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = reader.result;
    qaAttachedImages.push({
      b64: dataUrl.split(",")[1] || "",
      name: file.name || "photo.jpg",
      size: file.size,
      dataUrl,
    });
    qaAttachBtn.classList.add("active");
    renderAttachThumbs();
    qaInput.focus();
  };
  reader.readAsDataURL(file);
}

// 移除单张图片
function removeAttachImage(index) {
  qaAttachedImages.splice(index, 1);
  if (!qaAttachedImages.length) qaAttachBtn.classList.remove("active");
  renderAttachThumbs();
}

// 统一清理：清空全部已选图片、关闭面板、取消高亮
function resetAttachImage() {
  qaAttachedImages = [];
  qaAttachBtn.classList.remove("active");
  qaImageInput.value = "";
  hideAttachPanel();
  if (attachReady) attachReady.hidden = true;
  if (attachAddTile) attachAddTile.hidden = false;
}

// 附件按钮点击：每次点开都重新开始，不带上一次的图片
qaAttachBtn.addEventListener("click", () => {
  // 点击按压反馈
  qaAttachBtn.classList.add("btn-press");
  setTimeout(() => qaAttachBtn.classList.remove("btn-press"), 180);
  // 清空上一次的图片，重新开始
  resetAttachImage();
  // 打开上传面板
  showAttachPanel();
  // 弹出文件选择框
  qaImageInput.click();
});

// 面板 ✕：关闭并清除全部图片
attachPanelClose.addEventListener("click", () => {
  resetAttachImage();
  qaInput.focus();
});

// “+” 继续添加图片
attachAddTile.addEventListener("click", () => {
  qaImageInput.click();
});

// 拖拽上传：拖到面板或输入框上都行
function handleAttachDrop(event) {
  event.preventDefault();
  const files = event.dataTransfer && event.dataTransfer.files;
  if (!files || !files.length) return;
  // 显示面板并加入拖进来的图片
  showAttachPanel();
  addAttachFiles(files);
}

// 上传面板拖拽高亮
attachPanel.addEventListener("dragover", (event) => {
  event.preventDefault();
  attachPanel.classList.add("dragover");
});
attachPanel.addEventListener("dragleave", () => {
  attachPanel.classList.remove("dragover");
});
attachPanel.addEventListener("drop", (event) => {
  attachPanel.classList.remove("dragover");
  handleAttachDrop(event);
});

// 输入框拖拽（面板没打开时也能直接拖图）
qaForm.addEventListener("dragover", (event) => {
  event.preventDefault();
  qaForm.classList.add("dragover");
});
qaForm.addEventListener("dragleave", () => {
  qaForm.classList.remove("dragover");
});
qaForm.addEventListener("drop", (event) => {
  qaForm.classList.remove("dragover");
  handleAttachDrop(event);
});

// 选择图片后
qaImageInput.addEventListener("change", (e) => {
  // 先把文件复制成数组（清空选择框后 FileList 会失效，必须先复制）
  const files = Array.from(e.target.files || []);
  // 清空选择框
  e.target.value = "";
  // 显示面板并加入图片
  showAttachPanel();
  addAttachFiles(files);
});

/* ---------- 清空对话 / 快捷问题 ---------- */
// 清空当前对话
function clearChat() {
  // 记住旧会话 ID
  const previousSessionId = chatSessionId;
  // 换一个新的会话 ID
  saveChatSessionId(createChatSessionId());
  // 清空历史对话 ID
  saveConversationId("");
  // 请求后端删除旧会话记忆
  authFetch(`/api/chat/memory/${encodeURIComponent(previousSessionId)}`, {
    method: "DELETE",
  }).catch(() => {});
  // 清空消息列表
  messageList.innerHTML = "";
  // 隐藏对话区
  conversation.hidden = true;
  // 回到欢迎布局
  pageShell.classList.remove("chatting");
  // 清空输入框
  qaInput.value = "";
  // 统一清理已选图片并隐藏预览条
  resetAttachImage();
  // 重置高度
  autoResize();
  // 刷新历史列表
  loadHistory();
  qaInput.focus();
}

// 清空按钮点击
$("clearChatBtn").addEventListener("click", clearChat);

// 快捷问题按钮绑定
document.querySelectorAll("[data-question]").forEach((button) => {
  // 点击直接发问题
  button.addEventListener("click", () => sendQuestion(button.dataset.question));
});

// 页面加载时检查健康
checkHealth();
// 聚焦问答输入框
qaInput.focus();

/* ---------- 登录状态：请求带凭证、顶栏用户菜单、401 处理 ---------- */
// 登录 token 存储键
const AUTH_TOKEN_KEY = "medrag_token";
// 用户信息存储键
const AUTH_USER_KEY = "medrag_user";

// 读取登录 token
function getAuthToken() {
  try {
    // 从 localStorage 取
    return localStorage.getItem(AUTH_TOKEN_KEY) || "";
  } catch (_) {
    // 取不到返回空串
    return "";
  }
}

// 清除本地登录信息
function clearAuth() {
  try {
    // 删除 token
    localStorage.removeItem(AUTH_TOKEN_KEY);
    // 删除用户信息
    localStorage.removeItem(AUTH_USER_KEY);
  } catch (_) {
    // Storage may be disabled.
  }
}

// 带凭证的请求封装
async function authFetch(url, options = {}) {
  // 取 token
  const token = getAuthToken();
  // 没有 token 直接请求
  if (!token) return fetch(url, options);
  // 构造请求头
  const headers = new Headers(options.headers || {});
  // 加上登录头
  headers.set("Authorization", "Bearer " + token);
  // 发起请求
  return fetch(url, { ...options, headers });
}

// 显示登录提示
function handleAuthError(bubble, message) {
  // 提前结束
  if (!bubble) return;
  // 替换气泡文字
  bubble.textContent = message;
  // 错误提示元素
  const note = document.createElement("div");
  // 错误样式
  note.className = "error-note";
  // 提示文字
  note.append("请登录后继续使用：");
  const link = document.createElement("a");
  // 登录链接
  link.href = "/login";
  // 链接文字
  link.textContent = "去登录";
  // 加入提示
  note.appendChild(link);
  // 加入气泡
  bubble.appendChild(note);
}

// 渲染侧边栏用户卡片
function renderUserChip(user) {
  // 记录当前用户
  currentUser = user || null;
  const avatar = $("railUserAvatar");
  const name = $("railUserName");
  const sub = $("railUserSub");
  const logout = $("railLogoutBtn");
  // 提前结束
  if (!avatar || !name || !sub || !logout) return;
  // 已登录
  if (currentUser) {
    // 显示名
    const display = currentUser.display_name || currentUser.username || "已登录";
    // 头像取第一个字
    avatar.textContent = (display.trim().charAt(0) || "用").toUpperCase();
    // 用户名
    name.textContent = display;
    // 账号
    sub.textContent = currentUser.username || "";
    // 显示退出按钮
    logout.hidden = false;
  } else {
    // 头像取第一个字
    avatar.textContent = "未";
    // 文字“未登录”
    name.textContent = "未登录";
    // 提示点击登录
    sub.textContent = "点击登录";
    // 隐藏退出按钮
    logout.hidden = true;
  }
}

// 刷新用户状态
async function refreshUserState() {
  try {
    // 请求当前用户
    const response = await authFetch("/api/auth/me", { cache: "no-store" });
    // 登录失效处理
    if (response.status === 401) {
      // 未登录：跳转到登录页，必须先登录才能进入主页
      window.location.href = "/login";
      // 提前结束
      return;
    }
    // 解析 JSON
    const data = await response.json().catch(() => ({}));
    // 有用户就渲染用户
    renderUserChip(data.authenticated && data.user ? data.user : null);
  } catch (_) {
    // 渲染未登录
    renderUserChip(null);
  }
}

// 绑定退出按钮
function bindLogout() {
  const logout = $("railLogoutBtn");
  // 提前结束
  if (!logout) return;
  // 点击退出
  logout.addEventListener("click", async (event) => {
    // 阻止冒泡到用户卡片
    event.stopPropagation();
    try {
      // 调用登出接口
      await authFetch("/api/auth/logout", { method: "POST" });
    } catch (_) {
      // 即使服务端注销失败，也清除本地凭证。
    }
    // 清除本地凭证
    clearAuth();
    // 跳转登录页
    window.location.href = "/login";
  });
}

/* ---------- 历史记录：加载 / 打开 / 清空 ---------- */
// 历史类型图标配置
const HISTORY_TYPE_META = {
  // 问答类型
  qa: { icon: "问", cls: "blue", label: "问答" },
  // 图片类型
  image: { icon: "图", cls: "violet", label: "图片" },
};

// 格式化历史时间
function formatHistoryTime(ts) {
  // 取不到返回空串
  if (!ts) return "";
  // 时间戳转日期
  const date = new Date(ts * 1000);
  // 当前时间
  const now = new Date();
  // 补零函数
  const pad = (n) => String(n).padStart(2, "0");
  // 时分
  const hm = `${pad(date.getHours())}:${pad(date.getMinutes())}`;
  // 今天显示时间
  if (date.toDateString() === now.toDateString()) return `今天 ${hm}`;
  // 昨天计算
  const yesterday = new Date(now.getTime() - 86400000);
  // 昨天显示昨天
  if (date.toDateString() === yesterday.toDateString()) return `昨天 ${hm}`;
  // 更早显示月日
  return `${date.getMonth() + 1}月${date.getDate()}日`;
}

// 渲染历史列表
function renderHistory(items) {
  const list = $("historyList");
  const countEl = $("historyCount");
  // 提前结束
  if (!list || !countEl) return;
  // 更新数量
  countEl.textContent = items.length;
  // 清空列表
  list.innerHTML = "";
  // 没有记录
  if (!items.length) {
    // 空状态元素
    const empty = document.createElement("div");
    // 空状态元素
    empty.className = "empty-history";
    // 空状态提示
    empty.innerHTML =
      '<span class="empty-ico">🗂️</span>还没有历史记录<br />开始第一次问答吧';
    // 加入列表
    list.appendChild(empty);
    // 提前结束
    return;
  }
  // 逐个分类
  items.forEach((item) => {
    // 取类型配置（默认问答）
    const meta = HISTORY_TYPE_META[item.type] || HISTORY_TYPE_META.qa;
    // 条目按钮
    const row = document.createElement("button");
    // 按钮类型
    row.type = "button";
    // 条目样式
    row.className =
      "history-item" +
      // 当前会话高亮
      (String(item.id) === String(chatConversationId) ? " active" : "");
    // 内嵌结构：图标+文字+时间
    row.innerHTML = `
      <span class="h-icon ${meta.cls}">${meta.icon}</span>
      <span class="h-body">
        <span class="h-question"></span>
        <span class="h-meta">
          ${formatHistoryTime(item.updated_at)}
          <span class="h-type ${meta.cls}">${meta.label}</span>
        </span>
      </span>`;
    // 标题文字
    row.querySelector(".h-question").textContent = item.title || "新对话";
    // 单条删除按钮
    const del = document.createElement("button");
    // 按钮类型
    del.type = "button";
    // 删除按钮样式
    del.className = "h-delete";
    // 删除文字
    del.textContent = "✕";
    // 悬停提示
    del.title = "删除此条记录";
    // 点击删除（不触发打开对话）
    del.addEventListener("click", (event) => {
      // 阻止冒泡，避免打开该会话
      event.stopPropagation();
      // 调用单删逻辑
      deleteHistoryItem(item.id);
    });
    // 加入条目
    row.appendChild(del);
    // 点击打开该会话
    row.addEventListener("click", () => openHistory(item.id));
    // 加入列表
    list.appendChild(row);
  });
}

// 删除单条历史记录
async function deleteHistoryItem(conversationId) {
  // 确认弹窗
  if (!window.confirm("确定删除这条历史记录吗？")) return;
  try {
    // 调用单删接口
    const response = await authFetch(
      `/api/history/${encodeURIComponent(conversationId)}`,
      { method: "DELETE" }
    );
    // 非 401 失败返回
    if (!response.ok && response.status !== 401) return;
  } catch (_) {
    // 服务异常也继续刷新视图
  }
  // 删除的是当前打开的对话时，回到欢迎页
  if (String(conversationId) === String(chatConversationId)) {
    // 清空当前视图
    clearChat();
  } else {
    // 刷新历史列表
    loadHistory();
  }
}

// 刷新历史列表
async function loadHistory() {
  try {
    // 请求历史接口
    const response = await authFetch("/api/history", { cache: "no-store" });
    // 提前结束
    if (response.status === 401) return;
    // 解析 JSON
    const data = await response.json().catch(() => ({}));
    // 渲染列表
    renderHistory(data.items || []);
  } catch (_) {
    // 服务暂不可用时保持现有列表。
  }
}

// 打开某条历史会话
async function openHistory(conversationId) {
  try {
    const response = await authFetch(
      `/api/history/${encodeURIComponent(conversationId)}`,
      { cache: "no-store" }
    );
    // 提前结束
    if (!response.ok) return;
    // 解析 JSON
    const data = await response.json();
    // 保存为当前会话
    saveConversationId(conversationId);
    // 清空消息列表
    messageList.innerHTML = "";
    // 逐条渲染历史消息
    (data.messages || []).forEach((message) => {
      const bubble = createMessage(
        // 助手消息带头像
        message.role === "assistant" ? "assistant" : "user",
        message.content || "",
        // 附件（上传的图片/PDF），打开历史时原样显示
        message.images || []
      );
      // 助手消息带头像
      if (message.role === "assistant") {
        // 追加元信息
        appendAnswerMeta(bubble, { sources: ["历史记录"] });
      }
    });
    // 显示对话区域
    conversation.hidden = false;
    // 切到对话布局
    pageShell.classList.add("chatting");
    // 刷新历史列表
    loadHistory();
    // 滚动到底部
    scrollToLatest();
    // 聚焦问答输入框
    qaInput.focus();
  } catch (_) {
    // 打开失败时保持当前页面。
  }
}

// 绑定历史清空按钮
function bindHistoryClear() {
  const btn = $("historyClearBtn");
  // 提前结束
  if (!btn) return;
  // 点击清空
  btn.addEventListener("click", async () => {
    // 提前结束
    if (!window.confirm("确定清空全部历史记录吗？")) return;
    try {
      // 请求历史接口
      const response = await authFetch("/api/history", { method: "DELETE" });
      // 提前结束
      if (!response.ok && response.status !== 401) return;
    } catch (_) {
      // 服务异常也继续清空本地视图。
    }
    // 清空当前视图
    clearChat();
  });
}

/* ---------- 新增对话 / 侧边栏收起 ---------- */
// 新增对话
function newChat() {
  // 清空当前视图
  clearChat();
}

// 侧边栏收起状态存储键
const SIDEBAR_COLLAPSED_KEY = "medrag_sidebar_collapsed";

// 是否已收起
function isSidebarCollapsed() {
  try {
    // 读 localStorage
    return localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  } catch (_) {
    // 异常返回展开
    return false;
  }
}

// 设置侧边栏收起状态
function setSidebarCollapsed(collapsed) {
  // 给 body 加/去类名
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  const btn = $("railCollapseBtn");
  // 存在时更新图标
  if (btn) {
    // 按钮提示（收起时按钮隐藏，用 🗂️ 展开）
    btn.title = collapsed ? "展开侧栏" : "收起侧栏";
  }
  try {
    // 记住选择
    localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0");
  } catch (_) {
    // Storage may be disabled.
  }
}

// 展开侧边栏
function expandSidebar() {
  // 设为展开
  setSidebarCollapsed(false);
}

// 绑定侧边栏控件
function bindRailControls() {
  // 侧栏右上角：收起按钮
  const collapseBtn = $("railCollapseBtn");
  if (collapseBtn) {
    collapseBtn.addEventListener("click", () => {
      setSidebarCollapsed(true);
    });
  }
  // 侧栏内：开启新对话
  const newChatBtn = $("newChatBtn");
  if (newChatBtn) {
    newChatBtn.addEventListener("click", newChat);
  }
  // 左上角按钮：显示/隐藏侧栏
  const sidebarToggleBtn = $("sidebarToggleBtn");
  if (sidebarToggleBtn) {
    sidebarToggleBtn.addEventListener("click", () => {
      setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed"));
    });
  }
  // 左上角按钮：新增对话
  const sidebarNewBtn = $("sidebarNewBtn");
  if (sidebarNewBtn) {
    sidebarNewBtn.addEventListener("click", newChat);
  }

  const railUser = $("railUser");
  // 存在时绑定
  if (railUser) {
    // 点击未登录跳登录页
    railUser.addEventListener("click", () => {
      // 跳转登录页
      if (!currentUser) window.location.href = "/login";
    });
    // 回车同样跳转
    railUser.addEventListener("keydown", (event) => {
      // 跳转登录页
      if (event.key === "Enter" && !currentUser) window.location.href = "/login";
    });
  }
}

// 当前用户变量
let currentUser = null;
// 初始化用户状态
refreshUserState();
// 绑定退出
bindLogout();
// 绑定历史清空
bindHistoryClear();
// 绑定侧边栏控件
bindRailControls();
// 恢复收起状态
setSidebarCollapsed(isSidebarCollapsed());
// 刷新历史列表
loadHistory();
