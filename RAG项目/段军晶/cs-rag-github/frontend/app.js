/* ============================================================================
   智查 AI —— 前端交互逻辑

   改造要点（相对旧版）：
     1. 真实账号登录：账号存 MySQL（users 表），登录态存 localStorage；
        未登录时强制显示登录界面，登录后才能提问。
     2. 三栏布局：左＝历史记录 / 我的收藏，中＝问答（含身份切换），右＝溯源面板。
     3. 身份切换（学生 / 职场）：只把 role 传给后端，
        由后端在生成答案那一步替换 system 提示词，前端不了解提示词内容。
     4. 历史记录：按会话从 MySQL 读取，打开旧会话会把上下文回灌到后端，
        可以直接接着往下问。
     5. 收藏：一问一答（含溯源）整条收藏，可收藏 / 取消。

   ★ 溯源面板只展示：文档名称、来源段落、页码 ★
     不展示向量分数、重排得分、trace、chunk_id 等任何内部细节。
     历史记录接口返回的溯源数据里本身也已经不含这些字段。
   ============================================================================ */

(function () {
  'use strict';

  // ---------------------------------------------------------------- 接口地址
  const API = {
    login:     '/api/auth/login',
    me:        '/api/auth/me',
    logout:    '/api/auth/logout',
    query:     '/api/chat/query',
    stream:    '/api/chat/stream',
    kbDocState: '/api/kb/status',
    sessions:  '/api/history/sessions',
    messages:  '/api/history/messages',
    favorites: '/api/favorite/list',
    favorite:  '/api/favorite',
    favIds:    '/api/favorite/ids'
  };

  const STORE_USER = 'zhicha.user';
  const STORE_ROLE = 'zhicha.role';
  const STORE_PANEL = 'zhicha.panel';
  const STORE_AVATAR = 'zhicha.avatar';

  const ROLE_LABEL = { student: '学生身份', workplace: '职场身份' };
  const ROLE_DESC = {
    student: '轻松易懂，像学长讲题',
    workplace: '严谨权威，像合规顾问'
  };
  const NL = String.fromCharCode(10);

  // ---------------------------------------------------------------- 元素引用
  const el = {
    layout:        document.getElementById('layout'),
    userBox:       document.getElementById('userBox'),
    userName:      document.getElementById('userName'),
    userAvatar:    document.getElementById('userAvatar'),
    logoutBtn:     document.getElementById('logoutBtn'),
    welcomePage:   document.getElementById('welcomePage'),
    avatarPicker:  document.getElementById('avatarPicker'),
    avatarUploadBtn: document.getElementById('avatarUploadBtn'),
    avatarPreview: document.getElementById('avatarPreview'),
    avatarFile:    document.getElementById('avatarFile'),
    wpAccount:     document.getElementById('wpAccount'),
    wpEnter:       document.getElementById('wpEnter'),
    wpError:       document.getElementById('wpError'),

    tabHistory:    document.getElementById('tabHistory'),
    tabFavorite:   document.getElementById('tabFavorite'),
    historyList:   document.getElementById('historyList'),
    favoriteList:  document.getElementById('favoriteList'),
    newChatBtn:    document.getElementById('newChatBtn'),

    roleSwitch:    document.getElementById('roleSwitch'),
    roleDesc:      document.getElementById('roleDesc'),
    chat:          document.getElementById('chat'),
    welcome:       document.getElementById('welcome'),
    kbWarning:     document.getElementById('kbWarning'),
    examples:      document.getElementById('examples'),

    input:         document.getElementById('questionInput'),
    sendBtn:       document.getElementById('sendBtn'),
    composerTip:   document.getElementById('composerTip'),

    sourceBody:    document.getElementById('sourceBody'),
    sourcePanel:   document.getElementById('sourcePanel'),
    collapseBtn:   document.getElementById('collapsePanelBtn'),
    expandBtn:     document.getElementById('expandPanelBtn'),

    roleBar:       document.getElementById('roleBar'),
    composer:      document.getElementById('composer'),
    favBar:        document.getElementById('favBar'),
    favBarCount:   document.getElementById('favBarCount'),
    backToChatBtn: document.getElementById('backToChatBtn'),
    favView:       document.getElementById('favView'),
    favDetail:     document.getElementById('favDetail')
  };

  // ---------------------------------------------------------------- 页面状态
  const state = {
    user: null,             // {user_id, username, display_name, avatar}
    avatar: '',             // 头像：emoji 标识，或服务端返回的自定义头像地址
    avatarData: '',         // 从相册选中的图片（data URL，未登录时先在前端暂存）
    role: 'student',
    sessionId: null,        // 当前会话：提问时带回去，实现多轮上下文
    activeSessionId: null,  // 左侧高亮的会话
    favoriteIds: new Set(),
    favorites: [],          // 收藏列表缓存
    busy: false,
    view: 'chat',           // 'chat' 问答视图 / 'favorite' 收藏浏览视图
    panelBeforeFav: null    // 进入收藏视图前的溯源面板展开状态
  };

  // ================================================================ 工具函数
  function escapeHtml(text) {
    return String(text == null ? '' : text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function nl2br(text) {
    return escapeHtml(text).split(NL).join('<br>');
  }

  // 页码：单页「P.08」；连续「P.08-09」；不连续「P.08、11」
  function formatPages(pageNo, pageNums) {
    const pages = Array.isArray(pageNums) && pageNums.length
      ? pageNums.slice().sort(function (a, b) { return a - b; })
      : [pageNo];
    const pad = function (n) { return String(n).padStart(2, '0'); };
    if (pages.length === 1) return 'P.' + pad(pages[0]);
    const continuous = pages.every(function (p, i) { return i === 0 || p === pages[i - 1] + 1; });
    if (continuous) return 'P.' + pad(pages[0]) + '-' + pad(pages[pages.length - 1]);
    return 'P.' + pages.map(pad).join('、');
  }

  function fmtTime(value) {
    if (!value) return '';
    const d = new Date(value);
    if (isNaN(d.getTime())) return '';
    const pad = function (n) { return String(n).padStart(2, '0'); };
    return pad(d.getMonth() + 1) + '-' + pad(d.getDate()) + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
  }

  function nowTime() {
    const d = new Date();
    const pad = function (n) { return String(n).padStart(2, '0'); };
    return pad(d.getHours()) + ':' + pad(d.getMinutes());
  }

  function toast(message, kind) {
    const box = document.createElement('div');
    box.textContent = message;
    box.style.cssText = [
      'position:fixed', 'left:50%', 'bottom:88px', 'transform:translateX(-50%)',
      'z-index:80', 'padding:10px 20px', 'border-radius:12px',
      'font-size:13.5px', 'box-shadow:0 8px 22px rgba(150,120,40,.22)',
      'background:' + (kind === 'error' ? '#FBEFE9' : '#FFF3D4'),
      'color:' + (kind === 'error' ? '#A9663F' : '#8A6410'),
      'border:1px solid ' + (kind === 'error' ? '#F0D2C6' : '#F4DFA6')
    ].join(';');
    document.body.appendChild(box);
    setTimeout(function () {
      box.style.transition = 'opacity .25s';
      box.style.opacity = '0';
      setTimeout(function () { box.remove(); }, 260);
    }, 1900);
  }

  async function request(url, options) {
    const response = await fetch(url, options || {});
    let payload = null;
    try { payload = await response.json(); } catch (e) { payload = null; }
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : ('请求失败（HTTP ' + response.status + '）');
      throw new Error(detail);
    }
    return payload;
  }

  function jsonBody(data) {
    return {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    };
  }

  function uid() { return state.user ? state.user.user_id : ''; }

  // ================================================================ 登录
  function loadLocalUser() {
    try { state.user = JSON.parse(localStorage.getItem(STORE_USER) || 'null'); }
    catch (e) { state.user = null; }
    const savedRole = localStorage.getItem(STORE_ROLE);
    if (savedRole === 'workplace' || savedRole === 'student') state.role = savedRole;
    const savedAvatar = localStorage.getItem(STORE_AVATAR);
    if (savedAvatar) state.avatar = savedAvatar;
    if (state.user && state.user.avatar) state.avatar = state.user.avatar;
  }

  function avatarOf() {
    // 有自选头像就用自选头像，否则用账号首字兜底
    const name = (state.user && (state.user.display_name || state.user.username)) || '';
    return state.avatar || (name ? name.slice(0, 1) : '智');
  }

  function renderUser() {
    const logged = !!(state.user && state.user.user_id);
    el.userBox.hidden = !logged;
    if (logged) {
      const name = state.user.display_name || state.user.username || '用户';
      el.userName.textContent = name;
      if (state.avatar && state.avatar.indexOf('/avatars/') === 0) {
        el.userAvatar.innerHTML = '<img src="' + state.avatar + '" alt="头像">';
        el.userAvatar.classList.add('has-img');
        el.userAvatar.classList.remove('emoji');
      } else {
        el.userAvatar.textContent = avatarOf();
        el.userAvatar.classList.remove('has-img');
        el.userAvatar.classList.toggle('emoji', !!state.avatar);
      }
    }
    el.input.disabled = !logged;
    el.sendBtn.disabled = !logged;
    el.input.placeholder = logged
      ? '输入你的问题……（回车发送，Shift + 回车换行）'
      : '请先登录后提问';
  }

  // ---------------------------------------------------------- 欢迎页（独立页面）
  function renderAvatarPicker() {
    Array.prototype.slice.call(el.avatarPicker.querySelectorAll('.avatar-opt')).forEach(function (btn) {
      btn.classList.toggle('active', (btn.dataset.avatar || '') === (state.avatar || ''));
    });
  }

  // 圆形预览：优先显示从相册选的图片，其次是 emoji，最后用账号首字
  function renderAvatarPreview() {
    if (state.avatarData) {
      el.avatarPreview.innerHTML = '<img src="' + state.avatarData + '" alt="头像预览">';
      el.avatarPreview.classList.add('has-img');
      return;
    }
    el.avatarPreview.classList.remove('has-img');
    const account = (el.wpAccount.value || '').trim();
    el.avatarPreview.textContent = state.avatar || (account ? account.slice(0, 1) : '智');
  }

  function selectAvatar(value) {
    state.avatar = value || '';
    state.avatarData = '';           // 选了预设头像，就放弃之前从相册选的那张
    localStorage.setItem(STORE_AVATAR, state.avatar);
    renderAvatarPicker();
    renderAvatarPreview();
  }

  // 把相册里选的图片：居中裁成正方形 → 缩到 160×160 → 压成 JPEG
  // 这样做是为了让上传体积可控（一般 10KB 上下），也保证任何原图都能当头像用
  function fileToAvatarDataUrl(file) {
    return new Promise(function (resolve, reject) {
      if (!file) { reject(new Error('没有选择文件')); return; }
      if (file.type && file.type.indexOf('image/') !== 0) {
        reject(new Error('请选择图片文件（jpg / png / webp 等）'));
        return;
      }
      if (file.size > 8 * 1024 * 1024) {
        reject(new Error('图片太大了，请选 8MB 以内的图片'));
        return;
      }
      const reader = new FileReader();
      reader.onerror = function () { reject(new Error('图片读取失败，请换一张试试')); };
      reader.onload = function () {
        const img = new Image();
        img.onerror = function () { reject(new Error('这张图片无法解析，请换一张试试')); };
        img.onload = function () {
          try {
            const SIZE = 160;
            const canvas = document.createElement('canvas');
            canvas.width = SIZE;
            canvas.height = SIZE;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#ffffff';          // JPEG 不支持透明，先铺白底
            ctx.fillRect(0, 0, SIZE, SIZE);
            const side = Math.min(img.width, img.height);
            ctx.drawImage(
              img,
              (img.width - side) / 2, (img.height - side) / 2, side, side,
              0, 0, SIZE, SIZE
            );
            resolve(canvas.toDataURL('image/jpeg', 0.85));
          } catch (e) {
            reject(new Error('图片处理失败，请换一张试试'));
          }
        };
        img.src = reader.result;
      };
      reader.readAsDataURL(file);
    });
  }

  async function onAvatarFilePicked(file) {
    try {
      const dataUrl = await fileToAvatarDataUrl(file);
      state.avatarData = dataUrl;
      state.avatar = '';                 // 自定义图片优先，不再用预设 emoji
      localStorage.setItem(STORE_AVATAR, '');
      renderAvatarPicker();
      renderAvatarPreview();
      toast('头像已选好');
    } catch (err) {
      toast(err.message || '头像处理失败', 'error');
    }
  }

  function showWelcomePage(account, message) {
    el.welcomePage.hidden = false;
    el.wpAccount.value = account || (state.user ? state.user.username : '');
    // 自定义头像地址是按用户存的（/avatars/用户ID.jpg），换账号时要清掉，
    // 否则新账号会顶着一张别人的头像进来
    if (state.avatar && state.avatar.indexOf('/avatars/') === 0) {
      state.avatar = '';
      state.avatarData = '';
    }
    el.wpError.hidden = !message;
    if (message) el.wpError.textContent = message;
    renderAvatarPicker();
    renderAvatarPreview();
    setTimeout(function () { el.wpAccount.focus(); }, 80);
  }

  function hideWelcomePage() {
    el.welcomePage.hidden = true;
  }

  async function doLogin() {
    const name = (el.wpAccount.value || '').trim();
    if (!name) {
      el.wpError.textContent = '请先填一个账号';
      el.wpError.hidden = false;
      el.wpAccount.focus();
      return;
    }
    el.wpEnter.disabled = true;
    el.wpEnter.textContent = '正在进入…';
    try {
      const data = await request(API.login, jsonBody({
        username: name,
        avatar: state.avatar || '',
        avatar_image: state.avatarData || ''
      }));
      state.user = data;
      // 服务端返回的头像为准：可能是预设 emoji，也可能是自定义头像地址
      state.avatar = data.avatar || state.avatar || '';
      state.avatarData = '';
      localStorage.setItem(STORE_USER, JSON.stringify(data));
      localStorage.setItem(STORE_AVATAR, state.avatar);
      hideWelcomePage();
      renderUser();
      toast(data.created ? ('已为你创建账号：' + data.username) : ('欢迎回来，' + data.username));
      await refreshAll();
      el.input.focus();
    } catch (err) {
      el.wpError.textContent = err.message || '登录失败，请重试';
      el.wpError.hidden = false;
    } finally {
      el.wpEnter.disabled = false;
      el.wpEnter.textContent = '进入我的 AI 助手';
    }
  }

  // 切换账号：清理当前登录态，回到欢迎页（账号与头像预填，方便改一下再进）
  async function switchAccount() {
    const lastAccount = state.user ? state.user.username : '';
    try { await request(API.logout + '?user_id=' + encodeURIComponent(uid()), { method: 'POST' }); } catch (e) { /* 忽略 */ }
    state.user = null;
    state.sessionId = null;
    state.activeSessionId = null;
    state.favoriteIds = new Set();
    state.favorites = [];
    localStorage.removeItem(STORE_USER);
    if (state.view === 'favorite') leaveFavoriteView();
    clearChat();
    showWelcome();
    el.historyList.innerHTML = '';
    el.favoriteList.innerHTML = '';
    renderSources([]);
    renderUser();
    showWelcomePage(lastAccount);
  }

  // 用本地登录态回查一次，确认账号在数据库里真实存在（防止库被清后前端还"登录着"）
  async function verifyUser() {
    if (!uid()) return false;
    try {
      const me = await request(API.me + '?user_id=' + encodeURIComponent(uid()));
      state.user = me;
      state.avatar = me.avatar || state.avatar || '';
      localStorage.setItem(STORE_USER, JSON.stringify(me));
      localStorage.setItem(STORE_AVATAR, state.avatar);
      renderUser();
      return true;
    } catch (err) {
      state.user = null;
      localStorage.removeItem(STORE_USER);
      renderUser();
      return false;
    }
  }
  // ================================================================ 聊天区
  function scrollBottom() { el.chat.scrollTop = el.chat.scrollHeight; }
  function hideWelcome() { if (el.welcome) el.welcome.hidden = true; }
  function showWelcome() { if (el.welcome) el.welcome.hidden = false; }

  function clearChat() {
    Array.prototype.slice.call(el.chat.querySelectorAll('.qa-block')).forEach(function (n) { n.remove(); });
  }

  function autoResize() {
    if (!el.input) return;
    el.input.style.height = 'auto';
    el.input.style.height = Math.min(el.input.scrollHeight, 140) + 'px';
  }

  // 新建一组问答块（用户气泡 + AI 气泡 + 工具条）
  function appendQaBlock(question) {
    const block = document.createElement('div');
    block.className = 'qa-block';
    block.innerHTML =
      '<div class="msg user"><div class="bubble user-bubble">' + nl2br(question) + '</div></div>' +
      '<div class="msg ai">' +
        '<div class="ai-avatar">智</div>' +
        '<div class="ai-body">' +
          '<div class="bubble ai-bubble"><span class="thinking">正在检索知识库并生成答案</span></div>' +
          '<div class="msg-tools" hidden></div>' +
        '</div>' +
      '</div>';
    el.chat.appendChild(block);
    scrollBottom();
    return block;
  }

  // 渲染答案（实时问答与历史回放共用）
  function fillAnswer(block, data) {
    const bubble = block.querySelector('.ai-bubble');
    const roleKey = data.role === 'workplace' ? 'workplace' : 'student';
    const roleLabel = ROLE_LABEL[roleKey];
    const roleCls = roleKey === 'workplace' ? ' workplace' : '';
    const cacheTag = data.cached ? '<span class="cache-tag">缓存命中</span>' : '';
    const answerHtml = nl2br(data.answer || '（未返回内容）');

    bubble.innerHTML =
      '<div class="msg-meta-line">' +
        '<span class="role-tag' + roleCls + '">' + roleLabel + '</span>' + cacheTag +
      '</div>' +
      '<p class="answer-text">' + answerHtml + '</p>';

    const mid = data.message_id ? Number(data.message_id) : null;
    block.dataset.mid = mid ? String(mid) : '';

    const tools = block.querySelector('.msg-tools');
    tools.innerHTML = '';
    tools.hidden = false;

    // 收藏按钮：一问一答 + 溯源 整条收藏
    const favBtn = document.createElement('button');
    favBtn.type = 'button';
    favBtn.className = 'tool-btn';
    const faved = !!(mid && state.favoriteIds.has(mid));
    favBtn.textContent = faved ? '★ 已收藏' : '☆ 收藏';
    if (faved) favBtn.classList.add('fav-on');
    favBtn.addEventListener('click', function () { toggleFavorite(mid, favBtn); });
    tools.appendChild(favBtn);

    // 参考来源：把这条答案的溯源送进右侧面板
    const srcBtn = document.createElement('button');
    srcBtn.type = 'button';
    srcBtn.className = 'tool-btn';
    const srcCount = (data.sources || []).length;
    srcBtn.textContent = '参考来源' + (srcCount ? '（' + srcCount + '）' : '');
    srcBtn.addEventListener('click', function () {
      renderSources(data.sources || []);
      setPanelCollapsed(false);
    });
    tools.appendChild(srcBtn);

    const timeTag = document.createElement('span');
    timeTag.className = 'tool-time';
    timeTag.textContent = data.created_at ? fmtTime(data.created_at) : nowTime();
    tools.appendChild(timeTag);

    // 删除单条问答：仅历史记录里的问答可删（需要 message_id）
    if (mid) {
      const delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.className = 'tool-btn danger';
      delBtn.textContent = '删除';
      delBtn.addEventListener('click', function () { removeMessage(mid, block); });
      tools.appendChild(delBtn);
    }

    // 最新一条答案自动填进溯源面板
    renderSources(data.sources || []);
    scrollBottom();
  }

  function renderError(block, message) {
    const bubble = block.querySelector('.ai-bubble');
    if (bubble) {
      bubble.className = 'bubble msg-error';
      bubble.textContent = message;
    }
    scrollBottom();
  }

  // ================================================================ 溯源面板
  // ★ 只渲染：知识库文档名、来源章节、相关段落摘要、页码 ★
  // 不渲染 score / chunk_id / trace 等任何内部细节。
  function sourceCardsHtml(sources) {
    const list = Array.isArray(sources) ? sources : [];
    if (!list.length) return '';
    return list.map(function (s) {
      const page = formatPages(s.page_no, s.page_nums);
      const secHtml = s.section_title
        ? '<div class="src-sec">来源章节：' + escapeHtml(s.section_title) + '</div>' : '';
      const sumHtml = s.summary
        ? '<div class="src-sum">相关段落：' + escapeHtml(s.summary) + '</div>' : '';
      return '<div class="src-card">' +
        '<div class="src-row">' +
          '<span class="src-label">知识库文档</span>' +
          '<span class="src-page">' + page + '</span>' +
        '</div>' +
        '<div class="src-file">' + escapeHtml(s.file_name || '未命名文档') + '</div>' +
        secHtml + sumHtml +
      '</div>';
    }).join('');
  }

  function renderSources(sources) {
    const html = sourceCardsHtml(sources);
    el.sourceBody.innerHTML = html || '<div class="panel-empty">这条回答没有可展示的溯源信息</div>';
  }

  function setPanelCollapsed(collapsed) {
    el.layout.classList.toggle('panel-collapsed', !!collapsed);
    el.expandBtn.hidden = !collapsed;
    localStorage.setItem(STORE_PANEL, collapsed ? '1' : '0');
  }

  // ================================================================ 提问
  // 走流式接口：检索一完成先出溯源，随后答案逐字显示。
  // 若流式不可用（老浏览器 / 接口异常），自动回退到一次性接口，功能不受影响。

  function streamStatusHtml(roleKey, text) {
    const roleCls = roleKey === 'workplace' ? ' workplace' : '';
    return '<div class="msg-meta-line">' +
      '<span class="role-tag' + roleCls + '">' + ROLE_LABEL[roleKey] + '</span>' +
      '<span class="stream-status">' + escapeHtml(text) + '</span>' +
    '</div>';
  }

  // 流式过程中的答案渲染：正文 + 闪烁光标
  function renderStreamingAnswer(bubble, roleKey, statusText, answerText) {
    bubble.innerHTML =
      streamStatusHtml(roleKey, statusText) +
      '<p class="answer-text">' + nl2br(answerText) +
      '<span class="stream-cursor"></span></p>';
    scrollBottom();
  }

  async function askStreaming(question, block) {
    const bubble = block.querySelector('.ai-bubble');
    const startedAt = Date.now();
    let roleKey = state.role === 'workplace' ? 'workplace' : 'student';
    let answerText = '';
    let sources = [];
    let sessionId = state.sessionId;
    let messageId = null;
    let cached = false;
    let statusText = '正在检索知识库…';
    let sawDelta = false;
    let thinkingChars = 0;      // 模型思考进度（推理模型先"想"后"说"）
    const NL2 = NL + NL;

    // 等待期间每 0.5 秒刷新一次状态（让用户知道系统还在跑）
    const tick = setInterval(function () {
      const sec = Math.round((Date.now() - startedAt) / 1000);
      if (sawDelta) return;
      if (thinkingChars > 0) {
        // 推理模型正在思考：秒数 + 已推理字数一起跳，让等待有动静
        renderStreamingAnswer(
          bubble, roleKey,
          '模型正在思考… 已推理 ' + thinkingChars + ' 字（已 ' + sec + 's）',
          answerText
        );
      } else {
        renderStreamingAnswer(bubble, roleKey, statusText + '（已 ' + sec + 's）', answerText);
      }
    }, 500);
    renderStreamingAnswer(bubble, roleKey, statusText, answerText);

    try {
      const resp = await fetch(API.stream, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          session_id: state.sessionId,
          role: state.role,
          user_id: uid()
        })
      });
      if (!resp.ok || !resp.body || !resp.body.getReader) {
        throw new Error('流式接口不可用（HTTP ' + resp.status + '）');
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const piece = await reader.read();
        if (piece.done) break;
        buffer += decoder.decode(piece.value, { stream: true });

        let cut = buffer.indexOf(NL2);
        while (cut >= 0) {
          const rawEvent = buffer.slice(0, cut);
          buffer = buffer.slice(cut + NL2.length);
          const payload = rawEvent.split(NL)
            .filter(function (line) { return line.indexOf('data:') === 0; })
            .map(function (line) { return line.slice(5).trim(); })
            .join('');
          cut = buffer.indexOf(NL2);
          if (!payload) continue;

          let evt = null;
          try { evt = JSON.parse(payload); } catch (e) { continue; }
          if (!evt || !evt.type) continue;

          if (evt.type === 'meta') {
            sources = evt.sources || [];
            if (evt.role) roleKey = evt.role === 'workplace' ? 'workplace' : 'student';
            if (evt.session_id) {
              sessionId = evt.session_id;
              state.sessionId = evt.session_id;
              state.activeSessionId = evt.session_id;
            }
            cached = !!evt.cached;
            renderSources(sources);
            statusText = sources.length
              ? ('已命中 ' + sources.length + ' 处原文，正在生成答案…')
              : '正在生成答案…';
            renderStreamingAnswer(bubble, roleKey, statusText, answerText);
          } else if (evt.type === 'thinking') {
            thinkingChars = evt.chars || 0;
          } else if (evt.type === 'delta') {
            if (!sawDelta) sawDelta = true;
            answerText += (evt.text || '');
            renderStreamingAnswer(bubble, roleKey, '正在生成答案…', answerText);
          } else if (evt.type === 'saved') {
            messageId = evt.message_id || null;
          } else if (evt.type === 'error') {
            throw new Error(evt.detail || '问答服务处理失败');
          }
        }
      }
    } finally {
      clearInterval(tick);
    }

    if (!answerText.trim()) {
      throw new Error('未收到答案内容');
    }

    // 收尾：统一按最终态渲染（与历史回放完全一致，工具条/收藏按钮也就位了）
    if (sessionId) {
      state.sessionId = sessionId;
      state.activeSessionId = sessionId;
    }
    block.dataset.mid = messageId ? String(messageId) : '';
    fillAnswer(block, {
      answer: answerText,
      sources: sources,
      role: roleKey,
      message_id: messageId,
      cached: cached,
      latency_ms: Date.now() - startedAt
    });
  }

  // 回退方案：一次性接口（流式不可用时使用）
  async function askOnce(question, block) {
    const data = await request(API.query, jsonBody({
      question: question,
      session_id: state.sessionId,
      role: state.role,
      user_id: uid()
    }));
    if (data.session_id) {
      state.sessionId = data.session_id;
      state.activeSessionId = data.session_id;
    }
    fillAnswer(block, data);
  }

  async function ask(text) {
    if (state.busy) return;
    const question = (text || '').trim();
    if (!question) return;
    if (!uid()) { showWelcomePage('', '请先登录后再提问'); return; }

    state.busy = true;
    el.sendBtn.disabled = true;
    hideWelcome();
    el.input.value = '';
    autoResize();

    const block = appendQaBlock(question);
    try {
      await askStreaming(question, block);
    } catch (err) {
      // 流式失败（网络中断 / 浏览器不支持流读取）：回退到一次性接口再试一次
      try {
        await askOnce(question, block);
      } catch (err2) {
        renderError(block, err2.message || err.message || '提问失败，请稍后重试');
      }
    } finally {
      state.busy = false;
      el.sendBtn.disabled = false;
      el.input.focus();
      loadSessions();
    }
  }

  // ================================================================ 历史记录
  async function loadSessions() {
    if (!uid()) return;
    try {
      const data = await request(API.sessions + '?user_id=' + encodeURIComponent(uid()) + '&limit=100');
      state.sessions = data.sessions || [];
      renderSessions(state.sessions);
    } catch (err) {
      el.historyList.innerHTML = '<div class="side-empty">历史记录加载失败<br>' + escapeHtml(err.message) + '</div>';
    }
  }

  function renderSessions(list) {
    if (!list.length) {
      el.historyList.innerHTML = '<div class="side-empty">还没有历史记录<br>提问后会自动按会话保存到这里</div>';
      return;
    }
    el.historyList.innerHTML = list.map(function (s) {
      const sid = escapeHtml(s.session_id);
      const active = s.session_id === state.activeSessionId ? ' active' : '';
      return '<div class="hist-item' + active + '" data-sid="' + sid + '">' +
        '<div class="hist-title">' + escapeHtml(s.title || '（无标题会话）') + '</div>' +
        '<div class="hist-sub">' + (s.message_count || 0) + ' 条 · ' + fmtTime(s.updated_at) + '</div>' +
        '<button class="hist-del" data-del="' + sid + '" title="删除会话">×</button>' +
      '</div>';
    }).join('');
  }

  async function openSession(sid) {
    if (!sid) return;
    try {
      const data = await request(
        API.sessions + '/' + encodeURIComponent(sid) + '?user_id=' + encodeURIComponent(uid())
      );
      state.sessionId = sid;
      state.activeSessionId = sid;
      clearChat();
      hideWelcome();

      const messages = data.messages || [];
      if (!messages.length) showWelcome();
      messages.forEach(function (m) {
        const block = appendQaBlock(m.question);
        fillAnswer(block, {
          answer: m.answer,
          sources: m.sources,
          role: m.role,
          message_id: m.id,
          created_at: m.created_at,
          cached: false
        });
      });
      renderSessions(state.sessions || []);
      if (data.restored_turns) {
        toast('已恢复上下文 ' + data.restored_turns + ' 轮，可以直接接着问');
      }
      el.input.focus();
    } catch (err) {
      toast(err.message || '打开会话失败', 'error');
    }
  }

  async function confirmDeleteSession(sid) {
    if (!sid) return;
    if (!window.confirm('确定删除这个会话吗？会话内的全部问答都会被删除（已收藏的内容不受影响）。')) return;
    try {
      await request(
        API.sessions + '/' + encodeURIComponent(sid) + '?user_id=' + encodeURIComponent(uid()),
        { method: 'DELETE' }
      );
      if (state.activeSessionId === sid) {
        state.sessionId = null;
        state.activeSessionId = null;
        clearChat();
        showWelcome();
        renderSources([]);
      }
      await loadSessions();
      toast('会话已删除');
    } catch (err) {
      toast(err.message || '删除失败', 'error');
    }
  }

  async function removeMessage(mid, block) {
    if (!window.confirm('确定删除这条问答吗？（已收藏的内容不受影响）')) return;
    try {
      await request(
        API.messages + '/' + mid + '?user_id=' + encodeURIComponent(uid()),
        { method: 'DELETE' }
      );
      if (block && block.parentNode) block.remove();
      state.favoriteIds.delete(Number(mid));
      if (!el.chat.querySelector('.qa-block')) showWelcome();
      await loadSessions();
      toast('已删除这条问答');
    } catch (err) {
      toast(err.message || '删除失败', 'error');
    }
  }

  function newChat() {
    switchTab('history');
    state.sessionId = null;
    state.activeSessionId = null;
    clearChat();
    showWelcome();
    renderSources([]);
    renderSessions(state.sessions || []);
    el.input.focus();
    toast('已开始新对话（历史记录仍然保留）');
  }
  // ================================================================ 收藏
  function briefText(text, max) {
    const s = String(text == null ? '' : text).split(NL).join(' ').split('  ').join(' ').trim();
    return s.length > max ? (s.slice(0, max) + '…') : s;
  }

  async function loadFavorites() {
    if (!uid()) return;
    try {
      const data = await request(API.favorites + '?user_id=' + encodeURIComponent(uid()));
      state.favorites = data.items || [];
      renderFavorites(state.favorites);
    } catch (err) {
      el.favoriteList.innerHTML = '<div class="side-empty">收藏加载失败<br>' + escapeHtml(err.message) + '</div>';
    }
  }

  function renderFavorites(items) {
    if (!items.length) {
      el.favoriteList.innerHTML = '<div class="side-empty">还没有收藏<br>在答案下方点「☆ 收藏」，就会出现在这里</div>';
      return;
    }
    el.favoriteList.innerHTML = items.map(function (f) {
      const roleLabel = ROLE_LABEL[f.role] || ROLE_LABEL.student;
      const mid = f.message_id ? String(f.message_id) : '';
      return '<div class="fav-item" data-mid="' + mid + '">' +
        '<div class="fav-q">' + escapeHtml(briefText(f.question, 40)) + '</div>' +
        '<div class="fav-a">' + escapeHtml(briefText(f.answer, 58)) + '</div>' +
        '<div class="fav-meta">' +
          '<span>' + fmtTime(f.created_at) + ' · ' + roleLabel + '</span>' +
          '<span class="fav-unfav" data-unfav="' + mid + '">取消收藏</span>' +
        '</div>' +
      '</div>';
    }).join('');
  }

  async function loadFavoriteIds() {
    if (!uid()) return;
    try {
      const data = await request(API.favIds + '?user_id=' + encodeURIComponent(uid()));
      state.favoriteIds = new Set((data.message_ids || []).map(Number));
      refreshFavoriteButtons();
    } catch (err) { /* 静默：收藏状态拉取失败不影响问答 */ }
  }

  function refreshFavoriteButtons() {
    Array.prototype.slice.call(el.chat.querySelectorAll('.qa-block')).forEach(function (block) {
      const mid = Number(block.dataset.mid || 0);
      if (!mid) return;
      const btn = block.querySelector('.msg-tools .tool-btn');
      if (!btn) return;
      const on = state.favoriteIds.has(mid);
      btn.classList.toggle('fav-on', on);
      btn.textContent = on ? '★ 已收藏' : '☆ 收藏';
    });
  }

  async function toggleFavorite(messageId, btn) {
    if (!messageId) {
      toast('这条答案还在生成中或未保存，稍后再试', 'error');
      return;
    }
    const mid = Number(messageId);
    const faved = state.favoriteIds.has(mid);
    btn.disabled = true;
    try {
      if (faved) {
        await request(API.favorite + '/' + mid + '?user_id=' + encodeURIComponent(uid()), { method: 'DELETE' });
        state.favoriteIds.delete(mid);
        btn.classList.remove('fav-on');
        btn.textContent = '☆ 收藏';
        toast('已取消收藏');
      } else {
        await request(API.favorite, jsonBody({ user_id: uid(), message_id: mid }));
        state.favoriteIds.add(mid);
        btn.classList.add('fav-on');
        btn.textContent = '★ 已收藏';
        toast('已加入「我的收藏」');
      }
      await loadFavorites();
    } catch (err) {
      toast(err.message || '操作失败', 'error');
    } finally {
      btn.disabled = false;
    }
  }

  async function unfavorite(messageId) {
    const mid = Number(messageId);
    if (!mid) return;
    try {
      await request(API.favorite + '/' + mid + '?user_id=' + encodeURIComponent(uid()), { method: 'DELETE' });
      state.favoriteIds.delete(mid);
      refreshFavoriteButtons();
      const wasInFavView = state.view === 'favorite';
      await loadFavorites();
      if (wasInFavView) {
        const next = (state.favorites || [])[0];
        if (next) { renderFavoriteDetail(next); markActiveFavorite(next.message_id); }
        else { renderFavEmpty(); }
      }
      toast('已取消收藏');
    } catch (err) {
      toast(err.message || '操作失败', 'error');
    }
  }

  // ------------------------------------------------- 收藏浏览视图（独立页面）
  // 点「我的收藏」时，中间区域整块换成收藏详情页：
  //   直接展示 问题 + 答案 + 溯源，不再以问答气泡的形式呈现，
  //   同时隐藏身份切换与输入框（浏览模式，不是提问模式）；
  //   溯源已在这一页里，右侧溯源面板自动收起，避免同一份内容出现两次。
  function enterFavoriteView() {
    if (state.view === 'favorite') return;
    state.view = 'favorite';
    state.panelBeforeFav = el.layout.classList.contains('panel-collapsed');
    el.favBar.hidden = false;
    el.roleBar.hidden = true;
    el.composer.hidden = true;
    el.chat.hidden = true;
    el.favView.hidden = false;
    setPanelCollapsed(true);
  }

  function leaveFavoriteView() {
    if (state.view !== 'favorite') return;
    state.view = 'chat';
    el.favBar.hidden = true;
    el.roleBar.hidden = false;
    el.composer.hidden = false;
    el.chat.hidden = false;
    el.favView.hidden = true;
    if (state.panelBeforeFav === false) setPanelCollapsed(false);
  }

  function renderFavEmpty() {
    el.favBarCount.textContent = '共 0 条';
    el.favDetail.innerHTML =
      '<div class="fav-empty">' +
        '<div class="fav-empty-icon">☆</div>' +
        '<p>还没有收藏</p>' +
        '<p class="fav-empty-tip">在问答页的答案下方点「☆ 收藏」，收藏的内容会出现在这里，并完整保留来源文档、来源段落与页码。</p>' +
        '<button class="primary-btn" id="favEmptyBack" type="button">去提问</button>' +
      '</div>';
    const btn = document.getElementById('favEmptyBack');
    if (btn) btn.addEventListener('click', function () { switchTab('history'); });
  }

  function renderFavoriteDetail(item) {
    if (!item) { renderFavEmpty(); return; }
    const roleLabel = ROLE_LABEL[item.role] || ROLE_LABEL.student;
    const sources = Array.isArray(item.sources) ? item.sources : [];
    const srcHtml = sources.length
      ? sourceCardsHtml(sources)
      : '<div class="fav-no-source">这条收藏没有溯源信息</div>';

    el.favBarCount.textContent = '共 ' + (state.favorites || []).length + ' 条';
    el.favDetail.innerHTML =
      '<article class="fav-detail-card">' +
        '<div class="fd-block">' +
          '<div class="fd-label">问题</div>' +
          '<h2 class="fd-question">' + nl2br(item.question || '') + '</h2>' +
        '</div>' +
        '<div class="fd-block">' +
          '<div class="fd-label">答案' +
            '<span class="fd-meta">' + roleLabel + ' · 收藏于 ' + fmtTime(item.created_at) + '</span>' +
          '</div>' +
          '<div class="fd-answer">' + nl2br(item.answer || '（无内容）') + '</div>' +
        '</div>' +
        '<div class="fd-block">' +
          '<div class="fd-label">溯源<span class="fd-meta">' + sources.length + ' 处</span></div>' +
          '<div class="fd-sources">' + srcHtml + '</div>' +
        '</div>' +
        '<div class="fd-actions">' +
          '<button class="tool-btn fav-on" id="fdUnfav" type="button">★ 已收藏（点此取消）</button>' +
        '</div>' +
      '</article>';

    const unfavBtn = document.getElementById('fdUnfav');
    if (unfavBtn) {
      unfavBtn.addEventListener('click', function () { unfavorite(item.message_id); });
    }
    el.favView.scrollTop = 0;
  }

  function markActiveFavorite(messageId) {
    const mid = Number(messageId);
    Array.prototype.slice.call(el.favoriteList.querySelectorAll('.fav-item')).forEach(function (node) {
      node.classList.toggle('active', Number(node.dataset.mid) === mid);
    });
  }

  function openFavorite(messageId) {
    const mid = Number(messageId);
    const fav = (state.favorites || []).filter(function (f) { return Number(f.message_id) === mid; })[0];
    if (!fav) return;
    enterFavoriteView();
    renderFavoriteDetail(fav);
    markActiveFavorite(mid);
  }

  // ================================================================ 左栏 Tab
  function switchTab(tab) {
    const isHistory = tab !== 'favorite';
    el.tabHistory.classList.toggle('active', isHistory);
    el.tabFavorite.classList.toggle('active', !isHistory);
    el.historyList.hidden = !isHistory;
    el.favoriteList.hidden = isHistory;

    if (isHistory) {
      leaveFavoriteView();
      return;
    }
    // 进入收藏浏览视图：自动打开最近一条收藏
    loadFavorites().then(function () {
      enterFavoriteView();
      const first = (state.favorites || [])[0];
      if (first) {
        renderFavoriteDetail(first);
        markActiveFavorite(first.message_id);
      } else {
        renderFavEmpty();
      }
    });
  }

  // ================================================================ 身份切换
  function setRole(role, silent) {
    state.role = role === 'workplace' ? 'workplace' : 'student';
    localStorage.setItem(STORE_ROLE, state.role);
    Array.prototype.slice.call(el.roleSwitch.querySelectorAll('.seg-btn')).forEach(function (b) {
      b.classList.toggle('active', b.dataset.role === state.role);
    });
    el.roleDesc.textContent = ROLE_DESC[state.role];
    if (!silent) toast('已切换到' + ROLE_LABEL[state.role] + '，之后的回答会换一种讲法');
  }

  // ================================================================ 知识库自检
  // 静默检查，不在页面上放任何知识库管理控件（按需求移除）。
  // 只在知识库为空时给一句文字提示，避免用户面对「未找到相关内容」不知所措。
  async function checkKb() {
    try {
      const data = await request(API.kbDocState);
      if (!data.running && (data.doc_count || 0) === 0) {
        el.kbWarning.textContent = '提示：当前知识库还没有内容，请先让管理员加载文档后再提问。';
        el.kbWarning.hidden = false;
      }
    } catch (err) { /* 静默 */ }
  }

  // ================================================================ 数据刷新
  async function refreshAll() {
    await Promise.all([loadSessions(), loadFavorites(), loadFavoriteIds()]);
  }

  // ================================================================ 事件绑定
  function bindEvents() {
    el.wpEnter.addEventListener('click', doLogin);
    el.wpAccount.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); doLogin(); }
    });
    el.avatarPicker.addEventListener('click', function (e) {
      const btn = e.target.closest ? e.target.closest('.avatar-opt') : null;
      if (btn) selectAvatar(btn.dataset.avatar || '');
    });
    el.avatarUploadBtn.addEventListener('click', function () { el.avatarFile.click(); });
    el.avatarFile.addEventListener('change', function (e) {
      const file = e.target.files && e.target.files[0];
      onAvatarFilePicked(file);
      e.target.value = '';   // 允许重复选同一张
    });
    el.wpAccount.addEventListener('input', renderAvatarPreview);
    el.logoutBtn.addEventListener('click', switchAccount);

    el.sendBtn.addEventListener('click', function () { ask(el.input.value); });
    el.input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(el.input.value); }
    });
    el.input.addEventListener('input', autoResize);

    el.tabHistory.addEventListener('click', function () { switchTab('history'); });
    el.tabFavorite.addEventListener('click', function () { switchTab('favorite'); });
    el.newChatBtn.addEventListener('click', newChat);

    el.roleSwitch.addEventListener('click', function (e) {
      const btn = e.target.closest ? e.target.closest('.seg-btn') : null;
      if (btn) setRole(btn.dataset.role);
    });

    el.collapseBtn.addEventListener('click', function () { setPanelCollapsed(true); });
    el.expandBtn.addEventListener('click', function () { setPanelCollapsed(false); });
    el.backToChatBtn.addEventListener('click', function () { switchTab('history'); });

    if (el.examples) {
      el.examples.addEventListener('click', function (e) {
        if (e.target.classList && e.target.classList.contains('example')) ask(e.target.textContent);
      });
    }

    el.historyList.addEventListener('click', function (e) {
      const del = e.target.closest ? e.target.closest('.hist-del') : null;
      if (del) { e.stopPropagation(); confirmDeleteSession(del.dataset.del); return; }
      const item = e.target.closest ? e.target.closest('.hist-item') : null;
      if (item) openSession(item.dataset.sid);
    });

    el.favoriteList.addEventListener('click', function (e) {
      const unfav = e.target.closest ? e.target.closest('[data-unfav]') : null;
      if (unfav) { e.stopPropagation(); unfavorite(unfav.dataset.unfav); return; }
      const item = e.target.closest ? e.target.closest('.fav-item') : null;
      if (item) openFavorite(item.dataset.mid);
    });
  }

  // ================================================================ 初始化
  async function init() {
    loadLocalUser();
    setRole(state.role, true);
    setPanelCollapsed(localStorage.getItem(STORE_PANEL) === '1');
    renderUser();
    bindEvents();
    autoResize();
    checkKb();

    const verified = await verifyUser();
    if (!verified) {
      // 第一次打开（或登录态失效）：先看欢迎页，选头像 + 填账号后才能进
      showWelcomePage();
      return;
    }
    await refreshAll();
    el.input.focus();
  }

  init();

})();