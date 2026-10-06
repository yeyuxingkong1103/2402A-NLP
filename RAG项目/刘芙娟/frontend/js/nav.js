/* 导航栏：配置驱动 + 面板切换。
 *
 * ⚠️ 新增一个功能界面只需要做两件事：
 *      1. 在 index.html 里加一个 <section class="panel" id="panel-xxx" hidden>
 *      2. 在下面的 NAV_ITEMS 里追加一行
 *    问答面板的逻辑代码一行都不用改（规格 SC-005）。
 *
 * ⚠️ 切换靠切换 hidden 属性，**不重建任何面板的 DOM**。
 *    重建会让输入框里的草稿和已渲染的回答一起消失，切回来就没了
 *    （规格 US3-2 的可独立测试点）。
 */

const NAV_ITEMS = [
  { id: 'qa', label: '医疗问答', panel: '#panel-qa', default: true },
];

function initNav() {
  const nav = document.getElementById('nav');
  if (!nav) return;

  nav.innerHTML = '';

  NAV_ITEMS.forEach((item) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'nav-item';
    btn.dataset.navId = item.id;

    const dot = document.createElement('span');
    dot.className = 'nav-dot';
    dot.setAttribute('aria-hidden', 'true');

    const label = document.createElement('span');
    label.className = 'nav-label';
    label.textContent = item.label;

    btn.append(dot, label);
    btn.addEventListener('click', () => selectNavItem(item.id));
    nav.appendChild(btn);
  });

  const initial = NAV_ITEMS.find((i) => i.default) || NAV_ITEMS[0];
  if (initial) selectNavItem(initial.id);
}

function selectNavItem(id) {
  const target = NAV_ITEMS.find((i) => i.id === id);
  if (!target) return;

  // 只切 hidden，不动面板内部的任何节点。
  NAV_ITEMS.forEach((item) => {
    const panel = document.querySelector(item.panel);
    if (panel) panel.hidden = item.id !== id;
  });

  document.querySelectorAll('.nav-item').forEach((btn) => {
    const on = btn.dataset.navId === id;
    if (on) {
      btn.setAttribute('aria-current', 'page');
    } else {
      btn.removeAttribute('aria-current');
    }
  });
}
