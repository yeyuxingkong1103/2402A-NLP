/* 提交交互与状态机。
 *
 * 状态机（data-model.md §4.3）：
 *
 *   idle → submitting → streaming → done
 *                 ↘        ↘
 *                   failed ←┘
 *
 * ⚠️ submitting 与 streaming 必须分开。
 *    前者是"请求还没被接受"，后者是"已经被接受、正在处理"。合并会让
 *    "服务没起来"和"服务在算"看起来一模一样 —— 而用户对这两种情况该做的事
 *    完全不同（检查服务 vs 继续等）。
 */

const STATE = {
  IDLE: 'idle',
  SUBMITTING: 'submitting',
  STREAMING: 'streaming',
  DONE: 'done',
  FAILED: 'failed',
};

const BUSY_STATES = new Set([STATE.SUBMITTING, STATE.STREAMING]);

/* 与服务端 QUESTION_MAX_LEN 必须一致（backend/api/__init__.py）。
 * 前端这份拦截的目的是"不发无谓的请求"，服务端那份才是边界 ——
 * 两处都改了才算改对。验收会同时打这两条路径。 */
const QUESTION_MAX_LEN = 200;

let _state = STATE.IDLE;
let _lengthHintOn = false;   // 当前提示是否为"过长"提示，避免误清掉别的提示

/**
 * 校验问题。返回 { ok: true, value } 或 { ok: false, message }。
 *
 * ⚠️ 用 `raw.trim()` 而非只去 ASCII 空格：用户从文档里粘贴时常带进换行与
 * **全角空格**（中文输入法下极常见），那些都该算作空白，否则"看起来是空的"
 * 输入会被真的发到服务端。
 */
function validateQuestion(raw) {
  const question = raw.trim();

  if (question.length === 0) {
    return { ok: false, message: '请输入你的问题' };
  }

  if (question.length > QUESTION_MAX_LEN) {
    return {
      ok: false,
      message:
        `问题过长，请控制在 ${QUESTION_MAX_LEN} 个字符以内` +
        `（当前 ${question.length} 个字符）`,
    };
  }

  return { ok: true, value: question };
}

function initApp() {
  initNav();

  const form = document.getElementById('composer');
  const input = document.getElementById('question');
  if (!form || !input) return;

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    submitQuestion();
  });

  input.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;

    // ⚠️ 输入法组合态下回车是在选词，**不是提交**。
    // 不判断它，用拼音打字的人第一次选词就会把半截拼音发出去。
    // isComposing 是标准属性；keyCode === 229 是部分旧版浏览器的兜底。
    if (event.isComposing || event.keyCode === 229) return;

    if (event.shiftKey) return;   // Shift+Enter 换行

    event.preventDefault();
    submitQuestion();
  });

  // 输入时：自动增高 + 超长即时提示。
  //
  // 超长提示放在 input 事件里而不是等提交，是为了「粘贴一大段时立刻看到问题」
  // —— 等按了回车才报错，用户已经白等了一次交互（规格 Edge Case）。
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';

    const length = input.value.trim().length;
    if (length > QUESTION_MAX_LEN) {
      _lengthHintOn = true;
      setNotice(
        `问题过长，请控制在 ${QUESTION_MAX_LEN} 个字符以内（当前 ${length} 个字符）`,
        'error'
      );
    } else if (_lengthHintOn) {
      // 只清掉自己那条，不碰别的提示（如提交失败的错误）。
      _lengthHintOn = false;
      setNotice('');
    }
  });
}

function setState(next) {
  _state = next;
  const send = document.getElementById('send');
  if (send) send.disabled = BUSY_STATES.has(next);
}

function setNotice(message, kind) {
  const el = document.getElementById('notice');
  if (!el) return;
  if (!message) {
    el.hidden = true;
    el.textContent = '';
    return;
  }
  el.hidden = false;
  el.dataset.kind = kind || 'info';
  el.textContent = message;
}

async function submitQuestion() {
  if (BUSY_STATES.has(_state)) return;   // 进行中重复提交直接忽略（FR-013）

  const input = document.getElementById('question');
  if (!input) return;

  // ⚠️ 前端拦截必须在**发出任何请求之前**。
  // 空输入与超长输入 MUST NOT 产生服务端请求（FR-016、FR-017、SC-003）
  // —— 不只是"提示一下"，而是压根不发。
  //
  // 校验也在 resetResult() 之前：校验失败时不该把上一次的回答清掉。
  const verdict = validateQuestion(input.value);
  if (!verdict.ok) {
    _lengthHintOn = false;
    setNotice(verdict.message, 'error');
    return;
  }
  const question = verdict.value;

  _lengthHintOn = false;
  setNotice('');
  resetResult();
  setState(STATE.SUBMITTING);

  try {
    await streamAsk(question, {
      onStatus: () => setState(STATE.STREAMING),
      onCitations: (list) => renderCitations(list),
      onToken: (text) => appendToken(text),
      onDone: (envelope) => {
        renderDone(envelope);
        setState(STATE.DONE);
      },
    });
    setState(STATE.DONE);
  } catch (err) {
    setState(STATE.FAILED);
    const kind = err && err.kind;
    const message = (err && err.message) || '提交失败，请稍后重试。';

    showResultError(message);
    setNotice(message, 'error');

    if (kind === 'rejected') {
      // 请求未被接受：问题本身没能通过服务端校验，用户改问题即可。
      console.info('[医知源] 请求被拒绝：', message);
    }
  } finally {
    if (_state !== STATE.SUBMITTING) {
      const send = document.getElementById('send');
      if (send) send.disabled = false;
    }
  }
}

document.addEventListener('DOMContentLoaded', initApp);
