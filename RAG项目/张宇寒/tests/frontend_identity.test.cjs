// Run the real IIFE functions with storage/DOM boundary fixtures. No live accounts.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('frontend/src/app.js', 'utf8');
function context(names, values = {}) {
  const storage = new Map();
  const ctx = vm.createContext({
    console: {log() {}, warn(...args) {throw new Error(args.join(' '));}}, MAX_TASKS: 20, identityVersion: 0, settingsSaveVersion: 0,
    window: {localStorage: {getItem: key => storage.get(key) || '', setItem: (key, value) => storage.set(key, value)}},
    makeTask: () => ({id: 'new-task', sessionId: 'new-session', messages: []}),
    normalizeStoredMessage: message => message,
    ...values,
  });
  for (const name of names) {
    const fn = source.match(new RegExp(`^  (?:async )?function ${name}\\([\\s\\S]*?^  }`, 'm'));
    assert.ok(fn, `Missing production function: ${name}`);
    vm.runInContext(fn[0], ctx);
  }
  return {ctx, storage};
}
const taskFunctions = ['normalizeRetrievalMode', 'readStorage', 'taskStorageKey', 'activeTaskStorageKey', 'activeTaskSnapshotStorageKey', 'readStoredTasks', 'hasMeaningfulTask', 'loadTaskState'];
test('an empty account does not inherit guest messages or active session', () => {
  const {ctx, storage} = context(taskFunctions);
  storage.set('lawrag.tasks.guest', JSON.stringify([{id: 'guest-task', sessionId: 'guest-session', messages: [{role: 'user', content: 'private guest question'}]}]));
  storage.set('lawrag.active_task.guest', 'guest-task');
  const state = ctx.loadTaskState('user_42');
  assert.equal(state.tasks[0].sessionId, 'new-session');
  assert.equal(state.tasks[0].messages.length, 0);
  assert.equal(state.activeTaskId, 'new-task');
});
test('upload picker asks guests to log in and opens for authenticated users', () => {
  let loginRequests = 0;
  let pickerClicks = 0;
  const fileInput = {value: 'old', click() {pickerClicks += 1;}};
  const {ctx} = context(['openFilePicker'], {
    ENABLE_FILE_UPLOADS: true, authenticated: false, fileInput,
    openAuth(mode) {assert.equal(mode, 'login'); loginRequests += 1;},
  });
  ctx.openFilePicker();
  assert.equal(loginRequests, 1);
  assert.equal(pickerClicks, 0);
  ctx.authenticated = true;
  ctx.openFilePicker();
  assert.equal(pickerClicks, 1);
  assert.equal(fileInput.value, '');
});
test('the homepage evidence button uses the normal authenticated upload flow', () => {
  let clickHandler;
  let pickerClicks = 0;
  const trigger = {addEventListener(type, handler) {assert.equal(type, 'click'); clickHandler = handler;}};
  const fileInput = {value: '', click() {pickerClicks += 1;}};
  const {ctx} = context(['openFilePicker', 'bindFilePickerTrigger'], {
    ENABLE_FILE_UPLOADS: true, authenticated: true, fileInput, openAuth() {},
  });
  ctx.bindFilePickerTrigger(trigger);
  clickHandler();
  assert.equal(pickerClicks, 1);
});
test('the homepage shows uploaded evidence names and processing state', () => {
  const homeMaterialSummary = {hidden: true, innerHTML: ''};
  const {ctx} = context(['renderHomeMaterialSummary'], {
    homeMaterialSummary,
    escapeHtml: value => String(value),
    workspaceFileMeta: file => file.status === 'ready' ? '证据材料已就绪' : '证据材料解析中',
  });
  ctx.renderHomeMaterialSummary([
    {file_name: '借条.pdf', status: 'ready'},
    {file_name: '转账记录.png', status: 'processing'},
  ]);
  assert.equal(homeMaterialSummary.hidden, false);
  assert.match(homeMaterialSummary.innerHTML, /借条\.pdf/);
  assert.match(homeMaterialSummary.innerHTML, /证据材料已就绪/);
  assert.match(homeMaterialSummary.innerHTML, /转账记录\.png/);
  assert.match(homeMaterialSummary.innerHTML, /证据材料解析中/);
  ctx.renderHomeMaterialSummary([]);
  assert.equal(homeMaterialSummary.hidden, true);
});
test('guest activation starts clean without restoring persisted guest history', () => {
  const {ctx, storage} = context(taskFunctions);
  storage.set('lawrag.tasks.guest', JSON.stringify([{id: 'old', sessionId: 'old-session', messages: [{role: 'user', content: 'old question'}]}]));
  const state = ctx.loadTaskState('guest');
  assert.equal(state.tasks[0].sessionId, 'new-session');
  assert.equal(state.tasks[0].messages.length, 0);
  assert.ok(storage.has('lawrag.tasks.guest'), 'Old data is preserved, not deleted');
});
test('guest questions are not persisted as account history', () => {
  const {ctx, storage} = context(['taskStorageKey', 'activeTaskStorageKey', 'activeTaskSnapshotStorageKey', 'saveTasks'], {
    taskNamespace: 'guest', tasks: [{id: 'g', sessionId: 's', messages: []}], activeTaskId: 'g', activeTask: null,
    compactTaskForStorage: task => task,
  });
  ctx.saveTasks();
  assert.equal(storage.size, 0);
});
test('late history responses cannot populate a different account', async () => {
  let release;
  const {ctx} = context(['restoreServerConversationHistory'], {
    authenticated: true, currentUser: {user_id: 'A'}, taskNamespace: 'user_A', tasks: [],
    makeTask: sessionId => ({id: sessionId, sessionId, messages: []}),
    readDeletedSessions: () => new Set(), taskTimestamp: value => Number(value) || 0,
    loading: false, activeTask: null, activeTaskId: '', sessionId: '',
    taskHasAssistantAnswer: () => false,
    saveTasks() {}, renderTaskList() {}, renderTaskMessages() {},
    backendUrl: path => path,
    fetch: () => new Promise(resolve => {release = resolve;}),
  });
  const pending = ctx.restoreServerConversationHistory();
  ctx.identityVersion += 1;
  ctx.currentUser = {user_id: 'B'};
  ctx.taskNamespace = 'user_B';
  release({ok: true, json: async () => ({data: [{session_id: 'A-secret', messages: [{role: 'user', content: 'A private question'}]}]})});
  await pending;
  assert.equal(ctx.tasks.length, 0);
});
test('mobile navigation updates the backdrop and expanded state', () => {
  const classes = new Set();
  const attributes = {};
  const {ctx} = context(['setSidebarOpen'], {
    window: {matchMedia: () => ({matches: true})},
    document: {body: {classList: {toggle: (key, on) => on ? classes.add(key) : classes.delete(key), contains: key => classes.has(key)}}},
    sidebarBackdrop: {hidden: true},
    chatMain: {inert: false}, newChatButton: {focus() {}},
    sidebarToggle: {setAttribute: (key, value) => {attributes[key] = value;}},
  });
  ctx.setSidebarOpen(true);
  assert.equal(ctx.sidebarBackdrop.hidden, false);
  assert.equal(attributes['aria-expanded'], 'true');
  assert.equal(ctx.chatMain.inert, true);
  ctx.setSidebarOpen(false);
  assert.equal(ctx.sidebarBackdrop.hidden, true);
  assert.equal(attributes['aria-expanded'], 'false');
  assert.equal(ctx.chatMain.inert, false);
});
test('failed server logout keeps the account instead of pretending to be a guest', async () => {
  const {ctx} = context(['logout'], {
    authAttemptVersion: 0, authenticated: true, currentUser: {user_id: 'A'},
    questionController: null, authStatus: {textContent: ''},
    closeSettings() {}, closeAccountMenu() {}, activateTaskNamespace() {},
    loadLocalSettings() {}, renderAccountSummary() {}, renderAccountDeletionStatus() {},
    renderWorkspaceFiles() {}, syncThinkingToggle() {}, navigate() {},
    workspaceLabel: null, backendUrl: path => path,
    fetch: async () => {throw new Error('offline');},
  });
  await ctx.logout();
  assert.equal(ctx.authenticated, true);
  assert.equal(ctx.currentUser.user_id, 'A');
  assert.ok(ctx.authStatus.textContent.includes('退出失败'));
});
test('late settings response does not overwrite a new account settings', async () => {
  let release;
  const {ctx} = context(['loadUserSettings'], {
    authenticated: true, userSettings: {theme: 'cool_gray'}, taskNamespace: 'user_A',
    backendUrl: path => path, normalizeSettings: value => value,
    saveLocalSettings() {}, applySettingsToForm() {}, loadLocalSettings() {},
    fetch: () => new Promise(resolve => {release = resolve;}),
  });
  const pending = ctx.loadUserSettings();
  ctx.identityVersion += 1;
  ctx.taskNamespace = 'user_B';
  release({ok: true, json: async () => ({data: {theme: 'warm_gold'}})});
  await pending;
  assert.equal(ctx.userSettings.theme, 'cool_gray');
});
test('late account deletion status does not attach to a new user', async () => {
  let release;
  const {ctx} = context(['loadAccountDeletionStatus'], {
    authenticated: true, currentUser: {user_id: 'A'}, backendUrl: path => path,
    renderAccountDeletionStatus() {}, accountDeletionStatus: null, accountDeletionButton: null,
    fetch: () => new Promise(resolve => {release = resolve;}),
  });
  const pending = ctx.loadAccountDeletionStatus();
  ctx.identityVersion += 1;
  ctx.currentUser = {user_id: 'B'};
  release({ok: true, json: async () => ({data: {status: 'pending'}})});
  await pending;
  assert.equal(ctx.currentUser.account_deletion, undefined);
});
test('late activity does not display another account records', async () => {
  let release;
  const activityList = {textContent: '', innerHTML: ''};
  const {ctx} = context(['renderActivity', 'loadActivity'], {
    authenticated: true, activityList, backendUrl: path => path,
    escapeHtml: value => String(value), formatActivityTime: () => 'today',
    fetch: () => new Promise(resolve => {release = resolve;}),
  });
  const pending = ctx.loadActivity();
  ctx.identityVersion += 1;
  release({ok: true, json: async () => ({data: [{title: 'A private activity', detail: 'secret', created_at: 1}]})});
  await pending;
  assert.equal(activityList.innerHTML, '');
});
test('a solution finishing after identity switch cannot update the old task', async () => {
  let release;
  const task = {solution: null};
  const {ctx} = context(['openGeneratedLegalSolution'], {
    buildLegalSolution: () => ({markdown: 'fallback'}),
    generateLegalSolution: () => new Promise(resolve => {release = resolve;}),
    openSolutionDrawer() {},
  });
  const pending = ctx.openGeneratedLegalSolution(null, 'old question', {}, [], {
    onGenerated: solution => {task.solution = solution;},
  });
  ctx.identityVersion += 1;
  release({markdown: 'old account solution'});
  await pending;
  assert.equal(task.solution, null);
});
test('an already scheduled question is not sent under the next identity', () => {
  const timers = [];
  const sent = [];
  const {ctx} = context(['queueNextQuestion'], {
    queuedQuestions: ['old account question'],
    window: {setTimeout: callback => timers.push(callback)},
    submitQuestion: question => sent.push(question),
  });
  ctx.queueNextQuestion(0);
  ctx.identityVersion += 1;
  timers[0]();
  assert.equal(sent.length, 0);
  ctx.queuedQuestions.push('new account question');
  ctx.queueNextQuestion(1);
  timers[1]();
  assert.deepEqual(sent, ['new account question']);
});
test('late settings save cannot overwrite the next account preferences', async () => {
  let release;
  const {ctx} = context(['saveUserSettings'], {
    authenticated: true, taskNamespace: 'user_A', userSettings: {}, settingsSaveTimer: null, settingsStatus: null,
    window: {clearTimeout() {}}, readSettingsForm: () => ({theme: 'warm_gold'}),
    applySettingsToForm() {}, saveLocalSettings() {}, normalizeSettings: value => value, loadActivity: async () => [],
    backendUrl: path => path, fetch: () => new Promise(resolve => {release = resolve;}),
  });
  const pending = ctx.saveUserSettings();
  ctx.identityVersion += 1;
  ctx.taskNamespace = 'user_B';
  ctx.userSettings = {theme: 'cool_gray'};
  release({ok: true, json: async () => ({data: {theme: 'warm_gold'}})});
  await pending;
  assert.equal(ctx.userSettings.theme, 'cool_gray');
});
test('late deletion action cannot update a different account status', async () => {
  let release;
  const {ctx} = context(['handleAccountDeletionAction'], {
    authenticated: true, currentUser: {user_id: 'A'},
    accountDeletionButton: {disabled: false, dataset: {mode: 'cancel'}}, accountDeletionStatus: {textContent: ''},
    backendUrl: path => path, renderAccountDeletionStatus() {}, loadActivity: async () => [],
    fetch: () => new Promise(resolve => {release = resolve;}),
  });
  const pending = ctx.handleAccountDeletionAction();
  ctx.identityVersion += 1;
  ctx.currentUser = {user_id: 'B'};
  release({ok: true, json: async () => ({data: {status: 'active'}})});
  await pending;
  assert.equal(ctx.currentUser.account_deletion, undefined);
});
test('late diagnostics do not render records from the previous identity', async () => {
  let release;
  const {ctx} = context(['refreshDiagnostics'], {
    diagnosticsOutput: {textContent: ''},
    buildDiagnostics: () => new Promise(resolve => {release = resolve;}),
    formatDiagnostics: data => data.recent_activity[0].title,
  });
  const pending = ctx.refreshDiagnostics();
  ctx.identityVersion += 1;
  ctx.diagnosticsOutput.textContent = '';
  release({recent_activity: [{title: 'A private record'}]});
  await pending;
  assert.equal(ctx.diagnosticsOutput.textContent, '');
});

test('opening the selected consultation returns to chat without resetting its session or streaming view', () => {
  const task = {id: 'loan', sessionId: 'loan-session', messages: [{role: 'user', content: '借款问题'}]};
  const routes = [];
  let focused = false;
  const {ctx} = context(['switchTask'], {
    tasks: [task], activeTask: task, activeTaskId: 'loan', sessionId: 'loan-session',
    input: {focus() {focused = true;}}, navigate: route => routes.push(route),
    renderRetrievalMode() {},
    clearActiveTaskSnapshot() {assert.fail('Selected consultation must not be reset');},
    renderTaskMessages() {assert.fail('Do not replace a live streaming row');},
  });
  ctx.switchTask('loan');
  assert.deepEqual(routes, ['ask']);
  assert.equal(focused, true);
  assert.equal(ctx.activeTask, task);
  assert.equal(ctx.sessionId, 'loan-session');
  assert.equal(ctx.tasks.length, 1);
  assert.equal(ctx.activeTask.messages[0].content, '借款问题');
});

test('opening another consultation restores its own session and messages without creating a task', () => {
  const task = {id: 'rental', sessionId: 'rental-session', messages: [{role: 'user', content: '租房问题'}]};
  const routes = [];
  let rendered = '';
  const {ctx} = context(['switchTask'], {
    tasks: [{id: 'loan', sessionId: 'loan-session', messages: []}, task],
    activeTaskId: 'loan', activeTask: null, sessionId: 'loan-session', authenticated: false,
    clearActiveTaskSnapshot() {}, saveTasks() {}, renderTaskList() {}, renderWorkspaceFiles() {},
    renderTaskMessages() {rendered = ctx.activeTask.messages[0].content;},
    renderRetrievalMode() {},
    navigate: route => routes.push(route), input: {focus() {}},
  });
  ctx.switchTask('rental');
  assert.equal(ctx.sessionId, 'rental-session');
  assert.equal(ctx.activeTaskId, 'rental');
  assert.equal(ctx.activeTask, task);
  assert.equal(rendered, '租房问题');
  assert.equal(ctx.tasks.length, 2);
  assert.deepEqual(routes, ['ask']);
  ctx.switchTask('missing');
  assert.equal(ctx.sessionId, 'rental-session');
  assert.equal(routes.length, 1);
});

test('creating a consultation opens the same question workspace as current consultations', () => {
  const routes = [];
  let questionFocused = false;
  let homeFocused = false;
  const {ctx} = context(['createTask'], {
    tasks: [{id: 'old-task', sessionId: 'old-session', messages: []}],
    activeTask: null, activeTaskId: 'old-task', sessionId: 'old-session', authenticated: false,
    clearActiveTaskSnapshot() {}, saveTasks() {}, renderTaskList() {},
    renderTaskMessages() {}, renderWorkspaceFiles() {},
    renderRetrievalMode() {},
    navigate: route => routes.push(route),
    input: {focus() {questionFocused = true;}},
    homeInput: {focus() {homeFocused = true;}},
  });
  ctx.createTask();
  assert.deepEqual(routes, ['ask']);
  assert.equal(questionFocused, true);
  assert.equal(homeFocused, false);
  assert.equal(ctx.activeTaskId, 'new-task');
  assert.equal(ctx.sessionId, 'new-session');
});

test('settings tabs show only the requested panel and expose the selected tab to assistive technology', () => {
  const tabs = ['preferences', 'account'].map(name => ({dataset: {settingsTab: name}, attrs: {},
    setAttribute(key, value) {this.attrs[key] = value;}, tabIndex: 0}));
  const panels = ['preferences', 'account'].map(name => ({dataset: {settingsPanel: name}, hidden: false}));
  const {ctx} = context(['selectSettingsTab'], {settingsTabs: tabs, settingsPanels: panels, settingsSave: {hidden: false}});
  ctx.selectSettingsTab('account');
  assert.equal(panels[0].hidden, true);
  assert.equal(panels[1].hidden, false);
  assert.equal(tabs[1].attrs['aria-selected'], 'true');
  assert.equal(tabs[0].tabIndex, -1);
  assert.equal(ctx.settingsSave.hidden, true);
  ctx.selectSettingsTab('invalid');
  assert.equal(panels[0].hidden, false);
  assert.equal(panels[1].hidden, true);
  assert.equal(tabs[0].attrs['aria-selected'], 'true');
  assert.equal(ctx.settingsSave.hidden, false);
});

test('opening settings isolates the background and closing returns focus to its trigger', async () => {
  let focused = false;
  const trigger = {isConnected: true, focus() {focused = true;}};
  const {ctx} = context(['openSettings', 'closeSettings'], {
    settingsBackdrop: {hidden: true}, authBackdrop: {hidden: true}, appShell: {inert: false},
    settingsReturnFocus: null, document: {activeElement: trigger}, authenticated: false,
    settingsClose: {focus() {}}, settingsStatus: {textContent: ''}, activityList: null,
    closeAccountMenu() {}, setSidebarOpen() {}, applyBackendOriginToForm() {}, applySettingsToForm() {},
    renderAccountDeletionStatus() {}, selectSettingsTab() {},
    loadAccountDeletionStatus: async () => {}, loadActivity: async () => [],
  });
  await ctx.openSettings('account');
  assert.equal(ctx.settingsBackdrop.hidden, false);
  assert.equal(ctx.appShell.inert, true);
  ctx.closeSettings();
  assert.equal(ctx.settingsBackdrop.hidden, true);
  assert.equal(ctx.appShell.inert, false);
  assert.equal(focused, true);
});

test('settings opened from the disappearing account menu return focus to the visible account button', async () => {
  let focused = '';
  const {ctx} = context(['openSettings', 'closeSettings'], {
    settingsBackdrop: {hidden: true}, authBackdrop: {hidden: true}, appShell: {inert: false},
    settingsReturnFocus: null,
    document: {activeElement: {isConnected: true, closest: () => ({}), focus() {focused = 'hidden-menu';}}},
    loginButton: {isConnected: true, focus() {focused = 'account-button';}},
    authenticated: false, settingsClose: {focus() {}}, settingsStatus: null, activityList: null,
    closeAccountMenu() {}, setSidebarOpen() {}, applyBackendOriginToForm() {}, applySettingsToForm() {},
    renderAccountDeletionStatus() {}, selectSettingsTab() {},
    loadAccountDeletionStatus: async () => {}, loadActivity: async () => [],
  });
  await ctx.openSettings('account');
  ctx.closeSettings();
  assert.equal(focused, 'account-button');
});

function themeContext(names, values = {}) {
  const palette = source.match(/^  const THEME_TOKENS = \{[\s\S]*?^  \};/m);
  assert.ok(palette);
  const {ctx, storage} = context(names, values);
  vm.runInContext(palette[0], ctx);
  return {ctx, storage};
}

test('paper-white theme updates shell and sidebar to light surfaces without dark text on dark buttons', () => {
  const properties = new Map();
  const meta = {content: ''};
  const {ctx} = themeContext(['applyTheme'], {
    DEFAULT_USER_SETTINGS: {theme: 'warm_gold'}, settingTheme: {value: ''},
    document: {documentElement: {style: {setProperty: (key, value) => properties.set(key, value)}},
      body: {dataset: {}}, querySelector: () => meta},
  });
  ctx.applyTheme('cool_gray');
  assert.equal(properties.get('--shell-bg'), '#F8FAFC');
  assert.equal(properties.get('--shell-text'), '#1E293B');
  assert.equal(properties.get('--sidebar-start'), '#F1F5F9');
  assert.equal(properties.get('--sidebar-text'), '#1E293B');
  assert.equal(properties.get('--text-on-accent'), '#F8FAFC');
  assert.equal(meta.content, '#F8FAFC');
  ctx.applyTheme('warm_gold');
  assert.equal(properties.get('--shell-bg'), '#24231F');
  assert.equal(properties.get('--shell-text'), '#F5F1E8');
  assert.equal(properties.get('--sidebar-start'), '#1E1D19');
  assert.equal(ctx.document.body.dataset.theme, 'warm_gold');
});

test('changing the theme control applies immediately and saves a reloadable preference', async () => {
  const properties = new Map();
  const handlers = new Map();
  const control = value => ({value, addEventListener: (event, callback) => handlers.set(event, callback)});
  const themeControl = control('cool_gray');
  const {ctx, storage} = themeContext([
    'normalizeThemeKey', 'readThemePreference', 'normalizeSettings', 'readSettingsForm',
    'applyTheme', 'applySettingsToForm', 'settingsStorageKey', 'readStorage', 'writeStorage',
    'saveLocalSettings', 'queueUserSettingsSave', 'saveUserSettings',
  ], {
    DEFAULT_USER_SETTINGS: {theme: 'warm_gold'}, THEME_STORAGE_KEY: 'lawrag.theme',
    taskNamespace: 'user_A', userSettings: {theme: 'warm_gold'}, authenticated: false,
    settingTheme: themeControl, settingAutoExpand: null, settingShowProcess: null,
    settingAnswerDetailOptions: [], settingsStatus: null, settingsSaveTimer: null,
    document: {documentElement: {style: {setProperty: (key, value) => properties.set(key, value)}},
      body: {dataset: {}, classList: {toggle() {}}}, querySelector: () => null},
    renderSettingsPreview() {}, loadActivity: async () => [],
    window: {clearTimeout() {}, setTimeout(callback) {this.pending = callback;},
      localStorage: {getItem: key => storage.get(key) || '', setItem: (key, value) => storage.set(key, value)}},
  });
  // Execute the actual control registration, not a synthetic change callback.
  const registration = source.match(/^  \[settingAutoExpand, settingShowProcess, \.\.\.settingAnswerDetailOptions[^\n]*\][\s\S]*?^    \.forEach\([^\n]*\);/m);
  assert.ok(registration);
  vm.runInContext(registration[0], ctx);
  assert.equal(typeof handlers.get('change'), 'function', 'Theme select must participate in automatic saving');
  handlers.get('change')();
  assert.equal(properties.get('--shell-bg'), '#F8FAFC');
  assert.equal(ctx.document.body.dataset.theme, 'cool_gray');
  assert.equal(storage.get('lawrag.theme'), 'cool_gray', 'Persist before the debounce so immediate reload keeps the theme');
  assert.equal(storage.get('lawrag.settings.user_A.pending_theme'), 'cool_gray');
  await ctx.window.pending();
  assert.equal(JSON.parse(storage.get('lawrag.settings.user_A')).theme, 'cool_gray');
  assert.equal(ctx.readThemePreference(), 'cool_gray');
});

test('account reload preserves an unsynchronized local theme over an older server theme and schedules a retry', async () => {
  let retries = 0;
  const {ctx, storage} = context(['loadUserSettings', 'normalizeThemeKey', 'readStorage', 'settingsStorageKey'], {
    authenticated: true, taskNamespace: 'user_A', userSettings: {theme: 'cool_gray'},
    normalizeSettings: value => value, saveLocalSettings() {}, applySettingsToForm() {}, loadLocalSettings() {},
    queueUserSettingsSave() {retries += 1;}, backendUrl: path => path,
    fetch: async () => ({ok: true, json: async () => ({data: {theme: 'warm_gold'}})}),
  });
  storage.set('lawrag.settings.user_A.pending_theme', 'cool_gray');
  await ctx.loadUserSettings();
  assert.equal(ctx.userSettings.theme, 'cool_gray');
  assert.equal(retries, 1);
  ctx.taskNamespace = 'user_B';
  ctx.userSettings = {theme: 'cool_gray'};
  await ctx.loadUserSettings();
  assert.equal(ctx.userSettings.theme, 'warm_gold', 'Another account must not inherit A pending preference');
});

test('an older settings save response cannot revert a more recent theme selection', async () => {
  let release;
  const {ctx} = context(['saveUserSettings'], {
    authenticated: true, taskNamespace: 'user_A', userSettings: {}, settingsSaveTimer: null, settingsStatus: null,
    window: {clearTimeout() {}}, readSettingsForm: () => ({theme: 'warm_gold'}),
    applySettingsToForm() {}, saveLocalSettings() {}, normalizeSettings: value => value, loadActivity: async () => [],
    backendUrl: path => path, fetch: () => new Promise(resolve => {release = resolve;}),
  });
  const pending = ctx.saveUserSettings();
  ctx.settingsSaveVersion += 1;
  ctx.userSettings = {theme: 'cool_gray'};
  release({ok: true, json: async () => ({data: {theme: 'warm_gold'}})});
  await pending;
  assert.equal(ctx.userSettings.theme, 'cool_gray');
});

test('a confirmed account theme save clears only the matching pending preference', async () => {
  const {ctx, storage} = context(['saveUserSettings', 'readStorage', 'writeStorage', 'settingsStorageKey'], {
    authenticated: true, taskNamespace: 'user_A', userSettings: {}, settingsSaveTimer: null, settingsStatus: null,
    window: {clearTimeout() {}, localStorage: {getItem: key => storage.get(key) || '', setItem: (key, value) => storage.set(key, value)}},
    readSettingsForm: () => ({theme: 'cool_gray'}), applySettingsToForm() {}, saveLocalSettings() {},
    normalizeSettings: value => value, loadActivity: async () => [], backendUrl: path => path,
    fetch: async () => ({ok: true, json: async () => ({data: {theme: 'cool_gray'}})}),
  });
  storage.set('lawrag.settings.user_A.pending_theme', 'cool_gray');
  storage.set('lawrag.settings.user_B.pending_theme', 'warm_gold');
  await ctx.saveUserSettings();
  assert.equal(storage.get('lawrag.settings.user_A.pending_theme'), '');
  assert.equal(storage.get('lawrag.settings.user_B.pending_theme'), 'warm_gold');
});

test('ordinary questions default to local knowledge retrieval and preserve an explicit web preference', () => {
  const defaults = {
    include_web_default: false, auto_expand_professional: false,
    show_retrieval_process: true, answer_detail: 'standard',
    enable_long_memory: false, theme: 'warm_gold',
  };
  const {ctx} = context(['normalizeSettings'], {
    DEFAULT_USER_SETTINGS: defaults,
    normalizeThemeKey: value => value,
    readThemePreference: () => 'warm_gold',
  });
  assert.equal(ctx.normalizeSettings({}).include_web_default, false);
  assert.equal(ctx.normalizeSettings({include_web_default: true}).include_web_default, true);
});

test('question request uses the consultation retrieval mode and answer detail', async () => {
  let requestBody = null;
  const input = {value: '', placeholder: '', focus() {}};
  const {ctx} = context(['normalizeRetrievalMode', 'submitQuestion'], {
    loading: false, queuedQuestions: [], activeTask: {messages: [], retrievalMode: 'local'}, activeTaskId: 'task-1',
    identityVersion: 0, sessionId: 'session-1', currentRoute: 'ask', deepseekThinkingEnabled: false,
    userSettings: {include_web_default: false, answer_detail: 'detailed'}, input, sendButton: {disabled: false},
    questionController: null, AbortController: class { constructor() { this.signal = {}; } },
    navigate() {}, appendUserMessage() {}, setLoading() {}, appendLoadingMessage() {},
    backendUrl: path => path,
    fetch: async (_url, options) => {
      requestBody = JSON.parse(options.body);
      throw new Error('stop after observing request');
    },
    appendAssistantMessage() {}, fallbackClientAnswer: () => ({}), setServiceStatus() {},
    document: {getElementById: () => null}, queueNextQuestion() {},
  });

  await ctx.submitQuestion('合同解除需要什么条件？');

  assert.equal(requestBody.include_web, false);
  assert.equal(requestBody.retrieval_mode, 'local');
  assert.equal(requestBody.answer_detail, 'detailed');
  assert.equal(requestBody.thinking_enabled, false);
});
