const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync("app/web/static/app.js", "utf8").split("\nbindSegmentedControls();")[0];
const resultHistory = {
  conversation_id: "saved",
  title: "已完成的核查",
  last_label: "True",
  history_count: 2,
};

function response(payload, ok = true) {
  return { ok, status: ok ? 200 : 500, json: async () => payload };
}

function setup({ storage = new Map(), images = new Map(), fetch = async () => response({}) } = {}) {
  const elements = new Map();
  const element = (selector) => {
    if (!elements.has(selector)) {
      elements.set(selector, {
        value: "", files: [], classList: { toggle() {} }, dataset: {},
        setCustomValidity() {},
      });
    }
    return elements.get(selector);
  };
  let nextId = 0;
  const context = vm.createContext({
    FormData, URL, fetch,
    document: { querySelector: element },
    window: { crypto: { randomUUID: () => `new-${++nextId}` } },
    localStorage: { getItem: (key) => storage.get(key), setItem: (key, value) => storage.set(key, value) },
    DataTransfer: class {
      files = [];
      items = { add: (file) => this.files.push(file) };
    },
    images,
  });
  const run = (code) => vm.runInContext(code, context);
  run(source);
  run(`
    state.conversationId = "draft";
    renderConversation = () => {};
    setStatus = () => {};
    setRunningResultView = () => {};
    renderEmptyPreview = () => {};
    renderSelectedImagePreviews = () => {};
    updateImageSelectionMeta = () => {};
    setMobilePanel = () => {};
    ensureImagePreviewBeforeSubmit = () => {};
    selectSegmentValue = (field, value) => { state[field] = value; };
    setPrimaryInputsLocked = (locked) => { state.primaryInputLocked = locked; };
    setResult = (result, { persist = true } = {}) => {
      state.lastResult = result;
      if (Array.isArray(result.conversation_history)) state.conversationHistory = result.conversation_history;
      if (persist) upsertConversationFromResult(result);
    };
    draftFileOperation = async (_mode, operation) => operation({
      put: (files, id) => images.set(id, files),
      get: (id) => images.get(id),
      delete: (id) => images.delete(id),
    });
  `);
  return { context, run, element, storage, images };
}

test("refresh and history sync retain the active draft text, question, and unconfirmed link", async () => {
  const first = setup();
  first.context.saved = resultHistory;
  first.run("upsertConversation(saved)");
  first.element("#textInput").value = "第一行正文\n第二行正文";
  first.element("#questionInput").value = "消息可信吗？";
  first.element("#urlInput").value = "https://example.test/article";
  first.run('localStorage.setItem(CONVERSATION_ID_KEY, "draft"); persistCurrentConversationDraft();');

  const refreshed = setup({
    storage: first.storage,
    fetch: async (url) => response(url === "/conversations" ? { conversations: [resultHistory] } : {}),
  });
  await refreshed.run("loadConversationIndex(); activateConversation(localStorage.getItem(CONVERSATION_ID_KEY)); syncConversationList();");
  assert.equal(refreshed.element("#textInput").value, "第一行正文\n第二行正文");
  assert.equal(refreshed.element("#questionInput").value, "消息可信吗？");
  assert.equal(refreshed.element("#urlInput").value, "https://example.test/article");
  assert.equal(refreshed.run("isSubmittedConversation(findConversation('draft'))"), false);
  assert.equal(refreshed.run("Boolean(findConversation('saved'))"), true);
});

test("a delayed empty server response does not clear newly typed draft content", async () => {
  let complete;
  const app = setup({ fetch: () => new Promise((resolve) => { complete = resolve; }) });
  const syncing = app.run("syncConversation('draft')");
  app.element("#textInput").value = "正在输入";
  app.run("persistCurrentConversationDraft()");
  complete(response({ conversation_id: "draft", conversation: { title: "新对话" }, conversation_history: [], latest_input: {} }));
  await syncing;
  assert.equal(app.element("#textInput").value, "正在输入");
  assert.equal(app.run("findConversation('draft').draft_input.text"), "正在输入");
});

test("New Chat discards the unsubmitted draft and its local images without deleting recorded history", async () => {
  const requests = [];
  const app = setup({ fetch: async (url) => { requests.push(url); return response({}); } });
  app.context.saved = resultHistory;
  app.run("upsertConversation(saved)");
  app.element("#textInput").value = "将丢弃的正文";
  app.element("#imageInput").files = [{ name: "draft.png" }];
  app.run("persistCurrentConversationDraft()");
  assert.equal(app.images.get("draft")[0].name, "draft.png");

  app.run("newConversation(); loadConversationIndex();");
  assert.equal(app.run("state.conversationId"), "new-1");
  assert.equal(app.element("#textInput").value, "");
  assert.equal(app.run("Boolean(findConversation('draft'))"), false);
  assert.equal(app.run("findConversation('saved').last_label"), "True");
  assert.equal(app.images.has("draft"), false);
  assert.deepEqual(requests, []);
});

test("draft images are restored with the draft and can still be submitted after reload", async () => {
  const first = setup();
  first.element("#imageInput").files = [{ name: "restored.png" }];
  first.run("state.task = 'deepfake_detection'; persistCurrentConversationDraft();");
  const refreshed = setup({ storage: first.storage, images: first.images });
  refreshed.run("loadConversationIndex(); activateConversation('draft');");
  await Promise.resolve();
  assert.equal(refreshed.element("#imageInput").files[0].name, "restored.png");
  assert.equal(refreshed.run("hasPrimaryInput()"), true);
  assert.equal(refreshed.element("#submitBtn").disabled, false);
});

test("a submitted check stays in history and a late result does not replace the new conversation", async () => {
  let complete;
  const app = setup({ fetch: () => new Promise((resolve) => { complete = resolve; }) });
  app.element("#textInput").value = "已送出核查的正文";
  const detection = app.run("submitDetection()");
  app.run("newConversation()");
  assert.equal(app.run("findConversation('draft').submitted"), true);
  app.element("#textInput").value = "另一条正在编辑的草稿";
  app.run("persistCurrentConversationDraft()");

  complete(response({ conversation_id: "draft", label: "True", summary: "已完成", conversation_history: [] }));
  await detection;
  assert.equal(app.run("state.conversationId"), "new-1");
  assert.equal(app.element("#textInput").value, "另一条正在编辑的草稿");
  assert.equal(app.run("findConversation('draft').last_label"), "True");
  assert.equal(app.run("findConversation('draft').draft_input"), null);
});

test("a check that was submitted but failed can be reopened for retry", async () => {
  const app = setup({ fetch: async () => response({ decision_status: "technical_error", summary: "暂时失败" }) });
  app.element("#textInput").value = "失败后仍需可重试";
  await app.run("submitDetection()");
  app.run("newConversation(); activateConversation('draft');");
  assert.equal(app.run("findConversation('draft').submitted"), true);
  assert.equal(app.element("#textInput").value, "失败后仍需可重试");
  assert.equal(app.element("#submitBtn").disabled, false);
});
