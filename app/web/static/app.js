const state = {
  task: "multimodal_fact_checking",
  language: "zh",
  historyDeleteEnabled: false,
  lastResult: {},
  imageObjectUrls: [],
  imagePreviewSource: "",
  imagePreviewAssets: [],
  conversationId: "",
  conversationHistory: [],
  conversations: [],
  lastInputSignature: "",
  followupFailure: null,
  primaryInputLocked: false,
  imageSelectionNotice: "",
  confirmedImageUrl: "",
  confirmedWebpageUrl: "",
  urlImportPending: false,
  urlImportToken: 0,
  textFileReadToken: 0,
  textFileReading: false,
  webpageExtraction: null,
  draftImageRevision: 0,
  draftImagesRestoring: false,
};

const $ = (selector) => document.querySelector(selector);
const THEME_KEY = "jtm-theme";
const CONVERSATION_ID_KEY = "mid_demo_conversation_id";
const CONVERSATION_LIST_KEY = "mid_demo_conversations";
const MAX_CONVERSATION_LIST_ITEMS = 30;
const CONVERSATION_TITLE_LENGTH = 18;
const MAX_MMD_IMAGES = 4;
const pendingDetections = new Map();
const draftFileCache = new Map();
let draftFileDatabase;
let runtimeTimer;

function imageInputLimit() {
  return state.task === "deepfake_detection" ? 1 : MAX_MMD_IMAGES;
}

function updateTaskHeading() {
  const tasks = {
    multimodal_fact_checking: {
      zh: ["多模态事实核查", "输入待核查的文本，可附上图片，查看事实判断与图文依据。"],
      en: ["Multimodal fact-checking", "Submit text with optional images to review the verdict and supporting evidence."],
    },
    deepfake_detection: {
      zh: ["深度伪造检测", "上传一张图片或导入图片链接，查看是否存在伪造或合成痕迹。"],
      en: ["Deepfake detection", "Upload one image or import an image link to check for signs of manipulation or synthesis."],
    },
    aigc_detection: {
      zh: ["大模型生成检测", "大模型生成检测暂未开放。"],
      en: ["AI-generated content detection", "AI-generated content detection is not yet available."],
    },
  };
  const task = tasks[state.task] || tasks.multimodal_fact_checking;
  const [name, description] = task[state.language];
  $("#workspaceTaskName").textContent = name;
  $("#workspaceTaskDescription").textContent = description;
}

function updateThemeToggle() {
  const dark = document.documentElement.dataset.theme === "dark";
  const english = state.language === "en";
  const button = $("#themeToggleBtn");
  button.setAttribute("aria-pressed", String(dark));
  button.setAttribute("aria-label", english ? "Night mode" : "夜间模式");
  button.title = english
    ? (dark ? "Switch to light mode" : "Switch to night mode")
    : (dark ? "切换到浅色模式" : "切换到夜间模式");
}

function bindThemeToggle() {
  const root = document.documentElement;
  const updateBrowserColor = () => {
    $('meta[name="theme-color"]').content = getComputedStyle(root).getPropertyValue("--bg").trim();
  };
  updateBrowserColor();
  updateThemeToggle();
  $("#themeToggleBtn").addEventListener("click", () => {
    const theme = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = theme;
    try { localStorage.setItem(THEME_KEY, theme); } catch { /* Keep the theme active for this page. */ }
    updateBrowserColor();
    updateThemeToggle();
  });
}

function parsedImageUrls() {
  const value = $("#imageUrlInput")?.value || "";
  const url = value.trim();
  return url && url === state.confirmedImageUrl ? [url] : [];
}

function updateImageUrlConfirmButton() {
  updateUrlImportButton();
}

function updateUrlImportButton() {
  const input = $("#urlInput");
  const button = $("#importUrlBtn");
  if (!input || !button) return;
  const url = input.value.trim();
  const imported = Boolean(url && [state.confirmedImageUrl, state.confirmedWebpageUrl].includes(url));
  button.textContent = state.urlImportPending ? "解析中" : imported ? "已导入" : "导入";
  button.classList.toggle("confirmed", imported && !state.urlImportPending);
  button.disabled = state.primaryInputLocked || state.urlImportPending || state.textFileReading || state.draftImagesRestoring || !url || imported;
  input.readOnly = state.primaryInputLocked || state.urlImportPending;
  const textUpload = $("#uploadTextBtn");
  if (textUpload) textUpload.disabled = state.primaryInputLocked || state.urlImportPending || state.textFileReading;
}

function setUrlImportStatus(message, status = "") {
  const notice = $("#urlImportStatus");
  notice.textContent = message;
  notice.title = message;
  notice.dataset.status = status;
  notice.hidden = !message;
}

function resetUrlImport() {
  state.urlImportToken += 1;
  state.urlImportPending = false;
  $("#urlInput").value = "";
  $("#urlInput").setCustomValidity("");
  setUrlImportStatus("");
  updateUrlImportButton();
}

function selectedWebpageUrl() {
  const value = $("#webpageUrlInput")?.value.trim() || "";
  return value && value === state.confirmedWebpageUrl ? value : "";
}

function updateWebpageUrlConfirmButton() {
  updateUrlImportButton();
}

function selectedImageFiles() {
  return Array.from($("#imageInput")?.files || []).slice(0, imageInputLimit());
}

function selectedImageUrls() {
  const remaining = Math.max(0, imageInputLimit() - selectedImageFiles().length);
  return parsedImageUrls().slice(0, remaining);
}

function selectedImageCount() {
  return selectedImageFiles().length + selectedImageUrls().length + selectedWebpageImages().length;
}

function selectedWebpageImages() {
  const webpageUrl = selectedWebpageUrl();
  if (!webpageUrl || state.webpageExtraction?.url !== webpageUrl) return [];
  const directUrls = selectedImageUrls();
  const remaining = Math.max(0, imageInputLimit() - selectedImageFiles().length - directUrls.length);
  return (state.webpageExtraction?.images || []).filter((url) => !directUrls.includes(url)).slice(0, remaining);
}

function limitSelectedFiles() {
  const input = $("#imageInput");
  const files = Array.from(input?.files || []);
  const limit = imageInputLimit();
  if (!input || files.length <= limit) return;
  state.imageSelectionNotice = `已忽略超出上限的 ${files.length - limit} 张本地图片。`;
  if (typeof DataTransfer === "undefined") return;
  const transfer = new DataTransfer();
  files.slice(0, limit).forEach((file) => transfer.items.add(file));
  input.files = transfer.files;
}

function normalizeImageUrls() {
  const input = $("#imageUrlInput");
  if (!input) return;
  const files = selectedImageFiles();
  const urls = parsedImageUrls();
  const remaining = Math.max(0, imageInputLimit() - files.length);
  if (urls.length > remaining) {
    state.imageSelectionNotice = `已忽略超出上限的 ${urls.length - remaining} 个图片 URL。`;
    input.value = urls.slice(0, remaining).join("\n");
  }
}

function updateImageSelectionMeta() {
  const meta = $("#imageSelectionMeta");
  if (!meta) return;
  const count = selectedImageCount();
  const limit = imageInputLimit();
  meta.textContent = state.imageSelectionNotice;
  meta.hidden = !state.imageSelectionNotice;
  $("#addImageBtn").title = `选择图片（已选 ${count}/${limit} 张）`;
}

function syncImageInputMode() {
  updateTaskHeading();
  const input = $("#imageInput");
  if (!input) return;
  input.multiple = state.task !== "deepfake_detection";
  if (!input.multiple && input.files.length > 1) {
    limitSelectedFiles();
  }
  if (!input.multiple && selectedImageFiles().length && parsedImageUrls().length) {
    $("#imageUrlInput").value = "";
    state.confirmedImageUrl = "";
    state.imageSelectionNotice = "深度伪造检测只保留第一张本地图片。";
  }
  normalizeImageUrls();
  renderSelectedImagePreviews();
  updateImageSelectionMeta();
  updateImageUrlConfirmButton();
}

const labelText = {
  zh: {
    True: "真实",
    "Mostly True": "大部分真实",
    Mixture: "真伪混合",
    "Mostly False": "大部分虚假",
    False: "虚假",
    Completely_False: "完全虚假",
    Mostly_False: "大部分虚假",
    Mixed: "真假参半",
    Mostly_True: "大部分真实",
    Completely_True: "完全真实",
    Likely_False: "可能虚假",
    Likely_True: "可能真实",
    Cannot_Determine: "无法判断",
    真实图像: "真实图像",
    伪造图像: "伪造图像",
    合成图像: "合成图像",
    Real_Image: "真实图像",
    Forged_Image: "伪造图像",
    Synthetic_Image: "合成图像",
    完全真实: "完全真实",
    局部篡改: "存在伪造",
    存在伪造: "存在伪造",
    AI操纵: "完全AI生成",
    AI生成: "完全AI生成",
    完全AI生成: "完全AI生成",
    Completely_Real: "完全真实",
    Local_Tampering: "存在伪造",
    Forgery_Detected: "存在伪造",
    AI_Manipulated: "完全AI生成",
    AI_Generated: "完全AI生成",
    Completely_AI_Generated: "完全AI生成",
    Image_False: "图片存在问题",
    Cross_Modal_Mismatch: "图文不一致",
    Text_False: "文本存在问题",
    Consistent_True: "整体可信",
    Text_Anomaly: "文本异常",
    Image_Anomaly: "图像异常",
    No_Anomaly: "均无异常",
    All_Anomaly: "全部异常",
    文本异常: "文本异常",
    图像异常: "图像异常",
    均无异常: "均无异常",
    全部异常: "全部异常",
    Image_Out_Of_Context: "图片语境不符",
    Mismatch: "图文不一致",
    Weakly_Related: "关联较弱",
    Consistent: "图文一致",
    Partially_Consistent: "图文部分一致",
    Inconsistent: "图文不一致",
    Error: "错误",
  },
  en: {
    True: "True",
    "Mostly True": "Mostly True",
    Mixture: "Mixture",
    "Mostly False": "Mostly False",
    False: "False",
    Completely_False: "Completely False",
    Mostly_False: "Mostly False",
    Mixed: "Mixed",
    Mostly_True: "Mostly True",
    Completely_True: "Completely True",
    Likely_False: "Likely False",
    Likely_True: "Likely True",
    Cannot_Determine: "Cannot Determine",
    真实图像: "Real Image",
    伪造图像: "Forged Image",
    合成图像: "Synthetic Image",
    Real_Image: "Real Image",
    Forged_Image: "Forged Image",
    Synthetic_Image: "Synthetic Image",
    完全真实: "Completely Real",
    局部篡改: "Forgery Detected",
    存在伪造: "Forgery Detected",
    AI操纵: "Completely AI Generated",
    AI生成: "Completely AI Generated",
    完全AI生成: "Completely AI Generated",
    Completely_Real: "Completely Real",
    Local_Tampering: "Forgery Detected",
    Forgery_Detected: "Forgery Detected",
    AI_Manipulated: "Completely AI Generated",
    AI_Generated: "Completely AI Generated",
    Completely_AI_Generated: "Completely AI Generated",
    Image_False: "Image Issue",
    Cross_Modal_Mismatch: "Image-text Mismatch",
    Text_False: "Text Issue",
    Consistent_True: "Likely Credible",
    Text_Anomaly: "Text anomaly",
    Image_Anomaly: "Image anomaly",
    No_Anomaly: "No anomaly",
    All_Anomaly: "All anomaly",
    文本异常: "Text anomaly",
    图像异常: "Image anomaly",
    均无异常: "No anomaly",
    全部异常: "All anomaly",
    Image_Out_Of_Context: "Image Out of Context",
    Mismatch: "Mismatch",
    Weakly_Related: "Weakly Related",
    Consistent: "Consistent",
    Partially_Consistent: "Partially consistent",
    Inconsistent: "Inconsistent",
    Error: "Error",
  },
};

const uiText = {
  zh: {
    label: "标签",
    confidence: "置信度",
    runtime: "运行时间",
    status: "运行状态",
    tamper: "篡改风险",
    forgeryTypes: "伪造类型",
    detectionType: "类型",
    noSummary: "暂无摘要",
    running: "检测中...",
    followupPending: "正在回复…",
    followupFailed: "追问失败，请重试。",
    analysis: "分析过程",
    conclusion: "最终结论",
    finalJudgment: "最终判断",
    mainReasons: "主要原因",
    notes: "说明",
    textSide: "文本方面",
    imageSide: "图片方面",
    crossmodalSide: "跨模态关系",
    checkedClaims: "已核查关键事实点",
    foundEvidence: "检索到相关证据",
    newConversation: "新对话",
    conversation: "对话",
    history: "对话",
    historyList: "历史记录",
    historyCount: "条历史",
    noHistory: "暂无历史",
    noConversations: "暂无历史对话",
    enableDelete: "允许删除",
    finishDelete: "完成",
    deleteConversation: "删除",
    untitledConversation: "新对话",
    send: "发送",
    detecting: "开始检测",
  },
  en: {
    label: "Label",
    confidence: "Confidence",
    runtime: "Runtime",
    status: "Status",
    tamper: "Tamper Risk",
    forgeryTypes: "Forgery Types",
    detectionType: "Type",
    noSummary: "No summary",
    running: "Detecting...",
    followupPending: "Preparing a reply…",
    followupFailed: "Could not answer. Please try again.",
    analysis: "Analysis",
    conclusion: "Conclusion",
    finalJudgment: "Final judgment",
    mainReasons: "Main reasons",
    notes: "Notes",
    textSide: "Text",
    imageSide: "Image",
    crossmodalSide: "Cross-modal relation",
    checkedClaims: "Checked key claims",
    foundEvidence: "Found related evidence",
    newConversation: "New Chat",
    conversation: "Conversation",
    history: "Conversation",
    historyList: "History",
    historyCount: "history items",
    noHistory: "No history",
    noConversations: "No conversations",
    enableDelete: "Edit",
    finishDelete: "Done",
    deleteConversation: "Delete",
    untitledConversation: "New Chat",
    send: "Send",
    detecting: "Detect",
  },
};

function createConversationId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `conv_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function loadConversation() {
  loadConversationIndex();
  const conversationId = localStorage.getItem(CONVERSATION_ID_KEY) || createConversationId();
  activateConversation(conversationId);
  syncConversation(conversationId);
  syncConversationList();
}

function newConversation() {
  const current = findConversation(state.conversationId);
  if (current && !isSubmittedConversation(current)) {
    removeLocalConversation(current.conversation_id);
  }
  startEmptyConversation();
}

function setPrimaryInputsLocked(locked) {
  state.primaryInputLocked = Boolean(locked);
  const textInput = $("#textInput");
  const webpageUrlInput = $("#webpageUrlInput");
  const imageUrlInput = $("#imageUrlInput");
  const imageInput = $("#imageInput");
  const fileButton = $("#addImageBtn");
  if (textInput) textInput.readOnly = state.primaryInputLocked;
  if (webpageUrlInput) webpageUrlInput.readOnly = state.primaryInputLocked;
  if (imageUrlInput) imageUrlInput.readOnly = state.primaryInputLocked;
  if (imageInput) imageInput.disabled = state.primaryInputLocked;
  document.querySelectorAll('[data-field="task"] .segment:not([data-value="aigc_detection"])').forEach((button) => {
    button.disabled = state.primaryInputLocked;
  });
  if (fileButton) {
    fileButton.disabled = state.primaryInputLocked;
    fileButton.setAttribute("aria-disabled", state.primaryInputLocked ? "true" : "false");
  }
  document.querySelectorAll(".image-remove").forEach((button) => { button.hidden = state.primaryInputLocked; });
  updateImageUrlConfirmButton();
  updateWebpageUrlConfirmButton();
}

function loadConversationIndex() {
  try {
    const parsed = JSON.parse(localStorage.getItem(CONVERSATION_LIST_KEY) || "[]");
    state.conversations = Array.isArray(parsed)
      ? parsed.filter(isConversationMeaningful).slice(0, MAX_CONVERSATION_LIST_ITEMS)
      : [];
  } catch {
    state.conversations = [];
  }
  saveConversationIndex();
}

function saveConversationIndex() {
  try {
    localStorage.setItem(
      CONVERSATION_LIST_KEY,
      JSON.stringify(state.conversations.slice(0, MAX_CONVERSATION_LIST_ITEMS)),
    );
  } catch {
    // The current session can still keep a draft if browser storage is unavailable or full.
  }
}

function findConversation(conversationId) {
  return state.conversations.find((item) => item.conversation_id === conversationId);
}

function draftFileOperation(mode, operation) {
  if (typeof indexedDB === "undefined") return Promise.resolve(undefined);
  draftFileDatabase ??= new Promise((resolve, reject) => {
    const request = indexedDB.open("mid_demo_drafts", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("images");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => resolve(null);
  });
  return draftFileDatabase.then((database) => {
    if (!database) return undefined;
    return new Promise((resolve, reject) => {
      const transaction = database.transaction("images", mode);
      const request = operation(transaction.objectStore("images"));
      transaction.oncomplete = () => resolve(request.result);
      transaction.onabort = () => reject(transaction.error);
    });
  }).catch(() => undefined);
}

function rememberDraftFiles(conversationId, files) {
  const previous = draftFileCache.get(conversationId);
  if (previous && previous.length === files.length && previous.every((file, index) => file === files[index])) return;
  draftFileCache.set(conversationId, files);
  void draftFileOperation("readwrite", (store) =>
    files.length ? store.put(files, conversationId) : store.delete(conversationId));
}

function forgetDraftFiles(conversation) {
  if (!conversation) return;
  const id = conversation.conversation_id;
  const cached = draftFileCache.delete(id);
  if (cached || conversation.draft_input?.has_files) {
    void draftFileOperation("readwrite", (store) => store.delete(id));
  }
}

function removeLocalConversation(conversationId) {
  forgetDraftFiles(findConversation(conversationId));
  state.conversations = state.conversations.filter((item) => item.conversation_id !== conversationId);
  saveConversationIndex();
}

function hasRecordedConversation(item) {
  return Boolean(
    item?.conversation_history?.length || Number(item?.history_count || 0) > 0 ||
    item?.last_label || item?.last_summary || item?.latest_result?.label || item?.latest_result?.summary,
  );
}

function isSubmittedConversation(item) {
  return Boolean(item?.submitted || hasRecordedConversation(item));
}

function isDefaultConversationTitle(title) {
  return title === uiText.zh.untitledConversation || title === uiText.en.untitledConversation;
}

function isConversationMeaningful(item) {
  if (!item || !item.conversation_id) return false;
  const history = Array.isArray(item.conversation_history) ? item.conversation_history : [];
  const title = String(item.title || "").trim();
  return Boolean(
    item.draft_input || item.submitted || history.length ||
      Number(item.history_count || 0) > 0 ||
      String(item.last_label || "").trim() ||
      String(item.last_summary || "").trim() ||
      (title && !isDefaultConversationTitle(title)),
  );
}

function compactConversationTitle(value) {
  const normalized = String(value || "").replace(/\s+/g, " ").trim();
  const characters = Array.from(normalized);
  return characters.length > CONVERSATION_TITLE_LENGTH
    ? `${characters.slice(0, CONVERSATION_TITLE_LENGTH).join("")}...`
    : normalized;
}

function titleFromUserInput(userInput) {
  if (!userInput) return "";
  const firstImageUrl = Array.isArray(userInput.image_urls) ? userInput.image_urls[0] : "";
  const imageFile = userInput.image_asset?.filename || userInput.image_assets?.[0]?.filename;
  const candidates = [userInput.text, userInput.webpage_url, userInput.image_url, firstImageUrl, imageFile, userInput.question];
  const value = candidates.map((item) => String(item || "").trim()).find(Boolean);
  if (value) return compactConversationTitle(value);
  return userInput.has_image ? "图片检测" : "";
}

function titleFromCurrentInput() {
  const imageFile = $("#imageInput")?.files?.[0];
  const candidates = [
    $("#textInput")?.value,
    $("#webpageUrlInput")?.value,
    $("#imageUrlInput")?.value,
    imageFile?.name,
    $("#questionInput")?.value,
    $("#urlInput")?.value,
  ];
  const value = candidates.map((item) => String(item || "").trim()).find(Boolean);
  return value ? compactConversationTitle(value) : "";
}

function titleFromHistory(history) {
  const userItem = history?.find((item) => item.role === "user" && item.meta?.mode !== "followup");
  return titleFromUserInput(userItem?.meta);
}

function normalizeConversation(raw, existing = {}) {
  const conversation = raw?.conversation || raw || {};
  const conversationId = conversation.conversation_id || raw?.conversation_id || existing.conversation_id;
  const incomingHistory = Array.isArray(raw?.conversation_history)
    ? raw.conversation_history
    : Array.isArray(conversation.conversation_history)
      ? conversation.conversation_history
      : [];
  const history = incomingHistory.length ? incomingHistory : existing.conversation_history || [];
  const initialTitle = conversation.initial_title || existing.initial_title || titleFromHistory(history);
  const incomingTitle = conversation.title || "";
  const incomingHasTitle = Boolean(
    incomingTitle && incomingTitle !== uiText.zh.untitledConversation && incomingTitle !== uiText.en.untitledConversation,
  );
  const incomingHasContent = Boolean(
    incomingHasTitle ||
      incomingHistory.length ||
      conversation.history_count ||
      conversation.last_label ||
    conversation.last_summary,
  );
  return {
    ...existing,
    submitted: Boolean(conversation.submitted || existing.submitted),
    draft_input: conversation.draft_input !== undefined ? conversation.draft_input : existing.draft_input,
    conversation_id: conversationId,
    title: initialTitle || (incomingHasTitle ? incomingTitle : existing.title || uiText.zh.untitledConversation),
    initial_title: initialTitle,
    last_label: incomingHasContent ? conversation.last_label || "" : existing.last_label || "",
    last_summary: incomingHasContent ? conversation.last_summary || "" : existing.last_summary || "",
    created_at: conversation.created_at ?? existing.created_at ?? Date.now() / 1000,
    updated_at: incomingHasContent ? conversation.updated_at ?? Date.now() / 1000 : existing.updated_at ?? Date.now() / 1000,
    history_count: conversation.history_count ?? history.length,
    conversation_history: history,
    latest_input: raw?.latest_input || conversation.latest_input || existing.latest_input || {},
    latest_result: raw?.latest_result || conversation.latest_result || existing.latest_result || {},
  };
}

function upsertConversation(raw, options = {}) {
  const insertPosition = options.insertPosition || "top";
  const existingIndex = state.conversations.findIndex(
    (item) => item.conversation_id === (raw?.conversation_id || raw?.conversation?.conversation_id),
  );
  const existing = existingIndex >= 0 ? state.conversations[existingIndex] : {};
  const normalized = normalizeConversation(raw, existing);
  if (!normalized.conversation_id) return;
  if (!isConversationMeaningful(normalized)) return;
  if (hasRecordedConversation(normalized)) normalized.draft_input = null;
  if (existing.draft_input && !normalized.draft_input) forgetDraftFiles(existing);
  if (existingIndex >= 0) {
    state.conversations.splice(existingIndex, 1, normalized);
  } else if (insertPosition === "bottom") {
    state.conversations.push(normalized);
  } else {
    state.conversations.unshift(normalized);
  }
  state.conversations.splice(MAX_CONVERSATION_LIST_ITEMS).forEach(forgetDraftFiles);
  saveConversationIndex();
}

function upsertConversationFromResult(result) {
  if (!result?.conversation_id) return;
  const existing = findConversation(result.conversation_id);
  const { conversation_history: resultHistory, ...latestResult } = result;
  const history = Array.isArray(resultHistory) ? resultHistory : existing?.conversation_history || [];
  upsertConversation({
    conversation_id: result.conversation_id,
    submitted: true,
    draft_input: null,
    title: result.title || titleFromHistory(history) || titleFromCurrentInput(),
    last_label: result.label || "",
    last_summary: result.summary || result?.user_answer?.analysis || result?.user_answer?.judgment || result?.user_answer?.conclusion || "",
    updated_at: Date.now() / 1000,
    history_count: history.length,
    conversation_history: history,
    latest_result: latestResult,
  });
}

function persistCurrentConversationDraft({ submitted = false } = {}) {
  const conversationId = state.conversationId;
  const existing = findConversation(conversationId);
  if (!conversationId || pendingDetections.has(conversationId) ||
      hasRecordedConversation(existing) || state.conversationHistory.length) return;

  const files = selectedImageFiles();
  const draft = {
    task: state.task,
    text: $("#textInput").value,
    question: $("#questionInput").value,
    url: $("#urlInput").value,
    webpage_url: selectedWebpageUrl(),
    image_url: parsedImageUrls()[0] || "",
    webpage_images: state.webpageExtraction?.images || [],
    has_files: state.draftImagesRestoring ? Boolean(existing?.draft_input?.has_files) : Boolean(files.length),
  };
  const wasSubmitted = submitted || Boolean(existing?.submitted);
  const hasContent = draft.has_files || [draft.text, draft.question, draft.url, draft.webpage_url, draft.image_url].some((value) => value.trim());
  if (!hasContent && !wasSubmitted) {
    if (existing) removeLocalConversation(conversationId);
    return;
  }
  if (!state.draftImagesRestoring && (files.length || existing?.draft_input?.has_files)) {
    rememberDraftFiles(conversationId, files);
  }
  upsertConversation({
    conversation_id: conversationId,
    submitted: wasSubmitted,
    draft_input: draft,
    title: titleFromCurrentInput() || existing?.title || uiText.zh.untitledConversation,
    updated_at: Date.now() / 1000,
  });
}

async function syncConversationList() {
  const before = new Map(state.conversations.map((item) => [item.conversation_id, item]));
  try {
    const response = await fetch("/conversations");
    if (!response.ok) return;
    const payload = await response.json();
    const remote = payload.conversations || [];
    const remoteIds = new Set(remote.map((item) => item.conversation_id));
    state.conversations = state.conversations.filter((item) =>
      remoteIds.has(item.conversation_id) || item.draft_input ||
      (isSubmittedConversation(item) && before.get(item.conversation_id) !== item));
    remote.forEach((item) => upsertConversation(item, { insertPosition: "bottom" }));
    state.conversations.sort((a, b) => b.updated_at - a.updated_at);
    saveConversationIndex();
    renderConversation();
  } catch {
    renderConversation();
  }
}

async function syncConversation(conversationId) {
  if (!conversationId || pendingDetections.has(conversationId)) return;
  const local = findConversation(conversationId);
  if (local?.draft_input && !isSubmittedConversation(local)) return;
  try {
    const response = await fetch(`/conversations/${encodeURIComponent(conversationId)}`);
    if (!response.ok) return;
    const payload = await response.json();
    const currentResult = findConversation(conversationId)?.latest_result;
    if (pendingDetections.has(conversationId) ||
        (currentResult !== local?.latest_result && isCompletedDetection(currentResult))) return;
    const remoteHistory = Array.isArray(payload.conversation_history) ? payload.conversation_history : [];
    if (!hasRecordedConversation(payload) && !hasRecordedConversation(payload.conversation)) return;
    upsertConversation({
      ...payload,
      conversation: { ...payload.conversation, conversation_id: conversationId, submitted: true, draft_input: null },
    });
    if (state.conversationId === conversationId) {
      state.conversationHistory = remoteHistory;
      state.lastInputSignature = "";
      setResult({
        ...(payload.latest_result || {}),
        label: payload.latest_result?.label || payload.conversation?.last_label || "",
        summary: payload.latest_result?.summary || payload.conversation?.last_summary || "",
        conversation_id: conversationId,
        conversation_history: remoteHistory,
      }, { persist: false });
      restoreInputPreview(payload.latest_input || {});
    } else {
      renderConversation();
    }
  } catch {
    renderConversation();
  }
}

function activateConversation(conversationId) {
  state.conversationId = conversationId;
  state.followupFailure = null;
  localStorage.setItem(CONVERSATION_ID_KEY, conversationId);
  const conversation = findConversation(conversationId);
  state.conversationHistory = conversation?.conversation_history || [];
  state.lastInputSignature = "";
  setResult({
    ...(conversation?.latest_result || {}),
    conversation_id: conversationId,
    conversation_history: state.conversationHistory,
    label: conversation?.latest_result?.label || conversation?.last_label || "",
    summary: conversation?.latest_result?.summary || conversation?.last_summary || "",
  }, { persist: false });
  restoreInputPreview(conversation?.latest_input || {}, conversation?.draft_input);
  const pending = pendingDetections.get(conversationId);
  if (pending && !pending.question) {
    setPrimaryInputsLocked(true);
    setStatus("running", "running");
    setRunningResultView();
  }
  updateSubmitButtonLabel();
}

function selectConversation(conversationId) {
  if (!conversationId || conversationId === state.conversationId) return;
  persistCurrentConversationDraft();
  activateConversation(conversationId);
  syncConversation(conversationId);
}

function renderConversation() {
  const language = state.language === "en" ? "en" : "zh";
  const copy = uiText[language];
  const current = findConversation(state.conversationId);
  const shortId = state.conversationId ? state.conversationId.slice(0, 8) : "-";
  $("#newConversationBtn").textContent = copy.newConversation;
  const deleteToggle = $("#toggleHistoryDeleteBtn");
  deleteToggle.textContent = state.historyDeleteEnabled ? copy.finishDelete : copy.enableDelete;
  deleteToggle.setAttribute("aria-pressed", String(state.historyDeleteEnabled));
  const languageToggle = $("#languageToggleBtn");
  const languageAction = language === "en" ? "Current language: English. Switch to Chinese" : "当前语言：中文，点击切换为英文";
  languageToggle.setAttribute("aria-label", languageAction);
  languageToggle.title = languageAction;
  languageToggle.querySelector(".language-code").textContent = language === "en" ? "EN" : "中";
  updateSidebarToggle();
  $("#conversationListTitle").textContent = copy.historyList;
  const conversationTitle = $("#conversationTitle");
  if (conversationTitle) {
    conversationTitle.textContent = current?.title ? `${current.title} · ${shortId}` : `${copy.conversation} ${shortId}`;
  }
  $("#historyTitle").textContent = copy.history;
  $("#conclusionTitle").textContent = language === "en" ? "Verdict and evidence" : "核查结论与依据";
  const textUpload = $("#uploadTextBtn");
  textUpload.title = language === "en" ? "Choose a TXT file" : "选择 TXT 文件";
  textUpload.setAttribute("aria-label", textUpload.title);
  renderConversationList(language);
  renderConversationHistory(language);
  renderFollowupFeedback(language);
}

function renderConversationList(language) {
  const list = $("#conversationList");
  const copy = uiText[language];
  $("#conversationListCount").textContent = state.conversations.length;
  $("#conversationListCount").setAttribute("aria-label", language === "en" ? `${state.conversations.length} conversations` : `${state.conversations.length} 个历史对话`);
  list.classList.toggle("is-empty", !state.conversations.length);
  list.classList.toggle("delete-enabled", state.historyDeleteEnabled);
  if (!state.conversations.length) {
    list.textContent = copy.noConversations;
    return;
  }
  list.innerHTML = "";
  state.conversations.forEach((conversation) => {
    const button = document.createElement("div");
    button.className = `conversation-item${conversation.conversation_id === state.conversationId ? " active" : ""}`;
    button.dataset.conversationId = conversation.conversation_id;
    const openButton = document.createElement("button");
    openButton.className = "conversation-open";
    openButton.type = "button";
    openButton.setAttribute("aria-current", conversation.conversation_id === state.conversationId ? "true" : "false");

    const title = document.createElement("strong");
    title.textContent = conversation.title || copy.untitledConversation;
    openButton.title = title.textContent;
    const label = formatLabel(conversation.last_label, language);
    openButton.appendChild(title);
    if (label && label !== "-") {
      const badge = document.createElement("span");
      badge.className = "conversation-label";
      renderVerdictBadge(badge, conversation.last_label, language);
      openButton.appendChild(badge);
    } else if (conversation.draft_input && !isSubmittedConversation(conversation)) {
      const badge = document.createElement("span");
      badge.className = "conversation-label verdict-badge";
      badge.textContent = language === "en" ? "Draft" : "草稿";
      openButton.appendChild(badge);
    }
    button.appendChild(openButton);

    if (state.historyDeleteEnabled) {
      const deleteButton = document.createElement("button");
      deleteButton.className = "conversation-delete";
      deleteButton.type = "button";
      deleteButton.disabled = pendingDetections.has(conversation.conversation_id);
      deleteButton.dataset.conversationId = conversation.conversation_id;
      deleteButton.textContent = "×";
      deleteButton.title = copy.deleteConversation;
      deleteButton.setAttribute("aria-label", `${copy.deleteConversation} ${conversation.title || copy.untitledConversation}`);
      button.appendChild(deleteButton);
    }
    list.appendChild(button);
  });
}

function restoreInputPreview(latestInput, draft) {
  if (!latestInput || typeof latestInput !== "object") return;
  resetUrlImport();
  resetTextFileImport();
  setPrimaryInputsLocked(false);
  clearImageObjectUrl();
  state.draftImageRevision += 1;
  state.draftImagesRestoring = false;
  $("#imageInput").value = "";
  selectSegmentValue("task", draft?.task || latestInput.task || "multimodal_fact_checking");
  if (draft) {
    restoreDraftInput(draft);
    return;
  }
  $("#textInput").value = latestInput.text || "";
  $("#questionInput").value = "";
  $("#webpageUrlInput").value = latestInput.webpage_url || "";
  state.confirmedWebpageUrl = latestInput.webpage_url || "";
  state.webpageExtraction = latestInput.webpage_url
    ? { url: latestInput.webpage_url, text: latestInput.text || "", images: [] }
    : null;
  const storedAssets = Array.isArray(latestInput.image_assets)
    ? latestInput.image_assets.filter((item) => item?.url)
    : [];
  const storedSourceUrls = new Set(
    storedAssets.map((item) => item?.source_url).filter(Boolean),
  );
  const artifactRemoteUrls = Array.isArray(latestInput.image_artifacts)
    ? latestInput.image_artifacts
        .filter((item) => item?.source_url)
        .map((item) => item.source_url)
    : [];
  const legacyRemoteUrls = Array.isArray(latestInput.image_urls)
    ? latestInput.image_urls.filter(Boolean)
    : latestInput.image_url
      ? [latestInput.image_url]
      : [];
  const restoredRemoteUrls = [...new Set(
    artifactRemoteUrls.length ? artifactRemoteUrls : legacyRemoteUrls,
  )].filter((sourceUrl) => !storedSourceUrls.has(sourceUrl));
  const remoteAssets = restoredRemoteUrls.map((sourceUrl, index) => ({
    url: `/conversations/${encodeURIComponent(state.conversationId)}/images/${index}`,
    sourceUrl,
    filename: imageNameFromUrl(sourceUrl, index),
  }));
  const remoteUrls = remoteAssets.map((item) => item.sourceUrl);
  $("#imageUrlInput").value = remoteUrls[0] || latestInput.image_url || "";
  state.confirmedImageUrl = "";
  $("#urlInput").value = latestInput.webpage_url || remoteUrls[0] || latestInput.image_url || "";
  const previewAssets = [...storedAssets, ...remoteAssets].filter(
    (item, index, items) => items.findIndex((candidate) => candidate.url === item.url) === index,
  );
  const imageAssetUrl = latestInput.image_asset?.url || previewAssets[0]?.url || "";
  if (previewAssets.length > 1) {
    renderImagePreviews(
      previewAssets.map((item) => item.url),
      previewAssets.map((item, index) => item.filename || `Image ${index + 1}`),
    );
  } else if (imageAssetUrl) {
    renderImagePreview(imageAssetUrl, latestInput.image_asset?.filename || previewAssets[0]?.filename || "uploaded image");
  } else if (latestInput.image_url) {
    renderImagePreviewFromUrl(latestInput.image_url);
  } else {
    renderEmptyPreview();
  }
  updateImageSelectionMeta();
  updateImageUrlConfirmButton();
  updateWebpageUrlConfirmButton();
  const hasRestoredPrimary = Boolean(
    $("#textInput").value.trim() ||
      $("#webpageUrlInput").value.trim() ||
      $("#imageUrlInput").value.trim() ||
      imageAssetUrl ||
      latestInput.has_image,
  );
  setPrimaryInputsLocked(hasRestoredPrimary && hasDetectionResult());
  if (state.primaryInputLocked) {
    state.lastInputSignature = currentInputSignature();
  }
}

function restoreDraftInput(draft) {
  const conversationId = state.conversationId;
  const revision = state.draftImageRevision;
  $("#textInput").value = draft.text || "";
  $("#questionInput").value = draft.question || "";
  $("#urlInput").value = draft.url || "";
  $("#webpageUrlInput").value = state.confirmedWebpageUrl = draft.webpage_url || "";
  $("#imageUrlInput").value = state.confirmedImageUrl = draft.image_url || "";
  state.webpageExtraction = draft.webpage_url
    ? { url: draft.webpage_url, images: draft.webpage_images || [] }
    : null;
  state.imageSelectionNotice = "";
  state.draftImagesRestoring = Boolean(draft.has_files);
  renderSelectedImagePreviews();
  updateImageSelectionMeta();
  updateUrlImportButton();
  updateSubmitButtonLabel();
  if (!draft.has_files) return;

  const files = draftFileCache.has(conversationId)
    ? Promise.resolve(draftFileCache.get(conversationId))
    : draftFileOperation("readonly", (store) => store.get(conversationId));
  files.then((storedFiles) => {
    if (state.conversationId !== conversationId || state.draftImageRevision !== revision) return;
    state.draftImagesRestoring = false;
    if (storedFiles?.length && typeof DataTransfer !== "undefined") {
      const transfer = new DataTransfer();
      storedFiles.slice(0, imageInputLimit()).forEach((file) => transfer.items.add(file));
      $("#imageInput").files = transfer.files;
    }
    syncImageInputMode();
    persistCurrentConversationDraft();
    updateSubmitButtonLabel();
    renderConversation();
  });
}

async function deleteConversation(conversationId) {
  if (!state.historyDeleteEnabled || !conversationId || pendingDetections.has(conversationId)) return;
  const submitted = isSubmittedConversation(findConversation(conversationId));
  removeLocalConversation(conversationId);
  if (state.conversationId === conversationId) {
    startEmptyConversation();
  } else {
    renderConversation();
  }
  if (!submitted) return;
  try {
    await fetch(`/conversations/${encodeURIComponent(conversationId)}`, { method: "DELETE" });
  } catch {
    // Local cleanup still keeps the UI responsive if the backend item was already gone.
  }
}

function startEmptyConversation() {
  state.draftImageRevision += 1;
  state.draftImagesRestoring = false;
  resetUrlImport();
  resetTextFileImport();
  state.conversationId = createConversationId();
  state.conversationHistory = [];
  state.lastResult = {};
  state.lastInputSignature = "";
  state.followupFailure = null;
  setPrimaryInputsLocked(false);
  localStorage.setItem(CONVERSATION_ID_KEY, state.conversationId);
  $("#textInput").value = "";
  $("#questionInput").value = "";
  $("#webpageUrlInput").value = "";
  state.confirmedWebpageUrl = "";
  state.webpageExtraction = null;
  $("#imageUrlInput").value = "";
  state.confirmedImageUrl = "";
  $("#imageInput").value = "";
  state.imageSelectionNotice = "";
  clearImageObjectUrl();
  renderEmptyPreview();
  updateImageSelectionMeta();
  updateWebpageUrlConfirmButton();
  setResult({}, { persist: false });
  setStatus("idle", "idle");
  renderConversation();
}

function questionFromHistory(item, language) {
  const meta = item.meta || {};
  const question = typeof meta.question === "string" ? meta.question.trim() : "";
  if (question) return question;

  const content = typeof item.content === "string" ? item.content.trim() : "";
  const serialized = /^(?:text|question|image_url|webpage_url|image):[ \t]/.test(content);
  const hasInputMeta = Boolean(meta.text || meta.has_image || meta.image_url || meta.webpage_url || meta.image_urls?.length);
  if (!serialized && !hasInputMeta && content && content !== "empty user input") return content;
  if (serialized && !meta.text) {
    const legacyQuestion = Array.from(content.matchAll(/(?:^|\n)question:[ \t]*([^\n]+)/g)).at(-1)?.[1].trim();
    if (legacyQuestion) return legacyQuestion;
  }
  const task = meta.task || (meta.mode === "image_only" ? "deepfake_detection" : "");
  return questionForInput({
    task,
    hasWebpage: Boolean(meta.webpage_url || /(?:^|\n)webpage_url:/.test(content)),
    hasImage: Boolean(meta.has_image || meta.image_url || meta.image_urls?.length || /(?:^|\n)image(?:_url)?:/.test(content)),
    hasText: Boolean(meta.text || /(?:^|\n)text:/.test(content)),
  }, language) || (language === "en" ? "Please check the submitted content." : "请核查提交的内容。");
}

let renderedHistory;
let renderedHistoryLanguage;
let renderedHistoryResult;
function renderConversationHistory(language) {
  const list = $("#conversationHistory");
  const history = state.conversationHistory;
  const result = state.lastResult;
  if (renderedHistory === history && renderedHistoryLanguage === language && renderedHistoryResult === result) return;
  const newHistory = renderedHistory !== history;
  renderedHistory = history;
  renderedHistoryLanguage = language;
  renderedHistoryResult = result;
  list.classList.toggle("is-empty", !history.length);
  const empty = list.querySelector(".history-empty");
  const messages = list.querySelector(".history-messages");
  const conclusion = $("#conversationConclusion");
  list.insertBefore(conclusion, $("#followupFeedback"));
  messages.replaceChildren();
  empty.hidden = Boolean(history.length);
  if (!history.length) {
    empty.textContent = language === "en" ? "Your conversation will appear here after detection." : "检测完成后，可在这里查看对话并继续追问";
    return;
  }
  const hasEvidence = !conclusion.hidden && hasDetectionResult() && !$("#statusPill").classList.contains("running");
  const conclusionIndex = hasEvidence
    ? history.findLastIndex((item, index) => item.role === "assistant" && history[index - 1]?.meta?.mode !== "followup")
    : -1;
  if (hasEvidence && conclusionIndex === -1) messages.before(conclusion);
  const normalize = (text) => text.trim().replace(/\s+/g, " ");
  const evidenceText = Array.from($("#summaryBox").querySelectorAll("p, li"), (node) => normalize(node.textContent));
  history.forEach((item, index) => {
    const node = createHistoryMessage(item, language);
    if (index === conclusionIndex) {
      const message = node.querySelector(".message-text");
      message.before(conclusion);
      if (evidenceText.includes(normalize(message.textContent))) message.remove();
    }
    messages.append(node);
  });
  if (newHistory) list.scrollTop = history.length > 2 ? list.scrollHeight : 0;
}

function createHistoryMessage(item, language) {
  const user = item.role === "user";
  const node = document.createElement("article");
  node.className = `history-item ${user ? "user-message" : "assistant-message"}`;
  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  if (user) {
    avatar.append($("#userAvatarIcon").content.cloneNode(true));
  } else {
    const logo = document.createElement("img");
    logo.src = "/static/logo.png";
    logo.alt = "";
    avatar.append(logo);
  }
  const header = document.createElement("div");
  header.className = "message-header";
  const role = document.createElement("strong");
  role.textContent = user ? (language === "en" ? "User" : "用户") : (language === "en" ? "Assistant" : "惊堂木");
  header.append(avatar, role);
  const message = document.createElement("p");
  message.className = "message-text";
  message.textContent = user ? questionFromHistory(item, language) : item.content || item.label || "";
  node.append(header, message);
  return node;
}

let renderedFollowup;
let renderedFollowupLanguage;
function renderFollowupFeedback(language) {
  const pending = pendingDetections.get(state.conversationId);
  const followup = pending?.question ? pending : state.followupFailure;
  if (followup === renderedFollowup && language === renderedFollowupLanguage) return;
  renderedFollowup = followup;
  renderedFollowupLanguage = language;
  const container = $("#followupFeedback");
  container.replaceChildren();
  if (!followup) return;

  const copy = uiText[language];
  const status = document.createElement("p");
  const failed = followup.error !== undefined;
  status.className = failed ? "followup-status is-error" : "followup-status";
  status.setAttribute("role", "status");
  status.textContent = failed ? [copy.followupFailed, followup.error].filter(Boolean).join(" ") : copy.followupPending;
  container.append(createHistoryMessage({ role: "user", meta: { question: followup.question } }, language), status);
  const history = $("#conversationHistory");
  history.scrollTop = history.scrollHeight;
}

function bindSegmentedControls() {
  document.querySelectorAll(".segmented").forEach((group) => {
    group.addEventListener("click", (event) => {
      const button = event.target.closest(".segment");
      if (!button || button.disabled) return;
      group.querySelectorAll(".segment").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state[group.dataset.field] = button.dataset.value;
      group.querySelectorAll(".segment").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
      if (group.dataset.field === "task") {
        state.urlImportToken += 1;
        state.urlImportPending = false;
        setUrlImportStatus("");
        state.imageSelectionNotice = "";
        syncImageInputMode();
        persistCurrentConversationDraft();
        renderConversation();
        setMobilePanel("content");
        setSidebarOpen(false);
      }
      if (group.dataset.field === "language") {
        renderConversation();
      }
      updateSubmitButtonLabel();
    });
  });
}

function selectSegmentValue(field, value) {
  const group = document.querySelector(`.segmented[data-field="${field}"]`);
  const button = group?.querySelector(`.segment[data-value="${value}"]`);
  if (!group || !button || button.disabled) return;
  group.querySelectorAll(".segment").forEach((item) => item.classList.toggle("active", item === button));
  group.querySelectorAll(".segment").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
  state[field] = value;
  if (field === "task") syncImageInputMode();
}

function bindImagePreview() {
  $("#imageInput").addEventListener("change", () => {
    state.draftImageRevision += 1;
    state.draftImagesRestoring = false;
    state.imageSelectionNotice = "";
    limitSelectedFiles();
    if (state.task === "deepfake_detection" && selectedImageFiles().length) {
      $("#imageUrlInput").value = "";
      state.confirmedImageUrl = "";
    }
    normalizeImageUrls();
    renderSelectedImagePreviews();
    updateImageSelectionMeta();
    updateImageUrlConfirmButton();
    updateSubmitButtonLabel();
  });

}

function renderWebpageExtraction(payload, requestedUrl) {
  const extractedText = [payload?.title, payload?.text].filter(Boolean).join("\n\n").trim();
  const images = Array.isArray(payload?.images)
    ? [...new Set(payload.images.map((item) => String(item || "").trim()).filter(Boolean))].slice(0, imageInputLimit())
    : [];
  const fallbackImage = String(payload?.image_url || "").trim();
  if (!images.length && fallbackImage) images.push(fallbackImage);

  if (extractedText) {
    $("#textInput").value = extractedText;
    setTextFileStatus("");
  }
  state.webpageExtraction = { url: requestedUrl, text: extractedText, images };
  renderSelectedImagePreviews();
  state.imageSelectionNotice = "";
  updateImageSelectionMeta();
}

function probeImageUrl(url) {
  return new Promise((resolve) => {
    const image = new Image();
    image.referrerPolicy = "no-referrer";
    let settled = false;
    const finish = (isImage) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      image.onload = null;
      image.onerror = null;
      image.src = "";
      resolve(isImage);
    };
    const timer = setTimeout(() => finish(false), 8000);
    image.onload = () => finish(image.naturalWidth > 0 && image.naturalHeight > 0);
    image.onerror = () => finish(false);
    image.src = url;
  });
}

async function importUrl() {
  const input = $("#urlInput");
  const url = input.value.trim();
  if (!url || state.primaryInputLocked || state.urlImportPending || state.textFileReading || state.draftImagesRestoring) return;
  try {
    const parsed = new URL(url);
    if (!["http:", "https:"].includes(parsed.protocol)) throw new Error("unsupported protocol");
  } catch {
    input.setCustomValidity("请输入有效的 http 或 https 链接");
    input.reportValidity();
    return;
  }
  input.setCustomValidity("");
  const token = ++state.urlImportToken;
  const conversationId = state.conversationId;
  const task = state.task;
  const isCurrent = () => token === state.urlImportToken && conversationId === state.conversationId && task === state.task;
  state.urlImportPending = true;
  updateUrlImportButton();
  updateSubmitButtonLabel();
  setUrlImportStatus("正在识别链接内容…");
  try {
    const isImage = await probeImageUrl(url);
    if (!isCurrent()) return;
    if (isImage) {
      if (state.task !== "deepfake_detection" && selectedImageFiles().length >= imageInputLimit()) {
        throw new Error(`最多可添加 ${imageInputLimit()} 张图片，请先减少已选择的本地图片。`);
      }
      if (state.task === "deepfake_detection") $("#imageInput").value = "";
      $("#imageUrlInput").value = url;
      state.confirmedImageUrl = url;
      state.imageSelectionNotice = "";
      renderSelectedImagePreviews();
      updateImageSelectionMeta();
      setUrlImportStatus("已识别为图片，已添加到图片预览。", "success");
    } else {
      setUrlImportStatus("正在提取网页正文与图片…");
      const formData = new FormData();
      formData.append("webpage_url", url);
      const response = await fetch("/webpages/extract", { method: "POST", body: formData });
      const payload = await response.json();
      if (!isCurrent()) return;
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      $("#webpageUrlInput").value = url;
      state.confirmedWebpageUrl = url;
      renderWebpageExtraction(payload, url);
      const count = selectedWebpageImages().length;
      const extracted = state.webpageExtraction.text ? "已提取正文" : "未找到正文";
      setUrlImportStatus(`已识别为网页，${extracted}，已选 ${count} 张图片。`, "success");
    }
    persistCurrentConversationDraft();
    renderConversation();
  } catch (error) {
    if (isCurrent()) setUrlImportStatus(`导入失败：${String(error.message || error)}`, "error");
  } finally {
    if (token === state.urlImportToken) {
      state.urlImportPending = false;
      updateUrlImportButton();
      updateSubmitButtonLabel();
    }
  }
}

function bindUrlImport() {
  const input = $("#urlInput");
  input.addEventListener("input", () => {
    input.setCustomValidity("");
    setUrlImportStatus("");
    updateUrlImportButton();
    persistCurrentConversationDraft();
    renderConversation();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    $("#importUrlBtn").click();
  });
  $("#importUrlBtn").addEventListener("click", importUrl);
}

function imageNameFromUrl(url, index) {
  try {
    const parsed = new URL(url);
    const filename = parsed.pathname.split("/").filter(Boolean).at(-1);
    return filename ? decodeURIComponent(filename) : parsed.hostname || `remote-${index + 1}`;
  } catch {
    return `remote-${index + 1}`;
  }
}

function normalizePreviewAssets(assets) {
  const normalized = [];
  const seen = new Set();
  (Array.isArray(assets) ? assets : []).forEach((item, index) => {
    const url = String(item?.url || item?.public_url || "").trim();
    if (!url || seen.has(url)) return;
    seen.add(url);
    normalized.push({
      url,
      filename: String(item?.filename || item?.name || `Image ${index + 1}`).trim(),
    });
  });
  return normalized.slice(0, imageInputLimit());
}

function previewAssetsFromResult(result) {
  const candidates = [];
  const rawImages = result?.raw?.input_artifacts?.images;
  if (Array.isArray(rawImages)) candidates.push(...rawImages);

  const reportArtifacts = result?.raw?.report?.input_artifacts;
  if (Array.isArray(reportArtifacts)) {
    candidates.push(...reportArtifacts.filter((item) => item?.type === "image"));
  }

  const history = Array.isArray(result?.conversation_history) ? result.conversation_history : [];
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const assets = history[index]?.meta?.image_assets;
    if (!Array.isArray(assets) || !assets.length) continue;
    candidates.push(...assets);
    break;
  }

  return normalizePreviewAssets(candidates).filter((item) =>
    item.url.startsWith("/conversations/"),
  );
}

function renderRememberedImagePreviews() {
  const assets = normalizePreviewAssets(state.imagePreviewAssets);
  if (!assets.length) return false;
  state.imagePreviewAssets = assets;
  if (assets.length === 1) {
    renderImagePreview(assets[0].url, assets[0].filename, { remember: false });
  } else {
    renderImagePreviews(
      assets.map((item) => item.url),
      assets.map((item) => item.filename),
      { remember: false },
    );
  }
  return true;
}

function syncImagePreviewsFromResult(result) {
  const assets = previewAssetsFromResult(result);
  if (!assets.length) return false;
  clearImageObjectUrl();
  state.imagePreviewAssets = assets;
  return renderRememberedImagePreviews();
}

function renderSelectedImagePreviews() {
  const files = selectedImageFiles();
  const urls = [...selectedImageUrls(), ...selectedWebpageImages()];
  clearImageObjectUrl();
  state.imageObjectUrls = files.map((file) => URL.createObjectURL(file));
  const sources = [...state.imageObjectUrls, ...urls];
  const names = [
    ...files.map((file) => file.name),
    ...urls.map((url, index) => imageNameFromUrl(url, index)),
  ];
  const removers = [
    ...files.map((_, index) => () => removeSelectedImage(index)),
    ...urls.map((url) => () => removeSelectedImage(url)),
  ];
  renderImagePreviews(sources, names, { removers });
}

function removeSelectedImage(indexOrUrl) {
  if (state.primaryInputLocked || state.draftImagesRestoring) return;
  if (typeof indexOrUrl === "number") {
    const input = $("#imageInput");
    const remaining = selectedImageFiles().filter((_, index) => index !== indexOrUrl);
    if (remaining.length) {
      const transfer = new DataTransfer();
      remaining.forEach((file) => transfer.items.add(file));
      input.files = transfer.files;
    } else {
      input.value = "";
    }
  } else {
    if (state.confirmedImageUrl === indexOrUrl) {
      state.confirmedImageUrl = "";
      $("#imageUrlInput").value = "";
      if ($("#urlInput").value.trim() === indexOrUrl) {
        $("#urlInput").value = "";
        setUrlImportStatus("");
      }
    }
    if (state.webpageExtraction?.images.includes(indexOrUrl)) {
      state.webpageExtraction.images = state.webpageExtraction.images.filter((url) => url !== indexOrUrl);
      setUrlImportStatus(`网页已导入，已选 ${selectedWebpageImages().length} 张图片。`, "success");
    }
  }
  state.imageSelectionNotice = "";
  renderSelectedImagePreviews();
  updateImageSelectionMeta();
  updateUrlImportButton();
  persistCurrentConversationDraft();
  updateSubmitButtonLabel();
  renderConversation();
  ($(".image-remove:not([hidden])") || $("#addImageBtn")).focus();
}

function clearImageObjectUrl() {
  state.imageObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  state.imageObjectUrls = [];
}

function enableImageViewer(image) {
  image.tabIndex = 0;
  image.setAttribute("role", "button");
  image.setAttribute("aria-label", `放大查看：${image.alt || "图片"}`);
  image.title = "点击查看大图";
}

function bindImageViewer() {
  const preview = $("#imagePreview");
  const dialog = $("#imageViewer");
  let origin = null;
  const open = (image) => {
    if (!image?.naturalWidth || dialog.open) return;
    origin = image;
    const fullImage = $("#imageViewerImage");
    fullImage.referrerPolicy = image.referrerPolicy || "no-referrer";
    fullImage.src = image.currentSrc || image.src;
    fullImage.alt = image.alt;
    $("#imageViewerTitle").textContent = image.alt || "图片预览";
    dialog.showModal();
  };
  preview.addEventListener("click", (event) => open(event.target.closest("img")));
  preview.addEventListener("keydown", (event) => {
    if (!["Enter", " "].includes(event.key) || !event.target.matches("img[role='button']")) return;
    event.preventDefault();
    open(event.target);
  });
  $("#imageViewerCloseBtn").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener("close", () => {
    $("#imageViewerImage").removeAttribute("src");
    if (origin?.isConnected) origin.focus();
  });
}

function renderImagePreviewFromUrl(url) {
  if (!url) {
    renderEmptyPreview();
    return;
  }
  renderImagePreview(url, "image url", { noReferrer: true });
}

function renderEmptyPreview() {
  state.imagePreviewSource = "";
  state.imagePreviewAssets = [];
  const preview = $("#imagePreview");
  preview.classList.remove("has-image");
  preview.classList.remove("multi-image");
  preview.innerHTML = `<div class="preview-empty">
    <div class="upload-illustration" aria-hidden="true"><svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="5" width="24" height="22" rx="4"/><circle cx="11" cy="12" r="2.5"/><path d="m5 23 7-7 5 5 5-8 6 9"/></svg></div>
    <strong>添加图片，补充核查线索</strong>
    <p>点击右上角 + 添加图片，或通过上方链接导入<br>图片将以原始比例显示</p>
  </div>`;
}

function imageRemoveButton(name, remove) {
  const button = document.createElement("button");
  const label = state.language === "en" ? `Remove image: ${name}` : `移除图片：${name}`;
  button.type = "button";
  button.className = "image-remove";
  button.textContent = "×";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.hidden = state.primaryInputLocked;
  button.addEventListener("click", remove);
  return button;
}

function renderUnavailableImage(image) {
  const notice = document.createElement("span");
  notice.className = "preview-item-error";
  notice.textContent = "图片暂时无法加载";
  image.replaceWith(notice);
}

function renderImagePreview(src, alt, options = {}) {
  state.imagePreviewSource = src;
  if (options.remember !== false) {
    state.imagePreviewAssets = normalizePreviewAssets([{ url: src, filename: alt }]);
  }
  const preview = $("#imagePreview");
  preview.classList.add("has-image");
  preview.classList.remove("multi-image");
  preview.innerHTML = "";

  const img = document.createElement("img");
  img.className = "preview-image";
  img.alt = alt;
  enableImageViewer(img);
  if (options.noReferrer) {
    img.referrerPolicy = "no-referrer";
  }
  img.addEventListener("error", () => renderUnavailableImage(img));
  img.src = src;
  img.addEventListener("load", () => drawSuspiciousRegions(state.lastResult));
  preview.appendChild(img);
  if (options.onRemove) preview.appendChild(imageRemoveButton(alt, options.onRemove));

  const overlay = document.createElement("div");
  overlay.className = "region-overlay";
  preview.appendChild(overlay);
}

function renderImagePreviews(sources, names, options = {}) {
  if (!sources.length) {
    renderEmptyPreview();
    return;
  }
  if (sources.length === 1) {
    renderImagePreview(sources[0], names[0] || "Image 1", {
      remember: options.remember,
      noReferrer: /^https?:\/\//i.test(sources[0]),
      onRemove: options.removers?.[0],
    });
    return;
  }
  state.imagePreviewSource = sources.join("|");
  if (options.remember !== false) {
    state.imagePreviewAssets = normalizePreviewAssets(
      sources.map((url, index) => ({ url, filename: names[index] || `Image ${index + 1}` })),
    );
  }
  const preview = $("#imagePreview");
  preview.classList.add("has-image", "multi-image");
  preview.innerHTML = "";
  sources.forEach((src, index) => {
    const item = document.createElement("figure");
    item.className = "preview-item";
    const img = document.createElement("img");
    img.className = "preview-image";
    img.alt = names[index] || `Image ${index + 1}`;
    enableImageViewer(img);
    if (/^https?:\/\//i.test(src)) {
      img.referrerPolicy = "no-referrer";
    }
    img.addEventListener("error", () => renderUnavailableImage(img));
    img.src = src;
    const caption = document.createElement("figcaption");
    caption.textContent = `Image ${index + 1} · ${names[index] || "image"}`;
    item.appendChild(img);
    item.appendChild(caption);
    const remove = options.removers?.[index];
    if (remove) item.appendChild(imageRemoveButton(img.alt, remove));
    preview.appendChild(item);
  });
  const overlay = document.createElement("div");
  overlay.className = "region-overlay";
  preview.appendChild(overlay);
}

function setStatus(status, text) {
  const pill = $("#statusPill");
  pill.className = `status-pill ${status}`;
  const names = state.language === "en"
    ? { idle: "Ready", running: "Analyzing", done: "Completed", error: "Failed" }
    : { idle: "等待输入", running: "正在分析", done: "已完成", error: "检测失败" };
  pill.textContent = names[status] || text;
  $(".result-summary").setAttribute("aria-busy", String(status === "running"));
  $(".conversation-history").setAttribute("aria-busy", String(status === "running"));
  if (status !== "running" && runtimeTimer !== undefined) {
    clearInterval(runtimeTimer);
    runtimeTimer = undefined;
  }
}

function renderConclusion(result, language) {
  const details = $("#conversationConclusion");
  const container = $("#summaryBox");
  const hasResult = Boolean(result?.label || result?.summary || result?.error);
  const wasHidden = details.hidden;
  details.hidden = !hasResult;
  container.replaceChildren();
  if (!hasResult) {
    details.open = false;
    return;
  }
  if (wasHidden) details.open = true;

  const answer = formatUserAnswer(result, language);
  const copy = uiText[language];

  if (answer.message) {
    const status = document.createElement("p");
    status.className = "conclusion-status";
    status.textContent = answer.message;
    container.append(status);
    return;
  }

  function appendSection(title, entries, kind) {
    if (!entries.length) return;
    const section = document.createElement("section");
    section.className = `conclusion-section conclusion-${kind}`;
    const heading = document.createElement("h3");
    heading.textContent = title;
    section.append(heading);
    if (kind === "judgment") {
      const paragraph = document.createElement("p");
      paragraph.textContent = entries[0];
      section.append(paragraph);
    } else {
      const list = document.createElement(kind === "reasons" ? "ol" : "ul");
      for (const entry of entries) {
        const li = document.createElement("li");
        li.textContent = entry;
        list.append(li);
      }
      section.append(list);
    }
    container.append(section);
  }

  appendSection(copy.finalJudgment, [answer.conclusion], "judgment");
  appendSection(language === "en" ? "Key evidence" : "主要依据", answer.reasons, "reasons");
  appendSection(language === "en" ? "Notes and limitations" : "说明与局限", answer.notes, "notes");
}

function setRunningResultView() {
  const conclusion = $("#conversationConclusion");
  const history = $("#conversationHistory");
  history.insertBefore(conclusion, $("#followupFeedback"));
  conclusion.hidden = false;
  conclusion.open = true;
  const language = state.language === "en" ? "en" : "zh";
  const copy = uiText[language];
  renderResultMetrics({}, language);
  updateRunningTime();
  if (runtimeTimer === undefined) runtimeTimer = setInterval(updateRunningTime, 100);
  $("#summaryBox").textContent = copy.running;
  history.scrollTop = history.scrollHeight;
}

function updateRunningTime() {
  const pending = pendingDetections.get(state.conversationId);
  if (pending && !pending.question) {
    const seconds = Math.max(0, Date.now() - pending.startedAt) / 1000;
    $("#resultRuntime").textContent = formatRuntime(seconds, state.language);
  }
}

function setResult(result, options = {}) {
  const shouldPersist = options.persist !== false;
  state.lastResult = result || {};
  if (result?.conversation_id) {
    state.conversationId = result.conversation_id;
    localStorage.setItem(CONVERSATION_ID_KEY, state.conversationId);
  }
  if (Array.isArray(result?.conversation_history)) {
    state.conversationHistory = result.conversation_history;
  }
  if (shouldPersist) {
    upsertConversationFromResult(result);
  }
  renderExtractedWebpageInput(result);
  syncImagePreviewsFromResult(result);
  const language = state.language === "en" ? "en" : "zh";
  renderResultMetrics(result, language);
  const hasResult = Boolean(result?.label || result?.summary || result?.error);
  renderConclusion(result, language);
  setStatus(result?.label === "Error" || result?.decision_status === "technical_error" ? "error" : hasResult ? "done" : "idle");
  updateSubmitButtonLabel();
  renderConversation();
  drawSuspiciousRegions(result);
}

function renderResultMetrics(result, language) {
  const copy = uiText[language];
  const completed = isCompletedDetection(result);
  $("#labelTitle").textContent = copy.label;
  $("#confidenceTitle").textContent = copy.confidence;
  $("#aiTitle").textContent = copy.forgeryTypes;
  $("#runtimeTitle").textContent = copy.runtime;
  $("#statusTitle").textContent = copy.status;
  renderVerdictBadge($("#resultLabel"), completed ? result.label : "", language);
  $("#resultConfidence").textContent = completed ? formatConfidence(result.confidence, language) : "-";
  $("#aiScore").textContent = completed ? formatResultType(result, language) : "-";
  $("#resultRuntime").textContent = formatRuntime(result?.runtime_seconds, language);
}

function renderExtractedWebpageInput(result) {
  const webpage = result?.raw?.webpage;
  if (!webpage || typeof webpage !== "object") return;
  const extractedText = [webpage.title, webpage.text].filter(Boolean).join("\n\n").trim();
  const extractedImageUrls = Array.isArray(webpage.images)
    ? webpage.images.map((item) => String(item || "").trim()).filter(Boolean).slice(0, imageInputLimit())
    : [];
  const extractedImageUrl = String(webpage.image_url || "").trim();
  if (!extractedImageUrls.length && extractedImageUrl) extractedImageUrls.push(extractedImageUrl);
  const sourceUrl = String(webpage.url || webpage.final_url || "").trim();

  if (sourceUrl) {
    $("#webpageUrlInput").value = sourceUrl;
    state.confirmedWebpageUrl = sourceUrl;
    $("#urlInput").value = sourceUrl;
  }
  if (extractedText) {
    $("#textInput").value = extractedText;
  }
  state.webpageExtraction = {
    url: sourceUrl,
    text: extractedText,
    images: extractedImageUrls,
  };
  updateWebpageUrlConfirmButton();
  if (state.primaryInputLocked) {
    state.lastInputSignature = currentInputSignature();
  }
}

function formatResultType(result, language) {
  const verdict = result?.image_verdict || result?.raw?.image_forensics || {};
  const detection = result?.mmd_detection || result?.raw?.mmd_detection || {};
  const primaryType = result?.forgery_type ?? detection.forgery_type ?? verdict.forgery_type;
  if (primaryType) return formatForgeryTypes([primaryType], language);
  // Older image results used a single-item array; older MMD results have no issue type.
  if (result?.mmd_detection || result?.raw?.mmd_detection || result?.type === "multimodal_misinformation") return "-";
  return formatForgeryTypes(result.forgery_types || verdict.forgery_types, language);
}

function formatDetectionType(value, language) {
  const text = String(value || "").trim();
  const map = {
    rumor_detection: language === "en" ? "Rumor detection" : "谣言检测",
    multimodal_misinformation: language === "en" ? "Multimodal misinformation" : "多模态虚假信息检测",
    事实核查: language === "en" ? "Fact checking" : "事实核查",
    谣言检测: language === "en" ? "Rumor detection" : "谣言检测",
    Consistent: language === "en" ? "Image-text consistent" : "图文一致",
    Partially_Consistent: language === "en" ? "Image-text partially consistent" : "图文部分一致",
    Inconsistent: language === "en" ? "Image-text inconsistent" : "图文不一致",
    图文一致: language === "en" ? "Image-text consistent" : "图文一致",
    图文部分一致: language === "en" ? "Image-text partially consistent" : "图文部分一致",
    图文不一致: language === "en" ? "Image-text inconsistent" : "图文不一致",
    "Image-text consistent": language === "en" ? "Image-text consistent" : "图文一致",
    "Image-text partially consistent": language === "en" ? "Image-text partially consistent" : "图文部分一致",
    "Image-text inconsistent": language === "en" ? "Image-text inconsistent" : "图文不一致",
  };
  return map[text] || text || "-";
}

function formatForgeryTypes(types, language) {
  const values = Array.isArray(types) ? types : [];
  const cleaned = values.filter((item) => item && item !== "none");
  if (!cleaned.length) return language === "en" ? "None" : "无";
  const zh = {
    T1_fabrication: "事实虚构",
    T2_distortion: "夸大、删减或歪曲",
    T3_misleading_context: "错误语境",
    V1_synthetic_visual_content: "生成式画面",
    V2_visual_editing: "语义性编辑篡改",
    C1_semantic_inconsistency: "图文语义冲突",
    C2_contextual_inconsistency: "图文语境冲突",
    C3_pragmatic_inconsistency: "图文语用误导",
    splicing: "拼接",
    copy_move: "复制移动",
    inpainting: "内容移除",
    content_removal: "内容移除",
    content_deletion: "内容移除",
    content_replacement: "内容替换",
    face_swap: "人脸篡改",
    deepfake: "人脸篡改",
    face_tampering: "人脸篡改",
    local_repainting: "局部重绘",
    repainting: "局部重绘",
    ai_edited: "局部重绘",
    retouching: "全局编辑",
    global_editing: "全局编辑",
    ai_generated: "AI生成",
    cgi_rendering: "计算机渲染",
    other_synthetic: "其他合成",
    unknown: "未知",
    unknown_forgery: "未知",
    局部修复: "内容移除",
    内容删除: "内容移除",
    修饰改动: "全局编辑",
    AI辅助编辑: "局部重绘",
    AI局部编辑: "局部重绘",
    换脸: "人脸篡改",
    重绘: "局部重绘",
  };
  const en = {
    T1_fabrication: "Factual fabrication",
    T2_distortion: "Exaggeration, omission, or distortion",
    T3_misleading_context: "Misleading context",
    V1_synthetic_visual_content: "Synthetic visual content",
    V2_visual_editing: "Semantic visual editing",
    C1_semantic_inconsistency: "Image-text semantic conflict",
    C2_contextual_inconsistency: "Image-text contextual conflict",
    C3_pragmatic_inconsistency: "Image-text pragmatic misleadingness",
    splicing: "Splicing",
    copy_move: "Copy-move",
    inpainting: "Content removal",
    content_removal: "Content removal",
    content_deletion: "Content removal",
    content_replacement: "Content replacement",
    face_swap: "Face tampering",
    deepfake: "Face tampering",
    face_tampering: "Face tampering",
    local_repainting: "Local repainting",
    repainting: "Local repainting",
    ai_edited: "Local repainting",
    retouching: "Global editing",
    global_editing: "Global editing",
    ai_generated: "AI-generated",
    cgi_rendering: "Computer graphics rendering",
    other_synthetic: "Other synthesis",
    unknown: "Unknown",
    unknown_forgery: "Unknown",
    无: "None",
    拼接: "Splicing",
    复制移动: "Copy-move",
    内容移除: "Content removal",
    内容替换: "Content replacement",
    人脸篡改: "Face tampering",
    局部重绘: "Local repainting",
    全局编辑: "Global editing",
    AI生成: "AI-generated",
    计算机渲染: "Computer graphics rendering",
    其他合成: "Other synthesis",
    未知: "Unknown",
  };
  const map = language === "en" ? en : zh;
  const first = cleaned[0];
  return map[first] || first;
}

function formatScore(value) {
  if (value === undefined || value === null) return "-";
  const number = Number(value);
  if (Number.isNaN(number)) return "-";
  return `${Math.round(number * 100)}%`;
}

function formatLabel(label, language) {
  if (!label) return "-";
  return labelText[language][label] || label;
}

function verdictTone(label) {
  const value = formatLabel(label, "zh");
  const tones = {
    credible: ["真实", "完全真实", "大部分真实", "可能真实", "真实图像", "整体可信", "图文一致", "均无异常"],
    misleading: ["虚假", "完全虚假", "大部分虚假", "可能虚假", "伪造图像", "合成图像", "存在伪造", "完全AI生成", "图片存在问题", "图文不一致", "文本存在问题", "文本异常", "图像异常", "全部异常", "图片语境不符"],
    mixed: ["真假参半", "真伪混合", "图文部分一致", "关联较弱"],
  };
  return Object.keys(tones).find((tone) => tones[tone].includes(value)) || "neutral";
}

function renderVerdictBadge(element, label, language) {
  const text = formatLabel(label, language);
  element.classList.add("verdict-badge");
  element.classList.toggle("is-placeholder", text === "-");
  element.dataset.tone = verdictTone(label);
  element.title = text;
  const value = document.createElement("span");
  value.className = "verdict-text";
  value.textContent = text;
  element.replaceChildren(value);
}

function formatRuntime(seconds, language) {
  if (seconds === undefined || seconds === null) return "-";
  const value = Number(seconds);
  if (!Number.isFinite(value)) return "-";
  return `${value.toFixed(1)} ${language === "en" ? "s" : "秒"}`;
}

function formatConfidence(value, language) {
  if (value === undefined || value === null) return "-";
  const number = Number(value);
  if (Number.isNaN(number)) return String(value);
  const percent = number <= 1 ? number * 100 : number;
  return language === "en" ? `${Math.round(percent)}%` : `${Math.round(percent)}%`;
}

function formatUserAnswer(result, language) {
  const copy = uiText[language];
  if (!result || Object.keys(result).length === 0) return { message: copy.noSummary };
  if (result.decision_status === "technical_error") return { message: cleanText(result.summary || busyDetectionMessage(language)) };
  if (result.label === "Error") return { message: cleanText(result.summary || result.error || copy.noSummary) };
  if (result.mmd_detection || result.raw?.mmd_detection) {
    return formatMmdAnswer(result, language);
  }
  if (result.mode === "multimodal" || result.crossmodal_verdict) {
    return formatMultimodalAnswer(result, language);
  }
  if (result.mode === "image_only" || result.image_verdict || result.raw?.image_forensics) {
    return formatImageAnswer(result, language);
  }
  return formatTextAnswer(result, language);
}

function busyDetectionMessage(language) {
  return language === "en"
    ? "The server is busy and detection failed. Please try again later."
    : "服务器繁忙，检测暂时失败，请稍后重试。";
}

// New results share one answer contract; aliases only support saved older results.
function readUserAnswer(result) {
  const detection = result?.mmd_detection || result?.raw?.mmd_detection || {};
  const answers = [
    detection.user_answer,
    result?.user_answer,
    result?.image_verdict?.user_answer,
    result?.raw?.image_forensics?.user_answer,
    result?.text_verdict?.user_answer,
    detection.explanation,
    detection.user_explanation,
  ].filter((answer) => answer && typeof answer === "object" && !Array.isArray(answer));
  return {
    judgment: firstCleanText(
      ...answers.map((answer) => answer.judgment),
      ...answers.map((answer) => answer.verdict_statement || answer.conclusion || answer.headline),
    ),
    analysis: firstCleanText(
      ...answers.map((answer) => answer.analysis),
      ...answers.map((answer) => answer.reason_analysis || answer.summary),
    ),
    notes: dedupeText(answers.flatMap((answer) => cleanList(answer.notes ?? answer.limitations))),
  };
}

function formatMmdAnswer(result, language) {
  const copy = uiText[language];
  const detection = result?.mmd_detection || result?.raw?.mmd_detection || {};
  const answer = readUserAnswer(result);
  const conclusion = answer.judgment || fallbackConclusion(result?.label || detection.label, language);
  const reasons = [];
  const summary = firstCleanText(answer.analysis, result?.summary);
  if (summary) reasons.push(summary);

  const legacyAnswer = detection?.explanation || detection?.user_explanation || {};
  const findings = Array.isArray(legacyAnswer.key_findings) ? legacyAnswer.key_findings : [];
  findings.forEach((item) => {
    if (!item || typeof item !== "object") return;
    const title = cleanText(item.title);
    const findingResult = cleanText(item.result);
    const detail = cleanText(item.detail);
    const prefix = [title, findingResult].filter(Boolean).join(language === "en" ? ": " : "：");
    const line = [prefix, detail].filter(Boolean).join(language === "en" ? ". " : "。 ");
    if (line) reasons.push(line);
  });

  const notes = [...answer.notes];
  const decisionStatus = detection.decision_status || result?.decision_status;
  if (decisionStatus === "insufficient_evidence") {
    notes.push(language === "en" ? "This is the closest five-way label under limited evidence." : "这是在证据有限情况下给出的最接近五分类标签。");
  } else if (decisionStatus === "low_confidence") {
    notes.push(language === "en" ? "The current decision has relatively low confidence." : "当前判断置信度相对较低。");
  }
  return buildFriendlyBlock(copy, conclusion, dedupeText(reasons), dedupeText(notes));
}

function formatTextAnswer(result, language) {
  const copy = uiText[language];
  const answer = readUserAnswer(result);
  const conclusion = cleanText(answer.judgment) || fallbackConclusion(result?.label, language);
  const reasons = [];
  reasons.push(`${copy.textSide}：${formatLabel(result?.label, language)}`);

  const claimCount = Array.isArray(result?.claims) ? result.claims.length : 0;
  if (claimCount) reasons.push(`${copy.checkedClaims}：${claimCount}`);

  const evidenceCount = Array.isArray(result?.evidence) ? result.evidence.length : 0;
  if (evidenceCount) reasons.push(`${copy.foundEvidence}：${evidenceCount}`);

  const analysis = firstCleanText(answer.analysis, result?.summary, result?.appraisalResults);
  const splitAnalysis = splitReasonAndNotes(analysis);
  if (splitAnalysis.reason) reasons.push(splitAnalysis.reason);

  return buildFriendlyBlock(copy, conclusion, reasons, [...cleanList(answer.notes), ...splitAnalysis.notes]);
}

function formatImageAnswer(result, language) {
  const copy = uiText[language];
  const verdict = result?.image_verdict || result?.raw?.image_forensics || result || {};
  const answer = readUserAnswer(result);
  const label = result?.label || verdict?.label;
  const conclusion = cleanText(answer.judgment) || fallbackConclusion(label, language);
  const analysis = firstCleanText(answer.analysis, verdict.analysis, result?.summary);
  const { reason, notes } = splitReasonAndNotes(softenTechnicalAnalysis(analysis, language));
  return buildFriendlyBlock(copy, conclusion, [reason], [...cleanList(answer.notes), ...notes, ...collectLimitations(result)]);
}

function formatMultimodalAnswer(result, language) {
  const copy = uiText[language];
  const answer = readUserAnswer(result);
  const conclusion = cleanText(answer.judgment) || fallbackConclusion(result?.label, language);
  const statusParts = [];
  const reasons = [];
  const notes = cleanList(answer.notes);

  const textLabel = result?.text_verdict?.label || result?.raw?.textual_detection?.label;
  if (textLabel) statusParts.push(`${copy.textSide}：${formatLabel(textLabel, language)}`);

  const imageLabel = result?.image_verdict?.label || result?.raw?.visual_detection?.label;
  if (imageLabel) statusParts.push(`${copy.imageSide}：${formatLabel(imageLabel, language)}`);

  const crossmodalLabel = result?.crossmodal_verdict?.label;
  if (crossmodalLabel) statusParts.push(`${copy.crossmodalSide}：${formatLabel(crossmodalLabel, language)}`);

  const mismatch = cleanList(result?.crossmodal_verdict?.mismatched_elements);
  if (mismatch.length) {
    reasons.push(language === "en" ? `Main mismatch: ${mismatch.join("; ")}` : `主要不一致点：${mismatch.join("；")}`);
  }

  const matched = cleanList(result?.crossmodal_verdict?.matched_elements);
  if (matched.length) {
    reasons.push(language === "en" ? `Supported elements: ${matched.join("; ")}` : `能够对应的信息：${matched.join("；")}`);
  }

  const crossmodalReason = cleanText(result?.crossmodal_verdict?.reasoning || result?.crossmodal_verdict?.analysis);
  const splitCrossmodalReason = splitReasonAndNotes(crossmodalReason);
  if (splitCrossmodalReason.reason) reasons.push(splitCrossmodalReason.reason);
  notes.push(...splitCrossmodalReason.notes);

  if (reasons.length <= 1) {
    const analysis = firstCleanText(answer.analysis, result?.summary);
    const splitAnalysis = splitReasonAndNotes(softenTechnicalAnalysis(analysis, language));
    if (splitAnalysis.reason) reasons.push(splitAnalysis.reason);
    notes.push(...splitAnalysis.notes);
  }

  const statusSummary = statusParts.length ? `${statusParts.join(language === "en" ? ". " : "。 ")}${language === "en" ? "." : "。"}` : "";
  const finalConclusion = [conclusion, statusSummary].filter(Boolean).join("\n");

  return buildFriendlyBlock(copy, finalConclusion, reasons, [
    ...notes,
    ...collectLimitations(result),
  ]);
}

function buildFriendlyBlock(copy, conclusion, reasons, notes) {
  return {
    conclusion: conclusion || copy.noSummary,
    reasons: dedupeText(reasons).filter((reason) => cleanText(reason) !== cleanText(conclusion)),
    notes: dedupeText(notes),
  };
}

function firstCleanText(...values) {
  for (const value of values) {
    const text = cleanText(value);
    if (text) return text;
  }
  return "";
}

function cleanText(value) {
  if (value === undefined || value === null) return "";
  if (Array.isArray(value)) return cleanList(value).join("；");
  if (typeof value === "object") return "";
  return String(value)
    .replace(/\s+/g, " ")
    .replace(/^分析过程[:：]\s*/i, "")
    .replace(/^Analysis[:：]\s*/i, "")
    .trim();
}

function cleanList(items) {
  if (!Array.isArray(items)) return [];
  return items.map(cleanText).filter(Boolean);
}

function dedupeText(items) {
  const seen = new Set();
  return (items || []).filter((item) => {
    const text = cleanText(item);
    if (!text || seen.has(text)) return false;
    seen.add(text);
    return true;
  });
}

function splitReasonAndNotes(text) {
  const source = cleanText(text);
  if (!source) return { reason: "", notes: [] };
  const markers = ["需要注意：", "需要注意:", "局限：", "局限:", "限制：", "限制:", "Please note:", "Limitations:", "Limitation:"];
  let markerIndex = -1;
  let marker = "";
  markers.forEach((candidate) => {
    const index = source.toLowerCase().indexOf(candidate.toLowerCase());
    if (index >= 0 && (markerIndex < 0 || index < markerIndex)) {
      markerIndex = index;
      marker = candidate;
    }
  });
  if (markerIndex < 0) return { reason: source, notes: [] };

  const reason = source.slice(0, markerIndex).trim();
  const noteText = source.slice(markerIndex + marker.length).trim();
  return { reason, notes: splitNoteItems(noteText) };
}

function splitNoteItems(text) {
  return cleanText(text)
    .split(/[；;]\s*/)
    .map(cleanText)
    .filter(Boolean);
}

function collectLimitations(result) {
  const verdict = result?.image_verdict || result?.raw?.image_forensics || {};
  const rawImage = result?.raw?.image_forensics || result?.raw?.visual_detection?.raw?.image_forensics || {};
  const limitations = [
    ...(Array.isArray(verdict?.limitations) ? verdict.limitations : []),
    ...(Array.isArray(rawImage?.limitations) ? rawImage.limitations : []),
    ...(Array.isArray(result?.crossmodal_verdict?.limitations) ? result.crossmodal_verdict.limitations : []),
    ...readUserAnswer(result).notes,
  ];
  return cleanList(limitations);
}

function softenTechnicalAnalysis(text, language) {
  let output = cleanText(text);
  if (!output) return "";
  output = output
    .replace(/^(图片方面|Image)\s*[:：]\s*[^。.!?！？]+[。.!?！？]\s*/i, "")
    .replace(/Direct visual analysis:\s*/i, "")
    .replace(/模型直接视觉分析[:：]\s*/g, "")
    .replace(/画面观察[:：]\s*/g, "")
    .replace(/Visual observation:\s*/gi, "")
    .replace(/系统按最多三轮工具调用检查图像取证线索[:：]?\s*/g, "系统结合辅助检测结果进行核验。")
    .replace(/The system reviewed local forensic tool evidence and combined the observations into the final image verdict\./i, "Auxiliary checks were reviewed together with the visual observation.");
  output = localizeToolNames(output, language);
  output = simplifyTechnicalTerms(output, language);
  return output;
}

function localizeToolNames(text, language) {
  const zhNames = {
    SRM_Filter: "纹理噪声检查",
    Noise_Analyzer: "局部噪声检查",
    PatchDense_Scanner: "重复区域检查",
    CFA_Validator: "相机成像一致性检查",
    Spectrum_Grid_Analyzer: "整体生成痕迹检查",
  };
  const enNames = {
    SRM_Filter: "texture-noise check",
    Noise_Analyzer: "local-noise check",
    PatchDense_Scanner: "duplicate-region check",
    CFA_Validator: "camera-consistency check",
    Spectrum_Grid_Analyzer: "synthetic-pattern check",
  };
  const names = language === "en" ? enNames : zhNames;
  return Object.entries(names).reduce((output, [key, value]) => output.replaceAll(key, value), text);
}

function simplifyTechnicalTerms(text, language) {
  if (language === "en") {
    return text
      .replace(/\b[A-Za-z_]+_score\s*=\s*[\d.]+/g, "a related clue")
      .replace(/\bpeak_sparsity_score\b/g, "regular-pattern clue")
      .replace(/\benergy_hole_score\b/g, "missing-detail clue")
      .replace(/\bJPEG compression\b/gi, "image compression")
      .replace(/\bfrequency-domain\b/gi, "overall texture")
      .replace(/\bartifacts?\b/gi, "visible anomalies");
  }
  return text
    .replace(/[A-Za-z_]+_score\s*=\s*[\d.]+/g, "相关线索")
    .replace(/peak_sparsity_score/g, "规律纹理线索")
    .replace(/energy_hole_score/g, "细节缺失线索")
    .replace(/JPEG压缩/g, "图片压缩")
    .replace(/频域/g, "整体纹理")
    .replace(/伪影/g, "异常痕迹")
    .replace(/像素级/g, "细节级")
    .replace(/SRM/g, "纹理噪声");
}

function isAiGeneratedLabel(label) {
  return ["合成图像", "Synthetic_Image", "完全AI生成", "AI生成", "AI操纵", "Completely_AI_Generated", "AI_Generated", "AI_Manipulated"].includes(label);
}

function fallbackConclusion(label, language) {
  const zh = {
    True: "所有核心主张均有充分可靠证据支持，未发现错误或误导。",
    "Mostly True": "核心主张真实且有证据支持，仅存在不影响主要结论的局部错误。",
    Mixture: "内容中的真实与虚假成分同时存在，无法简单归为真或假。",
    "Mostly False": "主要结论或大部分核心主张为假，但仍含少量真实细节。",
    False: "核心主张完全没有可靠证据支持，或与外部证据完全矛盾。",
    Completely_False: "文本中的关键事实基本不成立。",
    Mostly_False: "文本中多数关键事实缺少可靠支持或与证据冲突。",
    Mixed: "当前内容真假信息混杂，需要结合更多证据继续判断。",
    Mostly_True: "文本中多数关键事实可以得到支持。",
    Completely_True: "文本中的关键事实整体可信。",
    Likely_False: "当前证据不足以完成严格事实核查，但内容更偏向不可信。",
    Likely_True: "当前证据不足以完成严格事实核查，但内容更偏向可信。",
    Cannot_Determine: "当前证据不足，暂时无法给出稳定判断。",
    真实图像: "图片未发现明显伪造迹象。",
    伪造图像: "图片存在可疑伪造迹象。",
    合成图像: "图片更像由 AI 或生成式方法合成。",
    完全真实: "图片未发现明显伪造迹象。",
    存在伪造: "图片存在可疑伪造迹象。",
    完全AI生成: "图片更像完全由 AI 生成。",
    Image_False: "该内容的主要风险来自图片本身。",
    Cross_Modal_Mismatch: "该内容存在图文不一致，配图不能支持文本核心信息。",
    Text_False: "该内容的主要风险来自文本事实问题。",
    Consistent_True: "文本和图片整体匹配，当前判断较可信。",
    文本异常: "文本检测发现异常，图像本身未见明确异常。",
    图像异常: "图像检测发现异常，文本本身未见明确异常。",
    均无异常: "文本核查未发现明确异常，且配图与文本不存在明显冲突。",
    全部异常: "文本和图像单模态检测均发现异常。",
  };
  const en = {
    True: "All core claims are fully supported by reliable evidence, with no factual error or misleading presentation.",
    "Mostly True": "Core claims are true and supported, with only minor inaccuracies that do not change the main conclusion.",
    Mixture: "Substantive true and false components coexist, preventing a simple true-or-false judgment.",
    "Mostly False": "The main conclusion or most core claims are false, although a small amount of true detail remains.",
    False: "Core claims have no reliable evidentiary support or completely contradict external evidence.",
    Completely_False: "The key text claims are not supported.",
    Mostly_False: "Most key text claims conflict with available evidence.",
    Mixed: "The content mixes supported and unsupported information.",
    Mostly_True: "Most key text claims are supported.",
    Completely_True: "The key text claims are generally credible.",
    Likely_False: "Strict fact-checking evidence is limited, but the content appears more likely false.",
    Likely_True: "Strict fact-checking evidence is limited, but the content appears more likely true.",
    Cannot_Determine: "Current evidence is insufficient for a stable judgment.",
    真实图像: "No obvious image forgery was found.",
    伪造图像: "The image shows suspicious signs of forgery.",
    合成图像: "The image appears synthetic or AI-generated.",
    Real_Image: "No obvious image forgery was found.",
    Forged_Image: "The image shows suspicious signs of forgery.",
    Synthetic_Image: "The image appears synthetic or AI-generated.",
    完全真实: "No obvious image forgery was found.",
    存在伪造: "The image shows suspicious signs of forgery.",
    完全AI生成: "The image appears fully AI-generated.",
    Completely_Real: "No obvious image forgery was found.",
    Forgery_Detected: "The image shows suspicious signs of forgery.",
    Completely_AI_Generated: "The image appears fully AI-generated.",
    Image_False: "The main risk comes from the image.",
    Cross_Modal_Mismatch: "The image does not support the core text.",
    Text_False: "The main risk comes from the text.",
    Consistent_True: "The text and image are broadly consistent and credible.",
    文本异常: "The text detector found an anomaly, while the image detector did not find a clear anomaly.",
    图像异常: "The image detector found an anomaly, while the text detector did not find a clear anomaly.",
    均无异常: "No clear text anomaly or image-text conflict was found.",
    全部异常: "Both the text and image detectors found anomalies.",
  };
  return (language === "en" ? en : zh)[label] || formatLabel(label, language);
}

function buildFormData() {
  const formData = new FormData();
  const text = $("#textInput").value.trim();
  const question = resolvedQuestionForSubmit();
  const webpageUrl = selectedWebpageUrl();
  const imageFiles = selectedImageFiles();
  const imageUrls = selectedImageUrls();

  if (question) formData.append("question", question);
  if (webpageUrl) {
    formData.append("webpage_url", webpageUrl);
    formData.append("webpage_image_urls", JSON.stringify(selectedWebpageImages()));
  }
  if (state.task === "deepfake_detection") {
    if (imageFiles[0]) formData.append("image", imageFiles[0]);
    if (imageUrls[0]) formData.append("image_url", imageUrls[0]);
  } else {
    if (text) formData.append("text", text);
    imageFiles.forEach((file) => formData.append("images", file));
    imageUrls.forEach((url) => formData.append("image_urls", url));
  }
  formData.append("language", state.language);
  formData.append("conversation_id", state.conversationId);
  formData.append("history", "[]");
  return formData;
}

function buildFollowupFormData(question) {
  const formData = new FormData();
  formData.append("question", question);
  formData.append("language", state.language);
  return formData;
}

function currentInputSignature() {
  const imageFiles = selectedImageFiles();
  return JSON.stringify({
    task: state.task,
    text: $("#textInput").value.trim(),
    webpage_url: selectedWebpageUrl(),
    webpage_image_urls: selectedWebpageImages(),
    image_urls: selectedImageUrls(),
    image_files: imageFiles.map(
      (file) => `${file.name}:${file.size}:${file.lastModified}`,
    ),
  });
}

function isCompletedDetection(result) {
  return Boolean(result?.label && result.label !== "Error" && result.decision_status !== "technical_error");
}

function hasDetectionResult() {
  return Boolean(state.conversationId && isCompletedDetection(state.lastResult));
}

function shouldUseFollowup() {
  const question = $("#questionInput").value.trim();
  if (!question || !hasDetectionResult()) return false;
  if (state.primaryInputLocked) return true;
  if (state.lastInputSignature) return currentInputSignature() === state.lastInputSignature;
  return !hasPrimaryInput();
}

function hasPrimaryInput() {
  if (state.task === "deepfake_detection") {
    return selectedImageCount() > 0;
  }
  if (selectedWebpageUrl()) return true;
  const hasText = Boolean($("#textInput").value.trim());
  return hasText;
}

function defaultQuestionForCurrentInput() {
  return questionForInput({
    task: state.task,
    hasImage: selectedImageCount() > 0,
    hasText: Boolean($("#textInput").value.trim()),
    hasWebpage: Boolean(selectedWebpageUrl()),
  }, state.language);
}

function questionForInput({ task, hasImage, hasText, hasWebpage }, language) {
  if (task === "deepfake_detection" || (hasImage && !hasText && !hasWebpage)) {
    return language === "en"
      ? "Please determine whether this image contains signs of deepfake manipulation."
      : "请判断这张图片是否存在深度伪造痕迹。";
  }
  if (hasWebpage) {
    return language === "en"
      ? "Analyze whether the multimodal information on this webpage is credible."
      : "分析这个网页中的文本、图片及图文关系是否真实可信。";
  }
  if (hasImage && hasText) {
    return language === "en"
      ? "Please assess whether this text and the supplied image(s) are credible, and analyze whether the images and text are consistent."
      : "请判断这段文字和所提供图片是否真实可信，并分析图文是否一致。";
  }
  if (hasText) {
    return language === "en"
      ? "Please fact-check the core factual claims in this text."
      : "请核查这段文字中的核心事实主张是否真实可信。";
  }
  return "";
}

function resolvedQuestionForSubmit() {
  return $("#questionInput").value.trim() || defaultQuestionForCurrentInput();
}

function updateSubmitButtonLabel() {
  const copy = uiText[state.language === "en" ? "en" : "zh"];
  const button = $("#submitBtn");
  if (!button) return;
  const question = $("#questionInput").value.trim();
  const samePrimaryInput =
    hasDetectionResult() && state.lastInputSignature && currentInputSignature() === state.lastInputSignature;
  const isFollowupContext = hasDetectionResult() && (state.primaryInputLocked || samePrimaryInput || !hasPrimaryInput());
  button.textContent = isFollowupContext ? copy.send : copy.detecting;
  button.disabled = state.urlImportPending || state.textFileReading || state.draftImagesRestoring ||
    pendingDetections.has(state.conversationId) || (isFollowupContext ? !question : !hasPrimaryInput());
}

function ensureImagePreviewBeforeSubmit(preserveExisting = false) {
  if (preserveExisting && state.imagePreviewAssets.length) {
    const preview = $("#imagePreview");
    if (!preview?.classList.contains("has-image")) {
      renderRememberedImagePreviews();
    }
    return;
  }
  normalizeImageUrls();
  renderSelectedImagePreviews();
  updateImageSelectionMeta();
}

function getSuspiciousRegions(result) {
  const regions =
    result?.user_answer?.forgery_regions ||
    result?.image_verdict?.user_answer?.forgery_regions ||
    result?.image_verdict?.suspicious_regions ||
    result?.raw?.image_forensics?.suspicious_regions ||
    result?.crossmodal_verdict?.suspicious_regions ||
    [];
  return Array.isArray(regions) ? regions : [];
}

function getAnnotatedImageUrl(result) {
  return (
    result?.image_verdict?.artifacts?.annotated_image_url ||
    result?.raw?.image_forensics?.artifacts?.annotated_image_url ||
    ""
  );
}

function parseRegion(region) {
  const text = String(region?.region || region || "");
  const values = text.match(/-?\d+(?:\.\d+)?/g);
  if (!values || values.length < 4) return null;
  const [x0, y0, x1, y1] = values.slice(0, 4).map(Number);
  if (![x0, y0, x1, y1].every(Number.isFinite)) return null;
  if (x1 <= x0 || y1 <= y0) return null;
  return { x0, y0, x1, y1 };
}

function drawSuspiciousRegions(result) {
  const preview = $("#imagePreview");
  const img = preview.querySelector(".preview-image:not(.annotated-preview-image)");
  const overlay = preview.querySelector(".region-overlay");
  if (!overlay) return;
  overlay.innerHTML = "";
  overlay.removeAttribute("style");
  overlay.hidden = false;
  const previousAnnotated = preview.querySelector(".annotated-preview-image");
  const annotatedUrl = getAnnotatedImageUrl(result);
  if (annotatedUrl && img) {
    if (previousAnnotated?.dataset.source === annotatedUrl) {
      if (previousAnnotated.complete && previousAnnotated.naturalWidth) {
        img.hidden = true;
        previousAnnotated.hidden = false;
        overlay.hidden = true;
      }
      return;
    }
    previousAnnotated?.remove();
    const annotated = document.createElement("img");
    annotated.className = "preview-image annotated-preview-image";
    annotated.alt = `${img.alt || "image"} · suspicious regions`;
    enableImageViewer(annotated);
    annotated.dataset.source = annotatedUrl;
    annotated.hidden = true;
    annotated.addEventListener("load", () => {
      img.hidden = true;
      annotated.hidden = false;
      overlay.innerHTML = "";
      overlay.hidden = true;
    });
    annotated.addEventListener("error", () => {
      annotated.remove();
      img.hidden = false;
      overlay.hidden = false;
      drawCoordinateRegionOverlay(preview, img, overlay, result);
    });
    preview.appendChild(annotated);
    annotated.src = annotatedUrl;
    return;
  }
  previousAnnotated?.remove();
  if (img) img.hidden = false;
  drawCoordinateRegionOverlay(preview, img, overlay, result);
}

function drawCoordinateRegionOverlay(preview, img, overlay, result) {
  overlay.innerHTML = "";
  overlay.removeAttribute("style");
  if (!img || !img.complete || !img.naturalWidth || !img.naturalHeight) return;

  const regions = getSuspiciousRegions(result).map(parseRegion).filter(Boolean);
  if (!regions.length) return;

  const previewRect = preview.getBoundingClientRect();
  const imageRect = img.getBoundingClientRect();
  const scaleX = imageRect.width / img.naturalWidth;
  const scaleY = imageRect.height / img.naturalHeight;

  overlay.style.left = `${imageRect.left - previewRect.left + preview.scrollLeft}px`;
  overlay.style.top = `${imageRect.top - previewRect.top + preview.scrollTop}px`;
  overlay.style.width = `${imageRect.width}px`;
  overlay.style.height = `${imageRect.height}px`;

  regions.forEach((region) => {
    const box = document.createElement("div");
    box.className = "region-box";
    box.style.left = `${region.x0 * scaleX}px`;
    box.style.top = `${region.y0 * scaleY}px`;
    box.style.width = `${(region.x1 - region.x0) * scaleX}px`;
    box.style.height = `${(region.y1 - region.y0) * scaleY}px`;
    overlay.appendChild(box);
  });
}

async function submitDetection() {
  const requestConversationId = state.conversationId;
  if (state.urlImportPending || state.textFileReading || state.draftImagesRestoring || pendingDetections.has(requestConversationId)) return;
  const submitBtn = $("#submitBtn");
  const isFollowup = shouldUseFollowup();
  const requestTitle = (isFollowup && findConversation(requestConversationId)?.title) || titleFromCurrentInput();
  const signature = currentInputSignature();
  const question = resolvedQuestionForSubmit();
  const previousResult = state.lastResult;
  if (hasDetectionResult() && !hasPrimaryInput() && !$("#questionInput").value.trim()) {
    updateSubmitButtonLabel();
    return;
  }
  if (!isFollowup && !hasPrimaryInput()) {
    updateSubmitButtonLabel();
    return;
  }
  setMobilePanel("dialogue");
  try {
    submitBtn.disabled = true;
    ensureImagePreviewBeforeSubmit(isFollowup);
    persistCurrentConversationDraft({ submitted: true });
    pendingDetections.set(requestConversationId, { startedAt: Date.now(), question: isFollowup ? question : "" });
    state.followupFailure = null;
    if (isFollowup) {
      $("#questionInput").value = "";
      renderConversation();
    } else {
      setPrimaryInputsLocked(true);
      setStatus("running", "running");
      setRunningResultView();
    }

    const detectionEndpoint = state.task === "deepfake_detection" ? "/detect/deepfake" : "/detect/mmd";
    const followupEndpoint = `/conversations/${encodeURIComponent(state.conversationId)}/followup`;
    const response = await fetch(isFollowup ? followupEndpoint : detectionEndpoint, {
      method: "POST",
      body: isFollowup ? buildFollowupFormData(question) : buildFormData(),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(response.status >= 500 ? busyDetectionMessage(state.language) : payload.detail || payload.error || `HTTP ${response.status}`);
    }

    const failed = payload.decision_status === "technical_error" || payload.label === "Error";
    const history = payload.conversation_history;
    if (isFollowup && (failed || !Array.isArray(history) || !history.length)) {
      throw new Error(payload.summary || payload.error || "");
    }
    if (isFollowup && history.at(-2)?.meta?.mode !== "auto") {
      upsertConversationFromResult({
        ...previousResult,
        conversation_id: requestConversationId,
        title: requestTitle,
        conversation_history: history,
      });
      if (state.conversationId === requestConversationId) state.conversationHistory = history;
      return;
    }

    if (state.conversationId !== requestConversationId) {
      if (!failed) {
        upsertConversationFromResult({ ...payload, conversation_id: requestConversationId, title: requestTitle });
      }
      return;
    }

    if (failed) {
      setResult(payload, { persist: false });
      if (!previousResult?.label) {
        setPrimaryInputsLocked(false);
        renderSelectedImagePreviews();
      }
      return;
    }

    payload.title = requestTitle || titleFromHistory(payload.conversation_history);
    if (!isFollowup) {
      state.lastInputSignature = signature;
      setPrimaryInputsLocked(true);
      $("#questionInput").value = "";
    }
    setResult(payload);
  } catch (error) {
    if (state.conversationId !== requestConversationId) return;
    if (isFollowup) {
      state.followupFailure = { question, error: String(error.message || "") };
      const input = $("#questionInput");
      if (!input.value.trim()) input.value = question;
      return;
    }
    setResult({
      mode: "auto",
      label: "Error",
      summary: error.message,
      error: String(error.message || error),
    }, { persist: false });
    if (!previousResult?.label) setPrimaryInputsLocked(false);
  } finally {
    pendingDetections.delete(requestConversationId);
    updateSubmitButtonLabel();
    renderConversation();
  }
}

function setMobilePanel(panel) {
  $(".workspace").dataset.mobilePanel = panel;
  document.querySelectorAll("[data-mobile-view]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.mobileView === panel));
  });
  window.requestAnimationFrame(() => drawSuspiciousRegions(state.lastResult));
}

function updateSidebarToggle() {
  const isDrawer = window.matchMedia("(max-width: 1100px)").matches;
  const expanded = isDrawer
    ? $("#sidebarPanel").classList.contains("is-open")
    : !$(".app-shell").classList.contains("sidebar-collapsed");
  const action = state.language === "en"
    ? (expanded ? "Collapse sidebar" : "Expand sidebar")
    : (expanded ? "收起侧边栏" : "展开侧边栏");
  const trigger = $("#sidebarToggleBtn");
  trigger.setAttribute("aria-expanded", String(expanded));
  trigger.setAttribute("aria-label", action);
  trigger.title = action;
  trigger.querySelector("svg path").setAttribute("d", expanded ? "M9 4v16M14 9l-3 3 3 3" : "M9 4v16M12 9l3 3-3 3");
}

function setSidebarOpen(open, restoreFocus = true) {
  const sidebar = $("#sidebarPanel");
  const trigger = $("#sidebarToggleBtn");
  const isDrawer = window.matchMedia("(max-width: 1100px)").matches;
  const wasOpen = sidebar.classList.contains("is-open");
  const visible = open && isDrawer;
  sidebar.classList.toggle("is-open", visible);
  sidebar.inert = isDrawer && !visible;
  $(".workspace").inert = visible;
  $("#sidebarBackdrop").hidden = !visible;
  updateSidebarToggle();
  if (visible) {
    sidebar.setAttribute("role", "dialog");
    sidebar.setAttribute("aria-modal", "true");
    $("#sidebarCloseBtn").focus();
    window.requestAnimationFrame(() => {
      if (sidebar.classList.contains("is-open")) $("#sidebarCloseBtn").focus();
    });
  } else {
    sidebar.removeAttribute("role");
    sidebar.removeAttribute("aria-modal");
    if (wasOpen && restoreFocus && isDrawer) trigger.focus();
  }
}

function bindResponsiveControls() {
  const sidebar = $("#sidebarPanel");
  const drawerQuery = window.matchMedia("(max-width: 1100px)");
  const syncDrawer = () => {
    const focusWasInSidebar = sidebar.contains(document.activeElement);
    setSidebarOpen(false, false);
    if (drawerQuery.matches && focusWasInSidebar) $("#sidebarToggleBtn").focus();
  };
  drawerQuery.addEventListener("change", syncDrawer);
  syncDrawer();
  $("#sidebarToggleBtn").addEventListener("click", () => {
    if (drawerQuery.matches) {
      setSidebarOpen(!sidebar.classList.contains("is-open"));
    } else {
      $(".app-shell").classList.toggle("sidebar-collapsed");
      updateSidebarToggle();
      window.requestAnimationFrame(() => drawSuspiciousRegions(state.lastResult));
    }
  });
  $("#sidebarCloseBtn").addEventListener("click", () => setSidebarOpen(false));
  $("#sidebarBackdrop").addEventListener("click", () => setSidebarOpen(false));
  document.querySelectorAll("[data-mobile-view]").forEach((button) => {
    button.addEventListener("click", () => setMobilePanel(button.dataset.mobileView));
  });
  $(".skip-link").addEventListener("click", () => setMobilePanel("content"));
  document.addEventListener("keydown", (event) => {
    if (!sidebar.classList.contains("is-open")) return;
    if (event.key === "Escape") {
      event.preventDefault();
      setSidebarOpen(false);
    } else if (event.key === "Tab") {
      const focusable = Array.from(sidebar.querySelectorAll("button:not(:disabled), [href], input:not(:disabled)"))
        .filter((element) => element.getClientRects().length);
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!sidebar.contains(document.activeElement)) {
        event.preventDefault();
        first?.focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    }
  });
  setMobilePanel(state.conversationHistory.length ? "dialogue" : "content");
}

function setTextFileStatus(message, status = "") {
  const notice = $("#textFileStatus");
  notice.textContent = message;
  notice.title = message;
  notice.dataset.status = status;
  notice.hidden = !message;
}

function resetTextFileImport() {
  state.textFileReadToken += 1;
  state.textFileReading = false;
  $("#textFileInput").value = "";
  setTextFileStatus("");
  updateUrlImportButton();
}

function decodeTextFile(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes[0] === 0xff && bytes[1] === 0xfe) return new TextDecoder("utf-16le", { fatal: true }).decode(bytes);
  if (bytes[0] === 0xfe && bytes[1] === 0xff) return new TextDecoder("utf-16be", { fatal: true }).decode(bytes);
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return new TextDecoder("gb18030", { fatal: true }).decode(bytes);
  }
}

async function importTextFile(file) {
  if (!file || state.primaryInputLocked || state.urlImportPending || state.textFileReading) return;
  const english = state.language === "en";
  if (!/\.txt$/i.test(file.name)) {
    setTextFileStatus(english ? "Choose a .txt file." : "请选择 .txt 文件。", "error");
    return;
  }
  const token = ++state.textFileReadToken;
  const textarea = $("#textInput");
  const originalText = textarea.value;
  state.textFileReading = true;
  updateUrlImportButton();
  updateSubmitButtonLabel();
  setTextFileStatus(english ? "Reading TXT…" : "正在读取 TXT…");
  try {
    const text = decodeTextFile(await file.arrayBuffer());
    if (token !== state.textFileReadToken || state.primaryInputLocked) return;
    if (!text.trim() || text.includes("\0")) {
      setTextFileStatus(english ? "The file contains no readable text." : "文件为空或没有可读取的文本。", "error");
      return;
    }
    if (textarea.value !== originalText) {
      setTextFileStatus(english ? "The text was edited while reading. Choose the file again to import it." : "读取时正文已被编辑，请重新选择文件以导入。", "error");
      return;
    }
    textarea.value = text;
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    setTextFileStatus(english ? `Imported: ${file.name}` : `已读取：${file.name}`, "success");
  } catch {
    if (token === state.textFileReadToken) {
      setTextFileStatus(english ? "Could not read the file. Try saving it as UTF-8." : "读取失败，请将文件另存为 UTF-8 后重试。", "error");
    }
  } finally {
    if (token === state.textFileReadToken) {
      state.textFileReading = false;
      updateUrlImportButton();
      updateSubmitButtonLabel();
    }
  }
}

function bindTextFileImport() {
  const picker = $("#textFileInput");
  $("#uploadTextBtn").addEventListener("click", () => picker.click());
  picker.addEventListener("change", () => {
    const file = picker.files[0];
    picker.value = "";
    importTextFile(file);
  });
}

function bindConversationDraftTitle() {
  ["#textInput", "#questionInput"].forEach((selector) => {
    $(selector).addEventListener("input", () => {
      persistCurrentConversationDraft();
      updateSubmitButtonLabel();
      renderConversation();
    });
  });
  $("#imageInput").addEventListener("change", () => {
    persistCurrentConversationDraft();
    updateSubmitButtonLabel();
    renderConversation();
  });
}

bindSegmentedControls();
bindThemeToggle();
document.querySelectorAll(".segment").forEach((button) => button.setAttribute("aria-pressed", String(button.classList.contains("active"))));
syncImageInputMode();
bindImagePreview();
bindUrlImport();
bindTextFileImport();
bindImageViewer();
updateWebpageUrlConfirmButton();
bindConversationDraftTitle();
loadConversation();
bindResponsiveControls();
updateSubmitButtonLabel();
$("#submitBtn").addEventListener("click", submitDetection);
$("#newConversationBtn").addEventListener("click", () => {
  newConversation();
  setMobilePanel("content");
  setSidebarOpen(false);
});
$("#toggleHistoryDeleteBtn").addEventListener("click", () => {
  state.historyDeleteEnabled = !state.historyDeleteEnabled;
  renderConversation();
});
$("#languageToggleBtn").addEventListener("click", () => {
  state.language = state.language === "zh" ? "en" : "zh";
  updateTaskHeading();
  const pill = $("#statusPill");
  const status = pill.classList.item(1);
  setStatus(status, pill.textContent);
  if (status === "running") {
    setRunningResultView();
  } else {
    renderConclusion(state.lastResult, state.language);
    renderResultMetrics(state.lastResult, state.language);
  }
  renderConversation();
  updateThemeToggle();
  updateSubmitButtonLabel();
});
$("#conversationList").addEventListener("click", (event) => {
  const deleteButton = event.target.closest(".conversation-delete");
  if (deleteButton) {
    if (!state.historyDeleteEnabled) return;
    event.preventDefault();
    event.stopPropagation();
    deleteConversation(deleteButton.dataset.conversationId);
    return;
  }
  const item = event.target.closest(".conversation-item");
  if (!item) return;
  const conversation = findConversation(item.dataset.conversationId);
  selectConversation(item.dataset.conversationId);
  setMobilePanel(conversation?.draft_input && !pendingDetections.has(item.dataset.conversationId) ? "content" : "dialogue");
  setSidebarOpen(false);
});
window.addEventListener("resize", () => drawSuspiciousRegions(state.lastResult));
$("#imagePreview").addEventListener("scroll", () => {
  window.requestAnimationFrame(() => drawSuspiciousRegions(state.lastResult));
});
$("#addImageBtn").addEventListener("click", () => {
  if (!state.primaryInputLocked) $("#imageInput").click();
});
window.addEventListener("beforeunload", clearImageObjectUrl);
