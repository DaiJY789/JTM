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
  primaryInputLocked: false,
  imageSelectionNotice: "",
  confirmedImageUrl: "",
  confirmedWebpageUrl: "",
  urlImportPending: false,
  urlImportToken: 0,
  webpageExtraction: null,
};

const $ = (selector) => document.querySelector(selector);
const CONVERSATION_ID_KEY = "mid_demo_conversation_id";
const CONVERSATION_LIST_KEY = "mid_demo_conversations";
const MAX_CONVERSATION_LIST_ITEMS = 30;
const CONVERSATION_TITLE_LENGTH = 18;
const MAX_MMD_IMAGES = 4;

function imageInputLimit() {
  return state.task === "deepfake_detection" ? 1 : MAX_MMD_IMAGES;
}

function updateTaskHeading() {
  const taskNames = {
    multimodal_fact_checking: "多模态事实核查",
    deepfake_detection: "深度伪造检测",
    aigc_detection: "大模型生成检测",
  };
  $("#workspaceTaskName").textContent = taskNames[state.task] || taskNames.multimodal_fact_checking;
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
  button.disabled = state.primaryInputLocked || state.urlImportPending || !url || imported;
  input.readOnly = state.primaryInputLocked || state.urlImportPending;
}

function setUrlImportStatus(message, status = "") {
  const notice = $("#urlImportStatus");
  notice.textContent = message;
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
  return selectedImageFiles().length + selectedImageUrls().length;
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
    tamper: "篡改风险",
    forgeryTypes: "伪造类型",
    detectionType: "类型",
    noSummary: "暂无摘要",
    running: "检测中...",
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
    tamper: "Tamper Risk",
    forgeryTypes: "Forgery Types",
    detectionType: "Type",
    noSummary: "No summary",
    running: "Detecting...",
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
  state.conversationId = localStorage.getItem(CONVERSATION_ID_KEY) || createConversationId();
  localStorage.setItem(CONVERSATION_ID_KEY, state.conversationId);
  const current = findConversation(state.conversationId);
  state.conversationHistory = current?.conversation_history || [];
  renderConversation();
  syncConversation(state.conversationId);
  syncConversationList();
}

function newConversation() {
  persistCurrentConversationDraft();
  resetUrlImport();
  state.conversationId = createConversationId();
  state.conversationHistory = [];
  state.lastResult = {};
  state.lastInputSignature = "";
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

function setPrimaryInputsLocked(locked) {
  state.primaryInputLocked = Boolean(locked);
  const textInput = $("#textInput");
  const webpageUrlInput = $("#webpageUrlInput");
  const imageUrlInput = $("#imageUrlInput");
  const imageInput = $("#imageInput");
  const fileButton = document.querySelector(".file-button");
  if (textInput) textInput.readOnly = state.primaryInputLocked;
  if (webpageUrlInput) webpageUrlInput.readOnly = state.primaryInputLocked;
  if (imageUrlInput) imageUrlInput.readOnly = state.primaryInputLocked;
  if (imageInput) imageInput.disabled = state.primaryInputLocked;
  document.querySelectorAll('[data-field="task"] .segment:not([data-value="aigc_detection"])').forEach((button) => {
    button.disabled = state.primaryInputLocked;
  });
  if (fileButton) {
    fileButton.disabled = state.primaryInputLocked;
    fileButton.classList.toggle("disabled", state.primaryInputLocked);
    fileButton.setAttribute("aria-disabled", state.primaryInputLocked ? "true" : "false");
  }
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
  localStorage.setItem(
    CONVERSATION_LIST_KEY,
    JSON.stringify(state.conversations.slice(0, MAX_CONVERSATION_LIST_ITEMS)),
  );
}

function findConversation(conversationId) {
  return state.conversations.find((item) => item.conversation_id === conversationId);
}

function ensureConversationInIndex(conversationId) {
  if (!conversationId || findConversation(conversationId)) return;
  const now = Date.now() / 1000;
  state.conversations.unshift({
    conversation_id: conversationId,
    title: uiText.zh.untitledConversation,
    last_label: "",
    last_summary: "",
    created_at: now,
    updated_at: now,
    history_count: state.conversationHistory.length,
    conversation_history: state.conversationHistory,
  });
  saveConversationIndex();
}

function isDefaultConversationTitle(title) {
  return title === uiText.zh.untitledConversation || title === uiText.en.untitledConversation;
}

function isConversationMeaningful(item) {
  if (!item || !item.conversation_id) return false;
  const history = Array.isArray(item.conversation_history) ? item.conversation_history : [];
  const title = String(item.title || "").trim();
  return Boolean(
    history.length ||
      Number(item.history_count || 0) > 0 ||
      String(item.last_label || "").trim() ||
      String(item.last_summary || "").trim() ||
      (title && !isDefaultConversationTitle(title)),
  );
}

function compactConversationTitle(value) {
  const normalized = String(value || "").replace(/\s+/g, " ").trim();
  if (!normalized) return "";
  return normalized.length > CONVERSATION_TITLE_LENGTH
    ? `${normalized.slice(0, CONVERSATION_TITLE_LENGTH)}...`
    : normalized;
}

function titleFromUserInput(userInput) {
  if (!userInput) return "";
  const firstImageUrl = Array.isArray(userInput.image_urls) ? userInput.image_urls[0] : "";
  const candidates = [userInput.text, userInput.question, userInput.image_url, firstImageUrl];
  const value = candidates.map((item) => String(item || "").trim()).find(Boolean);
  if (value) return compactConversationTitle(value);
  return userInput.has_image ? "图片检测" : "";
}

function titleFromCurrentInput() {
  const imageFile = $("#imageInput")?.files?.[0];
  const candidates = [
    $("#textInput")?.value,
    $("#webpageUrlInput")?.value,
    $("#questionInput")?.value,
    $("#imageUrlInput")?.value,
    imageFile?.name,
  ];
  const value = candidates.map((item) => String(item || "").trim()).find(Boolean);
  return value ? compactConversationTitle(value) : "";
}

function titleFromHistory(history) {
  const userItem = [...(history || [])].reverse().find((item) => item.role === "user");
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
  const incomingTitle = conversation.title || "";
  const incomingHasTitle = Boolean(
    incomingTitle && incomingTitle !== uiText.zh.untitledConversation && incomingTitle !== uiText.en.untitledConversation,
  );
  const incomingHasContent = Boolean(
    incomingHasTitle ||
      incomingHistory.length ||
      conversation.history_count ||
      conversation.last_label ||
      conversation.last_summary ||
      titleFromHistory(incomingHistory),
  );
  const fallbackTitle = titleFromHistory(history) || existing.title || uiText.zh.untitledConversation;
  return {
    ...existing,
    conversation_id: conversationId,
    title: incomingHasTitle ? incomingTitle : fallbackTitle,
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
  if (existingIndex >= 0) {
    state.conversations.splice(existingIndex, 1, normalized);
  } else if (insertPosition === "bottom") {
    state.conversations.push(normalized);
  } else {
    state.conversations.unshift(normalized);
  }
  state.conversations = state.conversations.slice(0, MAX_CONVERSATION_LIST_ITEMS);
  saveConversationIndex();
}

function upsertConversationFromResult(result) {
  if (!result?.conversation_id) return;
  const history = Array.isArray(result.conversation_history) ? result.conversation_history : state.conversationHistory;
  upsertConversation({
    conversation_id: result.conversation_id,
    title: result.title || titleFromHistory(history) || titleFromCurrentInput(),
    last_label: result.label || "",
    last_summary: result.summary || result?.user_answer?.conclusion || result?.user_answer?.analysis || "",
    updated_at: Date.now() / 1000,
    history_count: history.length,
    conversation_history: history,
    latest_result: result,
  });
}

function persistCurrentConversationDraft() {
  if (!state.conversationId) return;
  const existing = findConversation(state.conversationId) || {};
  const draftTitle = titleFromCurrentInput();
  const historyTitle = titleFromHistory(state.conversationHistory);
  const title = draftTitle || historyTitle || existing.title;
  const hasContent = Boolean(
    draftTitle ||
      historyTitle ||
      state.conversationHistory.length ||
      state.lastResult?.label ||
      state.lastResult?.summary ||
      existing.history_count,
  );
  if (!hasContent) return;
  upsertConversation({
    conversation_id: state.conversationId,
    title: title || uiText.zh.untitledConversation,
    last_label: state.lastResult?.label || existing.last_label || "",
    last_summary: state.lastResult?.summary || existing.last_summary || "",
    updated_at: Date.now() / 1000,
    history_count: state.conversationHistory.length || existing.history_count || 0,
    conversation_history: state.conversationHistory.length ? state.conversationHistory : existing.conversation_history || [],
    latest_input: existing.latest_input || {},
    latest_result: state.lastResult?.label ? state.lastResult : existing.latest_result || {},
  });
}

async function syncConversationList() {
  try {
    const response = await fetch("/conversations");
    if (!response.ok) return;
    const payload = await response.json();
    state.conversations = [];
    (payload.conversations || []).forEach((item) => upsertConversation(item, { insertPosition: "bottom" }));
    saveConversationIndex();
    renderConversation();
  } catch {
    renderConversation();
  }
}

async function syncConversation(conversationId) {
  if (!conversationId) return;
  const local = findConversation(conversationId);
  try {
    const response = await fetch(`/conversations/${encodeURIComponent(conversationId)}`);
    if (!response.ok) return;
    const payload = await response.json();
    const remoteHistory = Array.isArray(payload.conversation_history) ? payload.conversation_history : [];
    if (remoteHistory.length || !local?.conversation_history?.length) {
      upsertConversation(payload);
      if (state.conversationId === conversationId) {
        state.conversationHistory = remoteHistory;
        if (payload.latest_result && Object.keys(payload.latest_result).length) {
          state.lastResult = payload.latest_result;
          state.lastResult.conversation_id = conversationId;
          state.lastResult.conversation_history = remoteHistory;
          state.lastInputSignature = "";
          setResult(state.lastResult, { persist: false });
          restoreInputPreview(payload.latest_input || {});
          return;
        }
        restoreInputPreview(payload.latest_input || {});
        renderConversation();
      }
    }
  } catch {
    renderConversation();
  }
}

function selectConversation(conversationId) {
  if (!conversationId || conversationId === state.conversationId) return;
  persistCurrentConversationDraft();
  state.conversationId = conversationId;
  localStorage.setItem(CONVERSATION_ID_KEY, conversationId);
  const conversation = findConversation(conversationId);
  state.conversationHistory = conversation?.conversation_history || [];
  state.lastResult = {
    ...(conversation?.latest_result || {}),
    conversation_id: conversationId,
    conversation_history: state.conversationHistory,
    label: conversation?.latest_result?.label || conversation?.last_label || "",
    summary: conversation?.latest_result?.summary || conversation?.last_summary || "",
  };
  setResult(state.lastResult, { persist: false });
  restoreInputPreview(conversation?.latest_input || {});
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
  renderConversationList(language);
  renderConversationHistory(language);
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
    }
    button.appendChild(openButton);

    if (state.historyDeleteEnabled) {
      const deleteButton = document.createElement("button");
      deleteButton.className = "conversation-delete";
      deleteButton.type = "button";
      deleteButton.dataset.conversationId = conversation.conversation_id;
      deleteButton.textContent = "×";
      deleteButton.title = copy.deleteConversation;
      deleteButton.setAttribute("aria-label", `${copy.deleteConversation} ${conversation.title || copy.untitledConversation}`);
      button.appendChild(deleteButton);
    }
    list.appendChild(button);
  });
}

function restoreInputPreview(latestInput) {
  if (!latestInput || typeof latestInput !== "object") return;
  resetUrlImport();
  setPrimaryInputsLocked(false);
  clearImageObjectUrl();
  $("#imageInput").value = "";
  selectSegmentValue("task", latestInput.task || "multimodal_fact_checking");
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

async function deleteConversation(conversationId) {
  if (!state.historyDeleteEnabled || !conversationId) return;
  state.conversations = state.conversations.filter((item) => item.conversation_id !== conversationId);
  saveConversationIndex();
  if (state.conversationId === conversationId) {
    startEmptyConversation();
  } else {
    renderConversation();
  }
  try {
    await fetch(`/conversations/${encodeURIComponent(conversationId)}`, { method: "DELETE" });
  } catch {
    // Local cleanup still keeps the UI responsive if the backend item was already gone.
  }
}

function startEmptyConversation() {
  resetUrlImport();
  state.conversationId = createConversationId();
  state.conversationHistory = [];
  state.lastResult = {};
  state.lastInputSignature = "";
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

function renderConversationHistory(language) {
  const list = $("#conversationHistory");
  const copy = uiText[language];
  list.classList.toggle("is-empty", !state.conversationHistory.length);
  if (!state.conversationHistory.length) {
    list.textContent = language === "en" ? "Your conversation will appear here after detection." : "检测完成后，可在这里查看对话并继续追问";
    return;
  }
  list.innerHTML = "";
  state.conversationHistory.slice(-8).forEach((item) => {
    const node = document.createElement("div");
    node.className = `history-item ${item.role === "user" ? "user-message" : "assistant-message"}`;
    const role = document.createElement("strong");
    role.textContent = item.role === "user" ? (language === "en" ? "You" : "你") : (language === "en" ? "Assistant" : "惊堂木");
    const content = document.createElement("p");
    content.textContent = item.content || item.label || "";
    node.appendChild(role);
    node.appendChild(content);
    list.appendChild(node);
  });
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
    ? payload.images.map((item) => String(item || "").trim()).filter(Boolean).slice(0, imageInputLimit())
    : [];
  if (!images.length && payload?.image_url) images.push(String(payload.image_url).trim());

  if (extractedText) $("#textInput").value = extractedText;
  state.webpageExtraction = { url: requestedUrl, text: extractedText, images };
  if (!selectedImageCount()) {
    if (images.length === 1) {
      renderImagePreview(images[0], imageNameFromUrl(images[0], 0), { noReferrer: true });
    } else if (images.length > 1) {
      renderImagePreviews(images, images.map((url, index) => imageNameFromUrl(url, index)));
    } else {
      renderEmptyPreview();
    }
  }
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
  if (!url || state.primaryInputLocked || state.urlImportPending) return;
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
      setUrlImportStatus(`已识别为网页，已提取正文和 ${state.webpageExtraction.images.length} 张图片。`, "success");
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
    return new URL(url).hostname || `remote-${index + 1}`;
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
  const urls = selectedImageUrls();
  clearImageObjectUrl();
  state.imageObjectUrls = files.map((file) => URL.createObjectURL(file));
  const sources = [...state.imageObjectUrls, ...urls];
  const names = [
    ...files.map((file) => file.name),
    ...urls.map((url, index) => imageNameFromUrl(url, index)),
  ];
  if (!sources.length) {
    const webpageImages = selectedWebpageUrl()
      ? (state.webpageExtraction?.images || []).slice(0, imageInputLimit())
      : [];
    if (webpageImages.length === 1) {
      renderImagePreview(
        webpageImages[0],
        imageNameFromUrl(webpageImages[0], 0),
        { noReferrer: true },
      );
    } else if (webpageImages.length > 1) {
      renderImagePreviews(
        webpageImages,
        webpageImages.map((url, index) => imageNameFromUrl(url, index)),
      );
    } else {
      renderEmptyPreview();
    }
  } else if (sources.length === 1) {
    renderImagePreview(sources[0], names[0] || "Image 1", { noReferrer: urls.length === 1 });
  } else {
    renderImagePreviews(sources, names);
  }
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
    <p>点击右上角 + 添加图片，或在下方导入 URL<br>图片将以原始比例显示</p>
  </div>`;
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
  img.addEventListener("error", () => {
    preview.classList.remove("has-image");
    preview.innerHTML = "<span>历史图片暂时无法加载。</span>";
  });
  img.src = src;
  img.addEventListener("load", () => drawSuspiciousRegions(state.lastResult));
  preview.appendChild(img);

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
    renderImagePreview(sources[0], names[0] || "Image 1");
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
    img.addEventListener("error", () => {
      item.classList.add("preview-item-error");
      item.innerHTML = "<span>图片暂时无法加载</span>";
    });
    img.src = src;
    const caption = document.createElement("figcaption");
    caption.textContent = `Image ${index + 1} · ${names[index] || "image"}`;
    item.appendChild(img);
    item.appendChild(caption);
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
}

function setRunningResultView() {
  $("#conversationConclusion").hidden = false;
  $("#conversationConclusion").open = true;
  const language = state.language === "en" ? "en" : "zh";
  const copy = uiText[language];
  renderVerdictBadge($("#resultLabel"), "", language);
  $("#resultConfidence").textContent = "-";
  $("#resultRuntime").textContent = "-";
  $("#aiScore").textContent = "-";
  $("#summaryBox").textContent = copy.running;
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
  const copy = uiText[language];
  $("#labelTitle").textContent = copy.label;
  $("#confidenceTitle").textContent = copy.confidence;
  $("#aiTitle").textContent = result?.type ? copy.detectionType : copy.forgeryTypes;
  renderVerdictBadge($("#resultLabel"), result?.label, language);
  $("#resultConfidence").textContent = formatConfidence(result?.confidence, language);
  $("#resultRuntime").textContent = formatRuntime(result?.runtime_seconds, language);
  renderForensicMetrics(result, language);
  const hasResult = Boolean(result?.label || result?.summary || result?.error);
  $("#conversationConclusion").hidden = !hasResult;
  if (!hasResult) $("#aiScore").textContent = "-";
  $("#summaryBox").textContent = hasResult ? formatUserAnswer(result, language)
    : language === "en" ? "Add content to see the conclusion and supporting analysis here." : "输入待核查内容后，这里将展示检测结论与分析依据。";
  setStatus(result?.label === "Error" || result?.decision_status === "technical_error" ? "error" : hasResult ? "done" : "idle");
  updateSubmitButtonLabel();
  renderConversation();
  drawSuspiciousRegions(result);
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

function renderForensicMetrics(result, language) {
  const verdict = result?.image_verdict || result?.raw?.image_forensics || {};
  $("#aiScore").textContent = "-";
  if (result?.type) {
    $("#aiScore").textContent = formatDetectionType(result.type, language);
    return;
  }
  $("#aiScore").textContent = formatForgeryTypes(result?.forgery_types || verdict.forgery_types, language);
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
    ai_generated: "局部重绘",
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
    ai_generated: "Local repainting",
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
  if (Number.isNaN(value)) return "-";
  return language === "en" ? `${value.toFixed(3)} s` : `${value.toFixed(3)} 秒`;
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
  if (!result || Object.keys(result).length === 0) return copy.noSummary;
  if (result.decision_status === "technical_error") return cleanText(result.summary || busyDetectionMessage(language));
  if (result.label === "Error") return cleanText(result.summary || result.error || copy.noSummary);
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

function formatMmdAnswer(result, language) {
  const copy = uiText[language];
  const detection = result?.mmd_detection || result?.raw?.mmd_detection || {};
  const explanation = detection?.explanation || detection?.user_explanation || result?.user_answer || {};
  const conclusion = firstCleanText(
    explanation.verdict_statement,
    explanation.headline,
    result?.user_answer?.judgment,
    fallbackConclusion(result?.label, language),
  );
  const reasons = [];
  const summary = firstCleanText(explanation.analysis, explanation.summary, result?.user_answer?.reason_analysis, result?.summary);
  if (summary) reasons.push(summary);

  const findings = Array.isArray(explanation.key_findings) ? explanation.key_findings : [];
  findings.forEach((item) => {
    if (!item || typeof item !== "object") return;
    const title = cleanText(item.title);
    const findingResult = cleanText(item.result);
    const detail = cleanText(item.detail);
    const prefix = [title, findingResult].filter(Boolean).join(language === "en" ? ": " : "：");
    const line = [prefix, detail].filter(Boolean).join(language === "en" ? ". " : "。 ");
    if (line) reasons.push(line);
  });

  const notes = [
    ...cleanList(explanation.limitations),
    ...cleanList(result?.user_answer?.notes),
  ];
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
  const answer = result?.user_answer || result?.text_verdict?.user_answer || {};
  const conclusion = cleanText(answer.judgment) || cleanText(answer.conclusion) || fallbackConclusion(result?.label, language);
  const reasons = [];
  reasons.push(`${copy.textSide}：${formatLabel(result?.label, language)}`);

  const claimCount = Array.isArray(result?.claims) ? result.claims.length : 0;
  if (claimCount) reasons.push(`${copy.checkedClaims}：${claimCount}`);

  const evidenceCount = Array.isArray(result?.evidence) ? result.evidence.length : 0;
  if (evidenceCount) reasons.push(`${copy.foundEvidence}：${evidenceCount}`);

  const analysis = firstCleanText(answer.reason_analysis, answer.analysis, result?.summary, result?.appraisalResults);
  const splitAnalysis = splitReasonAndNotes(analysis);
  if (splitAnalysis.reason) reasons.push(splitAnalysis.reason);

  return buildFriendlyBlock(copy, conclusion, reasons, [...cleanList(answer.notes), ...splitAnalysis.notes]);
}

function formatImageAnswer(result, language) {
  const copy = uiText[language];
  const verdict = result?.image_verdict || result?.raw?.image_forensics || result || {};
  const answer = result?.user_answer || verdict?.user_answer || {};
  const label = result?.label || verdict?.label;
  const conclusion = cleanText(answer.judgment) || cleanText(answer.conclusion) || fallbackConclusion(label, language);
  return buildResultPanelBlock(copy, conclusion, cleanList(answer.notes));
}

function buildResultPanelBlock(copy, judgment, notes) {
  const lines = [copy.finalJudgment, judgment || copy.noSummary];
  const visibleNotes = dedupeText(notes || []);
  if (visibleNotes.length) {
    lines.push("", copy.notes, ...visibleNotes.map((item) => `- ${item}`));
  }
  return lines.join("\n");
}

function formatMultimodalAnswer(result, language) {
  const copy = uiText[language];
  const answer = result?.user_answer || {};
  const conclusion = cleanText(answer.judgment) || cleanText(answer.conclusion) || fallbackConclusion(result?.label, language);
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
    const analysis = firstCleanText(answer.reason_analysis, answer.analysis, result?.summary);
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
  const reasonParagraph = toReasonParagraph(reasons, copy.noSummary);
  const lines = [
    copy.finalJudgment,
    conclusion || copy.noSummary,
    "",
    copy.mainReasons,
    reasonParagraph,
  ];
  const visibleNotes = dedupeText(notes || []);
  if (visibleNotes.length) {
    lines.push("", copy.notes, ...visibleNotes.map((item) => `- ${item}`));
  }
  return lines.join("\n");
}

function toReasonParagraph(reasons, fallback) {
  const items = dedupeText(reasons).map(trimTrailingPunctuation).filter(Boolean);
  if (!items.length) return fallback;
  return items.map(ensureSentence).join(" ");
}

function ensureSentence(text) {
  const value = cleanText(text);
  if (!value) return "";
  if (/[。.!?！？]$/.test(value)) return value;
  return `${value}。`;
}

function trimTrailingPunctuation(text) {
  return cleanText(text).replace(/[。.!?！？]+$/g, "");
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
    ...(Array.isArray(result?.mmd_detection?.explanation?.limitations) ? result.mmd_detection.explanation.limitations : []),
    ...(Array.isArray(result?.mmd_detection?.user_explanation?.limitations) ? result.mmd_detection.user_explanation.limitations : []),
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
  if (webpageUrl) formData.append("webpage_url", webpageUrl);
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
    image_urls: selectedImageUrls(),
    image_files: imageFiles.map(
      (file) => `${file.name}:${file.size}:${file.lastModified}`,
    ),
  });
}

function hasDetectionResult() {
  return Boolean(state.conversationId && state.lastResult?.label && state.lastResult.label !== "Error");
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
    return Boolean(
      (selectedWebpageUrl() && state.webpageExtraction?.images?.length) ||
        selectedImageCount(),
    );
  }
  if (selectedWebpageUrl()) return true;
  const hasText = Boolean($("#textInput").value.trim());
  return hasText;
}

function defaultQuestionForCurrentInput() {
  const language = state.language === "en" ? "en" : "zh";
  const hasImage = selectedImageCount() > 0;
  const hasText = Boolean($("#textInput").value.trim());
  const hasWebpage = Boolean(selectedWebpageUrl());
  if (state.task === "deepfake_detection") {
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
  button.disabled = state.urlImportPending || (isFollowupContext ? !question : !hasPrimaryInput());
}

function ensureImagePreviewBeforeSubmit(preserveExisting = false) {
  if (preserveExisting && state.imagePreviewAssets.length) {
    const preview = $("#imagePreview");
    if (!preview?.classList.contains("has-image")) {
      renderRememberedImagePreviews();
    }
    return;
  }
  if (selectedWebpageUrl() && state.webpageExtraction?.images?.length && !selectedImageCount()) {
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
  if (state.urlImportPending) return;
  const submitBtn = $("#submitBtn");
  const requestTitle = titleFromCurrentInput();
  const isFollowup = shouldUseFollowup();
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
    persistCurrentConversationDraft();
    if (!isFollowup) {
      setPrimaryInputsLocked(true);
    }
    setStatus("running", "running");
    setRunningResultView();

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

    if (payload.decision_status === "technical_error") {
      setStatus("error", "error");
      setResult(payload, { persist: false });
      if (!isFollowup && !previousResult?.label) setPrimaryInputsLocked(false);
      return;
    }

    setStatus("done", "done");
    payload.title = requestTitle || titleFromHistory(payload.conversation_history);
    if (!isFollowup) {
      state.lastInputSignature = signature;
      setPrimaryInputsLocked(true);
    }
    setResult(payload);
    $("#questionInput").value = "";
    updateSubmitButtonLabel();
  } catch (error) {
    setStatus("error", "error");
    const errorResult = {
      ...(isFollowup && previousResult?.label ? previousResult : {}),
      mode: "auto",
      label: "Error",
      summary: error.message,
      error: String(error.message || error),
    };
    setResult(errorResult, { persist: false });
    if (!isFollowup && !previousResult?.label) {
      setPrimaryInputsLocked(false);
    }
  } finally {
    updateSubmitButtonLabel();
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
document.querySelectorAll(".segment").forEach((button) => button.setAttribute("aria-pressed", String(button.classList.contains("active"))));
syncImageInputMode();
bindImagePreview();
bindUrlImport();
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
  renderConversation();
  const label = $("#statusPill").classList.contains("running") ? "" : state.lastResult?.label;
  renderVerdictBadge($("#resultLabel"), label, state.language);
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
  selectConversation(item.dataset.conversationId);
  setMobilePanel("dialogue");
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
