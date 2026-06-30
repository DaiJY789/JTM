const state = {
  mode: "auto",
  language: "zh",
  lastResult: {},
  imageObjectUrl: "",
  imagePreviewSource: "",
  conversationId: "",
  conversationHistory: [],
  conversations: [],
  lastInputSignature: "",
  primaryInputLocked: false,
};

const $ = (selector) => document.querySelector(selector);
const CONVERSATION_ID_KEY = "mid_demo_conversation_id";
const CONVERSATION_LIST_KEY = "mid_demo_conversations";
const MAX_CONVERSATION_LIST_ITEMS = 30;
const CONVERSATION_TITLE_LENGTH = 18;

const labelText = {
  zh: {
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
    provenance: "来源凭证",
    noCredential: "未发现",
    credentialPresent: "已发现",
    aiDeclared: "声明AI生成",
    editDeclared: "声明编辑",
    noSummary: "暂无摘要",
    running: "检测中...",
    analysis: "分析过程",
    conclusion: "最终结论",
    finalJudgment: "最终判断",
    mainReasons: "主要原因",
    notes: "说明",
    textSide: "文本方面",
    imageSide: "图片方面",
    alignmentSide: "图文关系",
    checkedClaims: "已核查关键事实点",
    foundEvidence: "检索到相关证据",
    newConversation: "新对话",
    conversation: "对话",
    history: "对话历史",
    historyList: "历史记录",
    historyCount: "条历史",
    noHistory: "暂无历史",
    noConversations: "暂无历史对话",
    refresh: "刷新",
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
    provenance: "Provenance",
    noCredential: "None",
    credentialPresent: "Present",
    aiDeclared: "AI declared",
    editDeclared: "Edit declared",
    noSummary: "No summary",
    running: "Detecting...",
    analysis: "Analysis",
    conclusion: "Conclusion",
    finalJudgment: "Final judgment",
    mainReasons: "Main reasons",
    notes: "Notes",
    textSide: "Text",
    imageSide: "Image",
    alignmentSide: "Image-text relation",
    checkedClaims: "Checked key claims",
    foundEvidence: "Found related evidence",
    newConversation: "New Chat",
    conversation: "Conversation",
    history: "History",
    historyList: "History",
    historyCount: "history items",
    noHistory: "No history",
    noConversations: "No conversations",
    refresh: "Refresh",
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
  state.conversationId = createConversationId();
  state.conversationHistory = [];
  state.lastResult = {};
  state.lastInputSignature = "";
  setPrimaryInputsLocked(false);
  localStorage.setItem(CONVERSATION_ID_KEY, state.conversationId);
  $("#textInput").value = "";
  $("#questionInput").value = "";
  $("#webpageUrlInput").value = "";
  $("#imageUrlInput").value = "";
  $("#imageInput").value = "";
  clearImageObjectUrl();
  renderEmptyPreview();
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
  if (fileButton) {
    fileButton.classList.toggle("disabled", state.primaryInputLocked);
    fileButton.setAttribute("aria-disabled", state.primaryInputLocked ? "true" : "false");
  }
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
  const candidates = [userInput.text, userInput.question, userInput.image_url];
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
  $("#refreshConversationsBtn").textContent = copy.refresh;
  $("#conversationListTitle").textContent = copy.historyList;
  const conversationTitle = $("#conversationTitle");
  if (conversationTitle) {
    conversationTitle.textContent = current?.title ? `${current.title} · ${shortId}` : `${copy.conversation} ${shortId}`;
  }
  $("#historyTitle").textContent = copy.history;
  const conversationCount = $("#conversationCount");
  if (conversationCount) {
    conversationCount.textContent =
      language === "en"
        ? `${state.conversationHistory.length} ${copy.historyCount}`
        : `${state.conversationHistory.length} ${copy.historyCount}`;
  }
  renderConversationList(language);
  renderConversationHistory(language);
}

function renderConversationList(language) {
  const list = $("#conversationList");
  const copy = uiText[language];
  if (!state.conversations.length) {
    list.textContent = copy.noConversations;
    return;
  }
  list.innerHTML = "";
  state.conversations.forEach((conversation) => {
    const button = document.createElement("button");
    button.className = `conversation-item${conversation.conversation_id === state.conversationId ? " active" : ""}`;
    button.type = "button";
    button.dataset.conversationId = conversation.conversation_id;

    const title = document.createElement("strong");
    title.textContent = conversation.title || copy.untitledConversation;
    const meta = document.createElement("span");
    const label = formatLabel(conversation.last_label, language);
    const countText =
      language === "en"
        ? `${conversation.history_count || 0} ${copy.historyCount}`
        : `${conversation.history_count || 0} ${copy.historyCount}`;
    meta.textContent = label && label !== "-" ? `${label} · ${countText}` : countText;
    button.appendChild(title);
    button.appendChild(meta);

    const deleteButton = document.createElement("button");
    deleteButton.className = "conversation-delete";
    deleteButton.type = "button";
    deleteButton.dataset.conversationId = conversation.conversation_id;
    deleteButton.textContent = "×";
    deleteButton.title = copy.deleteConversation;
    button.appendChild(deleteButton);
    list.appendChild(button);
  });
}

function restoreInputPreview(latestInput) {
  if (!latestInput || typeof latestInput !== "object") return;
  setPrimaryInputsLocked(false);
  clearImageObjectUrl();
  $("#imageInput").value = "";
  $("#textInput").value = latestInput.text || "";
  $("#questionInput").value = "";
  $("#webpageUrlInput").value = latestInput.webpage_url || "";
  $("#imageUrlInput").value = latestInput.image_url || "";
  const imageAssetUrl = latestInput.image_asset?.url || "";
  if (imageAssetUrl) {
    renderImagePreview(imageAssetUrl, latestInput.image_asset?.filename || "uploaded image");
  } else if (latestInput.image_url) {
    renderImagePreviewFromUrl(latestInput.image_url);
  } else {
    renderEmptyPreview();
  }
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
  if (!conversationId) return;
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
  state.conversationId = createConversationId();
  state.conversationHistory = [];
  state.lastResult = {};
  state.lastInputSignature = "";
  setPrimaryInputsLocked(false);
  localStorage.setItem(CONVERSATION_ID_KEY, state.conversationId);
  $("#textInput").value = "";
  $("#questionInput").value = "";
  $("#webpageUrlInput").value = "";
  $("#imageUrlInput").value = "";
  $("#imageInput").value = "";
  clearImageObjectUrl();
  renderEmptyPreview();
  setResult({}, { persist: false });
  setStatus("idle", "idle");
  renderConversation();
}

function renderConversationHistory(language) {
  const list = $("#conversationHistory");
  const copy = uiText[language];
  if (!state.conversationHistory.length) {
    list.textContent = copy.noHistory;
    return;
  }
  list.innerHTML = "";
  state.conversationHistory.slice(-8).forEach((item) => {
    const node = document.createElement("div");
    node.className = `history-item ${item.role === "user" ? "user-message" : "assistant-message"}`;
    const role = document.createElement("strong");
    role.textContent = item.role || "-";
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
      if (!button) return;
      group.querySelectorAll(".segment").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state[group.dataset.field] = button.dataset.value;
      if (group.dataset.field === "language") {
        renderConversation();
      }
      updateSubmitButtonLabel();
    });
  });
}

function bindImagePreview() {
  $("#imageInput").addEventListener("change", () => {
    const file = $("#imageInput").files[0];
    if (!file) {
      clearImageObjectUrl();
      renderImagePreviewFromUrl($("#imageUrlInput").value.trim());
      return;
    }
    clearImageObjectUrl();
    state.imageObjectUrl = URL.createObjectURL(file);
    renderImagePreview(state.imageObjectUrl, file.name);
    updateSubmitButtonLabel();
  });

  $("#imageUrlInput").addEventListener("change", () => {
    if (!$("#imageInput").files[0]) {
      renderImagePreviewFromUrl($("#imageUrlInput").value.trim());
    }
    updateSubmitButtonLabel();
  });
}

function clearImageObjectUrl() {
  if (state.imageObjectUrl) {
    URL.revokeObjectURL(state.imageObjectUrl);
    state.imageObjectUrl = "";
  }
}

function renderImagePreviewFromUrl(url) {
  if (!url) {
    renderEmptyPreview();
    return;
  }
  renderImagePreview(url, "image url");
}

function renderEmptyPreview() {
  state.imagePreviewSource = "";
  const preview = $("#imagePreview");
  preview.classList.remove("has-image");
  preview.innerHTML = "<span>未选择图片</span>";
}

function renderImagePreview(src, alt) {
  state.imagePreviewSource = src;
  const preview = $("#imagePreview");
  preview.classList.add("has-image");
  preview.innerHTML = "";

  const img = document.createElement("img");
  img.className = "preview-image";
  img.alt = alt;
  img.src = src;
  img.addEventListener("load", () => drawSuspiciousRegions(state.lastResult));
  preview.appendChild(img);

  const overlay = document.createElement("div");
  overlay.className = "region-overlay";
  preview.appendChild(overlay);
}

function setStatus(status, text) {
  const pill = $("#statusPill");
  pill.className = `status-pill ${status}`;
  pill.textContent = text;
}

function setRunningResultView() {
  const language = state.language === "en" ? "en" : "zh";
  const copy = uiText[language];
  $("#resultLabel").textContent = "-";
  $("#resultConfidence").textContent = "-";
  $("#resultRuntime").textContent = "-";
  $("#aiScore").textContent = "-";
  $("#summaryBox").textContent = copy.running;
  $("#jsonOutput").textContent = JSON.stringify(state.lastResult || {}, null, 2);
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
  const language = state.language === "en" ? "en" : "zh";
  const copy = uiText[language];
  $("#labelTitle").textContent = copy.label;
  $("#confidenceTitle").textContent = copy.confidence;
  $("#aiTitle").textContent = result?.type ? copy.detectionType : copy.forgeryTypes;
  $("#resultLabel").textContent = formatLabel(result?.label, language);
  $("#resultConfidence").textContent = formatConfidence(result?.confidence, language);
  $("#resultRuntime").textContent = formatRuntime(result?.runtime_seconds, language);
  renderForensicMetrics(result, language);
  $("#summaryBox").textContent = formatUserAnswer(result, language);
  $("#jsonOutput").textContent = JSON.stringify(result || {}, null, 2);
  updateSubmitButtonLabel();
  renderConversation();
  drawSuspiciousRegions(result);
}

function renderExtractedWebpageInput(result) {
  const webpage = result?.raw?.webpage;
  if (!webpage || typeof webpage !== "object") return;
  const extractedText = [webpage.title, webpage.text].filter(Boolean).join("\n\n").trim();
  const extractedImageUrl = String(webpage.image_url || "").trim();
  const sourceUrl = String(webpage.url || webpage.final_url || "").trim();

  if (sourceUrl) {
    $("#webpageUrlInput").value = sourceUrl;
  }
  if (extractedText) {
    $("#textInput").value = extractedText;
  }
  if (extractedImageUrl) {
    $("#imageUrlInput").value = extractedImageUrl;
    if (!$("#imageInput").files[0]) {
      renderImagePreviewFromUrl(extractedImageUrl);
    }
  }
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

function formatProvenance(provenance, language) {
  const copy = uiText[language];
  if (!provenance || !provenance.has_c2pa) return copy.noCredential;
  if (provenance.declared_ai_generated) return copy.aiDeclared;
  if (provenance.declared_edited) return copy.editDeclared;
  return copy.credentialPresent;
}

function formatLabel(label, language) {
  if (!label) return "-";
  return labelText[language][label] || label;
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
  if (result.label === "Error") return cleanText(result.summary || result.error || copy.noSummary);
  if (result.mode === "multimodal" || result.alignment_verdict) {
    return formatMultimodalAnswer(result, language);
  }
  if (result.mode === "image_only" || result.image_verdict || result.raw?.image_forensics) {
    return formatImageAnswer(result, language);
  }
  return formatTextAnswer(result, language);
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

  const textLabel = result?.text_verdict?.label || result?.raw?.text_detection?.label;
  if (textLabel) statusParts.push(`${copy.textSide}：${formatLabel(textLabel, language)}`);

  const imageLabel = result?.image_verdict?.label || result?.raw?.image_detection?.label;
  if (imageLabel) statusParts.push(`${copy.imageSide}：${formatLabel(imageLabel, language)}`);

  const alignmentLabel = result?.alignment_verdict?.label;
  if (alignmentLabel) statusParts.push(`${copy.alignmentSide}：${formatLabel(alignmentLabel, language)}`);

  const mismatch = cleanList(result?.alignment_verdict?.mismatched_elements);
  if (mismatch.length) {
    reasons.push(language === "en" ? `Main mismatch: ${mismatch.join("; ")}` : `主要不一致点：${mismatch.join("；")}`);
  }

  const matched = cleanList(result?.alignment_verdict?.matched_elements);
  if (matched.length) {
    reasons.push(language === "en" ? `Supported elements: ${matched.join("; ")}` : `能够对应的信息：${matched.join("；")}`);
  }

  const alignmentReason = cleanText(result?.alignment_verdict?.reasoning || result?.alignment_verdict?.analysis);
  const splitAlignmentReason = splitReasonAndNotes(alignmentReason);
  if (splitAlignmentReason.reason) reasons.push(splitAlignmentReason.reason);
  notes.push(...splitAlignmentReason.notes);

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
  const rawImage = result?.raw?.image_forensics || result?.raw?.image_detection?.raw?.image_forensics || {};
  const limitations = [
    ...(Array.isArray(verdict?.limitations) ? verdict.limitations : []),
    ...(Array.isArray(rawImage?.limitations) ? rawImage.limitations : []),
    ...(Array.isArray(result?.alignment_verdict?.limitations) ? result.alignment_verdict.limitations : []),
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
    Facial_Biological_Auditor: "人脸结构检查",
    TruFor_Detector: "深度伪造检查",
    CATNet_Detector: "压缩痕迹检查",
    DIRE_Detector: "AI生成痕迹检查",
  };
  const enNames = {
    SRM_Filter: "texture-noise check",
    Noise_Analyzer: "local-noise check",
    PatchDense_Scanner: "duplicate-region check",
    CFA_Validator: "camera-consistency check",
    Spectrum_Grid_Analyzer: "synthetic-pattern check",
    Facial_Biological_Auditor: "face-structure check",
    TruFor_Detector: "deep-forgery check",
    CATNet_Detector: "compression-trace check",
    DIRE_Detector: "AI-generation trace check",
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
    均无异常: "文本和图像单模态检测均未发现明确异常。",
    全部异常: "文本和图像单模态检测均发现异常。",
  };
  const en = {
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
    均无异常: "No clear single-modal anomaly was found in either the text or the image.",
    全部异常: "Both the text and image detectors found anomalies.",
  };
  return (language === "en" ? en : zh)[label] || formatLabel(label, language);
}

function buildFormData() {
  const formData = new FormData();
  const text = $("#textInput").value.trim();
  const question = resolvedQuestionForSubmit();
  const webpageUrl = $("#webpageUrlInput").value.trim();
  const imageUrl = $("#imageUrlInput").value.trim();
  const imageFile = $("#imageInput").files[0];

  if (text) formData.append("text", text);
  if (question) formData.append("question", question);
  if (webpageUrl) formData.append("webpage_url", webpageUrl);
  if (imageUrl) formData.append("image_url", imageUrl);
  if (imageFile) formData.append("image", imageFile);
  formData.append("mode", state.mode);
  formData.append("language", state.language);
  formData.append("conversation_id", state.conversationId);
  formData.append("history", "[]");
  return formData;
}

function buildFollowupFormData(question) {
  const formData = new FormData();
  formData.append("question", question);
  formData.append("conversation_id", state.conversationId);
  formData.append("language", state.language);
  return formData;
}

function currentInputSignature() {
  const imageFile = $("#imageInput").files[0];
  return JSON.stringify({
    mode: state.mode,
    text: $("#textInput").value.trim(),
    webpage_url: $("#webpageUrlInput").value.trim(),
    image_url: $("#imageUrlInput").value.trim(),
    image_file: imageFile ? `${imageFile.name}:${imageFile.size}:${imageFile.lastModified}` : "",
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
  return Boolean(
    $("#textInput").value.trim() ||
      $("#webpageUrlInput").value.trim() ||
      $("#imageUrlInput").value.trim() ||
      $("#imageInput").files[0],
  );
}

function defaultQuestionForCurrentInput() {
  const language = state.language === "en" ? "en" : "zh";
  const hasImage = Boolean($("#imageInput").files[0] || $("#imageUrlInput").value.trim());
  const hasText = Boolean($("#textInput").value.trim());
  const hasWebpage = Boolean($("#webpageUrlInput").value.trim());
  if (hasWebpage) {
    return language === "en"
      ? "Analyze whether the multimodal information on this webpage is credible."
      : "分析这个网页中的文本、图片及图文关系是否真实可信。";
  }
  if (hasImage && hasText) {
    return language === "en"
      ? "Please assess whether this text and image are credible, and analyze whether the image and text are consistent."
      : "请判断这段文字和这张图片是否真实可信，并分析图文是否一致。";
  }
  if (hasImage && !hasText) {
    return language === "en"
      ? "Please determine whether this image is a real image, a forged image, or a synthetic image."
      : "请判断这张图片是真实图像，伪造图像还是合成图像。";
  }
  if (hasText && !hasImage) {
    return language === "en" ? "Please determine the authenticity of this text." : "请判断这段文字的真实性。";
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
  button.disabled = isFollowupContext ? !question : !hasPrimaryInput();
}

function ensureImagePreviewBeforeSubmit() {
  if ($("#imageInput").files[0]) return;
  const imageUrl = $("#imageUrlInput").value.trim();
  if (imageUrl && imageUrl !== state.imagePreviewSource) {
    renderImagePreviewFromUrl(imageUrl);
  }
}

function getSuspiciousRegions(result) {
  const regions =
    result?.user_answer?.forgery_regions ||
    result?.image_verdict?.user_answer?.forgery_regions ||
    result?.image_verdict?.suspicious_regions ||
    result?.raw?.image_forensics?.suspicious_regions ||
    result?.alignment_verdict?.suspicious_regions ||
    [];
  return Array.isArray(regions) ? regions : [];
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
  const img = preview.querySelector(".preview-image");
  const overlay = preview.querySelector(".region-overlay");
  if (!overlay) return;
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
  try {
    submitBtn.disabled = true;
    ensureImagePreviewBeforeSubmit();
    persistCurrentConversationDraft();
    if (!isFollowup) {
      setPrimaryInputsLocked(true);
    }
    setStatus("running", "running");
    setRunningResultView();

    const response = await fetch(isFollowup ? "/followup" : "/detect", {
      method: "POST",
      body: isFollowup ? buildFollowupFormData(question) : buildFormData(),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
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
      mode: state.mode,
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

function bindConversationDraftTitle() {
  ["#textInput", "#questionInput", "#webpageUrlInput", "#imageUrlInput"].forEach((selector) => {
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

function bindCopy() {
  $("#copyBtn").addEventListener("click", async () => {
    await navigator.clipboard.writeText(JSON.stringify(state.lastResult || {}, null, 2));
  });
}

bindSegmentedControls();
bindImagePreview();
bindConversationDraftTitle();
bindCopy();
loadConversation();
updateSubmitButtonLabel();
$("#submitBtn").addEventListener("click", submitDetection);
$("#newConversationBtn").addEventListener("click", newConversation);
$("#refreshConversationsBtn").addEventListener("click", syncConversationList);
$("#conversationList").addEventListener("click", (event) => {
  const deleteButton = event.target.closest(".conversation-delete");
  if (deleteButton) {
    event.preventDefault();
    event.stopPropagation();
    deleteConversation(deleteButton.dataset.conversationId);
    return;
  }
  const item = event.target.closest(".conversation-item");
  if (!item) return;
  selectConversation(item.dataset.conversationId);
});
window.addEventListener("resize", () => drawSuspiciousRegions(state.lastResult));
$("#imagePreview").addEventListener("scroll", () => {
  window.requestAnimationFrame(() => drawSuspiciousRegions(state.lastResult));
});
window.addEventListener("beforeunload", clearImageObjectUrl);
