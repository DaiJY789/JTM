const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync("app/web/static/app.js", "utf8").split("\nbindSegmentedControls();")[0];

function setup(result) {
  const elements = new Map();
  const node = (tag = "div") => ({
    tag, children: [], dataset: {}, hidden: true, textContent: "",
    classList: { add() {}, toggle() {} },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
  });
  const element = (selector) => {
    if (!elements.has(selector)) elements.set(selector, node());
    return elements.get(selector);
  };
  const context = vm.createContext({
    URL, FormData, result,
    document: { querySelector: element, createElement: node },
  });
  vm.runInContext(source, context);
  return {
    element,
    run: (code) => vm.runInContext(code, context),
    value: (code) => JSON.parse(vm.runInContext(`JSON.stringify(${code})`, context)),
  };
}

const answer = { judgment: "核查后的判断。", analysis: "核查后的依据。", notes: ["证据边界。"] };

for (const [name, result] of [
  ["MFC", { mode: "multimodal", label: "Mostly False", mfc_detection: { label: "Mostly False", user_answer: answer } }],
  ["text-only MFC", { mode: "text_only", label: "Mostly False", mfc_detection: { label: "Mostly False", user_answer: answer } }],
  ["IFD", { mode: "image_only", label: "伪造图像", user_answer: answer }],
  ["nested IFD", { mode: "image_only", label: "伪造图像", image_verdict: { user_answer: answer } }],
]) {
  test(`${name} maps judgment, analysis, and notes to the three rendered sections`, () => {
    const app = setup(result);
    assert.deepEqual(app.value('formatUserAnswer(result, "zh")'), {
      conclusion: answer.judgment, reasons: [answer.analysis], notes: answer.notes,
    });
    app.run('renderConclusion(result, "zh")');
    const sections = app.element("#summaryBox").children;
    assert.deepEqual(sections.map((section) => section.children[0].textContent), ["最终判断", "主要依据", "说明与局限"]);
    assert.equal(sections[0].children[1].textContent, answer.judgment);
    assert.equal(sections[1].children[1].children[0].textContent, answer.analysis);
    assert.equal(sections[2].children[1].children[0].textContent, answer.notes[0]);
  });
}

test("canonical fields win over stale aliases and duplicated notes appear once", () => {
  const app = setup({
    mode: "multimodal", label: "True",
    user_answer: { ...answer, reason_analysis: "旧依据" },
    mfc_detection: {
      user_answer: answer,
      explanation: { verdict_statement: "旧判断", analysis: "旧依据", limitations: answer.notes },
    },
  });
  assert.deepEqual(app.value('formatUserAnswer(result, "zh")'), {
    conclusion: answer.judgment, reasons: [answer.analysis], notes: answer.notes,
  });
});

test("saved legacy MFC and IFD answers remain readable", () => {
  const mfc = setup({ label: "True", mfc_detection: {
    verdict: "True", explanation: { verdict_statement: answer.judgment, analysis: answer.analysis, limitations: answer.notes },
  } });
  const ifd = setup({ mode: "image_only", label: "真实图像", user_answer: {
    judgment: answer.judgment, reason_analysis: answer.analysis, notes: answer.notes,
  } });
  const expected = { conclusion: answer.judgment, reasons: [answer.analysis], notes: answer.notes };
  assert.deepEqual(mfc.value('formatUserAnswer(result, "zh")'), expected);
  assert.deepEqual(ifd.value('formatUserAnswer(result, "zh")'), expected);
});

test("label and confidence still populate metrics and low confidence adds a note", () => {
  const app = setup({ label: "Mostly False", confidence: 0.83, type: "C2_contextual_inconsistency", decision_status: "low_confidence", mfc_detection: { user_answer: answer } });
  app.run('renderResultMetrics(result, "zh")');
  assert.equal(app.element("#resultLabel").children[0].textContent, "大部分虚假");
  assert.equal(app.element("#resultConfidence").textContent, "83%");
  assert.equal(app.element("#aiTitle").textContent, "伪造类型");
  assert.equal(app.element("#aiScore").textContent, "图文语境冲突");
  assert.deepEqual(app.value('formatUserAnswer(result, "zh").notes'), [answer.notes[0], "当前判断置信度相对较低。"]);
});

test("all eight MFC primary dimensions have Chinese and English display names", () => {
  const names = {
    T1_fabrication: "事实虚构", T2_distortion: "夸大、删减或歪曲", T3_misleading_context: "错误语境",
    V1_synthetic_visual_content: "生成式画面", V2_visual_editing: "语义性编辑篡改",
    C1_semantic_inconsistency: "图文语义冲突", C2_contextual_inconsistency: "图文语境冲突", C3_pragmatic_inconsistency: "图文语用误导",
  };
  for (const [key, name] of Object.entries(names)) {
    const app = setup({ type: "multimodal_misinformation", mfc_detection: { type: key } });
    assert.equal(app.run('formatResultType(result, "zh")'), name);
    assert.notEqual(app.run('formatResultType(result, "en")'), key);
  }
});

test("IFD primary methods take precedence over the task type and old type array", () => {
  const app = setup({ type: "deepfake_detection", forgery_type: "ai_generated", image_verdict: { forgery_types: ["无"] } });
  assert.equal(app.run('formatResultType(result, "zh")'), "AI生成");
  assert.equal(app.run('formatResultType(result, "en")'), "AI-generated");
  const nested = setup({ image_verdict: { forgery_type: "cgi_rendering" } });
  assert.equal(nested.run('formatResultType(result, "zh")'), "计算机渲染");
});

test("no risk, unknown, legacy, and failed results do not display a task name as forgery type", () => {
  for (const [result, expected] of [
    [{ type: "none", mfc_detection: {} }, "无"],
    [{ type: "multimodal_misinformation", mfc_detection: { forgery_type: "C2_contextual_inconsistency" } }, "图文语境冲突"],
    [{ forgery_type: "unknown", mfc_detection: {} }, "未知"],
    [{ type: "multimodal_misinformation", mfc_detection: {} }, "-"],
    [{ image_verdict: { forgery_types: ["拼接"] } }, "拼接"],
  ]) {
    const app = setup(result);
    assert.equal(app.run('formatResultType(result, "zh")'), expected);
  }
  const failed = setup({ decision_status: "technical_error", forgery_type: "splicing", label: null });
  failed.run('renderResultMetrics(result, "zh")');
  assert.equal(failed.element("#aiScore").textContent, "-");
});

test("technical failures show the error instead of a stale answer", () => {
  const app = setup({ label: null, decision_status: "technical_error", summary: "暂时失败", user_answer: answer, mfc_detection: { user_answer: answer } });
  assert.deepEqual(app.value('formatUserAnswer(result, "zh")'), { message: "暂时失败" });
});

function descendants(node, predicate) {
  return [node, ...(node.children || []).flatMap((child) => descendants(child, predicate))].filter(predicate);
}

const searchEvidence = {
  evidence_id: "S1", title: "Consumer Price Index release", site_name: "BLS",
  domain: "www.bls.gov", url: "https://www.bls.gov/news.release/cpi.htm",
  snippet: "The release reports the annual and monthly changes separately.",
};

for (const mode of ["multimodal", "text_only"]) {
  test(`${mode} renders actual retrieved sources inside Notes with safe links and existing evidence IDs`, () => {
    const app = setup({ mode, label: "Mixture", mfc_detection: { user_answer: answer, evidence: [searchEvidence] } });
    app.run('renderConclusion(result, "zh")');
    const sections = app.element("#summaryBox").children;
    assert.equal(sections.length, 3);
    const notes = sections[2];
    assert.equal(notes.children[0].textContent, "说明与局限");
    assert.equal(notes.children[1].children[0].textContent, answer.notes[0]);
    assert.equal(notes.children[2].children[0].textContent, "检索证据（1）");
    const [link] = descendants(notes, (node) => node.tag === "a");
    assert.equal(link.textContent, searchEvidence.title);
    assert.equal(link.href, searchEvidence.url);
    assert.equal(link.target, "_blank");
    assert.equal(link.rel, "noopener noreferrer");
    assert.equal(descendants(notes, (node) => node.className === "search-evidence-id")[0].textContent, "S1");
    assert.equal(descendants(notes, (node) => node.className === "search-evidence-source")[0].textContent, "BLS · www.bls.gov");
    assert.equal(descendants(notes, (node) => node.className === "search-evidence-snippet")[0].textContent, searchEvidence.snippet);
  });
}

test("legacy retrieved evidence is supported and duplicate URLs and non-search records are excluded", () => {
  const app = setup({ label: "Mixture", user_answer: answer, mfc_detection: { evidence: [searchEvidence] }, evidence: [
    { type: "external_search_result", title: "Duplicate source", url: `${searchEvidence.url}#section`, content: { evidence_id: "S9" } },
    { type: "mfc_atomic_point", title: "A query plan", url: "https://example.org/planned" },
    { type: "external_search_result", title: "Second source", source: "Example", url: "https://example.org/report", content: { evidence_id: "S2" } },
  ] });
  const sources = app.value("collectSearchEvidence(result)");
  assert.equal(sources.length, 2);
  assert.equal(sources[0].evidenceId, "S1");
  assert.equal(sources[1].evidenceId, "S2");
  app.run('renderConclusion(result, "zh")');
  assert.equal(descendants(app.element("#summaryBox"), (node) => node.tag === "a").length, 2);
});

test("evidence alone creates Notes, supports English, and disappears when reopening a result without search", () => {
  const app = setup({ label: "True", mfc_detection: { user_answer: { ...answer, notes: [] }, evidence: [searchEvidence] } });
  app.run('renderConclusion(result, "en")');
  const notes = app.element("#summaryBox").children[2];
  assert.equal(notes.children[0].textContent, "Notes and limitations");
  assert.equal(notes.children[1].children[0].textContent, "Retrieved evidence (1)");
  app.run('result.mfc_detection.evidence = []; renderConclusion(result, "en")');
  assert.equal(descendants(app.element("#summaryBox"), (node) => node.tag === "a").length, 0);
  assert.equal(app.element("#summaryBox").children.length, 2);
});

test("malformed evidence and unsafe URLs are ignored and source content stays plain text", () => {
  const title = '<img src=x onerror="alert(1)">';
  const app = setup({ label: "Mixture", mfc_detection: { user_answer: answer, evidence: [
    null, "not a source", {}, { url: "javascript:alert(1)" }, { url: "data:text/html,unsafe" },
    { ...searchEvidence, title, snippet: "<script>alert(1)</script>" },
  ] } });
  app.run('renderConclusion(result, "zh")');
  const nodes = app.element("#summaryBox");
  const links = descendants(nodes, (node) => node.tag === "a");
  assert.equal(links.length, 1);
  assert.equal(links[0].textContent, title);
  assert.equal(descendants(nodes, (node) => node.tag === "img" || node.tag === "script").length, 0);
  assert.equal(descendants(nodes, (node) => node.className === "search-evidence-snippet")[0].textContent, "<script>alert(1)</script>");
});

test("failed detections do not render stale search evidence", () => {
  const app = setup({ label: null, decision_status: "technical_error", summary: "暂时失败", mfc_detection: { evidence: [searchEvidence] } });
  app.run('renderConclusion(result, "zh")');
  assert.equal(descendants(app.element("#summaryBox"), (node) => node.tag === "a").length, 0);
});

for (const legacyLocation of ["root", "raw"]) {
  test(`saved MMD ${legacyLocation} results still show answers, types, and retrieved evidence`, () => {
    const legacy = { type: "T1_fabrication", user_answer: answer, evidence: [searchEvidence] };
    const result = { mode: "text_only", label: "Mixture" };
    if (legacyLocation === "root") result.mmd_detection = legacy;
    else result.raw = { mmd_detection: legacy };
    const app = setup(result);
    assert.equal(app.run('formatResultType(result, "zh")'), "事实虚构");
    assert.deepEqual(app.value('formatUserAnswer(result, "zh")'), {
      conclusion: answer.judgment, reasons: [answer.analysis], notes: answer.notes,
    });
    app.run('renderConclusion(result, "zh")');
    assert.equal(descendants(app.element("#summaryBox"), (node) => node.tag === "a").length, 1);
  });
}

for (const language of ["zh", "en"]) {
  test(`new IFD fields render the label, type, and actual tool findings in ${language}`, () => {
    const evidence = [{ step: 1, tool: "Noise_Analyzer", why: "检查文字区域", key_findings: ["局部噪声不一致。"], artifacts: ["/private/tool-output.png"] }];
    const app = setup({
      mode: "image_only", label: "Forged_Image", type: "content_replacement", confidence: 0.95,
      user_answer: answer, image_verdict: { type: "content_replacement", evidence },
      raw: { image_forensics: { evidence } },
      evidence: [{ type: "visual_tool_evidence", source: "Noise_Analyzer", content: ["局部噪声不一致。"] }],
    });
    app.run(`renderResultMetrics(result, "${language}"); renderConclusion(result, "${language}")`);
    assert.equal(app.element("#resultLabel").children[0].textContent, language === "zh" ? "伪造图像" : "Forged Image");
    assert.equal(app.element("#aiScore").textContent, language === "zh" ? "内容替换" : "Content replacement");
    const notes = app.element("#summaryBox").children[2];
    const nodes = descendants(notes, () => true);
    assert.ok(nodes.some(n => n.textContent === (language === "zh" ? "图像核查证据（1）" : "Image-check evidence (1)")));
    assert.ok(nodes.some(n => n.textContent === "局部噪声不一致。"));
    assert.ok(nodes.some(n => n.textContent.includes("检查文字区域")));
    assert.equal(nodes.filter(n => n.className === "search-evidence-card").length, 1);
    assert.equal(nodes.filter(n => n.tag === "a").length, 0);
    assert.ok(!nodes.some(n => n.textContent.includes("/private/tool-output.png")));
  });
}

test("old IFD tool_evidence still renders alongside the old forgery_type", () => {
  const app = setup({
    mode: "image_only", label: "伪造图像", forgery_type: "splicing", user_answer: answer,
    image_verdict: { evidence: ["旧文字线索"], tool_evidence: [{ tool: "SRM_Filter", why: "检查边界", key_findings: ["边界残差异常。"] }] },
  });
  app.run('renderConclusion(result, "zh")');
  assert.equal(app.run('formatResultType(result, "zh")'), "拼接");
  assert.ok(descendants(app.element("#summaryBox"), n => n.textContent === "边界残差异常。").length);
});
