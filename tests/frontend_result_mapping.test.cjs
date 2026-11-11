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
  ["MMD", { mode: "multimodal", label: "Mostly False", mmd_detection: { label: "Mostly False", user_answer: answer } }],
  ["text-only MMD", { mode: "text_only", label: "Mostly False", mmd_detection: { label: "Mostly False", user_answer: answer } }],
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
    mmd_detection: {
      user_answer: answer,
      explanation: { verdict_statement: "旧判断", analysis: "旧依据", limitations: answer.notes },
    },
  });
  assert.deepEqual(app.value('formatUserAnswer(result, "zh")'), {
    conclusion: answer.judgment, reasons: [answer.analysis], notes: answer.notes,
  });
});

test("saved legacy MMD and IFD answers remain readable", () => {
  const mmd = setup({ label: "True", mmd_detection: {
    verdict: "True", explanation: { verdict_statement: answer.judgment, analysis: answer.analysis, limitations: answer.notes },
  } });
  const ifd = setup({ mode: "image_only", label: "真实图像", user_answer: {
    judgment: answer.judgment, reason_analysis: answer.analysis, notes: answer.notes,
  } });
  const expected = { conclusion: answer.judgment, reasons: [answer.analysis], notes: answer.notes };
  assert.deepEqual(mmd.value('formatUserAnswer(result, "zh")'), expected);
  assert.deepEqual(ifd.value('formatUserAnswer(result, "zh")'), expected);
});

test("label and confidence still populate metrics and low confidence adds a note", () => {
  const app = setup({ label: "Mostly False", confidence: 0.83, type: "multimodal_misinformation", forgery_type: "C2_contextual_inconsistency", decision_status: "low_confidence", mmd_detection: { user_answer: answer } });
  app.run('renderResultMetrics(result, "zh")');
  assert.equal(app.element("#resultLabel").children[0].textContent, "大部分虚假");
  assert.equal(app.element("#resultConfidence").textContent, "83%");
  assert.equal(app.element("#aiTitle").textContent, "伪造类型");
  assert.equal(app.element("#aiScore").textContent, "图文语境冲突");
  assert.deepEqual(app.value('formatUserAnswer(result, "zh").notes'), [answer.notes[0], "当前判断置信度相对较低。"]);
});

test("all eight MMD primary dimensions have Chinese and English display names", () => {
  const names = {
    T1_fabrication: "事实虚构", T2_distortion: "夸大、删减或歪曲", T3_misleading_context: "错误语境",
    V1_synthetic_visual_content: "生成式画面", V2_visual_editing: "语义性编辑篡改",
    C1_semantic_inconsistency: "图文语义冲突", C2_contextual_inconsistency: "图文语境冲突", C3_pragmatic_inconsistency: "图文语用误导",
  };
  for (const [key, name] of Object.entries(names)) {
    const app = setup({ type: "multimodal_misinformation", mmd_detection: { forgery_type: key } });
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
    [{ forgery_type: "none", mmd_detection: {} }, "无"],
    [{ forgery_type: "unknown", mmd_detection: {} }, "未知"],
    [{ type: "multimodal_misinformation", mmd_detection: {} }, "-"],
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
  const app = setup({ label: null, decision_status: "technical_error", summary: "暂时失败", user_answer: answer, mmd_detection: { user_answer: answer } });
  assert.deepEqual(app.value('formatUserAnswer(result, "zh")'), { message: "暂时失败" });
});
