const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

// Exercise the real import flow without network access or paid detection calls.
function setup({ image = false, response = { title: '网页标题', text: '正文', images: ['https://example.test/photo'] }, ok = true } = {}) {
  const elements = new Map();
  const calls = [];
  const element = (selector) => {
    if (!elements.has(selector)) elements.set(selector, {
      value: '', files: [], textContent: '', dataset: {}, hidden: false,
      classList: { toggle() {} }, setCustomValidity(message) { this.validationMessage = message; }, reportValidity() {},
    });
    return elements.get(selector);
  };
  const context = vm.createContext({
    URL, FormData, setTimeout, clearTimeout,
    document: { querySelector: element },
    Image: class {
      set src(value) {
        if (!value) return;
        queueMicrotask(() => {
          this.naturalWidth = image ? 640 : 0;
          this.naturalHeight = image ? 480 : 0;
          if (image) this.onload?.(); else this.onerror?.();
        });
      }
    },
    fetch: async (url, options) => {
      calls.push({ url, data: Object.fromEntries(options.body) });
      return { ok, json: async () => response };
    },
  });
  const code = fs.readFileSync('app/web/static/app.js', 'utf8').split('\nbindSegmentedControls();')[0];
  vm.runInContext(code, context);
  vm.runInContext(`
    state.conversationId = 'test-conversation';
    renderSelectedImagePreviews = () => {};
    renderImagePreview = () => {};
    renderImagePreviews = () => {};
    renderEmptyPreview = () => {};
    persistCurrentConversationDraft = () => {};
    renderConversation = () => {};
  `, context);
  return { context, calls, element, run: (code) => vm.runInContext(code, context) };
}

test('extensionless image is imported as an image and included in the detection payload', async () => {
  const app = setup({ image: true });
  app.element('#urlInput').value = 'https://example.test/asset?id=3';
  await app.run('importUrl()');
  assert.equal(app.calls.length, 0);
  assert.equal(app.element('#imageUrlInput').value, 'https://example.test/asset?id=3');
  assert.equal(app.run('selectedImageUrls()[0]'), 'https://example.test/asset?id=3');
  assert.equal(app.run('buildFormData().get("image_urls")'), 'https://example.test/asset?id=3');
  assert.equal(app.element('#urlImportStatus').dataset.status, 'success');
  assert.equal(app.run('state.urlImportPending'), false);
});

test('a webpage with an image suffix still extracts text and images as a webpage', async () => {
  const app = setup();
  app.element('#urlInput').value = 'https://example.test/article.jpg';
  await app.run('importUrl()');
  assert.equal(app.calls[0].url, '/webpages/extract');
  assert.equal(app.calls[0].data.webpage_url, 'https://example.test/article.jpg');
  assert.equal(app.element('#textInput').value, '网页标题\n\n正文');
  assert.equal(app.run('state.webpageExtraction.images.length'), 1);
  assert.equal(app.run('buildFormData().get("webpage_url")'), 'https://example.test/article.jpg');
});

test('invalid URL schemes are rejected before importing', async () => {
  const app = setup();
  app.element('#urlInput').value = 'file:///tmp/image.png';
  await app.run('importUrl()');
  assert.match(app.element('#urlInput').validationMessage, /http/);
  assert.equal(app.calls.length, 0);
});

test('failed imports preserve existing input and allow retry', async () => {
  const app = setup({ ok: false, response: { detail: '来源无法访问' } });
  app.element('#textInput').value = '用户输入';
  app.element('#urlInput').value = 'https://example.test/unavailable';
  await app.run('importUrl()');
  assert.equal(app.element('#textInput').value, '用户输入');
  assert.equal(app.element('#urlImportStatus').dataset.status, 'error');
  assert.match(app.element('#urlImportStatus').textContent, /来源无法访问/);
  assert.equal(app.element('#importUrlBtn').disabled, false);
});

test('an import finishing after changing conversations cannot overwrite the new conversation', async () => {
  const app = setup({ image: true });
  app.element('#urlInput').value = 'https://example.test/photo';
  const pending = app.run('importUrl()');
  app.run('resetUrlImport(); state.conversationId = "different-conversation";');
  await pending;
  assert.equal(app.element('#imageUrlInput').value, '');
  assert.equal(app.element('#urlImportStatus').textContent, '');
});

test('image import respects the multimodal file limit', async () => {
  const app = setup({ image: true });
  app.element('#imageInput').files = Array.from({ length: 4 }, (_, n) => ({ name: `${n}.jpg` }));
  app.element('#urlInput').value = 'https://example.test/photo';
  await app.run('importUrl()');
  assert.equal(app.element('#imageUrlInput').value, '');
  assert.equal(app.element('#urlImportStatus').dataset.status, 'error');
});

test('a deepfake image URL replaces the previous local file', async () => {
  const app = setup({ image: true });
  app.run('state.task = "deepfake_detection";');
  let files = [{ name: 'old.jpg' }];
  Object.defineProperty(app.element('#imageInput'), 'files', { get: () => files });
  Object.defineProperty(app.element('#imageInput'), 'value', { set: () => { files = []; } });
  app.element('#urlInput').value = 'https://example.test/photo';
  await app.run('importUrl()');
  assert.equal(app.run('buildFormData().get("image_url")'), 'https://example.test/photo');
  assert.equal(files.length, 0);
});
