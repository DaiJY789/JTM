from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "app/web/static/index.html").read_text(encoding="utf-8")
APP_JS = (ROOT / "app/web/static/app.js").read_text(encoding="utf-8")
STYLES = (ROOT / "app/web/static/styles.css").read_text(encoding="utf-8")


def test_frontend_exposes_multi_image_inputs_for_mmd():
    assert 'id="imageInput"' in INDEX
    assert 'accept="image/*" multiple' in INDEX
    assert 'id="imageUrlInput"' in INDEX
    assert 'id="imageUrlInput"' in INDEX and 'type="url"' in INDEX
    assert 'id="urlInput" type="url"' in INDEX
    assert "图片 URL，每行一个" not in INDEX
    assert "const MAX_MMD_IMAGES = 4;" in APP_JS


def test_frontend_places_dialogue_before_composer_and_right_input_column():
    result_position = INDEX.index('<div class="panel result-summary"')
    history_position = INDEX.index('<section class="panel dialogue-column conversation-history"')
    composer_position = INDEX.index('<div class="prompt-panel dialogue-composer">')
    input_position = INDEX.index('id="textInput"')
    url_position = INDEX.index('id="urlInput"')
    image_position = INDEX.index('id="imagePreview"')

    assert result_position < history_position < composer_position < input_position < image_position < url_position
    assert '多模态信息检测' not in INDEX
    assert '待核查内容</h2>' not in INDEX
    assert '分析与对话</h2>' not in INDEX
    assert "结构化输出" not in INDEX
    assert 'id="jsonOutput"' not in INDEX
    assert 'id="copyBtn"' not in INDEX
    assert '#jsonOutput' not in APP_JS
    assert '#copyBtn' not in APP_JS


def test_frontend_uses_compact_single_url_image_controls():
    assert 'id="addImageBtn" type="button" aria-label="选择图片"' in INDEX
    assert "选择图片（最多4张）" not in INDEX
    assert 'id="importUrlBtn"' in INDEX
    assert '>导入</button>' in INDEX
    assert 'url === state.confirmedImageUrl' in APP_JS
    assert '$("#importUrlBtn").addEventListener("click", importUrl)' in APP_JS
    assert 'id="imageUrlInput" type="hidden"' in INDEX


def test_frontend_confirms_and_extracts_webpage_before_detection():
    assert 'id="webpageUrlInput" type="hidden"' in INDEX
    assert 'const isImage = await probeImageUrl(url);' in APP_JS
    assert 'fetch("/webpages/extract"' in APP_JS
    assert 'state.confirmedWebpageUrl = url;' in APP_JS
    assert '$("#textInput").value = extractedText' in APP_JS
    assert 'function selectedWebpageUrl()' in APP_JS
    assert 'const webpageUrl = selectedWebpageUrl();' in APP_JS


def test_frontend_hides_image_notices_by_default_and_uses_compact_preview():
    heading = INDEX.index('<div class="media-heading">')
    title = INDEX.index('>图片预览</label>', heading)
    hint = INDEX.index('id="imageSelectionMeta"', title)
    preview = INDEX.index('id="imagePreview"', hint)

    assert heading < title < hint < preview
    assert 'id="imageSelectionMeta" role="status" hidden' in INDEX
    assert "height: 240px;" not in STYLES
    assert ".text-preview-section textarea { flex: 1;" in STYLES
    assert ".confirmable-url-field" in STYLES
    assert "padding-right: 82px;" in STYLES
    assert "border-radius: 0 6px 6px 0;" in STYLES


def test_frontend_restores_remote_history_images_through_backend_proxy():
    assert '`/conversations/${encodeURIComponent(state.conversationId)}/images/${index}`' in APP_JS
    assert 'const remoteUrls = remoteAssets.map((item) => item.sourceUrl);' in APP_JS
    assert '图片预览受来源限制，仍可提交后由后端尝试读取。' not in APP_JS


def test_frontend_posts_task_specific_fields_to_split_endpoints():
    assert 'formData.append("task", state.task)' not in APP_JS
    assert 'formData.append("images", file)' in APP_JS
    assert 'formData.append("image_urls", url)' in APP_JS
    assert 'formData.append("image", imageFiles[0])' in APP_JS
    assert 'formData.append("image_url", imageUrls[0])' in APP_JS
    assert 'state.task === "deepfake_detection" ? "/detect/deepfake" : "/detect/mmd"' in APP_JS
    assert '`/conversations/${encodeURIComponent(state.conversationId)}/followup`' in APP_JS
    assert 'isFollowup ? followupEndpoint : detectionEndpoint' in APP_JS


def test_frontend_switches_between_mmd_and_single_image_deepfake_limits():
    assert 'state.task === "deepfake_detection" ? 1 : MAX_MMD_IMAGES' in APP_JS
    assert 'input.multiple = state.task !== "deepfake_detection"' in APP_JS
    assert 'data-value="multimodal_fact_checking"' in INDEX
    assert 'data-value="deepfake_detection"' in INDEX


def test_frontend_control_labels_are_chinese():
    assert '<span class="control-label">任务</span>' in INDEX
    assert 'id="languageToggleBtn"' in INDEX
    assert 'aria-label="当前语言：中文，点击切换为英文"' in INDEX
    assert '<span class="control-label">Task</span>' not in INDEX
    assert '<span class="control-label">Language</span>' not in INDEX


def test_frontend_panel_titles_and_result_meta_use_shared_styles():
    for title in ["文本预览", "URL", "图片预览"]:
        assert f'class="panel-title"' in INDEX and f'>{title}</label>' in INDEX
    assert '<span class="status-pill runtime-pill" id="resultRuntime">-</span>' in INDEX
    assert '<span class="status-pill idle" id="statusPill" role="status">等待输入</span>' in INDEX


def test_frontend_displays_exhausted_detection_failure_as_busy():
    assert 'payload.decision_status === "technical_error"' in APP_JS
    assert 'setResult(payload, { persist: false })' in APP_JS
    assert "服务器繁忙，检测暂时失败，请稍后重试。" in APP_JS


def test_frontend_uses_flexible_contained_preview_and_backend_annotation():
    assert "height: 240px;" not in STYLES
    assert ".text-preview-section textarea { flex: 1;" in STYLES
    assert "object-fit: contain;" in STYLES
    assert "annotated_image_url" in APP_JS
    assert 'preview.querySelector(".preview-image:not(.annotated-preview-image)")' in APP_JS


def test_frontend_keeps_conversation_images_visible_during_followups():
    assert "imagePreviewAssets: []" in APP_JS
    assert "function previewAssetsFromResult(result)" in APP_JS
    assert 'item.url.startsWith("/conversations/")' in APP_JS
    assert "syncImagePreviewsFromResult(result);" in APP_JS
    assert "if (preserveExisting && state.imagePreviewAssets.length)" in APP_JS
    assert "ensureImagePreviewBeforeSubmit(isFollowup);" in APP_JS
    assert "/static/app.js?v=20261015-unified-url-fluid" in INDEX
