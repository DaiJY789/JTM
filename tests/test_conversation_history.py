import json

from app.services.conversation_history import ConversationHistoryStore


def make_store(tmp_path, max_items=20):
    return ConversationHistoryStore(max_items=max_items, log_dir=tmp_path)


def test_conversation_history_store_appends_and_isolates_conversations(tmp_path):
    store = make_store(tmp_path, max_items=4)
    first_id = store.ensure_id("first")
    second_id = store.ensure_id("second")

    first_history = store.append_turn(
        first_id,
        {"text": "claim one", "question": "is it true?"},
        {"mode": "text_only", "label": "Mostly_True", "summary": "Supported."},
    )
    second_history = store.append_turn(
        second_id,
        {"text": "claim two"},
        {"mode": "text_only", "label": "Mostly_False", "summary": "Refuted."},
    )

    assert len(first_history) == 2
    assert len(second_history) == 2
    assert store.get(first_id)[0]["content"].startswith("text: claim one")
    assert store.get(second_id)[0]["content"].startswith("text: claim two")


def test_conversation_history_store_truncates_old_items(tmp_path):
    store = make_store(tmp_path, max_items=3)
    conversation_id = store.ensure_id("conv")

    store.append_turn(conversation_id, {"text": "one"}, {"summary": "first"})
    history = store.append_turn(conversation_id, {"text": "two"}, {"summary": "second"})

    assert len(history) == 3
    assert history[0]["content"] == "first"
    assert history[-1]["content"] == "second"


def test_default_store_keeps_full_history_beyond_the_model_context_window(tmp_path):
    store = ConversationHistoryStore(log_dir=tmp_path)
    conversation_id = store.ensure_id("long-conversation")
    result = {"label": "True", "summary": "Initial answer."}
    store.append_turn(conversation_id, {"text": "claim"}, result)
    for number in range(1, 13):
        history = store.append_followup_turn(
            conversation_id, f"Question {number}", result, "en", assistant_content=f"Answer {number}",
        )

    assert len(history) == 26
    assert history[1]["content"] == "Initial answer."
    assert history[3]["content"] == "Answer 1"
    reopened = ConversationHistoryStore(log_dir=tmp_path)
    assert reopened.snapshot(conversation_id)["conversation_history"] == history
    assert reopened.list_conversations()[0]["history_count"] == 26
    assert reopened.get(conversation_id, limit=20) == history[-20:]
    assert reopened.latest_context(conversation_id, limit=20)["conversation_history"] == history[-20:]


def test_conversation_history_store_lists_conversations_with_metadata(tmp_path):
    store = make_store(tmp_path, max_items=6)
    first_id = store.ensure_id("first")
    second_id = store.ensure_id("second")

    store.append_turn(
        first_id,
        {"text": "older claim"},
        {"mode": "text_only", "label": "Mostly_True", "summary": "Older summary."},
    )
    store.append_turn(
        second_id,
        {"question": "这张图片是真的假的？", "has_image": True},
        {"mode": "image_only", "label": "完全真实", "summary": "No manipulation found."},
    )

    conversations = store.list_conversations()

    assert [item["conversation_id"] for item in conversations] == [second_id, first_id]
    assert conversations[0]["title"] == "这张图片是真的假的？"
    assert conversations[0]["last_label"] == "完全真实"
    assert conversations[0]["history_count"] == 2


def test_conversation_history_store_returns_snapshot(tmp_path):
    store = make_store(tmp_path, max_items=4)
    conversation_id = store.ensure_id("snapshot")
    store.append_turn(
        conversation_id,
        {"text": "snapshot claim"},
        {"mode": "text_only", "label": "Cannot_Determine", "summary": "Insufficient evidence."},
    )

    snapshot = store.snapshot(conversation_id)

    assert snapshot["conversation_id"] == conversation_id
    assert snapshot["conversation"]["title"] == "snapshot claim"
    assert snapshot["conversation"]["history_count"] == 2
    assert snapshot["conversation_history"][0]["content"].startswith("text: snapshot claim")


def test_conversation_title_prefers_text_prefix_over_question(tmp_path):
    store = make_store(tmp_path, max_items=4)
    conversation_id = store.ensure_id("title")
    store.append_turn(
        conversation_id,
        {
            "text": "这是一段用于命名对话的新闻输入内容，后续会被截断",
            "question": "这是真的吗？",
        },
        {"mode": "text_only", "label": "Mixed", "summary": "Mixed evidence."},
    )

    snapshot = store.snapshot(conversation_id)

    assert snapshot["conversation"]["title"] == "这是一段用于命名对话的新闻输入内容，..."


def test_followup_explanation_does_not_replace_latest_result(tmp_path):
    store = make_store(tmp_path, max_items=6)
    conversation_id = store.ensure_id("followup")
    original_result = {"mode": "image_only", "label": "真实图像", "summary": "原始检测结果"}
    store.append_turn(conversation_id, {"question": "判断图片", "has_image": True}, original_result)

    explanation_result = {"mode": "image_only", "label": "伪造图像", "summary": "只作为解释内容"}
    history = store.append_followup_turn(
        conversation_id,
        "为什么这样判断？",
        explanation_result,
        "zh",
        assistant_content="这是对已有结果的解释。",
        update_latest_result=False,
    )
    snapshot = store.snapshot(conversation_id)

    assert history[-1]["content"] == "这是对已有结果的解释。"
    assert snapshot["latest_result"] == original_result
    assert snapshot["conversation"]["title"] == "判断图片"


def test_initial_title_survives_a_rerun_and_reload(tmp_path):
    store = make_store(tmp_path, max_items=2)
    result = {"label": "True"}
    store.append_turn("stable", {"image_asset": {"filename": "source.png"}, "question": "Inspect it"}, result)
    store.append_turn("stable", {"text": "Later input", "mode": "auto"}, result)

    reopened = make_store(tmp_path, max_items=2)
    assert reopened.snapshot("stable")["conversation"]["title"] == "source.png"


def test_legacy_title_is_restored_from_the_first_input(tmp_path):
    store = make_store(tmp_path)
    result = {"label": "True"}
    store.append_turn("legacy", {"text": "Original claim"}, result)
    store.append_followup_turn("legacy", "Why?", result, "en")
    meta_path = tmp_path / "legacy" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["initial_title"]
    meta["title"] = "Why?"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert make_store(tmp_path).snapshot("legacy")["conversation"]["title"] == "Original claim"
