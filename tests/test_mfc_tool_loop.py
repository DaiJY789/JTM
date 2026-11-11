import json
from threading import Barrier, Lock

import pytest

from app.agents.MFC_Agent import MFCDetectionAgent
from app.core.config import Settings


class ImageService:
    model_factory = None

    def build_image_content_part(self, image_content=None, image_url=None):
        if image_content or image_url:
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}


def final_answer():
    return {
        "label": "Mixture", "confidence": 0.5, "type": "unknown",
        "user_answer": {"analysis": "现有证据有限。", "notes": []},
        "evidence": [{"evidence_id": "S999", "url": "https://invented.invalid/"}],
    }


def search_action(*queries):
    return {"action": "call_tools", "tool_calls": [
        {"tool": "bocha_web_search", "arguments": {"query": query, "count": 3}}
        for query in queries
    ]}


class ModelInput(dict):
    pass


def inputs(prompt):
    assert "tool_context" not in prompt
    assert "tool_control" not in prompt
    payload = ModelInput(json.JSONDecoder().raw_decode(prompt.split("INPUT_JSON:\n", 1)[1])[0])
    payload.prompt = prompt
    return payload


class SearchTool:
    def __init__(self):
        self.calls = []
        self.lock = Lock()

    def run(self, query, count=3, *, timeout_seconds=None):
        with self.lock:
            self.calls.append((query, count, timeout_seconds))
        return {"status": "success", "results": [{
            "title": query, "url": f"https://example.gov/{query}", "snippet": "摘要",
        }]}


def make_agent(responses, *, search=None, config=None):
    prompts = []
    def evaluate(prompt, *_):
        prompts.append(inputs(prompt))
        index = len(prompts) - 1
        response = responses(index, prompts[-1]) if callable(responses) else responses[index]
        return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
    agent = MFCDetectionAgent(
        ImageService(), config or Settings(), search_tool=search, evaluator=evaluate,
    )
    return agent, prompts


def test_model_searches_before_evidence_is_available_and_final_evidence_is_not_model_authored():
    search = SearchTool()
    def responses(index, payload):
        if index == 0:
            assert search.calls == []
            assert payload["evidence"] == []
            assert payload["search_history"] == []
            assert "本轮执行提示：本次无法继续搜索" not in payload.prompt
            return search_action("first")
        assert payload["evidence"][0]["evidence_id"] == "S1"
        return final_answer()
    agent, prompts = make_agent(responses, search=search)
    result = agent.detect(text="需要核验的新闻", image_content=b"image")
    assert result["status"] == "success"
    assert [item["evidence_id"] for item in result["evidence"]] == ["S1"]
    assert result["evidence"][0]["url"] == "https://example.gov/first"
    assert result["tool_context"]["ocr"]["status"] == "not_run"
    assert result["timings"]["model_calls"] == 2
    assert result["timings"]["search_calls"] == 1
    assert result["timings"]["invalid_output_retries"] == 0
    assert result["timings"]["enrichment_seconds"] == 0
    assert search.calls[0][2] > 0


def test_model_can_finish_directly_without_search_or_fabricated_evidence():
    search = SearchTool()
    agent, _ = make_agent([final_answer()], search=search)
    result = agent.detect(text="直接检查输入")
    assert result["status"] == "success"
    assert result["evidence"] == []
    assert search.calls == []
    assert result["timings"]["model_calls"] == 1


def test_independent_model_selected_searches_run_in_parallel():
    barrier = Barrier(3)
    class ParallelSearch(SearchTool):
        def run(self, *args, **kwargs):
            barrier.wait(timeout=2)
            result = super().run(*args, **kwargs)
            result["results"][0]["url"] = f"https://{args[0]}.example/report"
            return result
    search = ParallelSearch()
    agent, prompts = make_agent([search_action("a", "b", "c"), final_answer()], search=search)
    result = agent.detect(text="多个独立主张")
    assert result["status"] == "success"
    assert len(search.calls) == 3
    assert [item["query_id"] for item in result["evidence"]] == ["Q1", "Q2", "Q3"]
    assert "本轮执行提示：本次无法继续搜索" in prompts[1].prompt


def test_two_rounds_keep_evidence_ids_stable_and_allow_query_refinement():
    search = SearchTool()
    agent, prompts = make_agent([
        search_action("first"), search_action("refined"), final_answer(),
    ], search=search)
    result = agent.detect(text="需要补充证据")
    first = prompts[1]["evidence"][0]
    assert prompts[2]["evidence"][0] == first
    assert [item["evidence_id"] for item in result["evidence"]] == ["S1", "S2"]
    assert result["timings"]["tool_rounds"] == 2
    assert result["timings"]["model_calls"] == 3
    assert result["timings"]["invalid_output_retries"] == 0
    assert "本轮执行提示：本次无法继续搜索" not in prompts[2].prompt


def test_tool_round_limit_rejects_another_request_without_executing_it():
    search = SearchTool()
    agent, _ = make_agent(
        [search_action("first"), search_action("second")], search=search,
        config=Settings(mfc_max_tool_rounds=1, mfc_invalid_output_max_retries=0),
    )
    result = agent.detect(text="新闻")
    assert result["status"] == "error"
    assert result["label"] is None
    assert len(search.calls) == 1
    assert result["evidence"][0]["evidence_id"] == "S1"


def test_total_search_budget_and_result_count_cap_are_enforced():
    search = SearchTool()
    action = search_action("a", "b", "c")
    action["tool_calls"][0]["arguments"]["count"] = 1000
    agent, prompts = make_agent([action, final_answer()], search=search,
        config=Settings(mfc_search_query_count=2, mfc_search_results_per_query=2))
    result = agent.detect(text="新闻")
    assert result["status"] == "success"
    assert len(search.calls) == 2
    assert all(count == 2 for _, count, _ in search.calls)
    assert "本轮执行提示：本次无法继续搜索" in prompts[1].prompt


def test_unknown_tools_invalid_arguments_and_duplicate_queries_are_not_executed():
    search = SearchTool()
    action = search_action(" same  query ", "same query")
    action["tool_calls"].extend([
        {"tool": "extract_image_ocr_entities", "arguments": {}},
        {"tool": "bocha_web_search", "arguments": {"query": "bad", "count": "3"}},
    ])
    agent, prompts = make_agent([action, search_action("SAME QUERY"), final_answer()], search=search)
    result = agent.detect(text="新闻")
    assert result["status"] == "success"
    assert [query for query, _, _ in search.calls] == ["same query"]
    assert len(prompts[1]["search_errors"]) == 3
    assert "repeated" in prompts[2]["search_errors"][0]


def test_failed_search_does_not_expose_partial_or_invented_evidence_and_can_recover():
    class FailedSearch(SearchTool):
        def run(self, query, **kwargs):
            if query == "bad":
                self.calls.append((query, 3, None))
                return {"status": "error", "error": "unavailable", "results": [
                    {"url": "https://invented.invalid/", "title": "not evidence"}
                ]}
            return super().run(query, **kwargs)
    search = FailedSearch()
    agent, prompts = make_agent([search_action("bad"), search_action("better"), final_answer()], search=search)
    result = agent.detect(text="新闻")
    assert prompts[1]["evidence"] == []
    assert prompts[1]["search_history"][0]["error"] == "unavailable"
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["query_id"] == "Q2"
    assert result["evidence"][0]["evidence_id"] == "S1"


def test_tool_exceptions_return_to_model_as_errors_instead_of_crashing():
    class BrokenSearch:
        def run(self, *args, **kwargs):
            raise TimeoutError("search timed out")
    agent, prompts = make_agent([search_action("a"), final_answer()], search=BrokenSearch())
    result = agent.detect(text="新闻")
    assert result["status"] == "success"
    assert result["evidence"] == []
    assert prompts[1]["search_history"][0]["status"] == "error"


@pytest.mark.parametrize("config", [
    Settings(mfc_external_evidence_enabled=False),
    Settings(mfc_max_tool_rounds=0),
    Settings(mfc_search_query_count=0),
])
def test_disabled_tools_or_zero_budget_only_allow_final_answers(config):
    search = SearchTool()
    agent, prompts = make_agent([final_answer()], search=search, config=config)
    result = agent.detect(text="新闻")
    assert result["status"] == "success"
    assert "本轮执行提示：本次无法继续搜索" in prompts[0].prompt
    assert search.calls == []


def test_retry_counts_do_not_include_normal_tool_continuations():
    search = SearchTool()
    agent, _ = make_agent([search_action("a"), "{}", final_answer()], search=search)
    result = agent.detect(text="新闻")
    assert result["status"] == "success"
    assert result["timings"]["model_calls"] == 3
    assert result["timings"]["invalid_output_retries"] == 1
    assert result["timings"]["search_calls"] == 1


def test_end_to_end_time_budget_stops_new_tools_and_model_turns(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("app.agents.MFC_Agent.perf_counter", lambda: clock[0])
    search = SearchTool()
    def slow_model(index, payload):
        clock[0] = 2.0
        return search_action("a")
    agent, _ = make_agent(slow_model, search=search, config=Settings(mfc_agent_timeout_seconds=1))
    result = agent.detect(text="新闻")
    assert result["status"] == "error"
    assert "time budget" in result["error"]
    assert search.calls == []
    assert result["timings"]["model_calls"] == 1


def test_evidence_budget_forces_final_without_another_search():
    agent, prompts = make_agent([search_action("a"), final_answer()], search=SearchTool(),
        config=Settings(mfc_search_total_results=1))
    result = agent.detect(text="新闻")
    assert result["status"] == "success"
    assert "本轮执行提示：本次无法继续搜索" in prompts[1].prompt


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("image_input", [
    {},
    {"image_content": b"image"},
    {"image_contents": [b"image"]},
    {"image_url": "https://example.org/image.png"},
    {"image_urls": ["https://example.org/image.png"]},
])
def test_production_client_routes_prompt_and_preserves_evidence_after_search(language, image_input):
    messages = []
    class Completion:
        def create(self, **kwargs):
            messages.append(kwargs["messages"])
            output = search_action("first") if len(messages) == 1 else final_answer()
            message = type("Message", (), {"content": json.dumps(output)})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()
    class Factory:
        def qwen_vision_client(self, **kwargs):
            return type("Client", (), {"chat": type("Chat", (), {"completions": Completion()})()})()
    search = SearchTool()
    agent = MFCDetectionAgent(ImageService(), Settings(mfc_search_query_count=1), model_factory=Factory(), search_tool=search)
    result = agent.detect(text="新闻", image_labels=["Image 7"], language=language, **image_input)
    assert result["status"] == "success"
    initial = inputs(messages[0][1]["content"][-1]["text"])
    continued = inputs(messages[1][1]["content"][-1]["text"])
    assert initial["evidence"] == []
    assert continued["evidence"][0]["evidence_id"] == "S1"
    for turn in messages:
        prompt = turn[1]["content"][-1]["text"]
        payload = inputs(prompt)
        assert "input_scope" not in payload
        assert payload["content_mode"] == ("multimodal" if image_input else "text_only")
        if image_input:
            assert "content_mode" not in payload["image_info"]
            assert payload["image_info"]["selected_image_count"] == 1
            assert payload["image_info"]["other_news_images_provided"] is False
            assert payload["image_info"]["image_labels"] == ["Image 7"]
            assert "Image 7" in prompt
            assert "V1_synthetic_visual_content" in prompt
            assert ("多模态" if language == "zh" else "multimodal") in turn[0]["content"]
        else:
            assert "V1_synthetic_visual_content" not in prompt
            assert "visual_observation" not in prompt
            assert "image_info" not in payload
            assert ("纯文本" if language == "zh" else "text-only") in turn[0]["content"]
            assert len(turn[1]["content"]) == 1
    if image_input:
        assert continued["image_info"]["image_labels"] == ["Image 7"]
        assert ("完成八维核查" if language == "zh" else "Complete the eight checks") in continued.prompt
    else:
        assert ("完成三维核查" if language == "zh" else "Complete the three checks") in continued.prompt


def test_three_sequential_calls_are_allowed_and_fourth_is_rejected():
    class DistinctDomains(SearchTool):
        def run(self, query, **kwargs):
            result = super().run(query, **kwargs)
            result["results"][0]["url"] = f"https://{query}.example/report"
            return result
    search = DistinctDomains()
    agent, prompts = make_agent([
        search_action("first"), search_action("second"), search_action("third"),
        search_action("fourth"),
    ], search=search, config=Settings(mfc_invalid_output_max_retries=0))
    result = agent.detect(text="新闻")
    assert result["status"] == "error"
    assert len(search.calls) == 3
    assert len(result["evidence"]) == 3
    assert "本轮执行提示：本次无法继续搜索" in prompts[3].prompt
    assert result["timings"]["tool_rounds"] == 3
