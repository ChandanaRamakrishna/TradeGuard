from langgraph.types import Command

from agent.graph import _explain, build_graph, plan_tools
from agent.rag import retrieve_policy


def test_planner_selects_minimal_tools_for_instrument_and_portfolio():
    assert plan_tools("What is Infosys market value and limit utilization?") == [
        "lookup_position", "check_limit_breaches"
    ]
    assert plan_tools("What is total portfolio exposure?") == ["calculate_exposure"]
    assert plan_tools("What is portfolio VaR?") == ["calculate_historical_var"]
    assert plan_tools("What does policy say about escalation?") == ["retrieve_policy"]


def test_graph_records_actual_tool_calls_and_keeps_portfolio_var_distinct():
    graph = build_graph(enable_llm=False)
    result = graph.invoke(
        {"question": "What is the 95% one-day portfolio VaR?"},
        config={"configurable": {"thread_id": "test-var-only"}},
    )
    assert [call["tool"] for call in result["tool_calls"]] == ["calculate_historical_var"]
    assert result["var"]["horizon"] == "1 trading day"
    assert "Portfolio" in result["explanation"]


def test_stress_planner_passes_explicit_sector_shocks_to_python_tool():
    graph = build_graph(enable_llm=False)
    result = graph.invoke(
        {"question": "Stress test IT down 10% and Banking up 5%."},
        config={"configurable": {"thread_id": "test-sector-stress"}},
    )
    call = result["tool_calls"][0]
    assert call["tool"] == "run_stress_test"
    assert call["arguments"] == {
        "market_shock": -0.05,
        "sector_shocks": {"IT": -0.1, "Banking": 0.05},
    }


def test_normal_instrument_does_not_pause_for_another_instruments_critical_alert():
    graph = build_graph(enable_llm=False)
    result = graph.invoke(
        {"question": "What is TCS limit status?"},
        config={"configurable": {"thread_id": "test-normal-instrument"}},
    )
    assert result["breach"]["status"] == "Normal"
    assert not result.get("__interrupt__")


def test_critical_instrument_still_pauses_and_resumes_for_human_review(monkeypatch):
    monkeypatch.setattr("agent.graph.retrieve_policy", lambda _question: [{
        "text": "Critical alerts should be prepared for risk analyst review.",
        "source": "data/policies/risk_policy.txt",
        "section": "4. Escalation and human review",
        "retrieval_method": "test_fixture",
    }])
    graph = build_graph(enable_llm=False)
    config = {"configurable": {"thread_id": "test-critical-review"}}
    pending = graph.invoke({"question": "Why did Infosys trigger an alert?"}, config=config)
    assert pending.get("__interrupt__")
    resumed = graph.invoke(
        Command(resume={"decision": "approved", "reviewer": "Test", "comment": "Reviewed"}),
        config=config,
    )
    assert resumed["approval"]["decision"] == "approved"
    assert resumed["status"] == "reviewed"


def test_groq_model_configuration_and_safe_failure_logging(monkeypatch, caplog):
    import logging

    captured = {}

    class ModelUnavailable(Exception):
        status_code = 404

    class FailingModel:
        def invoke(self, _messages):
            raise ModelUnavailable("model not found")

    def fake_chatgroq(**kwargs):
        captured.update(kwargs)
        return FailingModel()

    monkeypatch.setenv("GROQ_API_KEY", "test-key-must-not-be-logged")
    monkeypatch.setenv("GROQ_MODEL", "configured-test-model")
    monkeypatch.setattr("agent.graph.ChatGroq", fake_chatgroq)
    caplog.set_level(logging.WARNING, logger="agent.graph")
    result = _explain({"question": "Summarize verified results", "evidence": {}}, enable_llm=True)

    assert captured["model"] == "configured-test-model"
    assert result["llm_status"] == "model_not_found_or_unavailable"
    assert "deterministic explanation used" in result["explanation"]
    assert "category=model_not_found_or_unavailable" in caplog.text
    assert "test-key-must-not-be-logged" not in caplog.text


def test_policy_retrieval_falls_back_to_attributed_local_policy(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.rag.DB_PATH", tmp_path / "missing-chroma")
    docs = retrieve_policy("critical alert escalation policy", k=5)
    assert docs
    assert all(doc["source"] == "data/policies/risk_policy.txt" for doc in docs)
    assert all(doc["retrieval_method"] == "local_keyword_fallback" for doc in docs)
    sections = [doc["section"] for doc in docs]
    assert len(sections) == len(set(sections))
    assert sections[0] == "4. Escalation and human review"


def test_local_policy_retrieval_targets_distinct_sections(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.rag.DB_PATH", tmp_path / "missing-chroma")
    questions = {
        "position limits utilization warning threshold": "1. Position limits",
        "sector concentration greater than 40 percent": "2. Concentration monitoring",
        "critical alerts escalation authorized human reviewer": "4. Escalation and human review",
    }
    for question, expected_section in questions.items():
        docs = retrieve_policy(question, k=3)
        assert docs
        assert docs[0]["section"] == expected_section
        assert docs[0]["source"] == "data/policies/risk_policy.txt"


def test_policy_retrieval_falls_back_when_chroma_loading_fails(monkeypatch, tmp_path):
    db_path = tmp_path / "broken-chroma"
    db_path.mkdir()

    def fail_embeddings(**_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("agent.rag.DB_PATH", db_path)
    monkeypatch.setattr("langchain_huggingface.HuggingFaceEmbeddings", fail_embeddings)
    docs = retrieve_policy("critical escalation policy")
    assert docs
    assert docs[0]["retrieval_method"] == "local_keyword_fallback"


def test_chroma_policy_retrieval_deduplicates_overlapping_chunks(monkeypatch, tmp_path):
    from langchain_core.documents import Document

    class Collection:
        def count(self):
            return 5

    class FakeStore:
        _collection = Collection()

        def similarity_search(self, _question, k):
            assert k == 5
            return [
                Document(page_content="1. Position limits chunk A", metadata={"source": "policy.txt", "section": "1. Position limits"}),
                Document(page_content="1. Position limits overlapping chunk B", metadata={"source": "policy.txt", "section": "1. Position limits"}),
                Document(page_content="2. Concentration monitoring", metadata={"source": "policy.txt", "section": "2. Concentration monitoring"}),
                Document(page_content="4. Escalation and review", metadata={"source": "policy.txt", "section": "4. Escalation and human review"}),
                Document(page_content="4. Escalation and review", metadata={"source": "policy.txt", "section": "4. Escalation and human review"}),
            ]

    monkeypatch.setattr("agent.rag.DB_PATH", tmp_path / "present-chroma")
    (tmp_path / "present-chroma").mkdir()
    monkeypatch.setattr("agent.rag._embeddings", lambda local_files_only: object())
    monkeypatch.setattr("langchain_chroma.Chroma", lambda **_kwargs: FakeStore())
    docs = retrieve_policy("risk policy", k=3)
    assert [doc["section"] for doc in docs] == [
        "1. Position limits", "2. Concentration monitoring", "4. Escalation and human review"
    ]
    assert all(doc["source"] == "policy.txt" for doc in docs)
