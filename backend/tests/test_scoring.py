"""
Unit tests for the scoring maths. The numbers are the same worked examples
used in the README, so the code and the explanation can't drift apart.

Run:  docker compose run --rm backend pytest -q     (or just `pytest -q` in backend/)
"""

import pytest

from app import evaluation, fusion, lexical, llm_eval, pipeline


# ---------------------------------------------------------------- nDCG
def test_ndcg_example_from_readme():
    # our order has HR labels [3, 1, 2]; perfect order is [3, 2, 1]
    assert evaluation.ndcg_at_k([3, 1, 2], k=10) == pytest.approx(0.9721, abs=1e-4)


def test_ndcg_perfect_order_is_one():
    assert evaluation.ndcg_at_k([3, 2, 1, 0], k=10) == pytest.approx(1.0)


def test_ndcg_best_candidate_ranked_second():
    assert evaluation.ndcg_at_k([1, 3, 2], k=10) == pytest.approx(0.7364, abs=1e-4)


def test_weight_grid_has_66_combinations():
    grid = list(evaluation.weight_grid())
    assert len(grid) == 66
    assert all(abs(sum(w.values()) - 1) < 1e-9 for w in grid)
    assert len(list(evaluation.weight_grid(include_llm=False))) == 11


# ---------------------------------------------------------------- fusion
def test_min_max_example_from_readme():
    # lowest cosine 0.38, highest 0.88, Rahul 0.78  ->  0.80
    assert fusion.min_max([0.38, 0.78, 0.88])[1] == pytest.approx(0.80)


def test_min_max_all_equal_gives_half():
    assert fusion.min_max([0.7, 0.7]) == [0.5, 0.5]


def test_weighted_sum_example_from_readme():
    w = fusion.effective_weights(llm_available=True,
                                 weights={"semantic": 0.5, "lexical": 0.3, "llm": 0.2})
    final = 0.80 * w["semantic"] + 0.60 * w["lexical"] + 0.70 * w["llm"]
    assert final == pytest.approx(0.72)


def test_llm_weight_is_shared_when_llm_is_off():
    w = fusion.effective_weights(llm_available=False,
                                 weights={"semantic": 0.5, "lexical": 0.3, "llm": 0.2})
    assert w == pytest.approx({"semantic": 0.625, "lexical": 0.375, "llm": 0.0})


def test_penalty_is_capped():
    assert fusion.penalty_for(["hidden_text", "tiny_font", "jd_copy"]) == 0.5


# ---------------------------------------------------------------- lexical
def test_lexical_example_from_readme():
    resume = "Built REST APIs with Flask. Skills: Python, SQL, MySQL, Git"
    result = lexical.score(resume, ["python", "sql", "docker"], ["aws", "git"])
    assert result["score"] == pytest.approx(0.6167, abs=1e-4)  # 0.7*2/3 + 0.3*1/2
    assert result["must_missing"] == ["docker"]


def test_lexical_word_boundaries():
    text = "experience with mysql and html".lower()
    assert not lexical.has_skill(text, "sql")               # 'sql' inside 'mysql'
    assert not lexical.has_skill(text, "machine learning")  # 'ml' inside 'html'
    assert lexical.has_skill("used node.js and c++", "node.js")
    assert lexical.has_skill("used node.js and c++", "c++")


def test_skill_synonyms():
    assert lexical.parse_skill_list("Postgres, K8s; nodejs") == ["postgresql", "kubernetes", "node.js"]


# ---------------------------------------------------------------- LLM reply parsing
def test_llm_reply_parsing_example_from_readme():
    reply = '```json\n{"skills": 8, "experience": 6, "projects": 7, "education": 9, "reasoning": "ok"}\n```'
    assert llm_eval.parse_reply(reply)["score"] == pytest.approx(0.73)


class _FakeLLMResponse:
    def raise_for_status(self):
        pass

    def json(self):
        content = '{"skills": 8, "experience": 6, "projects": 7, "education": 9, "reasoning": "fits"}'
        return {"choices": [{"message": {"content": content}}]}


def _enable_llm(monkeypatch):
    from app import config
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_MODEL", "test-model")


def test_llm_signal_is_fused_when_enabled(monkeypatch):
    _enable_llm(monkeypatch)
    monkeypatch.setattr(llm_eval.httpx, "post", lambda *args, **kwargs: _FakeLLMResponse())
    data = pipeline.load_sample_data()
    result = pipeline.screen(data["jd_text"], data["files"][:2], data["must_have"], data["nice_to_have"])
    assert result["llm"]["status"] == "on"
    assert result["weights"] == {"semantic": 0.5, "lexical": 0.3, "llm": 0.2}
    assert result["candidates"][0]["scores"]["llm"]["raw"] == pytest.approx(0.73)


def test_llm_failure_falls_back_to_two_signals(monkeypatch):
    _enable_llm(monkeypatch)

    def broken(*args, **kwargs):
        raise ConnectionError("no network")

    monkeypatch.setattr(llm_eval.httpx, "post", broken)
    data = pipeline.load_sample_data()
    result = pipeline.screen(data["jd_text"], data["files"][:2], data["must_have"], data["nice_to_have"])
    assert result["llm"]["status"] == "failed"
    assert result["weights"]["llm"] == 0


# ---------------------------------------------------------------- integrity checks
@pytest.fixture
def sample_result():
    data = pipeline.load_sample_data()
    return pipeline.screen(data["jd_text"], data["files"], data["must_have"],
                           data["nice_to_have"], data["labels"], k=10)


def _by_name(result):
    return {c["filename"]: c for c in result["candidates"]}


def test_hidden_and_tiny_text_are_flagged(sample_result):
    rohan = _by_name(sample_result)["rohan_deshmukh.pdf"]
    assert {"hidden_text", "tiny_font"} <= set(rohan["flags"])
    assert rohan["scores"]["lexical"]["raw"] == 0  # the hidden keywords did not count


def test_white_text_on_dark_header_is_not_hidden(sample_result):
    assert "hidden_text" not in _by_name(sample_result)["ananya_sharma.pdf"]["flags"]


def test_jd_copy_is_flagged(sample_result):
    priya = _by_name(sample_result)["priya_joshi.pdf"]
    assert "jd_copy" in priya["flags"]
    assert len(priya["evidence"]["jd_copy"]) == 2


def test_ranking_and_evaluation(sample_result):
    ranking = [c["filename"] for c in sample_result["candidates"]]
    assert ranking[:2] == ["ananya_sharma.pdf", "rahul_verma.pdf"]
    # both resumes that tried to game the system end up at the bottom
    assert set(ranking[-2:]) == {"rohan_deshmukh.pdf", "priya_joshi.pdf"}
    assert sample_result["evaluation"]["combinations_tried"] == 11  # LLM off in tests
    assert 0 < sample_result["evaluation"]["ndcg_current"] <= 1
