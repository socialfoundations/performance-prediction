from itertools import permutations

import numpy as np
import pytest

from pproute import cascade


def test_curve_replaces_answers_and_can_fall_below_strong_endpoint():
    row = cascade.curve([0, 1, 1], [1, 0, 1], [.1, .2, .9])
    assert row["accuracy"] == pytest.approx([2/3, 1, 2/3, 2/3])
    assert row["auc"] == pytest.approx(7/9)
    assert row["repairs"] == row["regressions"] == 1


def test_ties_equal_mean_over_all_possible_orders():
    weak, strong = np.array([0, 1, 0, 1]), np.array([1, 0, 1, 1])
    expected = []
    for order in permutations(range(3)):
        ranks = np.empty(4)
        ranks[list(order) + [3]] = np.arange(4)
        expected.append(cascade.curve(weak, strong, ranks)["auc"])
    assert cascade.curve(weak, strong, [0, 0, 0, 1])["auc"] == pytest.approx(np.mean(expected))
    random = cascade.curve(weak, strong, np.zeros(4))
    assert random["auc"] == random["random_auc"]
    assert random["excess_auc"] == 0


def test_oracle_and_monotone_scores():
    weak, strong = np.array([0, 1, 0, 1]), np.array([1, 0, 1, 1])
    reference = cascade.curve(weak, strong, [.2, .8, .2, .7])
    assert reference == cascade.curve(weak, strong, [2, 8, 2, 7])
    oracle = cascade.curve(weak, strong, weak-strong)
    assert oracle["auc"] >= reference["auc"]
    assert cascade.curve(weak, weak, [.2, .8, .2, .7])["auc"] == weak.mean()


@pytest.mark.parametrize("weak,strong,score", [([], [], []), ([1], [0, 1], [0]),
    ([2], [0], [.1]), ([1], [0], [float("nan")])])
def test_invalid_curve_inputs(weak, strong, score):
    with pytest.raises(ValueError):
        cascade.curve(weak, strong, score)


def test_answer_extraction_and_failure_denominator():
    doc = dict(id=3, gold="B", options=["one", "two"])
    completions = [dict(text=x, n_tokens=10, finish_reason="stop") for x in
                   ("the answer is (B)", "the answer is (A)", "no answer", "uncertain")]
    row = cascade.summarize_answers("mmlu_pro", doc, completions)
    assert row["majority"] == "B" and row["share"] == .25 and row["n_fail"] == 2 and row["z"] == 1
    assert cascade.extract_answer("mmlu_pro", "the answer is (A), but the answer is (B)") == "A"
    assert cascade.extract_answer("mmlu_pro", "the answer is (J)") == "J"
    assert cascade.extract_answer("gsm8k", "We get 1,000. Final answer: 42.") == "42"
    with pytest.raises(ValueError, match="Unsupported cascade generation task"):
        cascade.extract_answer("commonsense_qa", "Final answer: (B)")


def test_all_failed_answers_preserve_cascade_record_format():
    row = cascade.summarize_answers("gsm8k", dict(id=0, gold="42"), [
        dict(text="no answer", n_tokens=2, finish_reason="length"),
        dict(text="unknown", n_tokens=1, finish_reason="stop"),
    ])
    assert row == dict(id=0, gold="42", majority=None, z=0, k=2, share=0.0,
                       counts={}, answers=[None, None], n_fail=2, cap_hits=1, output_tokens=3)


def test_sampler_import_without_optional_flashinfer(monkeypatch):
    import importlib.metadata
    from pathlib import Path
    import runpy

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/cascade_sample.py"))
    assert callable(module["validate_result"])


def test_publication_rejects_changed_inputs(tmp_path):
    path = tmp_path / "prompts.json"
    row = cascade.prompt_record("gsm8k", "development", [dict(id=0, prompt="Question", gold="1")], {})
    cascade.publish_once(row, path)
    cascade.publish_once(row, path)
    assert cascade.load_record(path) == row
    with pytest.raises(ValueError):
        cascade.publish_once(dict(row, split="test"), path)


def test_resuming_checks_sample_evidence():
    from scripts.cascade_sample import validate_result
    doc = dict(id=3, gold="B", options=["one", "two"])
    completions = [dict(text="the answer is (B)", n_tokens=3, finish_reason="stop")]
    row = dict(cascade.summarize_answers("mmlu_pro", doc, completions), run_identity="run")
    validate_result(row, doc, "run", 1)
    with pytest.raises(ValueError):
        validate_result(dict(row, z=0), doc, "run", 1)
    with pytest.raises(ValueError):
        validate_result(row, doc, "different", 1)


def test_prepare_rejects_development_generation_before_reading_data():
    from scripts.cascade_prepare import prepare
    with pytest.raises(ValueError, match="test split"):
        prepare("gsm8k", "missing", "development", "missing")
