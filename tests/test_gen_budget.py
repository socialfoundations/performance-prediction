"""The evaluator must not silently override a task's own generation budget.

lm_eval applies `gen_kwargs` unconditionally, so any value the Evaluator passes wins over the task
config. With the old default of 256 that truncated MMLU-Pro, which asks for 2048 to fit a 5-shot
chain of thought: the model would be cut off before emitting "the answer is (X)", the extraction
regex would find nothing, and `Z` would read 0 -- indistinguishable from being wrong, on every model
rather than only the thinking ones.

`max_new_tokens = -1` now means "leave the task alone"; a benchmark that wants a different budget
sets it explicitly.
"""
import inspect
from types import SimpleNamespace

import pytest
import torch

from pproute.evaluator import Evaluator


@pytest.mark.parametrize("available", [3, 4])
def test_generation_prompt_must_fit_without_left_truncation(monkeypatch, available):
    from pproute.evaluator import HFLM, HarnessHFLM
    encoded, mask = torch.arange(4).reshape(1, 4), torch.ones(1, 4)
    monkeypatch.setattr(HarnessHFLM, "tok_batch_encode", lambda *a, **k: (encoded, mask))
    lm = HFLM.__new__(HFLM)
    if available < 4:
        with pytest.raises(ValueError, match="refusing to truncate"):
            lm.tok_batch_encode(["a complete prompt"], left_truncate_len=available)
    else:
        actual, actual_mask = lm.tok_batch_encode(["a complete prompt"], left_truncate_len=available)
        assert actual is encoded and actual_mask is mask


def _gen_kwargs_for(max_new_tokens):
    """What Evaluator.evaluate would pass to simple_evaluate for this setting."""
    captured = {}

    class _FakeLM:
        def __init__(self, *a, **k): pass

    import lm_eval
    import pproute.evaluator as ev
    real_simple, real_hflm, real_stamp = lm_eval.simple_evaluate, ev.HFLM, ev.stamp_gold_index
    ev.HFLM = _FakeLM
    lm_eval.simple_evaluate = lambda **kw: captured.update(kw) or {}
    ev.stamp_gold_index = lambda results, task_dict, predictor_contexts: results
    try:
        Evaluator({}, None, max_new_tokens=max_new_tokens).evaluate(None, None)
    finally:
        lm_eval.simple_evaluate, ev.HFLM, ev.stamp_gold_index = real_simple, real_hflm, real_stamp
    return captured


def test_default_does_not_override_the_task():
    assert inspect.signature(Evaluator.__init__).parameters["max_new_tokens"].default == -1, \
        "the default must be -1 (leave the task alone), not a concrete token count"
    kw = _gen_kwargs_for(-1)
    assert "gen_kwargs" not in kw, \
        "a -1 budget must omit gen_kwargs entirely, or the task's own max_gen_toks is overridden"
    print("  PASS  default (-1) passes no gen_kwargs")


def test_explicit_budget_is_forwarded():
    kw = _gen_kwargs_for(8192)
    assert kw.get("gen_kwargs") == "max_gen_toks=8192", kw.get("gen_kwargs")
    print("  PASS  explicit budget forwarded as max_gen_toks=8192")


def test_fewshot_seed_is_explicitly_forwarded():
    kw = _gen_kwargs_for(-1)
    assert kw.get("fewshot_random_seed") == 1234


def test_none_is_also_treated_as_no_override():
    assert "gen_kwargs" not in _gen_kwargs_for(None)
    print("  PASS  None treated as no override")


def test_nested_model_context_window_is_forwarded_to_hflm():
    captured = {}

    class _FakeLM:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    import lm_eval
    import pproute.evaluator as ev
    real_simple, real_hflm, real_stamp = lm_eval.simple_evaluate, ev.HFLM, ev.stamp_gold_index
    ev.HFLM = _FakeLM
    lm_eval.simple_evaluate = lambda **_kw: {}
    ev.stamp_gold_index = lambda results, task_dict, predictor_contexts: results
    try:
        model = SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(max_position_embeddings=262144)
            )
        )
        Evaluator({}, None).evaluate(model, None)
    finally:
        lm_eval.simple_evaluate, ev.HFLM, ev.stamp_gold_index = real_simple, real_hflm, real_stamp
    assert captured["max_length"] == 262144


def test_distributed_evaluation_is_rejected_before_generation():
    class _FakeLM:
        rank = 0
        world_size = 2

        def __init__(self, *args, **kwargs):
            pass

    import lm_eval
    import pproute.evaluator as ev
    real_simple, real_hflm = lm_eval.simple_evaluate, ev.HFLM
    ev.HFLM = _FakeLM
    lm_eval.simple_evaluate = lambda **_kwargs: pytest.fail("generation should not start")
    try:
        with pytest.raises(ValueError, match="one-process evaluation"):
            Evaluator({}, None).evaluate(None, None)
    finally:
        lm_eval.simple_evaluate, ev.HFLM = real_simple, real_hflm
def test_model_generation_config_ceiling_is_cleared():
    """A checkpoint-shipped generation_config.max_new_tokens must not outrank the task budget.

    transformers gives generation_config.max_new_tokens precedence over the max_length lm-eval's
    HFLM passes to generate(), so a model shipping one (Olmo-3-7B-Instruct: 32768) silently
    escapes the frozen 2,048-token ceiling; the evaluator must clear it before HFLM wraps the
    model. Caught on 16 Aug 2026 after five GSM8K validation units ran at the 32,768 ceiling.
    """
    class _FakeLM:
        def __init__(self, *a, **k): pass

    import lm_eval
    import pproute.evaluator as ev
    real_simple, real_hflm, real_stamp = lm_eval.simple_evaluate, ev.HFLM, ev.stamp_gold_index
    ev.HFLM = _FakeLM
    lm_eval.simple_evaluate = lambda **kw: {}
    ev.stamp_gold_index = lambda results, task_dict, predictor_contexts: results
    model = SimpleNamespace(config=None,
                            generation_config=SimpleNamespace(max_new_tokens=32768))
    try:
        Evaluator({}, None).evaluate(model, None)
    finally:
        lm_eval.simple_evaluate, ev.HFLM, ev.stamp_gold_index = real_simple, real_hflm, real_stamp
    assert model.generation_config.max_new_tokens is None, \
        "the model-shipped ceiling must be cleared so the task budget governs generation"
    print("  PASS  a model-shipped max_new_tokens ceiling is cleared before evaluation")
