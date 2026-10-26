"""Unit tests for the pieces every number passes through: metrics, selection rules,
optimizer grouping, score provenance, and split fingerprints.

All offline and fast. Nothing here loads a model or touches the network.
"""
import json
import os

import joblib as jbl
import numpy as np
import pytest
import torch

from pproute import analysis, predictor, predictor_protocol, scores
from pproute.models import config_dtype, dtype_kwargs
from pproute.scores import GOLD_KEY, PREDICTOR_TEXT_KEY, bce
from pproute.tasks import split_fingerprint


# --------------------------------------------------------------------- bce
def test_bce_matches_the_closed_form():
    z = np.array([1.0, 0.0])
    p = np.array([0.75, 0.25])
    assert bce(z, p) == pytest.approx(-np.log(0.75), abs=1e-12)


def test_bce_is_clipped_not_infinite():
    """A confident wrong prediction must cost a large finite number, never inf -- one such example
    would otherwise swallow a whole split's loss."""
    assert np.isfinite(bce(np.array([1.0]), np.array([0.0])))
    assert bce(np.array([1.0]), np.array([0.0])) > 10


def test_bce_perfect_and_worst_case_ordering():
    z = np.array([1.0, 0.0, 1.0])
    good = bce(z, np.array([0.9, 0.1, 0.8]))
    bad = bce(z, np.array([0.1, 0.9, 0.2]))
    assert good < bad


# --------------------------------------------------------------------- score files
def _mc_sample(doc_id, gold, lls):
    """A standard-layout MC sample with an explicit gold stamp."""
    ctx = "Q: item %d\nA:" % doc_id
    return {"doc_id": doc_id, "doc": {}, "target": gold, GOLD_KEY: gold,
            "arguments": [(ctx, " opt%d" % i) for i in range(len(lls))],
            "filtered_resps": [(ll, False) for ll in lls]}


def _write_mc(dirpath, fname, samples, task="mnli"):
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, fname)
    jbl.dump({"samples": {task: samples}}, path)
    return path


def _write_gen(dirpath, fname, zs, task="gsm8k"):
    os.makedirs(dirpath, exist_ok=True)
    samples = [{"doc_id": i, "arguments": [("question %d" % i, {})],
                PREDICTOR_TEXT_KEY: "Question: question %d\nAnswer:" % i,
                "resps": [["answer %d" % i]], "exact_match": z}
               for i, z in enumerate(zs)]
    path = os.path.join(dirpath, fname)
    jbl.dump({"samples": {task: samples}}, path)
    return path


def _write_score_config(dirpath, name, config, apply_chat_template):
    """Write the config plus finalized Recorder pair produced by evaluate_model."""
    with open(dirpath / (name + ".json"), "w") as handle:
        json.dump(config, handle)
    with open(dirpath / (name + ".result"), "w") as handle:
        handle.write(json.dumps({"chat_template": {
            "apply_chat_template": apply_chat_template,
        }}) + "\n")
        handle.write(json.dumps({"meta_data.git_commit": "f" * 40}) + "\n")
        handle.write(json.dumps({"meta_data.run_dir": str(dirpath)}) + "\n$END$\n")


def test_mc_target_only_predictor_input_is_exact_and_contains_no_gold(tmp_path):
    sample = _mc_sample(7, 1, [0.0, 0.0])
    sample["doc"] = {"answerKey": "GOLD-SENTINEL"}
    path = _write_mc(str(tmp_path), "eval_scores.jbl", [sample])

    texts, z, lfz, losses, ids = predictor.build_mc(path)

    assert texts == [
        "Instruct: Predict whether the proposed answer to this multiple-choice question is "
        "correct. Use the target question, answer choices, and the model's answer distribution "
        "as evidence.\nQuery:\nQ: item 7\nA:\n\nAnswer choices:\nA. opt0\nB. opt1\n\n"
        "Model probabilities: A=0.500000 B=0.500000\nProposed answer: A. opt0"
    ]
    assert "GOLD-SENTINEL" not in texts[0]
    assert z.tolist() == [0.0] and ids == [7]
    assert lfz.tolist() == [0.0]
    assert losses.tolist() == pytest.approx([np.log(2)])


def test_winogrande_target_only_predictor_input_preserves_multiple_input_layout(tmp_path):
    sample = {
        "doc_id": 9,
        "doc": {"answer": "GOLD-SENTINEL"},
        "target": " is too large.",
        GOLD_KEY: 0,
        "arguments": [
            ("The trophy does not fit in the suitcase because the trophy", " is too large."),
            ("The trophy does not fit in the suitcase because the suitcase", " is too large."),
        ],
        "filtered_resps": [(0.0, False), (0.0, False)],
    }
    path = _write_mc(str(tmp_path), "eval_scores.jbl", [sample], task="winogrande")

    text = predictor.build_mc(path)[0][0]

    assert text == (
        "Instruct: Predict whether the proposed answer to this multiple-choice question is "
        "correct. Use the target question, answer choices, and the model's answer distribution "
        "as evidence.\nQuery:\nThe trophy does not fit in the suitcase because the ___ is too "
        "large.\n\nAnswer choices:\nA. trophy\nB. suitcase\n\n"
        "Model probabilities: A=0.500000 B=0.500000\nProposed answer: A. trophy"
    )
    assert "GOLD-SENTINEL" not in text


def test_self_metrics_agrees_with_bce(tmp_path):
    """self_metrics' L_Z(f) must be exactly bce against the model's own confidence -- the two were
    separate formulas before and could drift apart."""
    path = _write_mc(str(tmp_path), "test_scores.jbl",
                     [_mc_sample(0, 0, [-0.1, -2.0]), _mc_sample(1, 1, [-0.2, -3.0])])
    m = scores.self_metrics(path)
    examples = scores.mc_examples(path)
    assert m["LZ_self"] == pytest.approx(
        bce(np.array([e["Z"] for e in examples]), [e["fz"] for e in examples]), abs=1e-12)
    assert m["n"] == 2
    assert m["acc"] == pytest.approx(0.5)          # first right, second wrong


def test_generative_predictor_prefers_stamped_task_text_over_rendered_chat(tmp_path):
    path = _write_gen(str(tmp_path), "eval_scores.jbl", [1])
    payload = jbl.load(path)
    sample = payload["samples"]["gsm8k"][0]
    sample["arguments"][0] = ("<chat>five separate turns</chat>", {})
    normalized = (
        "Question: demo one\nAnswer: worked one\n\n"
        "Question: demo two\nAnswer: worked two\n\n"
        "Question: target only\nAnswer:"
    )
    sample[PREDICTOR_TEXT_KEY] = normalized
    jbl.dump(payload, path)

    example = scores.gen_examples(path)[0]
    assert example["question"] == "<chat>five separate turns</chat>"
    assert example["predictor_text"] == normalized


def test_scored_model_provenance_binds_explicit_compute_dtype(tmp_path):
    config = {"task_name": "gsm8k", "base_model_name": "org/model",
              "revision": "a" * 40, "model_dtype": "bfloat16"}
    _write_score_config(tmp_path, "validation", config, False)
    assert scores.scored_model_provenance(tmp_path)["model_dtype"] == "bfloat16"
    _write_score_config(tmp_path, "test", dict(config, model_dtype="auto"), False)
    with pytest.raises(ValueError, match="mixes scored model identities"):
        scores.scored_model_provenance(tmp_path)


def test_scored_model_provenance_includes_revision_and_refuses_mixing(tmp_path):
    revision = "a" * 40
    first = {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": revision,
        "dir_checkpoint": "",
        "random_seed": 0,
        "dataset_seed": 0,
        "fewshot_random_seed": 1234,
        "max_new_tokens": 2048,
        "eval_use_auto_batch_size": False,
        "eval_max_batch_size": 16,
        "dataset_param": {
            "num_fewshot": 5,
            "fewshot_as_multiturn": True,
            "apply_chat_template": True,
            "allow_untemplated_instruct": False,
        },
    }
    _write_score_config(tmp_path, "one", first, apply_chat_template=True)
    assert scores.scored_model_provenance(tmp_path) == {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": revision,
        "dir_checkpoint": "",
        "prompt_protocol": {
            "random_seed": 0,
            "dataset_seed": 0,
            "fewshot_random_seed": 1234,
            "num_fewshot": 5,
            "fewshot_as_multiturn": True,
            "apply_chat_template": True,
            "requested_apply_chat_template": True,
            "allow_untemplated_instruct": False,
            "max_new_tokens": 2048,
            "eval_use_auto_batch_size": False,
            "eval_max_batch_size": 16,
        },
    }
    _write_score_config(
        tmp_path, "two", dict(first, dir_checkpoint="/checkpoint/one"),
        apply_chat_template=True,
    )
    with pytest.raises(ValueError, match="mixes scored model identities"):
        scores.scored_model_provenance(tmp_path)


def test_recorded_split_fingerprint_uses_finalized_score_recorders(tmp_path):
    config = {
        "task_name": "task", "base_model_name": "org/model", "revision": "a" * 40,
        "dir_checkpoint": "", "dataset_param": {},
    }
    _write_score_config(tmp_path, "one", config, apply_chat_template=False)
    path = tmp_path / "one.result"
    text = path.read_text().replace(
        "$END$", json.dumps({"eval_split": "task:10:abcdef"}) + "\n$END$"
    )
    path.write_text(text)
    assert scores.recorded_split_fingerprint(tmp_path, "eval") == "task:10:abcdef"

    _write_score_config(tmp_path, "two", config, apply_chat_template=False)
    path = tmp_path / "two.result"
    text = path.read_text().replace(
        "$END$", json.dumps({"eval_split": "task:10:different"}) + "\n$END$"
    )
    path.write_text(text)
    with pytest.raises(ValueError, match="2 finalized"):
        scores.recorded_split_fingerprint(tmp_path, "eval")


def test_scored_model_provenance_accepts_resume_but_rejects_prompt_mixing(tmp_path):
    revision = "a" * 40
    config = {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": revision,
        "dir_checkpoint": "/checkpoint/one",
        "random_seed": 0,
        "dataset_seed": 0,
        "fewshot_random_seed": 1234,
        "max_new_tokens": 2048,
        "eval_use_auto_batch_size": False,
        "eval_max_batch_size": 16,
        "dataset_param": {"num_fewshot": 5, "fewshot_as_multiturn": True},
    }
    for name in ("validation", "test"):
        _write_score_config(tmp_path, name, config, apply_chat_template=True)
    assert scores.scored_model_provenance(tmp_path)["prompt_protocol"]["num_fewshot"] == 5

    changed = json.loads(json.dumps(config))
    changed["dataset_param"]["num_fewshot"] = 0
    _write_score_config(tmp_path, "wrong-shots", changed, apply_chat_template=True)
    with pytest.raises(ValueError, match="mixes scored model identities"):
        scores.scored_model_provenance(tmp_path)


def test_scored_model_provenance_rejects_fewshot_seed_mixing(tmp_path):
    config = {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": "a" * 40,
        "dir_checkpoint": "",
        "fewshot_random_seed": 1234,
        "dataset_param": {"num_fewshot": 5},
    }
    _write_score_config(tmp_path, "one", config, apply_chat_template=False)
    _write_score_config(
        tmp_path, "two", dict(config, fewshot_random_seed=7),
        apply_chat_template=False,
    )
    with pytest.raises(ValueError, match="mixes scored model identities"):
        scores.scored_model_provenance(tmp_path)


def test_scored_model_provenance_binds_dataset_seed_and_reads_legacy_seed(tmp_path):
    config = dict(task_name="gsm8k", base_model_name="org/model", revision="a" * 40,
                  dir_checkpoint="", random_seed=7, dataset_param={"num_fewshot": 5})
    _write_score_config(tmp_path, "old", config, apply_chat_template=False)
    assert scores.scored_model_provenance(tmp_path)["prompt_protocol"]["dataset_seed"] == 7
    _write_score_config(tmp_path, "new", dict(config, dataset_seed=7), apply_chat_template=False)
    assert scores.scored_model_provenance(tmp_path)["prompt_protocol"]["dataset_seed"] == 7
    _write_score_config(tmp_path, "different", dict(config, dataset_seed=0), apply_chat_template=False)
    with pytest.raises(ValueError, match="mixes scored model identities"):
        scores.scored_model_provenance(tmp_path)


def test_scored_model_provenance_rejects_mutable_revision(tmp_path):
    _write_score_config(tmp_path, "config", {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": "main",
        "dir_checkpoint": "",
    }, apply_chat_template=False)
    with pytest.raises(ValueError, match="pinned 40-hex"):
        scores.scored_model_provenance(tmp_path)


def test_scored_model_provenance_uses_resolved_not_requested_chat_mode(tmp_path):
    config = {
        "task_name": "gsm8k",
        "base_model_name": "org/model-instruct",
        "revision": "a" * 40,
        "dir_checkpoint": "",
        "random_seed": 0,
        "dataset_seed": 0,
        "max_new_tokens": 2048,
        "eval_use_auto_batch_size": False,
        "eval_max_batch_size": 8,
        "dataset_param": {
            "num_fewshot": 5,
            "fewshot_as_multiturn": True,
            "apply_chat_template": False,
            "allow_untemplated_instruct": False,
        },
    }
    _write_score_config(tmp_path, "eval", config, apply_chat_template=True)
    protocol = scores.scored_model_provenance(tmp_path)["prompt_protocol"]
    assert protocol["apply_chat_template"] is True
    assert protocol["requested_apply_chat_template"] is False
    assert protocol["eval_max_batch_size"] == 8


def test_scored_model_provenance_ignores_failed_attempt_if_retry_finalized(tmp_path):
    config = {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": "a" * 40,
        "dir_checkpoint": "",
    }
    with open(tmp_path / "config.json", "w") as handle:
        json.dump(config, handle)
    with pytest.raises(ValueError, match="no recorded scored-model provenance"):
        scores.scored_model_provenance(tmp_path)
    _write_score_config(tmp_path, "retry", config, apply_chat_template=False)
    assert scores.scored_model_provenance(tmp_path)["base_model_name"] == "org/model"


def test_scored_model_provenance_allows_runs_from_different_commits(tmp_path):
    config = {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": "a" * 40,
        "dir_checkpoint": "",
    }
    _write_score_config(tmp_path, "first", config, apply_chat_template=False)
    _write_score_config(tmp_path, "second", config, apply_chat_template=False)
    second = tmp_path / "second.result"
    second.write_text(second.read_text().replace("f" * 40, "e" * 40))
    assert scores.scored_model_provenance(tmp_path)["base_model_name"] == "org/model"


def test_scored_model_provenance_allows_explicit_no_git_development(tmp_path):
    config = {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": "a" * 40,
        "dir_checkpoint": "",
    }
    _write_score_config(tmp_path, "eval", config, apply_chat_template=False)
    result = tmp_path / "eval.result"
    lines = [
        line for line in result.read_text().splitlines()
        if "meta_data.git_commit" not in line
    ]
    result.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="has no git commit"):
        scores.scored_model_provenance(tmp_path)
    provenance = scores.scored_model_provenance(tmp_path, require_git=False)
    assert provenance["base_model_name"] == "org/model"


def test_scored_model_provenance_resolves_relative_checkpoint_from_run_dir(tmp_path):
    config = {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": "a" * 40,
        "dir_checkpoint": "results/adapter",
    }
    _write_score_config(tmp_path, "eval", config, apply_chat_template=False)
    assert scores.scored_model_provenance(tmp_path)["dir_checkpoint"] == str(
        tmp_path / "results" / "adapter"
    )


def test_embedding_loader_forwards_one_pinned_revision(monkeypatch):
    calls = {}

    class Loader:
        def __init__(self, kind):
            self.kind = kind

        def from_pretrained(self, name, **kwargs):
            calls[self.kind] = (name, kwargs)
            return self.kind

    monkeypatch.setattr(predictor, "AutoTokenizer", Loader("tokenizer"))
    monkeypatch.setattr(predictor, "AutoModel", Loader("model"))
    revision = "a" * 40
    tokenizer, model = predictor.load_embedding_backbone("org/embed", revision)

    assert (tokenizer, model) == ("tokenizer", "model")
    assert calls["tokenizer"][1]["revision"] == revision
    assert calls["model"][1]["revision"] == revision


# --------------------------------------------------------------------- pre-registered rules
def test_model_selection_ties_by_step_then_seed(tmp_path):
    candidates = []
    for seed, step in ((0, 600), (0, 300), (1, 300), (2, 300)):
        directory = os.path.join(str(tmp_path), "seed-%d-step-%d" % (seed, step))
        _write_mc(directory, "eval_scores.jbl", [_mc_sample(0, 0, [-0.1, -2.0])])
        candidates.append({"scores_dir": directory, "seed": seed, "step": step})
    best, table = analysis.select_model_candidate(candidates, "mc")
    assert len(table) == 4
    assert (best["step"], best["seed"]) == (300, 0)


def test_model_selection_optimizes_before_tie_breaking(tmp_path):
    candidates = []
    for seed, step, zs in ((0, 300, [1, 0]), (2, 600, [1, 1])):
        directory = os.path.join(str(tmp_path), "seed-%d-step-%d" % (seed, step))
        _write_gen(directory, "eval_scores.jbl", zs)
        candidates.append({"scores_dir": directory, "seed": seed, "step": step})
    best, _ = analysis.select_model_candidate(candidates, "generative")
    assert (best["step"], best["seed"]) == (600, 2)


def test_predictor_selection_uses_validation_loss_then_seed():
    candidates = [
        {"predictor_seed": 2, "LZ_pred": 0.4},
        {"predictor_seed": 1, "LZ_pred": 0.3},
        {"predictor_seed": 0, "LZ_pred": 0.3},
    ]
    assert analysis.select_predictor_candidate(candidates)["predictor_seed"] == 0


# --------------------------------------------------------------------- optimizer grouping
class _FakeVerifier:
    """Only what param_groups touches: named_parameters(), .head, .a, .b."""

    def __init__(self):
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.ones(1))
        self.head = torch.nn.Linear(4, 1)
        self.lora = torch.nn.Parameter(torch.zeros(2))
        self.frozen = torch.nn.Parameter(torch.zeros(2), requires_grad=False)

    def named_parameters(self):
        return [("a", self.a), ("b", self.b),
                ("head.weight", self.head.weight), ("head.bias", self.head.bias),
                ("enc.base_model.lora_A.default.weight", self.lora),
                ("enc.base_model.frozen.weight", self.frozen)]


def _group_of(groups, param):
    for g in groups:
        if any(p is param for p in g["params"]):
            return g
    raise AssertionError("parameter is in no optimizer group -- it would never be updated")


def test_offset_params_are_not_weight_decayed():
    """b starts at 1 and IS the logit f_Z passthrough, so decaying it toward 0 would pull the
    anchor apart. Its prior is 1, not 0."""
    m = _FakeVerifier()
    groups = predictor.param_groups(m)
    assert _group_of(groups, m.b)["weight_decay"] == 0.0
    assert _group_of(groups, m.a)["weight_decay"] == 0.0


def test_head_is_weight_decayed_and_lora_is_not():
    m = _FakeVerifier()
    groups = predictor.param_groups(m)
    assert _group_of(groups, m.head.weight)["weight_decay"] == predictor_protocol.HEAD_WEIGHT_DECAY
    assert _group_of(groups, m.lora)["weight_decay"] == 0.0


def test_frozen_params_are_not_optimized():
    m = _FakeVerifier()
    groups = predictor.param_groups(m)
    assert all(not any(p is m.frozen for p in g["params"]) for g in groups)






# --------------------------------------------------------------------- split fingerprint
class _FakeTask:
    def __init__(self, docs, name="winogrande"):
        self._docs = docs
        self.config = type("C", (), {"task": name})()

    @property
    def eval_docs(self):
        return self._docs


def test_split_fingerprint_is_stable_and_content_sensitive():
    a = _FakeTask([{"q": 1}, {"q": 2}])
    assert split_fingerprint(a) == split_fingerprint(_FakeTask([{"q": 1}, {"q": 2}]))
    # different content, same length -- the case a length-only or index-based hash would miss
    assert split_fingerprint(a) != split_fingerprint(_FakeTask([{"q": 1}, {"q": 3}]))
    # different order is a different split as far as the predictor is concerned
    assert split_fingerprint(a) != split_fingerprint(_FakeTask([{"q": 2}, {"q": 1}]))
    # and it carries the size and task name in the clear
    assert split_fingerprint(a).startswith("winogrande:2:")


def test_split_fingerprint_distinguishes_a_changed_valid_frac():
    """The scenario it exists for: TASK_VALID_FRAC changes, so the carve changes, so old and new
    scores are not comparable."""
    docs = [{"i": i} for i in range(100)]
    assert split_fingerprint(_FakeTask(docs[:20])) != split_fingerprint(_FakeTask(docs[:30]))


# --------------------------------------------------------------------- dtype plumbing
def test_dtype_kwargs_tracks_the_installed_transformers():
    import transformers
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    assert list(dtype_kwargs("auto")) == [key]


def test_config_dtype_reads_both_spellings_and_strings():
    """transformers 5 renamed the config field and may store it as a string; a None here means
    fp32 training, which is a 2x memory difference on a 9B model."""
    assert config_dtype(type("C", (), {"dtype": torch.bfloat16})()) is torch.bfloat16
    assert config_dtype(type("C", (), {"torch_dtype": torch.float16})()) is torch.float16
    assert config_dtype(type("C", (), {"dtype": "bfloat16"})()) is torch.bfloat16
    assert config_dtype(type("C", (), {})()) is None
    assert config_dtype(type("C", (), {"dtype": "not_a_dtype"})()) is None


def test_config_dtype_reads_nested_text_config():
    """The Qwen3.5-9B shape: bfloat16 declared ONLY in text_config, no top-level dtype. Reading
    the top level alone returns None and downgrades a 9B finetune to fp32 silently."""
    nested = type("C", (), {"text_config": type("T", (), {"dtype": torch.bfloat16})()})()
    assert config_dtype(nested) is torch.bfloat16
    # a string in the nested config too
    nested_str = type("C", (), {"text_config": type("T", (), {"torch_dtype": "bfloat16"})()})()
    assert config_dtype(nested_str) is torch.bfloat16
    # top level wins when both are present
    both = type("C", (), {"dtype": torch.float16,
                          "text_config": type("T", (), {"dtype": torch.bfloat16})()})()
    assert config_dtype(both) is torch.float16
