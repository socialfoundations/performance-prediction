"""Offline checks for the release's development-only selection entry point."""
import json
from pathlib import Path

import pytest

from pproute.selection import Selection


def test_selection_rejects_existing_test_artifacts(tmp_path):
    selection = Selection("commonsense_qa", tmp_path)
    path = tmp_path / "endpoints/commonsense_qa/stray/test_scores.jbl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"held out")
    with pytest.raises(SystemExit, match="held-out artifacts"):
        selection.require_development_only()


@pytest.mark.parametrize("mismatch", [None, "ids", "fingerprint"])
def test_development_endpoints_validate_shared_pool_once(tmp_path, monkeypatch, mismatch):
    import numpy as np
    from pproute import predictor_protocol
    from pproute.provenance import sha256_text

    selection = Selection("commonsense_qa", tmp_path)
    selection.models = selection.models[:1]
    selection.task.update(development_rows=2,
                          development_fingerprint_sha256=sha256_text("fixed"))
    calls = []

    def evidence(model, endpoint):
        calls.append(endpoint)
        ids = [1, 0] if mismatch == "ids" and endpoint == "after" else [0, 1]
        return ids, np.array([1, 0]), np.array([.5, -.5]), "source"

    monkeypatch.setattr(selection, "read_endpoint_record", lambda model, endpoint: endpoint)
    monkeypatch.setattr(selection, "development_evidence", evidence)
    monkeypatch.setattr(predictor_protocol, "endpoint_source_fingerprint",
                        lambda kind, path, split, record, conf:
                        "wrong" if mismatch == "fingerprint" and record == "after" else "fixed")
    if mismatch:
        with pytest.raises(ValueError, match="Development"):
            list(selection.development_endpoints())
    else:
        endpoints = list(selection.development_endpoints())
        assert [row[1] for row in endpoints] == ["before", "after"]
    assert calls == ["before", "after"]


@pytest.fixture
def strong_candidates(tmp_path, monkeypatch):
    from pproute import scores
    from pproute.provenance import sha256_text

    selection = Selection("commonsense_qa", tmp_path)
    monkeypatch.setattr(selection, "require_clean_git", lambda: None)
    model = "Qwen/Qwen3.5-122B-A10B"
    fingerprint = "commonsense_qa:development:fixed"
    selection.task["development_fingerprint_sha256"] = sha256_text(fingerprint)
    records, metrics = {}, {}
    for step in (0, 400, 800, 1200, 1600):
        directory = tmp_path / f"commonsense_qa/development/step-{step}"
        directory.mkdir(parents=True)
        (directory / "eval_scores.jbl").write_bytes(str(step).encode())
        checkpoint = tmp_path / f"model_candidates/commonsense_qa/{model}/seed-0/train/checkpoint-{step}"
        if step:
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            (checkpoint / "adapter_config.json").write_text("{}")
        records[str(directory)] = dict(
            task_name="commonsense_qa", base_model_name=model,
            revision="dc4d348443bc740c68e2d77492492c11606384d5",
            dir_checkpoint=str(checkpoint) if step else "", model_dtype="bfloat16",
            prompt_protocol=dict(random_seed=0, dataset_seed=0, fewshot_random_seed=1234, num_fewshot=5,
                                 apply_chat_template=True, fewshot_as_multiturn=True,
                                 max_new_tokens=-1),
        )
        metrics[str(directory / "eval_scores.jbl")] = dict(
            n=3342, L_f=.2 if step in (400, 800) else .4)
    monkeypatch.setattr(scores, "scored_model_provenance", lambda path: records[str(path)])
    monkeypatch.setattr(scores, "recorded_split_fingerprint", lambda *args: fingerprint)
    monkeypatch.setattr(scores, "self_metrics", lambda path: metrics[str(path)])
    return selection, records, metrics


def test_strong_selection_uses_development_loss_and_earlier_tie(strong_candidates):
    from pproute.provenance import validate_identity
    selection, _, _ = strong_candidates
    selection.select_strong()
    record = json.loads((Path(selection.common.RESULTS) / "commonsense_qa/selection.json").read_text())
    validate_identity(record, "strong selection")
    assert record["selected"]["step"] == 400
    assert record["selected"]["checkpoint"].endswith("/seed-0/train/checkpoint-400")


@pytest.mark.parametrize("field,value,match", [
    ("task_name", "winogrande", "records task"),
    ("model_dtype", "float32", "wrong model dtype"),
    ("revision", "a" * 40, "revision"),
    ("dir_checkpoint", "/wrong/checkpoint", "dir_checkpoint"),
    ("num_fewshot", 0, "prompt protocol"),
    ("fewshot_random_seed", 42, "prompt protocol"),
    ("random_seed", 1, "prompt protocol"),
    ("dataset_seed", 1, "prompt protocol"),
    ("apply_chat_template", False, "prompt protocol"),
    ("fewshot_as_multiturn", False, "prompt protocol"),
    ("max_new_tokens", 2048, "prompt protocol"),
])
def test_selection_rejects_off_protocol_scores(strong_candidates, field, value, match):
    selection, records, _ = strong_candidates
    record = next(iter(records.values()))
    target = record["prompt_protocol"] if field in record["prompt_protocol"] else record
    target[field] = value
    with pytest.raises(ValueError, match=match):
        selection.select_strong()


def test_strong_selection_rejects_same_size_wrong_split(strong_candidates, monkeypatch):
    from pproute import scores
    selection, _, _ = strong_candidates
    monkeypatch.setattr(scores, "recorded_split_fingerprint", lambda *args: "another split")
    with pytest.raises(ValueError, match="development fingerprint"):
        selection.select_strong()


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf")])
def test_strong_selection_rejects_nonfinite_loss(strong_candidates, loss):
    selection, _, metrics = strong_candidates
    next(iter(metrics.values()))["L_f"] = loss
    with pytest.raises(ValueError, match="must be finite"):
        selection.select_strong()


def test_strong_selection_stops_if_finetuning_does_not_win(strong_candidates):
    selection, _, metrics = strong_candidates
    next(iter(metrics.values()))["L_f"] = .1
    with pytest.raises(ValueError, match="Released strong model won"):
        selection.select_strong()


@pytest.mark.parametrize("fit,holdout", [([0], [0, 1]), ([0], []), ([0], [2]), ([[0]], [1])])
def test_predictor_rejects_invalid_fixed_split_before_model_loading(monkeypatch, fit, holdout):
    import numpy as np
    from pproute import predictor
    monkeypatch.setattr(predictor, "load_embedding_backbone",
                        lambda *a, **k: pytest.fail("loaded a model before split validation"))
    with pytest.raises(ValueError, match="fixed predictor partitions"):
        predictor.train_verifier(
            "unused", ["a", "b"], np.array([0., 1.]), ["c"], np.array([0.]),
            emb_revision="a" * 40, tr_lfz=np.zeros(2), te_lfz=np.zeros(1),
            fit_indices=fit, holdout_indices=holdout,
        )


def test_selection_keeps_the_output_path_as_given(tmp_path):
    """A results directory reached through a symlink must not be resolved: the fitting scripts
    record os.path.abspath of the paths they were given, and selection compares those strings."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    selection = Selection("commonsense_qa", link)
    assert selection.common.RESULTS == str(link)
    assert selection.split_manifest_path.startswith(str(link))
