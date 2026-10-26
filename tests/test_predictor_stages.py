"""Platt initialization and evidence binding through the fit and test entry points."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from pproute import consistency, predictor, predictor_protocol as protocol, scores
from pproute.provenance import identified, sha256_file
from scripts import evaluate_predictor, fit_predictor


def test_fit_and_test_keep_the_platt_anchor(tmp_path, monkeypatch):
    source = tmp_path / "scores"
    source.mkdir()
    for split in ("eval", "test"):
        (source / (split + "_scores.jbl")).write_bytes(split.encode())
    fingerprint = "task:20:development"
    split = protocol.make_split_manifest("task", list(range(20)), source / "eval_scores.jbl", fingerprint)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    endpoint = identified(dict(schema=protocol.ENDPOINT_SCHEMA, task="task", model="org/model",
        revision="a" * 40, endpoint="before", checkpoint="", selected_seed=None,
        selected_step=None, adapter_sha256=None, adapter_config_sha256=None))
    endpoint_path = tmp_path / "endpoint.json"
    endpoint_path.write_text(json.dumps(endpoint))
    raw = np.array([-13.81551, -2., 0., 2., 13.802318] * 4, dtype=np.float32)
    labels = np.array([0, 1] * 10, dtype=np.float32)
    data = dict(text=["evidence %d" % i for i in range(20)], Z=labels, raw_lfz=raw,
                L=np.arange(20, dtype=np.float64)/10, ids=list(range(20)))
    provenance = dict(task_name="task", base_model_name="org/model", revision="a" * 40,
                      dir_checkpoint="")
    monkeypatch.setattr(scores, "scored_model_provenance", lambda *a, **k: provenance)
    monkeypatch.setattr(scores, "recorded_split_fingerprint",
                        lambda path, split: fingerprint if split == "eval" else "task:20:test")
    monkeypatch.setattr(predictor, "load_scored_tokenizer", lambda *a: None)
    monkeypatch.setattr(predictor, "build_endpoint_data", lambda task, path, split, *a, **k:
                        dict(data, ids=list(range(20)) if split == "eval" else list(range(100, 120))))
    options = dict(exp_name=str(tmp_path / "fit"), task_name="task", task_type="mc",
        scores_dir=str(source), split_manifest=str(split_path), endpoint_manifest=str(endpoint_path),
        emb_model="org/embed", emb_revision="b"*40, backbone_tag="small", use_git=False,
        random_seed=4)
    class Config(dict):
        def show(self): pass
        def dump(self, path): pass
    class Recorder:
        def __init__(self, *a, **k): pass
        def add_with_logging(self, **k): pass
        def end_recording(self): pass
    for module in (fit_predictor, evaluate_predictor):
        monkeypatch.setattr(module, "Config", lambda default_config_dict, **k:
                            Config(default_config_dict, **options))
        monkeypatch.setattr(module, "Recorder", Recorder)
    calibrator = identified(dict(schema=protocol.CALIBRATOR_SCHEMA, task="task",
        model="org/model", endpoint="before", endpoint_identity=endpoint["identity"],
        split_manifest_sha256=sha256_file(split_path), source_fingerprint=fingerprint,
        source=str(source / "eval_scores.jbl"), source_sha256=sha256_file(source / "eval_scores.jbl"),
        a=.4, b=.7, n_fit=17, test_labels_used=False))
    path = tmp_path / "calibrator.json"
    path.write_text(json.dumps(calibrator))
    options["calibrator"] = str(path)
    expected_logits = protocol.calibrated_logits(raw, calibrator)
    def train(model, texts, z, te_texts, te_z, **kwargs):
        np.testing.assert_array_equal(texts, data["text"])
        np.testing.assert_array_equal(z, labels)
        np.testing.assert_array_equal(kwargs["tr_lfz"], expected_logits)
        _, selected = protocol.partition_indices(data["ids"], split)
        np.testing.assert_array_equal(kwargs["te_lfz"], expected_logits[selected])
        checkpoint = Path(kwargs["checkpoint_path"])
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text(json.dumps(kwargs["checkpoint_metadata"]))
        probabilities = expit(expected_logits[selected])
        loss = scores.bce(te_z, probabilities)
        return probabilities, dict(selected_epoch=0, holdout_init=loss, holdout_best=loss,
            effective_batch_size=16, micro_batch_size=16, gradient_accumulation_steps=1,
            input_max_len=384, input_train_max_tokens=10, input_test_max_tokens=10,
            anchor={"a": 0.0, "b": 1.0})
    monkeypatch.setattr(predictor, "train_verifier", train)
    assert fit_predictor.main() == 0
    validation = tmp_path / "fit/validation.json"
    candidate = protocol.read_predictor_candidate(validation)
    assert candidate["selected_epoch"] == 0
    assert candidate["training_protocol"]["anchor"] == "fixed_platt_fit_partition"
    assert candidate["test_artifacts_read"] is False
    assert candidate["fitted_anchor"] == {"a": 0.0, "b": 1.0}
    assert "anchor_learning_rate" not in candidate["training_protocol"]
    # A completed fit must validate its identity and skip training on rerun.
    monkeypatch.setattr(predictor, "train_verifier", lambda *a, **k: pytest.fail("retrained completed fit"))
    assert fit_predictor.main() == 0
    options.update(exp_name=str(tmp_path / "test"), expected_n=20)
    selected = dict(candidate, validation=str(validation), validation_sha256=sha256_file(validation))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(identified(dict(schema=protocol.SELECTION_SCHEMA, task="task",
        code_revision="c"*40, rows=[selected]))))
    options["predictor_selection"] = str(selection_path)
    def predict(model, texts, logits, checkpoint, **kwargs):
        np.testing.assert_array_equal(logits, expected_logits)
        return expit(logits).astype(np.float64), dict(checkpoint_metadata=json.loads(Path(checkpoint).read_text()),
            input_max_len=384, input_test_max_tokens=10, input_test_over_max_len=0)
    monkeypatch.setattr(predictor, "predict_verifier_checkpoint", predict)
    assert evaluate_predictor.main() == 0
    metric = json.loads((tmp_path / "test/metrics.json").read_text())
    np.testing.assert_array_equal(metric["test_z"], labels)
    np.testing.assert_array_equal(metric["test_y_loss"], data["L"])
    assert metric["LZ_f_raw"] == pytest.approx(scores.bce(labels, expit(raw)))
    assert metric["schema"] == protocol.TEST_METRICS_SCHEMA
    assert metric["candidate_validation"] == str(validation)
    monkeypatch.setattr(predictor, "predict_verifier_checkpoint", lambda *a, **k: pytest.fail("reevaluated completed test"))
    assert evaluate_predictor.main() == 0
    options["random_seed"] = 0
    with pytest.raises(ValueError, match="predictor seed"):
        evaluate_predictor.main()


def test_predictor_requires_a_calibrator():
    with pytest.raises(ValueError, match="requires a calibrator"):
        protocol.anchor_metadata("")


def test_confidence_source_checks_endpoint_and_score_hash(tmp_path, monkeypatch):
    score = tmp_path / "eval_scores.jbl"
    score.write_bytes(b"score")
    endpoint = dict(task="task", model="org/model", revision="a"*40, endpoint="before", identity="b"*64)
    meta = dict(endpoint, endpoint_identity=endpoint["identity"], scores_dir=str(tmp_path),
                scores_sha256={"eval": sha256_file(score)}, fingerprints={"eval_split": "fingerprint"})
    monkeypatch.setattr(consistency, "read_conf_scores", lambda *a, **k: {"meta": meta})
    assert protocol.endpoint_source_fingerprint("consistency", str(tmp_path), "eval", endpoint, "conf") == "fingerprint"
    meta["endpoint_identity"] = "c"*64
    with pytest.raises(ValueError, match="endpoint_identity"):
        protocol.endpoint_source_fingerprint("consistency", str(tmp_path), "eval", endpoint, "conf")
    meta["endpoint_identity"] = endpoint["identity"]
    score.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="scores_sha256"):
        protocol.endpoint_source_fingerprint("consistency", str(tmp_path), "eval", endpoint, "conf")
