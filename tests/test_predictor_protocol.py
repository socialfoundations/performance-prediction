import json

import numpy as np
import pytest

from pproute import predictor_protocol, scores
from pproute.provenance import identified, sha256_file


def _manifest(tmp_path, ids):
    source = tmp_path / "source.bin"
    source.write_bytes(b"frozen development artifact")
    return predictor_protocol.make_split_manifest(
        "task", ids, source, "task:%d:fingerprint" % len(ids)
    )


def test_predictor_split_is_fixed_85_15_and_typed(tmp_path):
    ids = list(range(20))
    manifest = _manifest(tmp_path, ids)
    assert manifest["n_fit"] == 17
    assert manifest["n_selection"] == 3
    assert "code_revision" not in manifest
    assert manifest["fit_ids"] == [4, 19, 6, 2, 13, 16, 3, 11, 10, 8, 0, 12, 7, 5, 18, 17, 14]
    predictor_protocol.validate_split_manifest(manifest, task="task", ids=ids)

    typed = _manifest(tmp_path, [1, "1", 2])
    assert typed["n_development"] == 3
    predictor_protocol.validate_split_manifest(typed, ids=[1, "1", 2])


def test_endpoint_manifest_identity_and_checkpoint_are_self_authenticating(tmp_path):
    core = {
        "schema": predictor_protocol.ENDPOINT_SCHEMA,
        "task": "task", "model": "org/model", "revision": "a" * 40,
        "endpoint": "after", "selected_seed": 1, "selected_step": 300,
        "checkpoint": "/fast/project/checkpoint-300", "adapter_sha256": "b" * 64,
        "adapter_config_sha256": "e" * 64,
    }
    payload = identified(core)
    path = tmp_path / "endpoint.json"
    path.write_text(json.dumps(payload))

    assert predictor_protocol.read_endpoint_manifest(
        path, task="task", model="org/model", revision="a" * 40,
        endpoint="after", checkpoint="/fast/project/checkpoint-300",
    ) == payload

    tampered = dict(payload, selected_step=600)
    with pytest.raises(ValueError, match="identity"):
        predictor_protocol.validate_endpoint_manifest(tampered)
    with pytest.raises(ValueError, match="checkpoint"):
        predictor_protocol.validate_endpoint_manifest(
            payload, checkpoint="/fast/project/checkpoint-600"
        )



def test_predictor_split_indices_follow_manifest_order(tmp_path):
    ids = ["c", "a", "b", "d"]
    manifest = _manifest(tmp_path, ids)
    fit, selection = predictor_protocol.partition_indices(ids, manifest)
    assert [ids[index] for index in fit] == manifest["fit_ids"]
    assert [ids[index] for index in selection] == manifest["selection_ids"]
    with pytest.raises(ValueError, match="IDs/order"):
        predictor_protocol.partition_indices(list(reversed(ids)), manifest)


def test_predictor_split_refuses_a_reidentified_nonseeded_partition(tmp_path):
    manifest = _manifest(tmp_path, list(range(20)))
    core = dict(manifest)
    core.pop("identity")
    core["fit_ids"][0], core["selection_ids"][0] = (
        core["selection_ids"][0], core["fit_ids"][0]
    )
    core["fit_ids_sha256"] = predictor_protocol.ids_sha256(core["fit_ids"])
    core["selection_ids_sha256"] = predictor_protocol.ids_sha256(core["selection_ids"])
    tampered = identified(core)
    with pytest.raises(ValueError, match="seed/fraction"):
        predictor_protocol.validate_split_manifest(tampered)


def test_predictor_split_validation_refuses_empty_partitions(tmp_path):
    for ids in ([], ["only-one"]):
        core = {
            "schema": predictor_protocol.SPLIT_SCHEMA,
            "task": "task",
            "method": "numpy.random.default_rng(seed).permutation",
            "seed": predictor_protocol.SPLIT_SEED,
            "fit_fraction": predictor_protocol.FIT_FRACTION,
            "n_development": len(ids),
            "n_fit": 0,
            "n_selection": len(ids),
            "development_ids": ids,
            "fit_ids": [],
            "selection_ids": ids,
            "development_ids_sha256": predictor_protocol.ids_sha256(ids),
            "fit_ids_sha256": predictor_protocol.ids_sha256([]),
            "selection_ids_sha256": predictor_protocol.ids_sha256(ids),
            "source_path": str((tmp_path / "source.bin").resolve()),
            "source_sha256": "a" * 64,
            "source_fingerprint": "task:%d:fingerprint" % len(ids),
        }
        with pytest.raises(ValueError, match="empty"):
            predictor_protocol.validate_split_manifest(
                identified(core)
            )


def test_predictor_split_create_once_is_read_only(tmp_path):
    manifest = _manifest(tmp_path, list(range(10)))
    path = tmp_path / "split.json"
    predictor_protocol.atomic_create_json(manifest, path)
    assert predictor_protocol.read_split_manifest(path, task="task") == manifest
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        predictor_protocol.atomic_create_json(manifest, path)


def test_platt_calibrator_roundtrip(tmp_path):
    raw = np.repeat(np.array([-1.0, 0.0, 1.0]), 20)
    z = np.concatenate((
        np.array([1] * 5 + [0] * 15),
        np.array([1] * 10 + [0] * 10),
        np.array([1] * 15 + [0] * 5),
    ))
    a, b, info = predictor_protocol.fit_platt(z, raw)
    payload = identified({
        "schema": predictor_protocol.CALIBRATOR_SCHEMA,
        "task": "task", "model": "org/model", "endpoint": "before",
        "a": a, "b": b, "n_fit": info["n_fit"],
        "source_fingerprint": "task:60:fingerprint",
        "split_manifest_sha256": "b" * 64, "test_labels_used": False,
    })
    path = tmp_path / "calibrator.json"
    path.write_text(json.dumps(payload))
    loaded = predictor_protocol.read_calibrator(
        path, task="task", model="org/model", endpoint="before",
        split_manifest_sha256="b" * 64,
    )
    assert loaded["a"] == pytest.approx(0.0, abs=1e-7)
    assert loaded["b"] == pytest.approx(np.log(3), abs=1e-6)
    assert predictor_protocol.calibrated_logits([-1, 1], loaded) == pytest.approx(
        [-np.log(3), np.log(3)], abs=1e-6
    )


def test_predictor_split_uses_finalized_dataset_fingerprint(tmp_path):
    score_file = tmp_path / "eval_scores.jbl"
    score_file.write_bytes(b"score bytes")
    score_config = {"base_model_name": "org/model"}
    (tmp_path / "unfinished.json").write_text(json.dumps(score_config))
    (tmp_path / "unfinished.result").write_text(
        json.dumps({"eval_split": "wrong-unfinished"}) + "\n"
    )
    (tmp_path / "finished.json").write_text(json.dumps(score_config))
    (tmp_path / "finished.result").write_text(
        json.dumps({"eval_split": "task:20:abc123"}) + "\n$END$\n"
    )
    assert scores.recorded_split_fingerprint(tmp_path, "eval") == "task:20:abc123"

    (tmp_path / "conflict.json").write_text(json.dumps(score_config))
    (tmp_path / "conflict.result").write_text(
        json.dumps({"eval_split": "task:20:different"}) + "\n$END$\n"
    )
    with pytest.raises(ValueError, match="2 finalized"):
        scores.recorded_split_fingerprint(tmp_path, "eval")


def test_predictor_candidate_reader_checks_inputs_and_development_only_claim(tmp_path):
    paths = {}
    for name in ("split_manifest", "calibrator", "development_source", "checkpoint"):
        path = (tmp_path / name).resolve()
        path.write_bytes(name.encode())
        paths[name] = path
    payload = {
        "schema": predictor_protocol.CANDIDATE_SCHEMA,
        "task": "task", "task_type": "consistency", "model": "org/model",
        "revision": "a" * 40, "endpoint": "before", "endpoint_identity": "b" * 64,
        "backbone_tag": "small", "embedding_model": "org/embed",
        "embedding_revision": "c" * 40, "predictor_seed": 2,
        "test_artifacts_read": False,
    }
    for name, path in paths.items():
        payload[name] = str(path)
        payload[name + "_sha256"] = sha256_file(path)
    candidate = tmp_path / "validation.json"
    candidate.write_text(json.dumps(payload))

    loaded = predictor_protocol.read_predictor_candidate(
        candidate, task="task", model="org/model", endpoint="before",
        backbone_tag="small",
    )
    assert loaded["predictor_seed"] == 2

    payload["test_artifacts_read"] = True
    candidate.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="development-only"):
        predictor_protocol.read_predictor_candidate(candidate)
