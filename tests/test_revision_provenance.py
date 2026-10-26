import types

import pytest

from peft import LoraConfig
from pproute import models, predictor, provenance, scores
from scripts.conf_sample import validate_prepared_endpoint


def test_atomic_json_dump_rejects_nonfinite_without_replacing(tmp_path):
    path = tmp_path / "artifact.json"
    provenance.atomic_json_dump({"value": 1.0}, str(path))

    with pytest.raises(ValueError, match="JSON compliant"):
        provenance.atomic_json_dump({"value": float("nan")}, str(path))

    assert path.read_text() == '{\n  "value": 1.0\n}\n'
    assert not list(tmp_path.glob("*.partial"))


def test_scored_tokenizer_uses_pinned_hub_revision_for_base(monkeypatch):
    calls = []
    monkeypatch.setattr(
        predictor,
        "AutoTokenizer",
        types.SimpleNamespace(
            from_pretrained=lambda source, **kwargs: calls.append((source, kwargs)) or object()
        ),
    )
    provenance = {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": "a" * 40,
        "dir_checkpoint": "",
    }
    predictor.load_scored_tokenizer(provenance)
    assert calls[0][0] == "org/model"
    assert calls[0][1]["revision"] == "a" * 40


def test_scored_tokenizer_uses_local_checkpoint_without_hub_revision(monkeypatch):
    calls = []
    monkeypatch.setattr(
        predictor,
        "AutoTokenizer",
        types.SimpleNamespace(
            from_pretrained=lambda source, **kwargs: calls.append((source, kwargs)) or object()
        ),
    )
    provenance = {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": "a" * 40,
        "dir_checkpoint": "/local/checkpoint-1500",
    }
    predictor.load_scored_tokenizer(provenance)
    assert calls[0][0] == "/local/checkpoint-1500"
    assert "revision" not in calls[0][1]


def test_requested_endpoint_must_match_score_checkpoint():
    provenance = {
        "task_name": "gsm8k",
        "base_model_name": "org/model",
        "revision": "a" * 40,
        "dir_checkpoint": "/checkpoint/selected",
    }
    scores.require_requested_model(
        provenance, "org/model", "a" * 40, "/checkpoint/selected"
    )
    with pytest.raises(ValueError, match="dir_checkpoint"):
        scores.require_requested_model(
            provenance, "org/model", "a" * 40, "/checkpoint/other"
        )


def test_confidence_sampling_accepts_equivalent_checkpoint_paths(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(checkpoint, target_is_directory=True)
    endpoint = {
        "task": "gsm8k", "model": "org/model", "revision": "a" * 40,
        "checkpoint": str(checkpoint), "identity": "b" * 64,
    }
    prepared = {
        "task": "gsm8k", "base_model_name": "org/model", "revision": "a" * 40,
        "dir_checkpoint": str(alias), "endpoint_identity": "b" * 64,
        "endpoint_manifest_sha256": "c" * 64,
    }

    validate_prepared_endpoint(prepared, endpoint, "c" * 64)


def test_checkpoint_loader_rejects_adapter_revision_mismatch(monkeypatch):
    monkeypatch.setattr(
        models,
        "PeftConfig",
        types.SimpleNamespace(
            from_pretrained=lambda _path: types.SimpleNamespace(
                base_model_name_or_path="org/base", revision="b" * 40
            )
        ),
    )
    with pytest.raises(ValueError, match="pins base revision"):
        models.load_model_and_tokenizer(
            "org/base",
            revision="a" * 40,
            dir_checkpoint="/checkpoint/adapter",
        )


def test_checkpoint_loader_rejects_adapter_base_model_mismatch(monkeypatch):
    monkeypatch.setattr(
        models,
        "PeftConfig",
        types.SimpleNamespace(
            from_pretrained=lambda _path: types.SimpleNamespace(
                base_model_name_or_path="org/other", revision="a" * 40
            )
        ),
    )
    with pytest.raises(ValueError, match="declares base model"):
        models.load_model_and_tokenizer(
            "org/base",
            revision="a" * 40,
            dir_checkpoint="/checkpoint/adapter",
        )


def test_lora_config_records_the_pinned_base_revision():
    revision = "a" * 40
    config = LoraConfig(task_type="CAUSAL_LM", target_modules=["q_proj"], revision=revision)
    assert config.to_dict()["revision"] == revision


def test_mixed_precision_config_uses_the_pinned_revision(monkeypatch):
    calls = []
    monkeypatch.setattr(
        models,
        "AutoConfig",
        types.SimpleNamespace(
            from_pretrained=lambda name, **kwargs: calls.append((name, kwargs))
            or types.SimpleNamespace(dtype=__import__("torch").bfloat16)
        ),
    )
    monkeypatch.setattr(models.torch.cuda, "is_bf16_supported", lambda: True)
    revision = "a" * 40
    assert models.get_mixed_precision("org/base", revision=revision) == (True, False)
    assert calls == [
        ("org/base", {"revision": revision, "trust_remote_code": True})
    ]


def test_explicit_compute_dtype_handles_missing_checkpoint_metadata(monkeypatch):
    monkeypatch.setattr(models.AutoConfig, "from_pretrained", lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(models.torch.cuda, "is_bf16_supported", lambda: True)
    with pytest.raises(ValueError, match="model_dtype"):
        models.get_mixed_precision("org/base")
    assert models.get_mixed_precision("org/base", model_dtype="bfloat16") == (True, False)
    assert models.requested_dtype("bfloat16") is models.torch.bfloat16
    with pytest.raises(ValueError, match="model_dtype"):
        models.requested_dtype("bf16")
