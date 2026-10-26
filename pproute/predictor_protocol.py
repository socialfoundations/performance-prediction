"""Frozen predictor partitions and calibrated anchors used by the experiment pipeline.

These records are shared across independently scheduled jobs.  Keeping their validation here
prevents a driver, calibrator, and predictor from each inventing a slightly different join rule.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from pproute.provenance import (
    canonical_json,
    identified,
    require_pinned_revision,
    sha256_file,
    sha256_text,
    validate_identity,
)


SPLIT_SCHEMA = "pproute-predictor-split-v1"
CALIBRATOR_SCHEMA = "pproute-platt-calibrator-v1"
ENDPOINT_SCHEMA = "pproute-endpoint-v1"
CANDIDATE_SCHEMA = "pproute-predictor-candidate-v1"
SELECTION_SCHEMA = "pproute-predictor-selection-v1"
TEST_METRICS_SCHEMA = "pproute-predictor-test-v1"
VERIFIER_CHECKPOINT_SCHEMA = "pproute-verifier-v1"
FIT_FRACTION = 0.85
SPLIT_SEED = 0
LORA_ALPHA_FACTOR = 2
LORA_DROPOUT = 0.05
HEAD_WEIGHT_DECAY = 1e-4


def ids_sha256(ids):
    return sha256_text(canonical_json(list(ids)))


def _id_key(value):
    """A typed, deterministic key (so integer 1 never aliases string ``"1"``)."""
    return canonical_json(value)


def validate_endpoint_manifest(payload, *, task=None, model=None, revision=None,
                               endpoint=None, checkpoint=None):
    """Validate a canonical before/after endpoint and its self-authenticating identity."""
    validate_identity(payload, "endpoint")
    if payload.get("schema") != ENDPOINT_SCHEMA:
        raise ValueError("unknown endpoint schema %r" % payload.get("schema"))
    for field, expected in (("task", task), ("model", model), ("revision", revision),
                            ("endpoint", endpoint)):
        if expected is not None and payload.get(field) != expected:
            raise ValueError("endpoint %s %r != %r" % (field, payload.get(field), expected))
    name = payload.get("endpoint")
    if name not in ("before", "after"):
        raise ValueError("endpoint must be before or after")
    require_pinned_revision(payload.get("revision"), "endpoint model revision")
    recorded_checkpoint = str(payload.get("checkpoint") or "")
    if recorded_checkpoint and not os.path.isabs(recorded_checkpoint):
        raise ValueError("endpoint checkpoint must be absolute")
    if checkpoint is not None and os.path.normpath(recorded_checkpoint) != os.path.normpath(
            str(checkpoint or "")):
        raise ValueError("endpoint checkpoint %r != %r" % (recorded_checkpoint, checkpoint))
    if name == "before":
        if recorded_checkpoint or payload.get("selected_seed") is not None or \
                payload.get("selected_step") is not None or payload.get("adapter_sha256") is not None or \
                payload.get("adapter_config_sha256") is not None:
            raise ValueError("before endpoint carries finetuned-checkpoint fields")
    else:
        if not recorded_checkpoint or not isinstance(payload.get("selected_seed"), int) or \
                not isinstance(payload.get("selected_step"), int):
            raise ValueError("after endpoint lacks selected checkpoint/seed/step")
        adapter_hash = str(payload.get("adapter_sha256") or "")
        config_hash = str(payload.get("adapter_config_sha256") or "")
        if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
               for value in (adapter_hash, config_hash)):
            raise ValueError("after endpoint adapter/config hash is not sha256")
    return payload


def read_endpoint_manifest(path, **expected):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_endpoint_manifest(payload, **expected)


def make_split_manifest(task, ids, source_path, source_fingerprint, *, seed=SPLIT_SEED,
                        fit_fraction=FIT_FRACTION):
    """Construct the one task-level 85/15 split from ordered development IDs."""
    ids = list(ids)
    keys = [_id_key(value) for value in ids]
    if not ids:
        raise ValueError("predictor development pool is empty")
    if len(keys) != len(set(keys)):
        raise ValueError("predictor development IDs are not unique")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be in (0, 1)")
    order = np.random.default_rng(seed).permutation(len(ids))
    cut = int(np.floor(fit_fraction * len(ids)))
    if cut == 0 or cut == len(ids):
        raise ValueError("predictor split would leave an empty partition")
    fit_ids = [ids[int(index)] for index in order[:cut]]
    selection_ids = [ids[int(index)] for index in order[cut:]]
    source_path = os.path.abspath(source_path)
    payload = {
        "schema": SPLIT_SCHEMA,
        "task": str(task),
        "method": "numpy.random.default_rng(seed).permutation",
        "seed": int(seed),
        "fit_fraction": float(fit_fraction),
        "n_development": len(ids),
        "n_fit": len(fit_ids),
        "n_selection": len(selection_ids),
        "development_ids": ids,
        "fit_ids": fit_ids,
        "selection_ids": selection_ids,
        "development_ids_sha256": ids_sha256(ids),
        "fit_ids_sha256": ids_sha256(fit_ids),
        "selection_ids_sha256": ids_sha256(selection_ids),
        "source_path": source_path,
        "source_sha256": sha256_file(source_path),
        "source_fingerprint": source_fingerprint,
    }
    return identified(payload)


def validate_split_manifest(payload, *, task=None, ids=None):
    """Validate the manifest and optionally require an exact ordered development pool."""
    validate_identity(payload, "predictor split")
    if payload.get("schema") != SPLIT_SCHEMA:
        raise ValueError("unknown predictor split schema %r" % payload.get("schema"))
    if task is not None and payload.get("task") != task:
        raise ValueError("predictor split is for %r, not %r" % (payload.get("task"), task))
    if payload.get("method") != "numpy.random.default_rng(seed).permutation":
        raise ValueError("predictor split has an unknown construction method")
    if payload.get("seed") != SPLIT_SEED or payload.get("fit_fraction") != FIT_FRACTION:
        raise ValueError("predictor split does not use the frozen seed/fraction")
    development = list(payload.get("development_ids", []))
    fit = list(payload.get("fit_ids", []))
    selection = list(payload.get("selection_ids", []))
    if not development:
        raise ValueError("predictor manifest development pool is empty")
    dev_keys = [_id_key(value) for value in development]
    fit_keys = [_id_key(value) for value in fit]
    selection_keys = [_id_key(value) for value in selection]
    if len(dev_keys) != len(set(dev_keys)):
        raise ValueError("predictor manifest development IDs are not unique")
    if len(fit_keys) != len(set(fit_keys)) or len(selection_keys) != len(set(selection_keys)):
        raise ValueError("predictor manifest partition IDs are not unique")
    if set(fit_keys) & set(selection_keys) or set(fit_keys) | set(selection_keys) != set(dev_keys):
        raise ValueError("predictor fit/selection IDs are not a disjoint exhaustive partition")
    order = np.random.default_rng(SPLIT_SEED).permutation(len(development))
    cut = int(np.floor(FIT_FRACTION * len(development)))
    if cut == 0 or cut == len(development):
        raise ValueError("predictor split would leave an empty partition")
    expected_fit = [dev_keys[int(index)] for index in order[:cut]]
    expected_selection = [dev_keys[int(index)] for index in order[cut:]]
    if fit_keys != expected_fit or selection_keys != expected_selection:
        raise ValueError("predictor partitions do not match the frozen seed/fraction construction")
    expected_counts = (len(development), len(fit), len(selection))
    observed_counts = (
        payload.get("n_development"), payload.get("n_fit"), payload.get("n_selection")
    )
    if observed_counts != expected_counts:
        raise ValueError("predictor split counts %r != %r" % (observed_counts, expected_counts))
    for field, values in (
        ("development_ids_sha256", development),
        ("fit_ids_sha256", fit),
        ("selection_ids_sha256", selection),
    ):
        if payload.get(field) != ids_sha256(values):
            raise ValueError("predictor split %s is wrong" % field)
    source_path = payload.get("source_path")
    source_hash = str(payload.get("source_sha256") or "")
    if not isinstance(source_path, str) or not os.path.isabs(source_path) or \
            len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
        raise ValueError("predictor split has invalid source provenance")
    if not isinstance(payload.get("source_fingerprint"), str) or not payload[
            "source_fingerprint"]:
        raise ValueError("predictor split has no source fingerprint")
    if "code_revision" in payload:
        require_pinned_revision(payload["code_revision"], "predictor split code revision")
    if ids is not None and [_id_key(value) for value in ids] != dev_keys:
        raise ValueError("endpoint development IDs/order differ from the frozen predictor split")
    return payload


def read_split_manifest(path, *, task=None, ids=None):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_split_manifest(payload, task=task, ids=ids)


def development_source(task_type, source):
    """Ordered development IDs and fingerprint from one canonical endpoint artifact."""
    if task_type == "mc":
        from pproute import scores

        examples = scores.mc_examples(source)
        fingerprint = scores.recorded_split_fingerprint(os.path.dirname(source), "eval")
        return [example["id"] for example in examples], fingerprint
    if task_type != "consistency":
        raise ValueError("task_type must be mc or consistency")
    from pproute import consistency

    payload = consistency.read_conf_scores(source, expected_splits=("eval",))
    rows = [row for row in payload["rows"] if row["split"] == "eval"]
    return [row["id"] for row in rows], payload["meta"]["fingerprints"]["eval_split"]


def freeze_split_manifest(task, task_type, source, out, expected_n):
    """Create or exactly validate the task-wide predictor split manifest."""
    ids, fingerprint = development_source(task_type, source)
    if len(ids) != int(expected_n):
        raise ValueError("development source has %d rows, expected %d" %
                         (len(ids), expected_n))
    payload = make_split_manifest(task, ids, source, fingerprint)
    if os.path.exists(out):
        existing = read_split_manifest(out, task=task, ids=ids)
        existing_contents = {key: value for key, value in existing.items()
                             if key not in ("identity", "code_revision")}
        payload_contents = {key: value for key, value in payload.items()
                            if key not in ("identity", "code_revision")}
        if existing_contents != payload_contents:
            raise ValueError("existing predictor split differs from the frozen construction")
        return existing, False
    atomic_create_json(payload, out)
    return payload, True


def partition_indices(ids, manifest):
    """Return fit/selection indices in the manifest's frozen order."""
    validate_split_manifest(manifest, ids=ids)
    by_id = {_id_key(value): index for index, value in enumerate(ids)}
    return (
        np.asarray([by_id[_id_key(value)] for value in manifest["fit_ids"]], dtype=np.int64),
        np.asarray([by_id[_id_key(value)] for value in manifest["selection_ids"]],
                   dtype=np.int64),
    )


def atomic_create_json(payload, path, *, read_only=True):
    """Publish an immutable JSON record without replacing an existing decision."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".partial",
                                     dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        if read_only:
            os.chmod(path, 0o444)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fit_platt(z, raw_logits):
    """Unregularized two-parameter Platt fit with the optimizer frozen by the protocol."""
    from scipy.optimize import minimize
    from scipy.special import expit
    from pproute import scores

    z = np.asarray(z, dtype=np.float64)
    raw = np.asarray(raw_logits, dtype=np.float64)
    if z.ndim != 1 or raw.ndim != 1 or len(z) != len(raw) or not len(z):
        raise ValueError("calibration labels/logits must be non-empty aligned vectors")
    if not np.all(np.isin(z, (0.0, 1.0))) or len(np.unique(z)) != 2:
        raise ValueError("calibration fit requires binary labels with both classes")
    if not np.all(np.isfinite(raw)):
        raise ValueError("calibration logits must be finite")
    design = np.column_stack((np.ones(len(raw)), raw))

    def objective(weight):
        score = design @ weight
        return np.mean(np.logaddexp(0, score) - z * score)

    def gradient(weight):
        return design.T @ (expit(design @ weight) - z) / len(z)

    result = minimize(objective, [0.0, 1.0], jac=gradient, method="L-BFGS-B",
                      options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 1000, "maxls": 50})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("anchor calibrator did not converge: %s" % result.message)
    a, b = (float(value) for value in result.x)
    calibrated = expit(a + b * raw)
    return a, b, {
        "optimizer": "scipy-L-BFGS-B",
        "optimizer_options": {"ftol": 1e-14, "gtol": 1e-9, "maxiter": 1000, "maxls": 50},
        "n_fit": len(z),
        "fit_positive": int(z.sum()),
        "fit_lz_raw": scores.bce(z, expit(raw)),
        "fit_lz_calibrated": scores.bce(z, calibrated),
    }


def validate_calibrator(payload, *, task=None, model=None, endpoint=None,
                        split_manifest_sha256=None):
    validate_identity(payload, "calibrator")
    if payload.get("schema") != CALIBRATOR_SCHEMA:
        raise ValueError("unknown calibrator schema %r" % payload.get("schema"))
    for field, expected in (("task", task), ("model", model), ("endpoint", endpoint),
                            ("split_manifest_sha256", split_manifest_sha256)):
        if expected is not None and payload.get(field) != expected:
            raise ValueError("calibrator %s %r != %r" % (field, payload.get(field), expected))
    for field in ("a", "b"):
        if not np.isfinite(float(payload.get(field, np.nan))):
            raise ValueError("calibrator %s must be finite" % field)
    if int(payload.get("n_fit", 0)) <= 0:
        raise ValueError("calibrator has no fit rows")
    if not isinstance(payload.get("source_fingerprint"), str) or not payload[
            "source_fingerprint"]:
        raise ValueError("calibrator has no development source fingerprint")
    if payload.get("test_labels_used") is not False:
        raise ValueError("calibrator does not certify that test labels were unused")
    return payload


def read_calibrator(path, **expected):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_calibrator(payload, **expected)


def calibrated_logits(raw_logits, calibrator):
    validate_calibrator(calibrator)
    raw = np.asarray(raw_logits, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError("calibrator input logits must be finite")
    return (float(calibrator["a"]) + float(calibrator["b"]) * raw).astype(np.float32)


def anchor_metadata(calibrator_path):
    """Bind the development-fitted Platt anchor used by every released predictor."""
    if not calibrator_path:
        raise ValueError("Predictor requires a calibrator")
    return {"calibrator_sha256": sha256_file(calibrator_path)}


def endpoint_source_fingerprint(task_type, scores_dir, split, endpoint, conf_file=""):
    """Bind predictor evidence to the requested endpoint and its recorded score split."""
    from pproute import consistency, scores

    if split not in ("eval", "test"):
        raise ValueError("split must be eval or test")
    if task_type == "mc":
        return scores.recorded_split_fingerprint(scores_dir, split)
    if task_type != "consistency":
        raise ValueError("task_type must be mc or consistency")
    meta = consistency.read_conf_scores(conf_file, expected_splits=(split,))["meta"]
    expected = {key: endpoint[key] for key in ("task", "model", "revision", "endpoint")}
    expected["endpoint_identity"] = endpoint["identity"]
    bad = {key: (meta.get(key), value) for key, value in expected.items()
           if meta.get(key) != value}
    if os.path.realpath(meta.get("scores_dir", "")) != os.path.realpath(scores_dir):
        bad["scores_dir"] = (meta.get("scores_dir"), scores_dir)
    digest = sha256_file(os.path.join(scores_dir, split + "_scores.jbl"))
    if meta.get("scores_sha256", {}).get(split) != digest:
        bad["scores_sha256"] = (meta.get("scores_sha256", {}).get(split), digest)
    if bad:
        raise ValueError("%s confidence evidence differs from endpoint: %s" % (split, bad))
    return meta["fingerprints"][split + "_split"]


def training_protocol(*, lora_r, lora_alpha, lora_dropout, learning_rate, max_epochs,
                      effective_batch_size, micro_batch_size, max_len, head_weight_decay):
    """The final predictor protocol recorded by fits and checked during selection."""
    return {
        "input_style": "target_only_explicit_correctness",
        "anchor": "fixed_platt_fit_partition",
        "lora_r": int(lora_r),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "effective_batch_size": int(effective_batch_size),
        "micro_batch_size": int(micro_batch_size),
        "max_len": int(max_len),
        "head_weight_decay": float(head_weight_decay),
    }


def read_predictor_candidate(path, **expected):
    """Read one fitted predictor candidate and verify its immutable inputs."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("unknown predictor candidate schema %r" % payload.get("schema"))
    bad = {field: (payload.get(field), value) for field, value in expected.items()
           if value is not None and payload.get(field) != value}
    if bad:
        raise ValueError("predictor candidate identity mismatch: %s" % bad)
    require_pinned_revision(payload.get("revision"), "predictor candidate model revision")
    require_pinned_revision(
        payload.get("embedding_revision"), "predictor candidate embedding revision"
    )
    if not isinstance(payload.get("predictor_seed"), int):
        raise ValueError("predictor candidate seed is not an integer")
    if payload.get("test_artifacts_read") is not False:
        raise ValueError("predictor candidate does not certify development-only fitting")
    if not os.path.isabs(str(payload.get("calibrator") or "")):
        raise ValueError("predictor candidate calibrator is not an absolute path")
    anchor = anchor_metadata(payload.get("calibrator"))
    if payload.get("calibrator_sha256") != anchor["calibrator_sha256"]:
        raise ValueError("predictor candidate calibrator changed or is missing")
    for source, digest in (
        ("split_manifest", "split_manifest_sha256"),
        ("development_source", "development_source_sha256"),
        ("checkpoint", "checkpoint_sha256"),
    ):
        source_path = payload.get(source)
        expected_hash = str(payload.get(digest) or "")
        if not isinstance(source_path, str) or not os.path.isabs(source_path):
            raise ValueError("predictor candidate %s is not an absolute path" % source)
        if not os.path.isfile(source_path) or sha256_file(source_path) != expected_hash:
            raise ValueError("predictor candidate %s changed or is missing" % source)
    return payload


def validate_predictor_selection(payload, *, task=None):
    validate_identity(payload, "predictor selection")
    if payload.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unknown predictor selection schema %r" % payload.get("schema"))
    if task is not None and payload.get("task") != task:
        raise ValueError("predictor selection is for %r, not %r" %
                         (payload.get("task"), task))
    require_pinned_revision(payload.get("code_revision"), "predictor selection code revision")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("predictor selection has no rows")
    keys = [(row.get("model"), row.get("endpoint"), row.get("backbone_tag")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("predictor selection has duplicate model/endpoint/backbone rows")
    return payload


def read_predictor_selection(path, *, task=None):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_predictor_selection(payload, task=task)


def selected_predictor_row(path, *, task, model, endpoint, backbone_tag):
    payload = read_predictor_selection(path, task=task)
    matches = [row for row in payload["rows"]
               if (row.get("model"), row.get("endpoint"), row.get("backbone_tag")) ==
               (model, endpoint, backbone_tag)]
    if len(matches) != 1:
        raise ValueError("predictor selection has %d rows for %s/%s/%s" %
                         (len(matches), model, endpoint, backbone_tag))
    return matches[0]
