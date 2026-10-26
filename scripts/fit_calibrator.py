"""Fit one endpoint's Platt calibrator on predictor-fit development rows only."""
from __future__ import annotations

import json
import os

import numpy as np
import scipy

from pproute import consistency, predictor_protocol, scores
from pproute.provenance import identified, sha256_file


def endpoint_arrays(task_type, scores_dir, conf_file):
    if task_type == "mc":
        examples = scores.mc_examples(os.path.join(scores_dir, "eval_scores.jbl"))
        return (
            [example["id"] for example in examples],
            np.asarray([example["Z"] for example in examples], dtype=np.float64),
            np.asarray([example["lfz"] for example in examples], dtype=np.float64),
            os.path.join(scores_dir, "eval_scores.jbl"),
        )
    payload = consistency.read_conf_scores(conf_file, expected_splits=("eval",))
    rows = [row for row in payload["rows"] if row["split"] == "eval"]
    return (
        [row["id"] for row in rows],
        np.asarray([row["z"] for row in rows], dtype=np.float64),
        # The same anchor log-odds the predictor receives (pproute.predictor.build_endpoint_data).
        np.asarray([consistency.anchor_logit(consistency.bucket_estimates(row)[0]) for row in rows],
                   dtype=np.float64),
        conf_file,
    )


def main():
    from zarth_utils.config import Config
    from zarth_utils.logger import get_logger
    from zarth_utils.nn_utils import get_all_paths
    from zarth_utils.recorder import Recorder

    config = Config(default_config_dict={
        "exp_name": "", "task_name": "", "task_type": "mc", "scores_dir": "",
        "conf_file": "", "split_manifest": "", "endpoint_manifest": "",
        "use_git": True,
    }, use_argparse=True)
    config.show()
    for required in ("exp_name", "task_name", "scores_dir", "split_manifest",
                     "endpoint_manifest"):
        if not config[required]:
            raise SystemExit("--%s is required" % required)
    if config["task_type"] not in ("mc", "consistency"):
        raise SystemExit("--task_type must be mc or consistency")
    if config.task_type == "consistency" and not config.conf_file:
        raise SystemExit("--conf_file is required for consistency calibration")
    result_dir = os.path.join(os.getcwd(), config.exp_name)
    out_path = os.path.join(result_dir, "calibrator.json")

    provenance = scores.scored_model_provenance(
        config.scores_dir, require_git=config.use_git
    )
    if provenance["task_name"] != config.task_name:
        raise ValueError("score task %r != %r" %
                         (provenance["task_name"], config.task_name))
    endpoint = predictor_protocol.read_endpoint_manifest(
        config.endpoint_manifest, task=config.task_name, model=provenance["base_model_name"],
        revision=provenance["revision"],
    )
    scores.require_requested_model(
        provenance, endpoint["model"], endpoint["revision"], endpoint["checkpoint"]
    )

    ids, z, raw_logits, source = endpoint_arrays(
        config.task_type, config.scores_dir, config.conf_file
    )
    split = predictor_protocol.read_split_manifest(
        config.split_manifest, task=config.task_name, ids=ids
    )
    if config.task_type == "mc":
        source_fingerprint = scores.recorded_split_fingerprint(config.scores_dir, "eval")
    else:
        confidence = consistency.read_conf_scores(
            config.conf_file, expected_splits=("eval",),
        )
        meta = confidence["meta"]
        expected_meta = {
            "task": config.task_name, "model": provenance["base_model_name"],
            "revision": provenance["revision"], "endpoint": endpoint["endpoint"],
            "endpoint_identity": endpoint["identity"],
        }
        bad = {key: (meta.get(key), value) for key, value in expected_meta.items()
               if meta.get(key) != value}
        if os.path.realpath(meta.get("scores_dir", "")) != os.path.realpath(config.scores_dir):
            bad["scores_dir"] = (meta.get("scores_dir"), config.scores_dir)
        score_path = os.path.join(config.scores_dir, "eval_scores.jbl")
        score_hash = sha256_file(score_path)
        if meta.get("scores_sha256", {}).get("eval") != score_hash:
            bad["scores_sha256"] = (meta.get("scores_sha256", {}).get("eval"), score_hash)
        if bad:
            raise ValueError("development confidence evidence differs from endpoint: %s" % bad)
        source_fingerprint = meta["fingerprints"]["eval_split"]
    if source_fingerprint != split["source_fingerprint"]:
        raise ValueError("endpoint development fingerprint differs from predictor split")
    fit_indices, _ = predictor_protocol.partition_indices(ids, split)
    a, b, info = predictor_protocol.fit_platt(z[fit_indices], raw_logits[fit_indices])
    payload = identified({
        "schema": predictor_protocol.CALIBRATOR_SCHEMA,
        "task": config.task_name,
        "task_type": config.task_type,
        "model": provenance["base_model_name"],
        "revision": provenance["revision"],
        "endpoint": endpoint["endpoint"],
        "endpoint_identity": endpoint["identity"],
        "a": a,
        "b": b,
        **info,
        "split_manifest": os.path.abspath(config.split_manifest),
        "split_manifest_sha256": sha256_file(config.split_manifest),
        "fit_ids_sha256": split["fit_ids_sha256"],
        "source": os.path.abspath(source),
        "source_sha256": sha256_file(source),
        "source_fingerprint": source_fingerprint,
        "endpoint_manifest": os.path.abspath(config.endpoint_manifest),
        "endpoint_manifest_sha256": sha256_file(config.endpoint_manifest),
        "sampling_temperature": consistency.TEMPERATURE
        if config.task_type == "consistency" else None,
        "sampling_k": consistency.K if config.task_type == "consistency" else None,
        "test_labels_used": False,
        "software": {"numpy": np.__version__, "scipy": scipy.__version__},
    })
    predictor_protocol.validate_calibrator(
        payload, task=config.task_name, model=provenance["base_model_name"],
        endpoint=endpoint["endpoint"],
        split_manifest_sha256=payload["split_manifest_sha256"],
    )
    existing = None
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        predictor_protocol.validate_calibrator(
            existing, task=config.task_name, model=provenance["base_model_name"],
            endpoint=endpoint["endpoint"],
            split_manifest_sha256=payload["split_manifest_sha256"],
        )
        existing_contents = {key: value for key, value in existing.items()
                             if key not in ("identity", "code_revision", "split_manifest")}
        payload_contents = {key: value for key, value in payload.items()
                            if key not in ("identity", "code_revision", "split_manifest")}
        if existing_contents != payload_contents:
            raise SystemExit("existing calibrator differs from this endpoint fit")

    recorder_dir = os.path.join(result_dir, "recorder")
    all_paths = get_all_paths(recorder_dir)
    config.dump(all_paths["path_config"])
    recorder = Recorder(all_paths["path_record"], config=config, use_git=config.use_git)
    get_logger(all_paths["path_log"], force_add_handler=True)
    recorder.add_with_logging(key="calibrator", value={
        key: value for key, value in payload.items()
        if key not in ("fit_ids",)
    })
    recorder.end_recording()
    if existing is None:
        predictor_protocol.atomic_create_json(payload, out_path)
    else:
        print("COMPLETE %s" % out_path)
        return 0
    print("DONE %s  n=%d raw=%.6f calibrated=%.6f" %
          (out_path, info["n_fit"], info["fit_lz_raw"], info["fit_lz_calibrated"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
