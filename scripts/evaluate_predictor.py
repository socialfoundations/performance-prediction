"""Evaluate the development-selected predictor on held-out test data."""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from zarth_utils.config import Config
from zarth_utils.logger import get_logger
from zarth_utils.nn_utils import get_all_paths, set_random_seed
from zarth_utils.recorder import Recorder

from pproute import consistency, predictor, predictor_protocol, scores
from pproute.provenance import atomic_json_dump, sha256_file, sha256_text


def main():
    config = Config(
        default_config_dict={
            "exp_name": "",
            "task_name": "",
            "task_type": "mc",
            "scores_dir": "",
            "conf_file": "",
            "split_manifest": "",
            "calibrator": "",
            "endpoint_manifest": "",
            "predictor_selection": "",
            "emb_model": "",
            "emb_revision": "",
            "backbone_tag": "",
            "micro_bs": 0,
            "max_len": 384,
            "expected_n": -1,
            "expected_fingerprint_sha256": "",
            "use_git": True,
            "random_seed": 0,
        },
        use_argparse=True,
    )
    config.show()
    for required in ("exp_name", "task_name", "scores_dir", "split_manifest",
                     "endpoint_manifest", "emb_model", "emb_revision",
                     "backbone_tag"):
        if not config[required]:
            raise SystemExit("--%s is required" % required)
    if config["task_type"] not in ("mc", "consistency"):
        raise SystemExit("--task_type must be mc or consistency")
    if config["task_type"] == "consistency" and not config["conf_file"]:
        raise SystemExit("--conf_file is required for consistency predictors")

    anchor_inputs = predictor_protocol.anchor_metadata(
        config["calibrator"]
    )
    if config["task_type"] == "consistency":
        # Bind the shared confidence estimator used by this predictor.
        anchor_inputs["consistency_estimator"] = consistency.ESTIMATOR
    if not config["predictor_selection"]:
        raise ValueError("evaluation requires predictor_selection")
    selection_hash = sha256_file(config["predictor_selection"])

    provenance = scores.scored_model_provenance(
        config["scores_dir"], require_git=config["use_git"]
    )
    if provenance["task_name"] != config["task_name"]:
        raise ValueError("score task %r != %r" % (provenance["task_name"], config["task_name"]))
    endpoint_manifest = predictor_protocol.read_endpoint_manifest(
        config["endpoint_manifest"], task=config["task_name"],
        model=provenance["base_model_name"], revision=provenance["revision"],
    )
    scores.require_requested_model(
        provenance, endpoint_manifest["model"], endpoint_manifest["revision"],
        endpoint_manifest["checkpoint"],
    )
    endpoint = endpoint_manifest["endpoint"]
    test_score_path = os.path.join(config["scores_dir"], "test_scores.jbl")
    test_source = test_score_path if config["task_type"] == "mc" else config["conf_file"]
    test_fingerprint = predictor_protocol.endpoint_source_fingerprint(
        config["task_type"], config["scores_dir"], "test", endpoint_manifest, config["conf_file"]
    )
    test_fingerprint_hash = sha256_text(test_fingerprint)
    if config["expected_fingerprint_sha256"] and \
            test_fingerprint_hash != config["expected_fingerprint_sha256"]:
        raise ValueError("test fingerprint hash %s != frozen %s" %
                         (test_fingerprint_hash, config["expected_fingerprint_sha256"]))

    selected = predictor_protocol.selected_predictor_row(
        config["predictor_selection"], task=config["task_name"],
        model=provenance["base_model_name"], endpoint=endpoint,
        backbone_tag=config["backbone_tag"],
    )
    selected_expected = {
        "revision": provenance["revision"],
        "endpoint_identity": endpoint_manifest["identity"],
        "embedding_model": config["emb_model"],
        "embedding_revision": config["emb_revision"],
    }
    bad = {key: (selected.get(key), value) for key, value in selected_expected.items()
           if selected.get(key) != value}
    if bad:
        raise ValueError("selected predictor identity differs: %s" % bad)
    candidate_path = os.path.abspath(selected["validation"])
    candidate = predictor_protocol.read_predictor_candidate(
        candidate_path,
        task=config["task_name"], task_type=config["task_type"],
        model=provenance["base_model_name"], revision=provenance["revision"],
        endpoint=endpoint, endpoint_identity=endpoint_manifest["identity"],
        backbone_tag=config["backbone_tag"], embedding_model=config["emb_model"],
        embedding_revision=config["emb_revision"],
        split_manifest_sha256=sha256_file(config["split_manifest"]),
        calibrator_sha256=anchor_inputs["calibrator_sha256"],
        consistency_estimator=anchor_inputs.get("consistency_estimator"),
    )
    candidate = dict(candidate, validation=candidate_path,
                     validation_sha256=sha256_file(candidate_path))
    for field in ("validation", "checkpoint"):
        if candidate[field + "_sha256"] != selected[field + "_sha256"]:
            raise ValueError("selected predictor %s hash changed" % field)
    if int(config["random_seed"]) != int(candidate["predictor_seed"]):
        raise ValueError("--random_seed %d != predictor seed %d" %
                         (config["random_seed"], candidate["predictor_seed"]))

    result_dir = os.path.join(os.getcwd(), config["exp_name"])
    metrics_path = os.path.join(result_dir, "metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        expected = {
            "schema": predictor_protocol.TEST_METRICS_SCHEMA,
            "task": config["task_name"],
            "task_type": config["task_type"],
            "model": endpoint_manifest["model"],
            "revision": endpoint_manifest["revision"],
            "endpoint": endpoint,
            "endpoint_identity": endpoint_manifest["identity"],
            "backbone_tag": config["backbone_tag"],
            "embedding_model": config["emb_model"],
            "embedding_revision": config["emb_revision"],
            "split_manifest_sha256": sha256_file(
                config["split_manifest"]),
            **anchor_inputs,
            "predictor_selection_sha256": selection_hash,
            "test_source_sha256": sha256_file(test_source),
            "test_fingerprint_sha256": test_fingerprint_hash,
            "candidate_validation_sha256": candidate["validation_sha256"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
        }
        bad = {key: (existing.get(key), value) for key, value in expected.items()
               if existing.get(key) != value}
        if config["expected_n"] != -1 and existing.get("n_test") != config["expected_n"]:
            bad["n_test"] = (existing.get("n_test"), config["expected_n"])
        if bad:
            raise ValueError("existing predictor test identity mismatch: %s" % bad)
        print("COMPLETE %s" % metrics_path)
        return 0
    all_paths = get_all_paths(os.path.join(result_dir, "recorder"))
    config.dump(all_paths["path_config"])
    recorder = Recorder(all_paths["path_record"], config=config, use_git=config["use_git"])
    get_logger(all_paths["path_log"], force_add_handler=True)
    set_random_seed(config["random_seed"])

    scored_tokenizer = predictor.load_scored_tokenizer(provenance)
    data = predictor.build_endpoint_data(
        config["task_type"], config["scores_dir"], "test", scored_tokenizer,
        conf_path=config["conf_file"],
    )
    if config["expected_n"] != -1 and len(data["ids"]) != config["expected_n"]:
        raise ValueError("predictor test has %d rows, expected %d" %
                         (len(data["ids"]), config["expected_n"]))
    split_hash = sha256_file(config["split_manifest"])
    split = predictor_protocol.read_split_manifest(
        config["split_manifest"], task=config["task_name"]
    )
    calibrator = predictor_protocol.read_calibrator(
        config["calibrator"], task=config["task_name"], model=provenance["base_model_name"],
        endpoint=endpoint, split_manifest_sha256=split_hash,
    )
    if calibrator.get("endpoint_identity") != endpoint_manifest.get("identity"):
        raise ValueError("calibrator and endpoint manifest identities differ")
    if calibrator.get("source_fingerprint") != split["source_fingerprint"]:
        raise ValueError("calibrator development fingerprint differs from predictor split")
    anchor_logits = predictor_protocol.calibrated_logits(data["raw_lfz"], calibrator)
    development_hash = calibrator["source_sha256"]
    probabilities, info = predictor.predict_verifier_checkpoint(
        config["emb_model"], data["text"], anchor_logits, candidate["checkpoint"],
        emb_revision=config["emb_revision"], micro_bs=config["micro_bs"],
        max_len=config["max_len"],
    )
    metadata = info["checkpoint_metadata"]
    expected_metadata = {
        "task": config["task_name"],
        "model": provenance["base_model_name"],
        "endpoint": endpoint,
        "endpoint_identity": endpoint_manifest["identity"],
        "backbone_tag": config["backbone_tag"],
        "predictor_seed": int(candidate["predictor_seed"]),
        "split_manifest_sha256": split_hash,
        **anchor_inputs,
        "development_source_sha256": development_hash,
    }
    bad = {key: (metadata.get(key), value) for key, value in expected_metadata.items()
           if metadata.get(key) != value}
    if bad:
        raise ValueError("restored predictor metadata mismatch: %s" % bad)

    anchor_probabilities = expit(anchor_logits)
    lz_raw = scores.bce(data["Z"], expit(data["raw_lfz"]))
    lz_anchor = scores.bce(data["Z"], anchor_probabilities)
    lz_pred = scores.bce(data["Z"], probabilities)
    row = {
        "schema": predictor_protocol.TEST_METRICS_SCHEMA,
        "task": config["task_name"],
        "task_type": config["task_type"],
        "model": provenance["base_model_name"],
        "revision": provenance["revision"],
        "endpoint": endpoint,
        "endpoint_identity": endpoint_manifest["identity"],
        "backbone_tag": config["backbone_tag"],
        "embedding_model": config["emb_model"],
        "embedding_revision": config["emb_revision"],
        "predictor_seed": int(candidate["predictor_seed"]),
        "n_test": len(data["ids"]),
        "acc_test": float(np.mean(data["Z"])),
        "L_Y": float(np.mean(data["L"])),
        "LZ_f_raw": lz_raw,
        "LZ_f_calibrated": lz_anchor,
        "LZ_g": lz_pred,
        "edge": lz_anchor - lz_pred,
        "auc_f_raw": float(roc_auc_score(data["Z"], expit(data["raw_lfz"])))
        if len(np.unique(data["Z"])) > 1 else None,
        "auc_f_calibrated": float(roc_auc_score(data["Z"], anchor_probabilities))
        if len(np.unique(data["Z"])) > 1 else None,
        "auc_g": float(roc_auc_score(data["Z"], probabilities))
        if len(np.unique(data["Z"])) > 1 else None,
        "test_ids": data["ids"],
        "test_ids_sha256": predictor_protocol.ids_sha256(data["ids"]),
        "test_z": [int(value) for value in data["Z"]],
        "test_fz_raw": [float(value) for value in expit(data["raw_lfz"])],
        "test_fz_calibrated": [float(value) for value in anchor_probabilities],
        "test_g": [float(value) for value in probabilities],
        "test_y_loss": [float(value) for value in data["L"]],
        "test_source": os.path.abspath(test_source),
        "test_source_sha256": sha256_file(test_source),
        "test_fingerprint": test_fingerprint,
        "test_fingerprint_sha256": test_fingerprint_hash,
        "split_manifest": os.path.abspath(config["split_manifest"]),
        "split_manifest_sha256": split_hash,
        "fit_ids_sha256": split["fit_ids_sha256"],
        "selection_ids_sha256": split["selection_ids_sha256"],
        "calibrator": os.path.abspath(config["calibrator"]),
        "calibrator_sha256": expected_metadata["calibrator_sha256"],
        "predictor_selection": os.path.abspath(config["predictor_selection"]),
        "predictor_selection_sha256": selection_hash,
        "candidate_validation": candidate["validation"],
        "candidate_validation_sha256": candidate["validation_sha256"],
        "checkpoint": candidate["checkpoint"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "input_max_len": info["input_max_len"],
        "input_test_max_tokens": info["input_test_max_tokens"],
        "input_test_over_max_len": info["input_test_over_max_len"],
    }
    if "consistency_estimator" in anchor_inputs:
        row["consistency_estimator"] = anchor_inputs["consistency_estimator"]
    recorder.add_with_logging(key="predictor_test", value={
        key: value for key, value in row.items()
        if key not in ("test_ids", "test_z", "test_fz_raw", "test_fz_calibrated", "test_g",
                       "test_y_loss")
    })
    recorder.end_recording()
    atomic_json_dump(row, metrics_path)
    print("DONE %s  LZ(anchor)=%.6f LZ(g)=%.6f edge=%.6f" %
          (metrics_path, lz_anchor, lz_pred, lz_anchor - lz_pred))
    return 0


if __name__ == "__main__":
    main()
