"""Train one predictor candidate and score its fixed selection partition.

The command cannot read a test artifact: it accepts only canonical development scores (and, for
consistency tasks, a development-only confidence file).  It saves restorable weights plus
full-precision selection probabilities. The caller determines whether to select or average seeds.
"""
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
from pproute.provenance import (atomic_json_dump, require_pinned_revision,
                                sha256_file)


def main():
    config = Config(
        default_config_dict={
            "exp_name": "",
            "task_name": "",
            "task_type": "mc",  # mc | consistency
            "scores_dir": "",
            "conf_file": "",
            "split_manifest": "",
            "calibrator": "",
            "endpoint_manifest": "",
            "emb_model": "",
            "emb_revision": "",
            "backbone_tag": "",
            "lora_r": 16,
            "epochs": 3,
            "lr": 1e-4,
            "bs": 16,
            "micro_bs": 0,
            "max_len": 384,
            "use_git": True,
            "random_seed": 0,
        },
        use_argparse=True,
    )
    config.show()
    for required in ("exp_name", "task_name", "scores_dir", "split_manifest",
                     "endpoint_manifest", "emb_model", "emb_revision", "backbone_tag"):
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
    require_pinned_revision(config["emb_revision"], "embedding revision")
    protocol = predictor_protocol.training_protocol(
        lora_r=config["lora_r"],
        lora_alpha=predictor_protocol.LORA_ALPHA_FACTOR * config["lora_r"],
        lora_dropout=predictor_protocol.LORA_DROPOUT,
        learning_rate=config["lr"],
        max_epochs=config["epochs"],
        effective_batch_size=config["bs"],
        micro_batch_size=config["micro_bs"] or config["bs"],
        max_len=config["max_len"],
        head_weight_decay=predictor_protocol.HEAD_WEIGHT_DECAY,
    )

    result_dir = os.path.join(os.getcwd(), config["exp_name"])
    validation_path = os.path.join(result_dir, "validation.json")
    checkpoint_path = os.path.join(result_dir, "checkpoint", "model.pt")
    if os.path.isfile(validation_path) and os.path.isfile(checkpoint_path):
        endpoint = predictor_protocol.read_endpoint_manifest(
            config["endpoint_manifest"], task=config["task_name"],
        )
        with open(validation_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        expected = {
            "schema": predictor_protocol.CANDIDATE_SCHEMA,
            "task": config["task_name"],
            "task_type": config["task_type"],
            "model": endpoint["model"],
            "revision": endpoint["revision"],
            "endpoint": endpoint["endpoint"],
            "endpoint_identity": endpoint["identity"],
            "backbone_tag": config["backbone_tag"],
            "embedding_model": config["emb_model"],
            "embedding_revision": config["emb_revision"],
            "predictor_seed": config["random_seed"],
            "split_manifest_sha256": sha256_file(
                config["split_manifest"]),
            **anchor_inputs,
            "development_source_sha256": sha256_file(
                os.path.join(config["scores_dir"], "eval_scores.jbl")
                if config["task_type"] == "mc" else config["conf_file"]
            ),
            "training_protocol": protocol,
        }
        bad = {key: (existing.get(key), value) for key, value in expected.items()
               if existing.get(key) != value}
        if bad:
            raise ValueError("existing predictor candidate identity mismatch: %s" % bad)
        if sha256_file(checkpoint_path) != existing.get("checkpoint_sha256"):
            raise ValueError("existing predictor checkpoint hash does not match validation.json")
        print("COMPLETE %s" % validation_path)
        return 0

    recorder_dir = os.path.join(result_dir, "recorder")
    all_paths = get_all_paths(recorder_dir)
    config.dump(all_paths["path_config"])
    recorder = Recorder(all_paths["path_record"], config=config, use_git=config["use_git"])
    get_logger(all_paths["path_log"], force_add_handler=True)
    set_random_seed(config["random_seed"])

    provenance = scores.scored_model_provenance(
        config["scores_dir"], require_git=config["use_git"]
    )
    if provenance["task_name"] != config["task_name"]:
        raise ValueError("score task %r != %r" % (provenance["task_name"], config["task_name"]))
    scored_tokenizer = predictor.load_scored_tokenizer(provenance)
    data = predictor.build_endpoint_data(
        config["task_type"], config["scores_dir"], "eval", scored_tokenizer,
        conf_path=config["conf_file"],
    )
    split = predictor_protocol.read_split_manifest(
        config["split_manifest"], task=config["task_name"], ids=data["ids"]
    )
    fit_indices, selection_indices = predictor_protocol.partition_indices(data["ids"], split)
    split_hash = sha256_file(config["split_manifest"])

    endpoint = predictor_protocol.read_endpoint_manifest(
        config["endpoint_manifest"], task=config["task_name"],
        model=provenance["base_model_name"], revision=provenance["revision"],
    )
    scores.require_requested_model(
        provenance, endpoint["model"], endpoint["revision"], endpoint["checkpoint"]
    )
    endpoint_name = endpoint.get("endpoint")
    development_source = (os.path.join(config["scores_dir"], "eval_scores.jbl")
                          if config["task_type"] == "mc" else config["conf_file"])
    source_fingerprint = predictor_protocol.endpoint_source_fingerprint(
        config["task_type"], config["scores_dir"], "eval", endpoint, config["conf_file"]
    )
    if source_fingerprint != split["source_fingerprint"]:
        raise ValueError("endpoint development fingerprint differs from predictor split")
    calibrator = predictor_protocol.read_calibrator(
        config["calibrator"], task=config["task_name"], model=provenance["base_model_name"],
        endpoint=endpoint_name, split_manifest_sha256=split_hash,
    )
    if calibrator.get("endpoint_identity") != endpoint.get("identity"):
        raise ValueError("calibrator and endpoint manifest identities differ")
    if calibrator.get("source_fingerprint") != split["source_fingerprint"]:
        raise ValueError("calibrator development fingerprint differs from predictor split")
    if os.path.realpath(calibrator.get("source", "")) != os.path.realpath(development_source) or \
            calibrator.get("source_sha256") != sha256_file(development_source):
        raise ValueError("calibrator source does not match the predictor development evidence")
    anchor_logits = predictor_protocol.calibrated_logits(data["raw_lfz"], calibrator)
    selection_logits = anchor_logits[selection_indices]
    selection_z = data["Z"][selection_indices]
    checkpoint_metadata = {
        "task": config["task_name"],
        "model": provenance["base_model_name"],
        "endpoint": endpoint_name,
        "endpoint_identity": endpoint["identity"],
        "backbone_tag": config["backbone_tag"],
        "predictor_seed": int(config["random_seed"]),
        "split_manifest_sha256": split_hash,
        **anchor_inputs,
        "development_source_sha256": sha256_file(development_source),
        "training_protocol": protocol,
    }
    probabilities, info = predictor.train_verifier(
        config["emb_model"], data["text"], data["Z"],
        [data["text"][index] for index in selection_indices], selection_z,
        emb_revision=config["emb_revision"],
        tr_lfz=anchor_logits, te_lfz=selection_logits,
        lora_r=config["lora_r"], epochs=config["epochs"], lr=config["lr"],
        bs=config["bs"], micro_bs=config["micro_bs"], max_len=config["max_len"],
        return_info=True, fit_indices=fit_indices, holdout_indices=selection_indices,
        checkpoint_path=checkpoint_path, checkpoint_metadata=checkpoint_metadata,
    )
    anchor_probabilities = expit(selection_logits)
    anchor_lz = scores.bce(selection_z, anchor_probabilities)
    if not np.isclose(info["holdout_init"], anchor_lz, rtol=0.0, atol=1e-6):
        raise ValueError("step-0 verifier loss %.12g != confidence anchor %.12g"
                         % (info["holdout_init"], anchor_lz))
    prediction_lz = scores.bce(selection_z, probabilities)
    row = {
        "schema": predictor_protocol.CANDIDATE_SCHEMA,
        "task": config["task_name"],
        "task_type": config["task_type"],
        "model": provenance["base_model_name"],
        "revision": provenance["revision"],
        "endpoint": endpoint_name,
        "endpoint_identity": endpoint["identity"],
        "backbone_tag": config["backbone_tag"],
        "embedding_model": config["emb_model"],
        "embedding_revision": config["emb_revision"],
        "predictor_seed": int(config["random_seed"]),
        "n_fit": len(fit_indices),
        "n_selection": len(selection_indices),
        "selection_ids": [data["ids"][index] for index in selection_indices],
        "selection_ids_sha256": split["selection_ids_sha256"],
        "split_manifest": os.path.abspath(config["split_manifest"]),
        "split_manifest_sha256": split_hash,
        "calibrator": os.path.abspath(config["calibrator"]),
        "calibrator_sha256": checkpoint_metadata["calibrator_sha256"],
        "development_source": os.path.abspath(development_source),
        "development_source_sha256": sha256_file(development_source),
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "LZ_anchor": anchor_lz,
        "LZ_pred": prediction_lz,
        "edge": anchor_lz - prediction_lz,
        "auc": float(roc_auc_score(selection_z, probabilities))
        if len(np.unique(selection_z)) > 1 else None,
        "selected_epoch": int(info["selected_epoch"]),
        "holdout_init": float(info["holdout_init"]),
        "holdout_best": float(info["holdout_best"]),
        "effective_batch_size": info["effective_batch_size"],
        "micro_batch_size": info["micro_batch_size"],
        "gradient_accumulation_steps": info["gradient_accumulation_steps"],
        "input_max_len": info["input_max_len"],
        "input_development_max_tokens": info["input_train_max_tokens"],
        "input_selection_max_tokens": info["input_test_max_tokens"],
        "fitted_anchor": info["anchor"],
        "training_protocol": protocol,
        "selection_g": [float(value) for value in probabilities],
        "selection_z": [int(value) for value in selection_z],
        "selection_fz_calibrated": [float(value) for value in anchor_probabilities],
        "test_artifacts_read": False,
    }
    if "consistency_estimator" in anchor_inputs:
        row["consistency_estimator"] = anchor_inputs["consistency_estimator"]
    recorder.add_with_logging(key="predictor_candidate", value={
        key: value for key, value in row.items()
        if key not in ("selection_ids", "selection_g", "selection_z",
                       "selection_fz_calibrated")
    })
    recorder.end_recording()
    atomic_json_dump(row, validation_path)
    print("DONE %s  LZ(anchor)=%.6f LZ(g)=%.6f epoch=%d" %
          (validation_path, anchor_lz, prediction_lz, info["selected_epoch"]))
    return 0


if __name__ == "__main__":
    main()
