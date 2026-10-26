"""Extract one endpoint's greedy prompts into a plain JSON file for the sampling arm.

Why this stage exists: an MMLU-Pro score file carries live ``lm_eval`` task references and is not a
portable sampling request. Extracting it once into plain JSON freezes the exact prompts and keeps
the vLLM stage from unpickling or re-rendering benchmark objects. GSM8K follows the same boundary
so both generative tasks use one confidence artifact contract.

The prompts are copied byte-identically -- this stage renders nothing.

    python -m scripts.conf_prepare --exp_name <dir> --scores_dir <endpoint scores> \\
        --task_name mmlu_pro [--no-use_git]
"""
from __future__ import annotations

import json
import os
import sys

from pproute import consistency, mmlu_pro, predictor_protocol, trace_sft
from pproute.provenance import atomic_json_dump, sha256_file, sha256_text
SPLIT_FILES = {"eval": "eval_scores.jbl", "test": "test_scores.jbl"}
PROMPTS_NAME = "conf_prompts.json"


def main():
    from zarth_utils.config import Config
    from zarth_utils.logger import get_logger
    from zarth_utils.nn_utils import get_all_paths, set_random_seed
    from zarth_utils.recorder import Recorder
    from pproute import scores as scores_mod

    config = Config(
        default_config_dict={
            "exp_name": "", "scores_dir": "", "task_name": mmlu_pro.TASK_NAME,
            "split": "",  # eval | test; prepared independently across the test gate
            "endpoint_manifest": "",
            "expected_n": -1,
            "expected_fingerprint_sha256": "",
            "use_git": True, "random_seed": 0,
        },
        use_argparse=True,
    )
    config.show()
    for required in ("exp_name", "scores_dir", "split", "endpoint_manifest",
                     "expected_fingerprint_sha256"):
        if not config[required]:
            raise SystemExit("--%s is required" % required)
    if config["split"] not in ("eval", "test"):
        raise SystemExit("--split must be eval or test")
    if config["expected_n"] <= 0:
        raise SystemExit("--expected_n must be positive")
    if len(config["expected_fingerprint_sha256"]) != 64:
        raise SystemExit("--expected_fingerprint_sha256 must be a SHA-256 digest")
    try:
        adapter = trace_sft.adapter_for(config["task_name"])
    except ValueError as error:
        raise SystemExit(str(error)) from error

    out_path = os.path.join(config["exp_name"], PROMPTS_NAME)
    existing = os.path.exists(out_path)
    os.makedirs(config["exp_name"], exist_ok=True)
    dir_results = os.path.join(os.getcwd(), config["exp_name"])
    all_paths = get_all_paths(dir_results)
    config.dump(all_paths["path_config"])
    recorder = Recorder(all_paths["path_record"], config=config, use_git=config["use_git"])
    get_logger(all_paths["path_log"], force_add_handler=True)
    set_random_seed(config["random_seed"])

    # The endpoint identity the sampler must agree with; also proves the scores are consumable.
    provenance = scores_mod.scored_model_provenance(
        config["scores_dir"], require_git=config["use_git"])
    endpoint = predictor_protocol.read_endpoint_manifest(
        config["endpoint_manifest"], task=config["task_name"],
        model=provenance["base_model_name"], revision=provenance["revision"],
    )
    scores_mod.require_requested_model(
        provenance, endpoint["model"], endpoint["revision"], endpoint["checkpoint"]
    )

    splits, stops = {}, None
    selected_splits = (config["split"],)
    fingerprints = {}
    for split in selected_splits:
        name = SPLIT_FILES[split]
        fingerprint = scores_mod.recorded_split_fingerprint(config["scores_dir"], split)
        fingerprint_hash = sha256_text(fingerprint)
        if fingerprint_hash != config["expected_fingerprint_sha256"]:
            raise ValueError("%s fingerprint hash %s != frozen %s" %
                             (split, fingerprint_hash,
                              config["expected_fingerprint_sha256"]))
        fingerprints[split + "_split"] = fingerprint
        docs, until = consistency.extract_prompts(
            os.path.join(config["scores_dir"], name), adapter)
        if stops is not None and until != stops:
            raise SystemExit(f"stop strings differ between splits: {stops} vs {until}")
        if len(docs) != config["expected_n"]:
            raise ValueError("%s prompt source has %d rows, expected %d" %
                             (split, len(docs), config["expected_n"]))
        stops, splits[split] = until, docs

    payload = {
        "meta": {
            "schema": consistency.prompts_schema_for(config["task_name"]),
            "task": config["task_name"],
            "scores_dir": os.path.abspath(config["scores_dir"]),
            "stop": stops,
            "splits": list(selected_splits),
            "n_eval": len(splits.get("eval", [])), "n_test": len(splits.get("test", [])),
            "base_model_name": provenance["base_model_name"],
            "revision": provenance["revision"],
            "dir_checkpoint": provenance["dir_checkpoint"],
            "endpoint_identity": endpoint["identity"],
            "endpoint_manifest_sha256": sha256_file(
                config["endpoint_manifest"]),
            "scores_sha256": {
                split: sha256_file(
                    os.path.join(config["scores_dir"], SPLIT_FILES[split]))
                for split in selected_splits
            },
            "fingerprints": fingerprints,
            "fingerprint_sha256": {
                split: sha256_text(fingerprints[split + "_split"])
                for split in selected_splits
            },
        },
        "splits": splits,
    }
    if existing:
        with open(out_path, encoding="utf-8") as handle:
            recorded = json.load(handle)
        if recorded != payload:
            raise ValueError("existing prepared prompts differ from current source/protocol")
    else:
        atomic_json_dump(payload, out_path)
    recorder.add_with_logging(key="conf_prompts", value=payload["meta"])
    recorder.end_recording()
    if existing:
        print(f"COMPLETE {out_path}: validated", flush=True)
        return 0
    print(f"DONE {out_path}  eval={len(splits.get('eval', []))} "
          f"test={len(splits.get('test', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
