"""Sampling-consistency confidence for one endpoint of a generative benchmark.

Runs through vLLM in the integrated ``pp-route`` environment. For every requested document it replays the exact
prompt stored in the greedy score files -- byte-identical, no rendering of any kind -- draws
k = 128 samples at T = 0.9 / top_p = 1.0 by default under per-request seeds, extracts each answer
with the one frozen extractor, and reduces them with the pre-registered rules in
pproute.consistency. Output is a create-once conf_scores.json holding, per document: the answer
distribution, the majority answer and its share (= f_Z), Z against gold, cap-hit and failure
counts, and the first majority-supporting trace for the predictor.

    python -m scripts.conf_sample --exp_name <out dir> --prompts <conf_prompts.json> \\
        --endpoint_manifest <endpoint.json> [--model_path <merged dir>] \\
        [--max_model_len 6144] [--no-use_git]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version

# Pin the native torch top-k/top-p sampler instead of accepting vLLM's backend-dependent default.
# It is slower than the fused implementation but keeps sampling behavior stable across workers.
# This must be set before vllm is imported anywhere.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# vLLM 0.27.1 pins a FlashInfer comm module whose evaluated type annotations require Python 3.12.
# Confidence sampling is single-GPU and does not use that communication module, so make vLLM take
# its existing ImportError fallback while leaving FlashInfer's generation kernels available.
try:
    _flashinfer_version = version("flashinfer-python")
except PackageNotFoundError:
    _flashinfer_version = None
if sys.version_info < (3, 12) and _flashinfer_version == "0.6.16.post3":
    sys.modules["flashinfer.comm"] = None

from pproute import consistency, mmlu_pro, predictor_protocol, scores, trace_sft
from pproute.provenance import sha256_file


def resolve_snapshot(model_name, revision):
    """The pinned local HF snapshot dir; offline by construction, exactly like the teacher."""
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    path = os.path.join(hub, "models--" + model_name.replace("/", "--"), "snapshots", revision)
    if not os.path.isdir(path):
        raise SystemExit(f"pinned snapshot missing: {path}")
    return path


def validate_prepared_endpoint(prepared_meta, endpoint_record, endpoint_manifest_hash):
    """Require prepared prompts to name the endpoint that sampling will replay."""
    scores.require_requested_model(
        prepared_meta,
        endpoint_record["model"],
        endpoint_record["revision"],
        endpoint_record["checkpoint"],
    )
    expected = {
        "task": endpoint_record["task"],
        "endpoint_identity": endpoint_record["identity"],
        "endpoint_manifest_sha256": endpoint_manifest_hash,
    }
    bad = {key: (prepared_meta.get(key), value) for key, value in expected.items()
           if prepared_meta.get(key) != value}
    if bad:
        raise ValueError("prepared prompts differ from endpoint: %s" % bad)


def main():
    from zarth_utils.config import Config
    from zarth_utils.logger import get_logger
    from zarth_utils.nn_utils import get_all_paths, set_random_seed
    from zarth_utils.recorder import Recorder

    config = Config(
        default_config_dict={
            "exp_name": "", "endpoint_manifest": "", "prompts": "",
            # Empty for before; required merged endpoint for after.
            "model_path": "",
            "max_model_len": 6144,
            "num_shards": 1,
            "shard_index": 0,
            "use_git": True,
            # Sampling is seeded per request from (endpoint, doc id) in pproute.consistency, so
            # this seeds only the engine; recorded for completeness rather than for control.
            "random_seed": 0,
        },
        use_argparse=True,
    )
    config.show()
    for required in ("exp_name", "endpoint_manifest", "prompts"):
        if not config[required]:
            raise SystemExit("--%s is required" % required)
    if config["num_shards"] <= 0:
        raise SystemExit("--num_shards must be positive")
    if not 0 <= config["shard_index"] < config["num_shards"]:
        raise SystemExit("--shard_index must be in [0, num_shards)")

    endpoint_record = predictor_protocol.read_endpoint_manifest(config["endpoint_manifest"])
    task_name = endpoint_record["task"]
    model_name = endpoint_record["model"]
    revision = endpoint_record["revision"]
    endpoint = endpoint_record["endpoint"]
    endpoint_identity = endpoint_record["identity"]
    endpoint_manifest_hash = sha256_file(config["endpoint_manifest"])
    try:
        adapter = trace_sft.adapter_for(task_name)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    with open(config["prompts"], encoding="utf-8") as handle:
        prepared = json.load(handle)
    prepared_meta = prepared.get("meta", {})
    if not consistency.supports_prompts_schema(task_name, prepared_meta.get("schema")):
        raise ValueError("prepared prompts do not implement the current task protocol")
    selected_splits = tuple(prepared_meta.get("splits", ()))
    if len(selected_splits) != 1 or selected_splits[0] not in ("eval", "test") or \
            set(prepared.get("splits", {})) != set(selected_splits):
        raise ValueError("prepared prompts must contain exactly one eval/test split")
    split = selected_splits[0]
    if task_name == mmlu_pro.TASK_NAME:
        for doc in prepared["splits"][split]:
            mmlu_pro.option_contents(doc.get("options"))
    scores_dir = prepared_meta.get("scores_dir", "")
    validate_prepared_endpoint(prepared_meta, endpoint_record, endpoint_manifest_hash)
    score_path = os.path.join(
        scores_dir, "eval_scores.jbl" if split == "eval" else "test_scores.jbl"
    )
    if prepared_meta.get("scores_sha256", {}).get(split) != sha256_file(score_path):
        raise ValueError("prepared prompts no longer match the score file")
    fingerprint = scores.recorded_split_fingerprint(scores_dir, split)
    if prepared_meta.get("fingerprints", {}).get(split + "_split") != fingerprint:
        raise ValueError("prepared prompts carry the wrong split fingerprint")
    fingerprints = {split + "_split": fingerprint}

    out_path = os.path.join(config["exp_name"], "conf_scores.json")
    os.makedirs(config["exp_name"], exist_ok=True)

    if endpoint == "after":
        if not config["model_path"]:
            raise SystemExit("--model_path is required for an after endpoint")
        provenance_path = os.path.join(config["model_path"], "merge_provenance.json")
        if not os.path.exists(provenance_path):
            raise SystemExit(f"merged model is incomplete (no {provenance_path})")
        with open(provenance_path, encoding="utf-8") as handle:
            merge_provenance = json.load(handle)
        if merge_provenance.get("endpoint_identity") != endpoint_identity or \
                merge_provenance.get("endpoint_manifest_sha256") != endpoint_manifest_hash:
            raise SystemExit("merged model provenance does not match the endpoint manifest")
        model_path = config["model_path"]
    else:
        if config["model_path"]:
            raise SystemExit("--model_path is not valid for a before endpoint")
        model_path = resolve_snapshot(model_name, revision)

    score_paths = {split: score_path}
    if os.path.exists(out_path):
        payload = consistency.read_conf_scores(
            out_path, expected_splits=selected_splits,
        )
        meta = payload["meta"]
        expected = {
            "task": task_name, "model": model_name,
            "revision": revision, "endpoint": endpoint,
            "endpoint_identity": endpoint_identity,
            "endpoint_manifest_sha256": endpoint_manifest_hash,
            "scores_sha256": {split: sha256_file(path)
                              for split, path in score_paths.items()},
            "fingerprints": fingerprints,
            "prompts_sha256": sha256_file(config["prompts"]),
            "splits": list(selected_splits),
            "shard": {"method": "position-modulo", "num_shards": config["num_shards"],
                      "shard_index": config["shard_index"]},
        }
        bad = {key: (meta.get(key), value) for key, value in expected.items()
               if meta.get(key) != value}
        if os.path.realpath(meta.get("scores_dir", "")) != os.path.realpath(scores_dir):
            bad["scores_dir"] = (meta.get("scores_dir"), scores_dir)
        if bad:
            raise ValueError("existing confidence artifact differs from invocation: %s" % bad)
        print(f"COMPLETE {out_path}: validated", flush=True)
        return 0

    # The run record lands beside conf_scores.json so the artifact directory is self-describing:
    # config dump, log, metrics, and -- the reason it matters most here -- a conda environment
    # snapshot, including the integrated environment used by every other experiment stage.
    dir_results = os.path.join(os.getcwd(), config["exp_name"])
    all_paths = get_all_paths(dir_results)
    config.dump(all_paths["path_config"])
    recorder = Recorder(all_paths["path_record"], config=config, use_git=config["use_git"])
    get_logger(all_paths["path_log"], force_add_handler=True)
    # no_torch=True is REQUIRED here, not a micro-optimisation. set_random_seed would call
    # torch.cuda.is_available(), which on a GPU node touches the CUDA driver in THIS process; vLLM
    # then forks its engine core and the child dies with "Cannot re-initialize CUDA in forked
    # subprocess". On 19-20 Aug that killed 28 of 32 MMLU-Pro conf units. Nothing in this parent
    # process does torch compute -- all generation happens inside the engine core, which vLLM seeds
    # itself from the explicit seed=0 passed to LLM() below -- so skipping the torch seeding here
    # costs no reproducibility. random/numpy are still seeded.
    set_random_seed(config["random_seed"], no_torch=True)

    splits = {"eval": [], "test": []}
    splits[split] = consistency.shard_documents(
        prepared["splits"][split], config["num_shards"], config["shard_index"]
    )
    stops = prepared_meta["stop"]
    if not any(splits.values()):
        raise SystemExit("sampling subset is empty")
    print(f"replaying {len(splits['eval'])} eval + {len(splits['test'])} test prompts "
          f"(shard {config['shard_index'] + 1}/{config['num_shards']}) "
          f"with k={consistency.K} T={consistency.TEMPERATURE} "
          f"top_p={consistency.TOP_P}",
          flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=config["max_model_len"],
              gpu_memory_utilization=0.90, seed=0, enable_prefix_caching=True,
              # Use Qwen3.5's in-tree Triton/FLA gated-delta-rule prefill path explicitly.
              # Ignored by every non-GDN architecture.
              additional_config={"gdn_prefill_backend": "triton"})

    rows = []
    for split, docs in splits.items():
        if not docs:
            continue
        params = [SamplingParams(
            n=consistency.K, temperature=consistency.TEMPERATURE, top_p=consistency.TOP_P,
            max_tokens=consistency.MAX_NEW_TOKENS, stop=list(stops),
            seed=consistency.request_seed(endpoint_identity, split, doc["id"]),
        ) for doc in docs]
        outputs = llm.generate([doc["prompt"] for doc in docs], params)
        if len(outputs) != len(docs):
            raise SystemExit(f"{split}: vLLM returned {len(outputs)} outputs for {len(docs)}")
        for doc, output in zip(docs, outputs):
            texts = [completion.text for completion in output.outputs]
            if len(texts) != consistency.K:
                raise SystemExit(f"{split} doc {doc['id']}: {len(texts)} samples != k")
            extracted = [adapter.extract_answer(text) for text in texts]
            answers = [adapter.normalize(a) if a is not None else None
                       for a in extracted]
            majority, share, counts = consistency.aggregate(answers, task=task_name)
            categorized = [answer if answer is not None else consistency.FAILURE
                           for answer in answers]
            first_seen = {}
            for position, answer in enumerate(categorized):
                first_seen.setdefault(answer, position)
            z = int(majority not in (None, consistency.FAILURE)
                    and majority == doc["gold"])
            trace = next((text for text, answer in zip(texts, answers)
                          if answer is not None and answer == majority), None)
            finish_reasons = dict(Counter(
                str(completion.finish_reason or "<none>") for completion in output.outputs
            ))
            row = {
                "split": split, "id": doc["id"], "gold": doc["gold"],
                "predictor_text": doc["predictor_text"],
                "majority": majority, "share": share, "z": z, "counts": counts,
                "first_seen": first_seen,
                "n_fail": counts.get(consistency.FAILURE, 0),
                "cap_hits": sum(1 for completion in output.outputs
                                if completion.finish_reason == "length"),
                "finish_reasons": finish_reasons,
                "majority_trace": trace,
            }
            if task_name == mmlu_pro.TASK_NAME:
                row["majority_option"] = (
                    None if majority is None
                    else mmlu_pro.selected_option_content(doc["options"], majority)
                )
            rows.append(row)
        print(f"{split}: {len(docs)} documents aggregated", flush=True)

    meta = {
        "schema": consistency.CONF_SCHEMA,
        "k": consistency.K, "temperature": consistency.TEMPERATURE,
        "top_p": consistency.TOP_P, "max_new_tokens": consistency.MAX_NEW_TOKENS,
        "stop": list(stops),
        "task": task_name,
        "model": model_name, "revision": revision,
        "endpoint": endpoint, "model_path": os.path.abspath(model_path),
        "endpoint_identity": endpoint_identity,
        "endpoint_manifest": os.path.abspath(config["endpoint_manifest"]),
        "endpoint_manifest_sha256": endpoint_manifest_hash,
        "scores_dir": os.path.abspath(scores_dir),
        "scores_sha256": {split: sha256_file(path) for split, path in score_paths.items()},
        "fingerprints": fingerprints,
        "prompts_file": os.path.abspath(config["prompts"]),
        "prompts_sha256": sha256_file(config["prompts"]),
        "splits": list(selected_splits),
        "n_eval": len(splits["eval"]), "n_test": len(splits["test"]),
        "subset": {
            "method": "all",
            "eval_ids_sha256": consistency.ids_sha256(splits["eval"]),
            "test_ids_sha256": consistency.ids_sha256(splits["test"]),
        },
        "shard": {
            "method": "position-modulo",
            "num_shards": config["num_shards"],
            "shard_index": config["shard_index"],
        },
    }
    acc = {s: sum(r["z"] for r in rows if r["split"] == s) /
              max(1, sum(1 for r in rows if r["split"] == s)) for s in ("eval", "test")}
    share = {s: sum(r["share"] for r in rows if r["split"] == s) /
                max(1, sum(1 for r in rows if r["split"] == s)) for s in ("eval", "test")}
    for split in ("eval", "test"):
        if not splits[split]:
            continue
        recorder.add_with_logging(key=f"{split}_acc", value=acc[split])
        recorder.add_with_logging(key=f"{split}_mean_share", value=share[split])
    recorder.add_with_logging(key="conf_meta", value=meta)
    recorder.end_recording()
    # Published last: a shard is complete only after both its environment record and rows exist.
    consistency.write_conf_scores(out_path, meta, rows)
    print(f"DONE {out_path}  acc(eval)={acc['eval']:.4f} acc(test)={acc['test']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
