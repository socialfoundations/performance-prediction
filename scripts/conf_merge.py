"""Validate and merge deterministic document shards from ``scripts.conf_sample``.

    python -m scripts.conf_merge --exp_name <merged dir> --shards_dir <shards dir> \
        --num_shards <n> --expected_eval <n> --expected_test <n> [--no-use_git]
"""
from __future__ import annotations

import json
import os

from pproute import consistency
from pproute.provenance import sha256_file

COMMON_META = (
    "k", "temperature", "top_p", "max_new_tokens", "task", "model",
    "revision", "endpoint", "model_path", "scores_dir", "fingerprints", "prompts_file", "stop",
    "schema", "endpoint_identity", "endpoint_manifest", "endpoint_manifest_sha256",
    "scores_sha256", "prompts_sha256", "splits",
)


def merged_conf_shards(paths, *, expected_eval, expected_test):
    if not paths:
        raise ValueError("at least one confidence shard is required")
    if expected_eval < 0 or expected_test < 0:
        raise ValueError("expected split sizes must be non-negative")
    expected_splits = tuple(split for split, count in (("eval", expected_eval),
                                                        ("test", expected_test)) if count)
    if not expected_splits:
        raise ValueError("at least one expected split must be non-empty")
    payloads = [consistency.read_conf_scores(path, expected_splits=expected_splits)
                for path in paths]
    first = payloads[0]["meta"]
    for index, payload in enumerate(payloads):
        meta = payload["meta"]
        shard = meta.get("shard")
        expected_shard = {
            "method": "position-modulo", "num_shards": len(paths), "shard_index": index,
        }
        if shard != expected_shard:
            raise ValueError(f"{paths[index]} has wrong shard metadata: {shard}")
        def shard_count(total):
            return len(range(index, total, len(paths)))
        expected_counts = (shard_count(expected_eval), shard_count(expected_test))
        if (meta["n_eval"], meta["n_test"]) != expected_counts:
            raise ValueError(
                f"{paths[index]} row counts {(meta['n_eval'], meta['n_test'])} "
                f"!= {expected_counts}")
        bad = {key: (meta.get(key), first.get(key)) for key in COMMON_META
               if meta.get(key) != first.get(key)}
        if bad:
            raise ValueError(f"{paths[index]} disagrees with shard 0: {bad}")

    rows = []
    for split in ("eval", "test"):
        members = [[row for row in payload["rows"] if row["split"] == split]
                   for payload in payloads]
        for offset in range(max(map(len, members), default=0)):
            rows.extend(shard[offset] for shard in members if offset < len(shard))
    identities = [(row["split"], json.dumps(row["id"], sort_keys=True)) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("confidence shards contain duplicate split/document ids")
    n_eval = sum(row["split"] == "eval" for row in rows)
    n_test = sum(row["split"] == "test" for row in rows)
    if (n_eval, n_test) != (expected_eval, expected_test):
        raise ValueError(
            f"merged row counts {(n_eval, n_test)} != {(expected_eval, expected_test)}")

    meta = dict(first)
    meta["n_eval"], meta["n_test"] = n_eval, n_test
    meta["subset"] = dict(first.get("subset", {}))
    meta["subset"]["eval_ids_sha256"] = consistency.ids_sha256(
        [row for row in rows if row["split"] == "eval"])
    meta["subset"]["test_ids_sha256"] = consistency.ids_sha256(
        [row for row in rows if row["split"] == "test"])
    meta.pop("shard", None)
    meta["merged_shards"] = {
        "method": "position-modulo",
        "num_shards": len(paths),
        "files": [os.path.abspath(path) for path in paths],
        "sha256": [sha256_file(path) for path in paths],
    }
    return meta, rows


def main():
    from zarth_utils.config import Config
    from zarth_utils.logger import get_logger
    from zarth_utils.nn_utils import get_all_paths, set_random_seed
    from zarth_utils.recorder import Recorder

    config = Config(default_config_dict={
        "exp_name": "", "shards_dir": "", "num_shards": 0,
        "expected_eval": -1, "expected_test": -1,
        "use_git": True, "random_seed": 0,
    }, use_argparse=True)
    config.show()
    for required in ("exp_name", "shards_dir"):
        if not config[required]:
            raise SystemExit("--%s is required" % required)
    if config["num_shards"] <= 0:
        raise ValueError("--num_shards must be positive")
    if config["expected_eval"] < 0 or config["expected_test"] < 0:
        raise ValueError("expected split sizes must be non-negative")
    if not (config["expected_eval"] or config["expected_test"]):
        raise ValueError("at least one expected split must be non-empty")
    paths = [os.path.join(config["shards_dir"], f"{index:02d}", "conf_scores.json")
             for index in range(config["num_shards"])]
    result_dir = os.path.join(os.getcwd(), config["exp_name"])
    out_path = os.path.join(result_dir, "conf_scores.json")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        expected_splits = tuple(
            split for split, count in (("eval", config["expected_eval"]),
                                       ("test", config["expected_test"])) if count
        )
        payload = consistency.read_conf_scores(out_path, expected_splits=expected_splits)
        recorded_counts = (payload["meta"]["n_eval"], payload["meta"]["n_test"])
        expected_counts = (config["expected_eval"], config["expected_test"])
        if recorded_counts != expected_counts:
            raise ValueError("existing merged row counts %s != %s" %
                             (recorded_counts, expected_counts))
        expected_merge = {
            "method": "position-modulo", "num_shards": config["num_shards"],
            "files": [os.path.abspath(path) for path in paths],
            "sha256": [sha256_file(path) for path in paths],
        }
        if payload["meta"].get("merged_shards") != expected_merge:
            raise ValueError("existing merged confidence file differs from current shards")
        print(f"COMPLETE {out_path}")
        return 0

    os.makedirs(result_dir, exist_ok=True)
    all_paths = get_all_paths(result_dir)
    config.dump(all_paths["path_config"])
    recorder = Recorder(all_paths["path_record"], config=config, use_git=config["use_git"])
    get_logger(all_paths["path_log"], force_add_handler=True)
    set_random_seed(config["random_seed"])
    meta, rows = merged_conf_shards(
        paths, expected_eval=config["expected_eval"], expected_test=config["expected_test"]
    )
    result = {"n_eval": meta["n_eval"], "n_test": meta["n_test"]}
    recorder.add_with_logging(key="conf_merge", value={
        **result,
        "schema": meta["schema"],
        "output": out_path,
        "merged_shards": meta["merged_shards"],
    })
    recorder.end_recording()
    # Published last: presence implies validated inputs and a finalized merge record.
    consistency.write_conf_scores(out_path, meta, rows)
    print(f"DONE {out_path}  eval={result['n_eval']} test={result['n_test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
