import os

import pytest

from pproute import consistency
from scripts import conf_merge


def _row(split, doc_id):
    return {
        "split": split, "id": doc_id, "gold": "A", "predictor_text": f"question {doc_id}",
        "majority": "A", "share": 1.0, "z": 1,
        "counts": {"A": consistency.K}, "first_seen": {"A": 0}, "n_fail": 0,
        "cap_hits": 0, "finish_reasons": {"stop": consistency.K},
        "majority_trace": "the answer is (A).", "majority_option": "option A",
    }


def _write_shard(root, index, eval_ids, test_ids):
    rows = [_row("eval", doc_id) for doc_id in eval_ids]
    rows += [_row("test", doc_id) for doc_id in test_ids]
    splits = [split for split, ids in (("eval", eval_ids), ("test", test_ids)) if ids]
    path = root / f"{index:02d}" / "conf_scores.json"
    path.parent.mkdir(parents=True)
    meta = {
        "schema": consistency.CONF_SCHEMA,
        "k": consistency.K, "temperature": consistency.TEMPERATURE,
        "top_p": consistency.TOP_P, "max_new_tokens": consistency.MAX_NEW_TOKENS,
        "task": "mmlu_pro", "model": "org/model",
        "revision": "a" * 40, "endpoint": "before", "model_path": "/model",
        "scores_dir": "/scores", "prompts_file": "/prompts",
        "endpoint_manifest": "/endpoint.json", "stop": ["Question:"],
        "endpoint_identity": "b" * 64, "endpoint_manifest_sha256": "c" * 64,
        "scores_sha256": {split: "d" * 64 for split in splits},
        "prompts_sha256": "e" * 64, "splits": splits,
        "fingerprints": {split + "_split": split + "-fp" for split in splits},
        "n_eval": len(eval_ids), "n_test": len(test_ids),
        "subset": {
            "method": "all",
            "eval_ids_sha256": consistency.ids_sha256(rows[:len(eval_ids)]),
            "test_ids_sha256": consistency.ids_sha256(rows[len(eval_ids):]),
        },
        "shard": {"method": "position-modulo", "num_shards": 2, "shard_index": index},
    }
    consistency.write_conf_scores(str(path), meta, rows)
    return str(path)


def test_merge_conf_shards_restores_order_and_protocol(tmp_path):
    shards = tmp_path / "shards"
    paths = [_write_shard(shards, 0, ["e0", "e2"], ["t0"]),
             _write_shard(shards, 1, ["e1", "e3"], ["t1"])]
    meta, rows = conf_merge.merged_conf_shards(paths, expected_eval=4, expected_test=2)
    payload = dict(meta=meta, rows=rows)
    assert [(row["split"], row["id"]) for row in payload["rows"]] == [
        ("eval", "e0"), ("eval", "e1"), ("eval", "e2"), ("eval", "e3"),
        ("test", "t0"), ("test", "t1"),
    ]
    assert payload["meta"]["merged_shards"]["num_shards"] == 2
    assert "shard" not in payload["meta"]
    assert "git_commit" not in payload["meta"]
    assert all(os.path.isabs(path) for path in payload["meta"]["merged_shards"]["files"])


def test_merge_conf_shards_accepts_uneven_single_split(tmp_path):
    shards = tmp_path / "shards"
    paths = [_write_shard(shards, 0, ["e0", "e2"], []),
             _write_shard(shards, 1, ["e1"], [])]
    _, rows = conf_merge.merged_conf_shards(paths, expected_eval=3, expected_test=0)
    assert [row["id"] for row in rows] == ["e0", "e1", "e2"]


def test_merge_conf_shards_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="at least one confidence shard"):
        conf_merge.merged_conf_shards(
            [], expected_eval=1, expected_test=0
        )


def test_existing_merge_must_match_requested_row_counts(tmp_path, monkeypatch):
    shards = tmp_path / "shards"
    paths = [_write_shard(shards, 0, ["e0"], ["t0"]),
             _write_shard(shards, 1, ["e1"], ["t1"])]
    out = tmp_path / "merged" / "conf_scores.json"
    meta, rows = conf_merge.merged_conf_shards(paths, expected_eval=2, expected_test=2)
    out.parent.mkdir()
    consistency.write_conf_scores(str(out), meta, rows)
    monkeypatch.setattr(
        "sys.argv",
        ["conf_merge", "--shards_dir", str(shards), "--num_shards", "2",
         "--exp_name", str(out.parent),
         "--expected_eval", "3", "--expected_test", "2"],
    )
    with pytest.raises(ValueError, match="existing merged row counts"):
        conf_merge.main()


def test_main_finalizes_recorder_before_publishing(tmp_path, monkeypatch):
    import zarth_utils.logger
    import zarth_utils.nn_utils
    import zarth_utils.recorder

    shards = tmp_path / "shards"
    _write_shard(shards, 0, ["e0"], [])
    _write_shard(shards, 1, ["e1"], [])
    out_dir = tmp_path / "merged"
    events = []

    class FakeRecorder:
        def __init__(self, *args, **kwargs):
            events.append("start")

        def add_with_logging(self, *, key, value):
            assert key == "conf_merge"
            assert value["n_eval"] == 2 and value["schema"] == consistency.CONF_SCHEMA
            events.append("record")

        def end_recording(self):
            events.append("end")

    monkeypatch.setattr(zarth_utils.recorder, "Recorder", FakeRecorder)
    monkeypatch.setattr(zarth_utils.logger, "get_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(zarth_utils.nn_utils, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(zarth_utils.nn_utils, "get_all_paths", lambda directory: {
        "path_config": str(out_dir / "run.json"),
        "path_record": str(out_dir / "run.result"),
        "path_log": str(out_dir / "run.log"),
    })
    real_write = consistency.write_conf_scores

    def checked_write(path, meta, rows):
        assert events == ["start", "record", "end"]
        real_write(path, meta, rows)

    monkeypatch.setattr(consistency, "write_conf_scores", checked_write)
    monkeypatch.setattr(
        "sys.argv",
        ["conf_merge", "--exp_name", str(out_dir), "--shards_dir", str(shards),
         "--num_shards", "2", "--expected_eval", "2", "--expected_test", "0",
         "--no-use_git"],
    )

    assert conf_merge.main() == 0
    assert consistency.read_conf_scores(
        out_dir / "conf_scores.json", expected_splits=("eval",)
    )["meta"]["n_eval"] == 2
