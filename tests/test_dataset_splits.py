"""The train/predictor-train/test carve, against real tasks.

Why this matters enough to need real data: `get_dataset` is called once per process -- by
finetune_model, then again by evaluate_model for each checkpoint, then implicitly by whatever reads
those scores. Nothing passes the split between them; each re-derives it from
(task, seed, TASK_VALID_FRAC, len(training_docs)). If that ever stops being reproducible, the
predictor trains on examples the checkpoint was selected on and nothing raises.

Needs the benchmark datasets, so it is opt-in:  pytest --live
"""
import pytest

from pproute.tasks import get_dataset, split_fingerprint

pytestmark = pytest.mark.live

TASK = "winogrande"        # carves valid out of train (no labelled test split) and is cached


def _splits(seed=0, max_num_train=16, max_num_valid=8, max_num_test=8):
    out = get_dataset(TASK, random_seed=seed, max_num_train=max_num_train,
                      max_num_valid=max_num_valid, max_num_test=max_num_test,
                      tokenizer_name="dummy", tokenizer=None, build_sft=True)
    train_ds, valid_ds, test_ds = out[0], out[1], out[2]
    train_task, valid_task, test_task = out[3][TASK], out[4][TASK], out[5][TASK]
    return {"datasets": (train_ds, valid_ds, test_ds),
            "tasks": (train_task, valid_task, test_task)}


def test_same_seed_gives_the_same_split():
    a, b = _splits(seed=0), _splits(seed=0)
    for i, name in enumerate(("train", "valid", "test")):
        assert split_fingerprint(a["tasks"][i]) == split_fingerprint(b["tasks"][i]), name


def test_different_seed_changes_the_carved_splits():
    """The carve is seeded, so a different seed must give different train/valid sets -- otherwise
    the seed is not doing what the multi-seed protocol assumes."""
    a, b = _splits(seed=0), _splits(seed=1)
    assert split_fingerprint(a["tasks"][1]) != split_fingerprint(b["tasks"][1])


def test_train_and_valid_are_disjoint():
    """The predictor trains on `valid`; any overlap with the finetuning set leaks the label."""
    s = _splits(max_num_train=64, max_num_valid=64)
    train_task, valid_task, _ = s["tasks"]
    key = lambda d: d["sentence"]                                    # noqa: E731
    train_keys = {key(d) for d in train_task.eval_docs}
    valid_keys = {key(d) for d in valid_task.eval_docs}
    assert train_keys and valid_keys
    assert not (train_keys & valid_keys)


def test_zero_max_num_yields_an_empty_split_without_raising():
    """evaluate_model asks for no train split by default; lm_eval raises if asked to build
    requests for zero docs, so get_dataset has to short-circuit instead."""
    s = _splits(max_num_train=0)
    train_ds, valid_ds, test_ds = s["datasets"]
    assert len(train_ds) == 0
    assert len(valid_ds) > 0 and len(test_ds) > 0


def test_sft_rows_are_built_for_a_multiple_input_task():
    """winogrande's gold comes from doc_to_text, not doc_to_target. If that regressed, the
    finetuning set would be empty or wrong for every Winograd-shaped task."""
    s = _splits(max_num_train=32)
    train_ds = s["datasets"][0]
    assert len(train_ds) > 0
    assert all(row["prompt"].strip() and row["completion"].strip() for row in train_ds)
