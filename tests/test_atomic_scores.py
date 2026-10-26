import pytest

from scripts import evaluate_model


def _scores(*task_names, date=1.0):
    task_maps = {
        key: {name: {"value": "%s:%s" % (key, name)} for name in task_names}
        for key in evaluate_model.TASK_RESULT_FIELDS
    }
    return {
        **task_maps,
        "date": date,
        "config": {"batch_size": 8},
        "group_subtasks": {},
        "pretty_env_info": "same environment",
    }


def test_atomic_joblib_dump_publishes_complete_file(tmp_path):
    path = tmp_path / "scores.jbl"
    value = {"samples": [1, 2, 3]}

    evaluate_model.atomic_joblib_dump(value, str(path))

    assert evaluate_model.jbl.load(path) == value
    assert not list(tmp_path.glob("*.partial"))


def test_atomic_joblib_dump_cleans_up_after_failed_write(tmp_path, monkeypatch):
    path = tmp_path / "scores.jbl"

    def fail(_value, tmp):
        with open(tmp, "wb") as f:
            f.write(b"incomplete")
        raise RuntimeError("evicted")

    monkeypatch.setattr(evaluate_model.jbl, "dump", fail)
    with pytest.raises(RuntimeError, match="evicted"):
        evaluate_model.atomic_joblib_dump({}, str(path))

    assert not path.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_merge_task_scores_matches_one_complete_payload():
    merged = evaluate_model.merge_task_scores([
        ("task_a", _scores("task_a")),
        ("task_b", _scores("task_b")),
    ])

    assert merged == _scores("task_a", "task_b")


def test_merge_preserves_git_stamps_when_head_changes_during_evaluation():
    first = dict(_scores("task_a"), git_hash="before")
    second = dict(_scores("task_b"), git_hash="after")

    merged = evaluate_model.merge_task_scores([("task_a", first), ("task_b", second)])

    assert merged["git_hash"] is None
    assert merged["pproute_git_hash_by_task"] == {"task_a": "before", "task_b": "after"}
    assert merged["samples"] == _scores("task_a", "task_b")["samples"]
    assert first["git_hash"] == "before" and second["git_hash"] == "after"


def test_merge_keeps_a_common_git_stamp_and_rejects_changed_settings():
    first = dict(_scores("task_a"), git_hash="same")
    second = dict(_scores("task_b"), git_hash="same")
    merged = evaluate_model.merge_task_scores([("task_a", first), ("task_b", second)])
    assert merged["git_hash"] == "same"
    assert "pproute_git_hash_by_task" not in merged

    second["config"] = {"batch_size": 1}
    with pytest.raises(ValueError, match="shared field config"):
        evaluate_model.merge_task_scores([("task_a", first), ("task_b", second)])


def test_multitask_evaluation_resumes_after_an_interruption(tmp_path, monkeypatch):
    calls = []
    fail_task_b = [True]

    class FakeEvaluator:
        def __init__(self, task_dict, _task_manager, **_kwargs):
            self.task_name = next(iter(task_dict))

        def evaluate(self, _model, _tokenizer):
            calls.append(self.task_name)
            if self.task_name == "task_b" and fail_task_b[0]:
                raise RuntimeError("evicted")
            return _scores(self.task_name)

    monkeypatch.setattr(evaluate_model, "Evaluator", FakeEvaluator)
    task_dict = {"task_a": object(), "task_b": object()}
    metadata = {
        name: {"schema": evaluate_model.SCORE_PART_SCHEMA, "task": name}
        for name in task_dict
    }
    parts_dir = tmp_path / "parts"

    with pytest.raises(RuntimeError, match="evicted"):
        evaluate_model.evaluate_resumable(
            task_dict, None, None, None, {}, str(parts_dir), metadata
        )
    assert calls == ["task_a", "task_b"]
    assert (parts_dir / "part-000.jbl").is_file()
    assert not (parts_dir / "part-001.jbl").exists()

    fail_task_b[0] = False
    resumed = evaluate_model.evaluate_resumable(
        task_dict, None, None, None, {}, str(parts_dir), metadata
    )

    assert calls == ["task_a", "task_b", "task_b"]
    assert resumed == _scores("task_a", "task_b")


def test_multitask_resume_refuses_stale_part_metadata(tmp_path):
    parts_dir = tmp_path / "parts"
    part = parts_dir / "part-000.jbl"
    evaluate_model.atomic_joblib_dump(
        {"meta": {"task": "old"}, "scores": _scores("task_a")}, str(part)
    )

    with pytest.raises(ValueError, match="stale or malformed"):
        evaluate_model.evaluate_resumable(
            {"task_a": object()}, None, None, None, {}, str(parts_dir),
            {"task_a": {"task": "task_a"}},
        )
