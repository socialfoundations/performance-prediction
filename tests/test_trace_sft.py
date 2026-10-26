import json

import pytest

from pproute import trace_sft


class Instance:
    def __init__(self, doc, prompt):
        self.doc = doc
        self.arguments = (prompt, {})


class Task:
    def __init__(self):
        self.eval_docs = [
            {"question_id": "q1", "answer": "B"},
            {"question_id": "q2", "answer": "C"},
        ]
        self.instances = [Instance(row, "rendered:" + row["question_id"])
                          for row in self.eval_docs]

    def doc_to_target(self, doc):
        return doc["answer"]

    def get_config(self, key):
        return " :: " if key == "target_delimiter" else None


def adapter():
    return trace_sft.TaskAdapter(
        task_name="multiple_choice_fixture",
        example_id=lambda doc: doc["question_id"],
        is_correct=lambda text, gold: text.rstrip().endswith("(%s)." % gold),
        split_description="the fixture's registered SFT split",
    )


def write_traces(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_non_gsm_adapter_preserves_prompt_completion(tmp_path):
    rows = [
        {"example_id": "q1", "gold": "B", "completion": "reason\nThe answer is (B).",
         "provenance": {"task": "multiple_choice_fixture", "teacher": "fixture"}},
        {"example_id": "q2", "gold": "C", "completion": "reason\nThe answer is (C).",
         "provenance": {"task": "multiple_choice_fixture", "teacher": "fixture"}},
    ]
    path = tmp_path / "traces.jsonl"
    write_traces(path, rows)

    teacher, info = trace_sft.build_trace_sft_dataset(
        Task(), path, "teacher", adapter()
    )

    assert teacher.column_names == ["prompt", "completion", "example_id"]
    assert teacher["prompt"] == ["rendered:q1", "rendered:q2"]
    assert teacher["completion"][0] == " :: reason\nThe answer is (B)."
    assert info["task_name"] == "multiple_choice_fixture"
    assert info["id_field"] == "example_id"


def test_shared_loader_rejects_leak_wrong_answer_and_mixed_provenance(tmp_path):
    path = tmp_path / "traces.jsonl"
    base = {"example_id": "q1", "gold": "B", "completion": "The answer is (B).",
            "provenance": {"task": "multiple_choice_fixture", "teacher": "fixture"}}
    write_traces(path, [base])
    with pytest.raises(ValueError, match="NOT in the fixture's registered SFT split"):
        trace_sft.load_accepted_traces(path, adapter(), allowed_ids={"q2"})

    write_traces(path, [dict(base, completion="The answer is (C).")])
    with pytest.raises(ValueError, match="not gold"):
        trace_sft.load_accepted_traces(path, adapter(), allowed_ids={"q1"})

    write_traces(path, [base, {
        "example_id": "q2", "gold": "C", "completion": "The answer is (C).",
        "provenance": {"task": "multiple_choice_fixture", "teacher": "different"},
    }])
    with pytest.raises(ValueError, match="mixes trace provenance"):
        trace_sft.load_accepted_traces(path, adapter(), allowed_ids={"q1", "q2"})


def test_shared_builder_rechecks_trace_against_current_task_gold(tmp_path):
    path = tmp_path / "traces.jsonl"
    write_traces(path, [{
        "example_id": "q1", "gold": "B", "completion": "The answer is (B).",
        "provenance": {"task": "multiple_choice_fixture", "teacher": "fixture"},
    }])
    task = Task()
    task.eval_docs = [{"question_id": "q1", "answer": "C"}]
    task.instances = [Instance(task.eval_docs[0], "rendered:q1")]
    with pytest.raises(ValueError, match="current task gold"):
        trace_sft.build_trace_sft_dataset(task, path, "teacher", adapter())


def test_shared_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "traces.jsonl"
    row = {"example_id": "q1", "gold": "B", "completion": "The answer is (B).",
           "provenance": {"task": "multiple_choice_fixture", "teacher": "fixture"}}
    write_traces(path, [row, row])
    with pytest.raises(ValueError, match="duplicate example_id"):
        trace_sft.load_accepted_traces(path, adapter(), allowed_ids={"q1"})


def test_trace_sft_params_are_train_only_without_mutating_caller():
    original = {"max_num_train": -1, "max_num_valid": 3, "max_num_test": 4}
    actual = trace_sft.train_only_dataset_params("traces.jsonl", original)
    assert actual == {"max_num_train": -1, "max_num_valid": 0, "max_num_test": 0}
    assert original == {"max_num_train": -1, "max_num_valid": 3, "max_num_test": 4}


def test_shared_builder_requires_one_rendered_request_per_trace(tmp_path):
    path = tmp_path / "traces.jsonl"
    write_traces(path, [{
        "example_id": "q1", "gold": "B", "completion": "The answer is (B).",
        "provenance": {"task": "multiple_choice_fixture", "teacher": "fixture"},
    }])

    missing = Task()
    missing.instances = []
    with pytest.raises(ValueError, match="absent from the current SFT split"):
        trace_sft.build_trace_sft_dataset(missing, path, "teacher", adapter())

    duplicate = Task()
    duplicate.eval_docs = duplicate.eval_docs[:1]
    duplicate.instances = [Instance(duplicate.eval_docs[0], "one"),
                           Instance(duplicate.eval_docs[0], "two")]
    with pytest.raises(ValueError, match="multiple SFT requests"):
        trace_sft.build_trace_sft_dataset(duplicate, path, "teacher", adapter())


def test_shared_builder_rejects_unknown_completion_source(tmp_path):
    with pytest.raises(ValueError, match="completion_source"):
        trace_sft.build_trace_sft_dataset(Task(), tmp_path / "unused", "text", adapter())


def test_shared_training_diagnostics_count_masking_and_truncation():
    class Tokenizer:
        eos_token = " <eos>"

        def __call__(self, text):
            return {"input_ids": text.split()}

    dataset = [
        {"prompt": "p0 p1 p2 p3", "completion": " c0"},
        {"prompt": "p0", "completion": " c0"},
    ]
    diagnostics = trace_sft.training_diagnostics(dataset, Tokenizer(), max_length=3)
    assert diagnostics == {
        "n": 2,
        "max_length": 3,
        "max_untruncated_tokens": 6,
        "truncated_examples": 1,
        "truncation_rate": 0.5,
        "fully_masked_examples": 1,
        "completion_tokens_lost": 2,
    }
