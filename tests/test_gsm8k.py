import json

import pytest

from pproute import gsm8k, trace_sft
from datasets import Dataset

from scripts import evaluate_model
from scripts.finetune_model import trace_sft_dataset_params
from pproute.tasks import _build_sft_dataset, reserve_fewshot_data


class FakeInstance:
    def __init__(self, doc, prompt):
        self.doc = doc
        self.arguments = (prompt, {})


class FakeGenerateTask:
    instances = []

    def __init__(self, docs, prompts=None):
        prompts = prompts or ["Question: %s\nAnswer:" % d["question"] for d in docs]
        self._docs = docs
        self.instances = [FakeInstance(d, p) for d, p in zip(docs, prompts)]

    @property
    def eval_docs(self):
        return self._docs

    def get_config(self, key):
        return "generate_until" if key == "output_type" else None

    def doc_to_target(self, doc):
        return doc["answer"]


def doc(question="How many?", answer="Reasoning.\n#### 12"):
    return {"question": question, "answer": answer}


def test_flexible_extractor_takes_the_last_number_like_eval():
    text = "First try 7, then 1,234. The requested ending is #### 12"
    assert gsm8k.extract_answer(text) == "12"
    assert gsm8k.is_correct(text, "work\n#### 12")
    assert not gsm8k.is_correct(text, "#### 7")
    assert gsm8k.extract_answer("no numeric answer") is None


def test_extracted_answer_that_normalizes_to_empty_is_a_failure():
    assert gsm8k.extract_answer("final answer: $$") == "$$"
    assert gsm8k.normalize_extracted_answer("$$") is None
    assert not gsm8k.is_correct("final answer: $$", "#### 0")


def test_generative_sft_requires_teacher_traces():
    with pytest.raises(ValueError, match="requires teacher traces"):
        _build_sft_dataset(FakeGenerateTask([doc()]))


def test_trace_sft_forces_train_only_dataset_construction():
    params = trace_sft_dataset_params(
        "gsm8k",
        "/tmp/traces.jsonl",
        {"max_num_train": -1, "max_num_valid": 7, "max_num_test": 9},
    )
    assert params == {"max_num_train": -1, "max_num_valid": 0, "max_num_test": 0}


def test_trace_sft_rejects_other_tasks_before_dataset_construction():
    with pytest.raises(ValueError, match="no generated-trace adapter"):
        trace_sft_dataset_params("arc_easy", "/tmp/traces.jsonl", {})


def test_train_only_gsm8k_never_accesses_or_builds_held_out_splits(monkeypatch):
    import pproute.tasks as tasks
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Sampler:
        def replace_df(self, docs):
            self.docs = docs

    class Task:
        def __init__(self):
            self.config = type("TaskConfig", (), {"task": "gsm8k"})()
            self.sampler = Sampler()
            self.instances = []

        def has_training_docs(self):
            return True

        def has_validation_docs(self):
            return False

        def has_test_docs(self):
            return True

        def training_docs(self):
            return Dataset.from_list([
                {"id": i, "question": "question %d" % i, "answer": "#### %d" % i}
                for i in range(30)
            ])

        @property
        def eval_docs(self):
            raise AssertionError("official GSM8K test docs were accessed")

        def get_config(self, key):
            return "generate_until" if key == "output_type" else None

        def doc_to_target(self, row):
            return row["answer"]

    request_splits = []

    def build_requests(task, *args, **kwargs):
        docs = list(task.eval_docs)
        request_splits.append([row["id"] for row in docs])
        task.instances = [
            FakeInstance(row, "Question: %s\nAnswer:" % row["question"])
            for row in docs
        ]
        return task

    monkeypatch.setattr(
        tasks,
        "get_tasks",
        lambda **kwargs: ({"gsm8k": Task()}, object()),
    )
    monkeypatch.setattr(tasks, "_build_requests", build_requests)

    train_only = tasks.get_dataset(
        "gsm8k",
        max_num_train=-1,
        max_num_valid=0,
        max_num_test=0,
        completion_only=True,
    )
    train_only_ids = [row["id"] for row in train_only[3]["gsm8k"].eval_docs]

    assert len(request_splits) == 1
    assert len(train_only_ids) > 0
    assert len(train_only[0]) == 0
    assert len(train_only[1]) == len(train_only[2]) == 0
    assert len(train_only[4]["gsm8k"].eval_docs) == 0
    assert len(train_only[5]["gsm8k"].eval_docs) == 0

    request_splits.clear()
    with_validation = tasks.get_dataset(
        "gsm8k",
        max_num_train=-1,
        max_num_valid=-1,
        max_num_test=0,
        completion_only=True,
    )
    assert [row["id"] for row in with_validation[3]["gsm8k"].eval_docs] == train_only_ids
    assert len(request_splits) == 2


def test_fewshot_reserve_is_disjoint_deterministic_and_complete():
    data = Dataset.from_list([{"id": i} for i in range(20)])
    remaining, demos = reserve_fewshot_data(data, 5, seed=1234)
    remaining_again, demos_again = reserve_fewshot_data(data, 5, seed=1234)
    r, d = set(remaining["id"]), set(demos["id"])
    assert not (r & d)
    assert r | d == set(range(20))
    assert remaining["id"] == remaining_again["id"]
    assert demos["id"] == demos_again["id"]


def write_traces(path, docs, completions=None):
    completions = completions or [d["answer"] for d in docs]
    provenance = {
        "task": "gsm8k",
        "teacher_model": "Qwen/Qwen3.5-122B-A10B",
        "dtype": "bfloat16",
    }
    with open(path, "w") as handle:
        for d, completion in zip(docs, completions):
            handle.write(json.dumps({
                "example_id": gsm8k.example_id(d),
                "gold": gsm8k.gold_answer(d["answer"]),
                "completion": completion,
                "provenance": provenance,
            }) + "\n")


def test_trace_loader_rejects_non_sft_and_wrong_answers(tmp_path):
    d = doc()
    path = tmp_path / "traces.jsonl"
    write_traces(path, [d])
    with pytest.raises(ValueError, match="NOT in GSM8K's SFT split"):
        trace_sft.load_accepted_traces(
            str(path), gsm8k.TRACE_SFT_ADAPTER, allowed_ids={"someone-else"}
        )
    write_traces(path, [d], ["step by step\n#### 13"])
    with pytest.raises(ValueError, match="not gold"):
        trace_sft.load_accepted_traces(
            str(path), gsm8k.TRACE_SFT_ADAPTER, allowed_ids={gsm8k.example_id(d)}
        )


def test_teacher_traces_preserve_prompts_and_accepted_ids(tmp_path):
    docs = [doc("one", "gold work\n#### 1"), doc("two", "gold work\n#### 2")]
    path = tmp_path / "traces.jsonl"
    write_traces(path, docs, ["teacher work\n#### 1", "teacher work\n#### 2"])
    task = FakeGenerateTask(docs, ["five-shot prompt 1", "five-shot prompt 2"])
    teacher, _ = trace_sft.build_trace_sft_dataset(
        task, str(path), "teacher", gsm8k.TRACE_SFT_ADAPTER
    )
    assert len(set(teacher["example_id"])) == 2
    assert teacher["prompt"] == ["five-shot prompt 1", "five-shot prompt 2"]
    assert all(c.startswith(" teacher") for c in teacher["completion"])


@pytest.mark.parametrize(
    "change, message",
    [
        ({"num_fewshot": 0}, "5-shot"),
        ({"fewshot_as_multiturn": False}, "multiturn"),
    ],
)
def test_gsm8k_eval_protocol_refuses_silent_misconfiguration(change, message):
    params = dict(max_num_train=0, max_num_valid=8, max_num_test=0,
                  num_fewshot=5, fewshot_as_multiturn=True)
    max_new_tokens = change.pop("max_new_tokens", 2048)
    params.update(change)
    with pytest.raises(ValueError, match=message):
        gsm8k.validate_eval_protocol(
            "gsm8k", params, max_new_tokens, apply_chat_template=True
        )


def test_gsm8k_eval_protocol_requires_full_generation_budget():
    params = dict(max_num_train=0, max_num_valid=8, max_num_test=0,
                  num_fewshot=5, fewshot_as_multiturn=True)
    with pytest.raises(ValueError, match="2048"):
        gsm8k.validate_eval_protocol("gsm8k", params, 256, True)


# The validation_only rail is task-neutral -- every benchmark's driver uses it for the validation
# stage -- so it lives on the entry script rather than in one benchmark module.
@pytest.mark.parametrize(
    "change", [{"max_num_test": 1}, {"max_num_train": 1}, {"max_num_valid": 0}]
)
def test_validation_only_refuses_to_touch_the_held_out_split(change):
    params = dict(max_num_train=0, max_num_valid=8, max_num_test=0)
    params.update(change)
    with pytest.raises(ValueError, match="validation_only"):
        evaluate_model.validate_validation_only(params, validation_only=True)


class FakeTokenizer:
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def __call__(self, text):
        return {"input_ids": text.split()}


def sample(doc_id, response, z, questions=6):
    return {
        "doc_id": doc_id,
        "filter": "flexible-extract",
        "arguments": [(" ".join(["Question:"] * questions), {})],
        "resps": [[response]],
        "exact_match": z,
    }


def test_evaluation_diagnostics_pin_artifact_and_failures():
    results = {"samples": {"gsm8k": [
        sample(0, "work #### 2", 1.0),
        sample(1, "no number", 0.0, questions=1),
    ]}}
    report = gsm8k.evaluation_diagnostics(results, FakeTokenizer(), max_new_tokens=3)
    assert report["accuracy"] == 0.5
    assert report["extraction_failures"] == 1
    assert report["generation_cap_hits"] == 1
    assert report["prompt_protocol_failures"] == 1


def test_evaluation_diagnostics_allow_question_marker_in_question_text():
    row = sample(0, "work #### 2", 1.0)
    context = row["arguments"][0][0] + ' What does the label "Question:" mean?'
    row["arguments"] = [(context, {})]
    report = gsm8k.evaluation_diagnostics(
        {"samples": {"gsm8k": [row]}}, FakeTokenizer(), max_new_tokens=2048)
    assert report["prompt_protocol_failures"] == 0


@pytest.mark.live
def test_real_gsm8k_trace_sft_loader_builds_only_the_registered_train_carve():
    from pproute.tasks import get_dataset

    output = get_dataset(
        "gsm8k",
        random_seed=0,
        max_num_train=-1,
        max_num_valid=0,
        max_num_test=0,
        num_fewshot=0,
        tokenizer_name="train-only-probe",
        tokenizer=None,
        completion_only=True,
    )
    assert (len(output[0]), len(output[1]), len(output[2])) == (0, 0, 0)
    assert len(output[3]["gsm8k"].eval_docs) == 5227
    assert len(output[4]["gsm8k"].eval_docs) == 0
    assert len(output[5]["gsm8k"].eval_docs) == 0


@pytest.mark.live
def test_real_gsm8k_prompts_are_five_shot_multiturn_without_validation_leakage():
    from transformers import AutoTokenizer

    from pproute.evaluator import build_predictor_contexts
    from pproute.tasks import get_dataset

    model = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    output = get_dataset(
        "gsm8k",
        random_seed=0,
        max_num_train=8,
        max_num_valid=8,
        max_num_test=0,
        apply_chat_template=True,
        num_fewshot=5,
        fewshot_as_multiturn=True,
        tokenizer_name=model,
        tokenizer=tokenizer,
        completion_only=True,
    )
    train_task, valid_task, test_task = (
        output[3]["gsm8k"], output[4]["gsm8k"], output[5]["gsm8k"]
    )
    # Generative SFT rows come from teacher traces, so the scored requests are the evidence here.
    assert tuple(len(task.eval_docs) for task in (train_task, valid_task, test_task)) == (8, 8, 0)
    assert all(inst.arguments[0].count("Question:") == 6 for inst in train_task.instances)
    assert all(inst.arguments[0].count("Question:") == 6 for inst in valid_task.instances)
    demonstrations = {gsm8k.example_id(doc) for doc in valid_task._pproute_fewshot_docs}
    train_ids = {gsm8k.example_id(doc) for doc in train_task.eval_docs}
    valid_ids = {gsm8k.example_id(doc) for doc in valid_task.eval_docs}
    assert len(demonstrations) == 5
    assert not demonstrations & train_ids
    assert not demonstrations & valid_ids

    # Predictor input is the same full task-native five-shot text for raw and chat models. The
    # scored request may carry vendor role markers, but the stamp must never collapse to the target
    # question or vary with that rendering choice.
    raw_output = get_dataset(
        "gsm8k",
        random_seed=0,
        max_num_train=0,
        max_num_valid=8,
        max_num_test=0,
        apply_chat_template=False,
        num_fewshot=5,
        fewshot_as_multiturn=False,
        tokenizer_name="raw-context-probe",
        tokenizer=None,
        completion_only=True,
    )
    raw_valid_task = raw_output[4]["gsm8k"]
    chat_contexts = build_predictor_contexts({"gsm8k": valid_task}, 5, 1234)
    raw_contexts = build_predictor_contexts({"gsm8k": raw_valid_task}, 5, 1234)
    assert chat_contexts == raw_contexts
    assert all(text.count("Question:") == 6 for text in raw_contexts["gsm8k"].values())
    raw_requests = {inst.doc_id: inst.arguments[0] for inst in raw_valid_task.instances}
    assert raw_requests == raw_contexts["gsm8k"]
