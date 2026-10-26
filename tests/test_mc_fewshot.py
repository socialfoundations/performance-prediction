"""MC five-shot contracts: unchanged query splits, fixed demos, gold-only SFT and target inputs."""
from types import SimpleNamespace

from datasets import Dataset
import joblib
import numpy as np
import pytest
from lm_eval.api.samplers import ContextSampler

from pproute import scores, tasks
from pproute.evaluator import _mc_predictor_target, stamp_gold_index
from pproute.predictor import build_mc


def document(task_name, index):
    if task_name == "winogrande":
        return {"id": index, "sentence": f"Question {index}: Sarah thanked Maria because _ helped.",
                "option1": "Sarah", "option2": "Maria", "answer": "2"}
    return {"id": index, "question": f"Question {index}: Where is the fork?",
            "choices": {"label": list("ABCDE"),
                        "text": ["garage", "kitchen", "ocean", "forest", "desert"]},
            "answerKey": "B"}


class FakeMCTask:
    def __init__(self, task_name, num_fewshot):
        self.config = SimpleNamespace(task=task_name, num_fewshot=num_fewshot,
                                      output_type="multiple_choice")
        self.multiple_input = task_name == "winogrande"
        self.sampler = ContextSampler([])
        self.instances = []

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return Dataset.from_list([document(self.config.task, i) for i in range(100)])

    @property
    def eval_docs(self):
        return Dataset.from_list([document(self.config.task, i) for i in range(100, 110)])

    def get_config(self, key):
        return getattr(self.config, key, None)

    def doc_to_text(self, doc):
        return int(doc["answer"]) - 1 if self.multiple_input else doc["question"]

    def doc_to_choice(self, doc):
        if self.multiple_input:
            prefix = doc["sentence"].split("_", 1)[0]
            return [prefix + doc["option1"], prefix + doc["option2"]]
        return list("ABCDE")

    def doc_to_target(self, doc):
        return doc["sentence"].split("_", 1)[1].strip() if self.multiple_input else doc["answerKey"]


@pytest.fixture
def fake_requests(monkeypatch):
    monkeypatch.setattr(tasks.torch.cuda, "is_available", lambda: False)

    def get_tasks(**kwargs):
        name = kwargs["tasks"][0]
        return {name: FakeMCTask(name, kwargs["num_fewshot"])}, object()

    def build_requests(task, *args):
        task.instances = []
        for doc_id, doc in enumerate(task.eval_docs):
            demos = task.sampler.sample(task.config.num_fewshot)
            prefix = "".join(f"Demo {d['id']}: solved.\n" for d in demos)
            for idx, choice in enumerate(task.doc_to_choice(doc)):
                prompt = prefix + (choice if task.multiple_input else task.doc_to_text(doc) + "\nAnswer:")
                completion = " " + (task.doc_to_target(doc) if task.multiple_input else choice)
                task.instances.append(SimpleNamespace(doc=doc, doc_id=doc_id, idx=idx,
                                                      arguments=(prompt, completion)))
        return task

    monkeypatch.setattr(tasks, "get_tasks", get_tasks)
    monkeypatch.setattr(tasks, "_build_requests", build_requests)


@pytest.mark.parametrize("task_name", ["commonsense_qa", "winogrande"])
def test_mc_demos_leave_development_test_membership_and_order_unchanged(fake_requests, task_name):
    old = tasks.get_dataset(task_name, num_fewshot=0, build_sft=True)
    new = tasks.get_dataset(task_name, num_fewshot=5, completion_only=True, build_sft=True)
    again = tasks.get_dataset(task_name, num_fewshot=5, completion_only=True, build_sft=True)
    train, valid, test = [new[i][task_name] for i in (3, 4, 5)]
    demos = train._pproute_fewshot_docs
    demo_ids = {d["id"] for d in demos}
    old_train_ids = list(old[3][task_name].eval_docs["id"])
    assert len(demo_ids) == 5
    assert demo_ids <= set(old_train_ids)
    assert list(train.eval_docs["id"]) == [i for i in old_train_ids if i not in demo_ids]
    for index in (4, 5):
        assert list(new[index][task_name].eval_docs["id"]) == list(old[index][task_name].eval_docs["id"])
        assert tasks.split_fingerprint(new[index][task_name]).split(":fewshot:")[0] == \
            tasks.split_fingerprint(old[index][task_name])
    for task in (train, valid, test):
        assert not demo_ids & set(task.eval_docs["id"])
        assert task._pproute_fewshot_docs == demos
        assert task.sampler.sample(5) == task.sampler.sample(5) == demos
        task.sampler.set_rnd(999)
        assert task.sampler.sample(5) == demos
        assert tasks.split_fingerprint(task).endswith(":order:fixed")
    assert list(new[0]) == list(again[0])
    assert len(new[0]) == len(old[0]) - 5
    assert len(new[1]) == len(old[1]) and len(new[2]) == len(old[2])


@pytest.mark.parametrize("task_name", ["commonsense_qa", "winogrande"])
def test_evaluation_skips_sft_but_preserves_requests(fake_requests, monkeypatch, task_name):
    training = tasks.get_dataset(task_name, num_fewshot=5, build_sft=True)
    monkeypatch.setattr(tasks, "_build_sft_dataset",
                        lambda *a: pytest.fail("evaluation built unused SFT rows"))
    evaluation = tasks.get_dataset(task_name, num_fewshot=5)
    assert all(len(dataset) == 0 for dataset in evaluation[:3])
    for index in (3, 4, 5):
        before, after = training[index][task_name], evaluation[index][task_name]
        assert tasks.split_fingerprint(before) == tasks.split_fingerprint(after)
        assert [(r.doc_id, r.idx, r.arguments) for r in before.instances] == [
            (r.doc_id, r.idx, r.arguments) for r in after.instances]


@pytest.mark.parametrize("task_name", ["commonsense_qa", "winogrande"])
def test_completion_only_sft_uses_exact_gold_request(fake_requests, task_name):
    output = tasks.get_dataset(task_name, num_fewshot=5, completion_only=True, build_sft=True,
                               max_num_train=2, max_num_valid=0, max_num_test=0)
    task = output[3][task_name]
    assert len(output[0]) == len(task.eval_docs)
    for row, doc in zip(output[0], task.eval_docs):
        gold = int(doc["answer"]) - 1 if task.multiple_input else list("ABCDE").index(doc["answerKey"])
        request = next(inst for inst in task.instances if inst.doc == doc and inst.idx == gold)
        assert (row["prompt"], row["completion"]) == request.arguments
        assert row["prompt"].count("Demo ") == 5
        assert row["completion"] == (" helped." if task.multiple_input else " B")
        if task.multiple_input:
            assert row["prompt"].endswith("because Maria")


@pytest.mark.parametrize("task_name", ["commonsense_qa", "winogrande"])
def test_five_shot_score_roundtrip_has_only_native_target_fields(tmp_path, task_name):
    doc = document(task_name, 7)
    task = FakeMCTask(task_name, 5)
    choices = task.doc_to_choice(doc)
    sample = {"doc_id": 7, "doc": doc,
              "arguments": [("<chat>Demo 999: leaked label</chat>", str(i)) for i in range(len(choices))],
              "filtered_resps": [(-float(i + 1), False) for i in range(len(choices))]}
    result = {"samples": {task_name: [sample]}, "configs": {task_name: {"num_fewshot": 5}}}
    stamp_gold_index(result, {task_name: task})
    target = sample[scores.MC_TARGET_KEY]
    without_gold = {k: v for k, v in doc.items() if k not in ("answer", "answerKey")}
    assert _mc_predictor_target(task_name, without_gold) == target
    path = tmp_path / "scores.jbl"
    joblib.dump(result, path)
    texts, z, logits, loss, ids = build_mc(path)
    assert ids == [7] and z.tolist() == [0]
    assert "Demo 999" not in texts[0] and "<chat>" not in texts[0]
    assert target["question"] in texts[0]
    for choice in target["choices"]:
        assert choice in texts[0]
    probabilities = scores.softmax(-np.arange(1, len(choices) + 1, dtype=float))
    assert loss[0] == pytest.approx(-np.log(probabilities[1]))
    assert logits[0] == pytest.approx(np.log(probabilities[0] / (1 - probabilities[0])))

    del sample[scores.MC_TARGET_KEY]
    joblib.dump(result, path)
    with pytest.raises(ValueError, match="missing pproute_mc_target"):
        build_mc(path)


@pytest.mark.live
@pytest.mark.parametrize("task_name,expected", [("commonsense_qa", (6394, 3342, 1221)),
                                               ("winogrande", (32313, 8080, 1267))])
def test_cached_mc_carves_preserve_original_query_ids(monkeypatch, task_name, expected):
    # Inspect all real query IDs without constructing tens of thousands of requests.
    with monkeypatch.context() as patch:
        patch.setattr(tasks, "_build_requests", lambda task, *args: task)
        patch.setattr(tasks, "_build_sft_dataset", lambda *args: Dataset.from_list([]))
        old_full = tasks.get_dataset(task_name, num_fewshot=0)
        new_full = tasks.get_dataset(task_name, num_fewshot=5, completion_only=True, build_sft=True)
    assert tuple(len(new_full[i][task_name].eval_docs) for i in (3, 4, 5)) == expected
    for i in (4, 5):
        assert list(new_full[i][task_name].eval_docs) == list(old_full[i][task_name].eval_docs)
    demos = new_full[3][task_name]._pproute_fewshot_docs
    assert len(demos) == 5
    original_train = list(old_full[3][task_name].eval_docs)
    assert all(doc in original_train for doc in demos)
    for i in (3, 4, 5):
        assert not any(doc in new_full[i][task_name].eval_docs for doc in demos)
    # Also build actual lm-eval requests with the reserved demonstration sampler.
    old = tasks.get_dataset(task_name, num_fewshot=0, max_num_train=1, max_num_valid=4, max_num_test=0)
    new = tasks.get_dataset(task_name, num_fewshot=5, completion_only=True, build_sft=True,
                            max_num_train=1, max_num_valid=4, max_num_test=0)
    assert list(old[4][task_name].eval_docs) == list(new[4][task_name].eval_docs)
    assert len(new[0]) == 1 and len(new[1]) == 4 and len(new[2]) == 0


@pytest.mark.live
@pytest.mark.parametrize("task_name", ["commonsense_qa", "winogrande"])
@pytest.mark.parametrize("model,revision", [
    ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("google/gemma-3-12b-it", "96b6f1eccf38110c56df3a15bffe176da04bfd80"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71"),
])
def test_cached_chat_sft_and_eval_share_five_demos_and_token_boundaries(task_name, model, revision):
    from transformers import AutoTokenizer
    from pproute.models import (configure_chat_sft_tokenizer,
                                disable_chat_template_thinking, tokenizer_load_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True,
                                             **tokenizer_load_kwargs(model))
    disable_chat_template_thinking(tokenizer)
    configure_chat_sft_tokenizer(tokenizer, True)
    output = tasks.get_dataset(task_name, num_fewshot=5, completion_only=True, build_sft=True,
                               max_num_train=2, max_num_valid=2, max_num_test=0,
                               apply_chat_template=True, fewshot_as_multiturn=True,
                               tokenizer=tokenizer, tokenizer_name=model)
    for dataset, task_dict in zip(output[:2], output[3:5]):
        task = task_dict[task_name]
        initial = [(inst.doc_id, inst.idx, inst.arguments) for inst in task.instances]
        task.set_fewshot_seed(0)
        tasks._build_requests(task, True, True, tokenizer, model)
        assert [(inst.doc_id, inst.idx, inst.arguments) for inst in task.instances] == initial
        for row in dataset:
            assert any(inst.arguments == (row["prompt"], row["completion"])
                       for inst in task.instances)
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            full_ids = tokenizer.encode(row["prompt"] + row["completion"], add_special_tokens=False)
            assert full_ids[:len(prompt_ids)] == prompt_ids
            assert len(full_ids) > len(prompt_ids)
        assert len(task._pproute_fewshot_docs) == 5
        # Changing query golds must not change any evaluation request, including Wino's
        # two candidate contexts. Demonstration answers deliberately stay unchanged.
        changed = []
        for doc in task.eval_docs:
            doc = dict(doc)
            if task_name == "winogrande":
                doc["answer"] = "1" if doc["answer"] == "2" else "2"
            else:
                doc["answerKey"] = "A" if doc["answerKey"] != "A" else "B"
            changed.append(doc)
        task._eval_docs = Dataset.from_list(changed)
        tasks._build_requests(task, True, True, tokenizer, model)
        assert [(inst.doc_id, inst.idx, inst.arguments) for inst in task.instances] == initial
