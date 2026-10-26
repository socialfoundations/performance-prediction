"""The one prompt extractor, and why sampling uses a separate artifact stage.

An MMLU-Pro score file contains live lm_eval task references. Prompts are projected once into plain
JSON so vLLM never unpickles or re-renders benchmark objects. Both active generative tasks use this
one protocol path.
"""
import joblib
import pytest

from pproute import consistency, gsm8k, mmlu_pro, trace_sft
from pproute.consistency import extract_prompts


def test_final_task_prompt_schemas():
    assert consistency.prompts_schema_for("gsm8k") == consistency.GSM8K_PROMPTS_SCHEMA
    assert consistency.prompts_schema_for("mmlu_pro") == consistency.PROMPTS_SCHEMA


def score_file(tmp_path, samples_by_task, name="eval_scores.jbl", until=("Question:",)):
    payload = {"samples": samples_by_task,
               "configs": {k: {"generation_kwargs": {"until": list(until)}}
                           for k in samples_by_task}}
    path = tmp_path / name
    joblib.dump(payload, str(path))
    return str(path)


def sample(doc_id, answer, filt, options=None):
    return {"doc_id": doc_id, "filter": filt, "arguments": [["prompt-%s" % doc_id, {}]],
            "doc": {"answer": answer,
                    "options": options or ["choice-%s" % letter for letter in "ABCDEFGHIJ"]},
            "pproute_predictor_text": "Question: target %s\nAnswer:" % doc_id}


def test_subtasks_are_pooled_with_namespaced_ids(tmp_path):
    path = score_file(tmp_path, {
        "mmlu_pro_law": [sample(0, "C", "custom-extract"), sample(1, "A", "custom-extract")],
        "mmlu_pro_biology": [sample(0, "B", "custom-extract")]})
    docs, until = extract_prompts(path, trace_sft.adapter_for(mmlu_pro.TASK_NAME))
    ids = [d["id"] for d in docs]
    assert sorted(ids) == ["mmlu_pro_biology:0", "mmlu_pro_law:0", "mmlu_pro_law:1"]
    assert len(set(ids)) == 3, "doc_ids collided across subtasks"
    assert until == ["Question:"]


def test_single_task_ids_stay_bare(tmp_path):
    """GSM8K must keep raw doc_ids: its existing artifacts and joins already use them."""
    path = score_file(tmp_path, {"gsm8k": [sample(0, "w\n#### 4", "flexible-extract")]})
    docs, _ = extract_prompts(path, trace_sft.adapter_for(gsm8k.TASK_NAME))
    assert [d["id"] for d in docs] == [0]


def test_prompts_are_copied_not_rendered(tmp_path):
    """This stage must never construct a prompt -- parity with the greedy run is by byte identity."""
    path = score_file(tmp_path, {"mmlu_pro_law": [sample(7, "D", "custom-extract")]})
    docs, _ = extract_prompts(path, trace_sft.adapter_for(mmlu_pro.TASK_NAME))
    assert docs[0]["prompt"] == "prompt-7"
    assert docs[0]["options"] == ["choice-%s" % letter for letter in "ABCDEFGHIJ"]


def test_mmlu_pro_prompt_preparation_requires_structured_options(tmp_path):
    row = sample(7, "D", "custom-extract")
    row["doc"].pop("options")
    path = score_file(tmp_path, {"mmlu_pro_law": [row]})

    with pytest.raises(ValueError, match="options"):
        extract_prompts(path, trace_sft.adapter_for(mmlu_pro.TASK_NAME))


def test_prompt_preparation_requires_task_native_predictor_text(tmp_path):
    row = sample(7, "D", "custom-extract")
    row.pop("pproute_predictor_text")
    path = score_file(tmp_path, {"mmlu_pro_law": [row]})
    with pytest.raises(ValueError, match="predictor-text stamp"):
        extract_prompts(path, trace_sft.adapter_for(mmlu_pro.TASK_NAME))

    row["pproute_predictor_text"] = "Question: target only\nAnswer:"
    path = score_file(tmp_path, {"mmlu_pro_law": [row]}, name="stamped_scores.jbl")
    docs, _ = extract_prompts(path, trace_sft.adapter_for(mmlu_pro.TASK_NAME))
    assert docs[0]["predictor_text"] == "Question: target only\nAnswer:"


def test_the_wrong_filter_yields_nothing(tmp_path):
    path = score_file(tmp_path, {"mmlu_pro_law": [sample(0, "C", "custom-extract")]})
    with pytest.raises(ValueError, match="flexible-extract"):
        extract_prompts(path, trace_sft.adapter_for(gsm8k.TASK_NAME))


def test_disagreeing_stop_strings_refused(tmp_path):
    path = tmp_path / "eval_scores.jbl"
    joblib.dump({"samples": {"a": [sample(0, "C", "custom-extract")],
                             "b": [sample(0, "C", "custom-extract")]},
                 "configs": {"a": {"generation_kwargs": {"until": ["Question:"]}},
                             "b": {"generation_kwargs": {"until": ["Q:"]}}}}, str(path))
    with pytest.raises(ValueError, match="stop strings"):
        extract_prompts(str(path), trace_sft.adapter_for(mmlu_pro.TASK_NAME))


def test_sampling_shards_are_balanced_and_restore_order():
    docs = [{"id": i} for i in range(11)]
    shards = [consistency.shard_documents(docs, 3, index) for index in range(3)]
    restored = [shard[offset] for offset in range(4) for shard in shards
                if offset < len(shard)]
    assert [len(shard) for shard in shards] == [4, 4, 3]
    assert restored == docs
    with pytest.raises(ValueError, match="num_shards"):
        consistency.shard_documents(docs, 0, 0)
    with pytest.raises(ValueError, match="shard_index"):
        consistency.shard_documents(docs, 3, 3)
