"""The sampling-consistency arm's confidence rules.

These pins matter because the aggregation defines Z and f_Z for a whole confidence arm: a
silent change to tie-breaking or failure accounting would relabel documents and shift every
share, with nothing else failing.
"""
import json
import os

import numpy as np
import pytest

from pproute import consistency, mmlu_pro
from pproute.predictor import (build_consistency, input_length_diagnostics,
                               train_verifier)


def test_aggregate_majority_and_share():
    majority, share, counts = consistency.aggregate(
        ["8", "8", "9", "8"], task="gsm8k"
    )
    assert majority == "8" and share == 0.75
    assert counts == {"8": 3, "9": 1}


def test_request_seeds_are_typed_and_split_namespaced():
    seed = consistency.request_seed("endpoint", "eval", 1)
    assert seed == consistency.request_seed("endpoint", "eval", 1)
    assert seed != consistency.request_seed("endpoint", "test", 1)
    assert seed != consistency.request_seed("endpoint", "eval", "1")
    with pytest.raises(ValueError, match="split"):
        consistency.request_seed("endpoint", "train", 1)


def test_aggregate_tie_breaks_to_earliest_first_occurrence():
    majority, share, _ = consistency.aggregate(
        ["9", "8", "8", "9"], task="gsm8k"
    )
    assert majority == "9", "ties must break to the answer seen first in seeded sample order"
    assert share == 0.5


def test_gsm8k_failures_stay_in_denominator_but_cannot_be_the_prediction():
    majority, share, counts = consistency.aggregate(
        [None, None, None, "7"], task="gsm8k"
    )
    assert majority == "7"
    assert share == 0.25
    assert counts == {"7": 1, consistency.FAILURE: 3}


def test_mmlu_pro_failures_stay_in_denominator_but_cannot_be_the_prediction():
    majority, share, counts = consistency.aggregate(
        [None, None, None, "A"], task="mmlu_pro"
    )
    assert majority == "A"
    assert share == 0.25
    assert counts == {"A": 1, consistency.FAILURE: 3}


def test_gsm8k_all_failures_has_no_prediction_and_zero_confidence():
    majority, share, counts = consistency.aggregate([None, None], task="gsm8k")
    assert majority is None and share == 0.0
    assert counts == {consistency.FAILURE: 2}


def test_mmlu_pro_all_failures_has_no_prediction_and_zero_confidence():
    majority, share, counts = consistency.aggregate([None, None], task="mmlu_pro")
    assert majority is None and share == 0.0
    assert counts == {consistency.FAILURE: 2}


def test_aggregate_rejects_an_empty_normalized_answer():
    with pytest.raises(ValueError, match="must not be empty"):
        consistency.aggregate([""], task="gsm8k")


def _payload(tmp_path, rows, task=None):
    splits = [split for split in ("eval", "test")
              if any(row["split"] == split for row in rows)]
    meta = {"schema": consistency.CONF_SCHEMA,
            "k": consistency.K, "temperature": consistency.TEMPERATURE,
            "top_p": consistency.TOP_P, "max_new_tokens": consistency.MAX_NEW_TOKENS,
            "task": task or "gsm8k",
            "scores_dir": "/scores", "model_path": "/model",
            "endpoint_manifest": "/endpoint.json", "prompts_file": "/prompts.json",
            "stop": ["Question:"], "model": "org/model",
            "revision": "a" * 40, "endpoint": "before",
            "endpoint_identity": "b" * 64, "endpoint_manifest_sha256": "c" * 64,
            "scores_sha256": {split: "d" * 64 for split in splits},
            "prompts_sha256": "e" * 64, "splits": splits,
            "fingerprints": {split + "_split": split + "-fingerprint" for split in splits},
            "n_eval": sum(row["split"] == "eval" for row in rows),
            "n_test": sum(row["split"] == "test" for row in rows),
            "subset": {
                "method": "all",
                "eval_ids_sha256": consistency.ids_sha256(
                    [row for row in rows if row["split"] == "eval"]),
                "test_ids_sha256": consistency.ids_sha256(
                    [row for row in rows if row["split"] == "test"]),
            }}
    if task is not None:
        meta["task"] = task
    path = str(tmp_path / "conf_scores.json")
    consistency.write_conf_scores(path, meta, rows)
    return path


def _row(split, doc_id, answers, gold="8", trace=None,
         predictor_text="Question: 3+5?\nAnswer:", task="gsm8k", options=None):
    if len(answers) != consistency.K:
        assert consistency.K % len(answers) == 0
        answers = [answer for answer in answers for _ in range(consistency.K // len(answers))]
    majority, share, counts = consistency.aggregate(answers, task=task)
    categorized = [answer if answer is not None else consistency.FAILURE for answer in answers]
    first_seen = {answer: categorized.index(answer) for answer in set(categorized)}
    if trace is None and majority is not None:
        trace = ("steps... the answer is (%s)." % majority if task == "mmlu_pro"
                 else "steps... #### %s" % majority)
    row = {"split": split, "id": doc_id, "gold": gold, "predictor_text": predictor_text,
           "majority": majority, "share": share,
           "z": int(majority is not None and majority == gold),
           "counts": counts,
           "n_fail": counts.get(consistency.FAILURE, 0), "cap_hits": 0,
           "first_seen": first_seen, "finish_reasons": {"stop": consistency.K},
           "majority_trace": trace if majority is not None else None}
    if task == "mmlu_pro":
        options = options or ["option %s" % letter for letter in "ABCDEFGHIJ"]
        row["majority_option"] = (
            None if majority is None else mmlu_pro.selected_option_content(options, majority)
        )
    return row


def test_conf_scores_roundtrip_and_validation(tmp_path):
    rows = [_row("eval", 0, ["8", "8", "9", "8"]), _row("eval", 1, [None, None, None, None]),
            _row("test", 0, ["8", "7", "8", "8"])]
    path = _payload(tmp_path, rows)
    payload = consistency.read_conf_scores(path)
    assert len(payload["rows"]) == 3

    with pytest.raises(FileExistsError):
        consistency.write_conf_scores(path, payload["meta"], rows)   # create-once

    bad = [dict(rows[0], counts={"8": 3}), rows[1], rows[2]]         # counts do not sum to k
    path2 = str(tmp_path / "bad" )
    os.makedirs(path2)
    path2 = os.path.join(path2, "conf_scores.json")
    consistency.write_conf_scores(path2, payload["meta"], bad)
    with pytest.raises(ValueError, match="sum to k"):
        consistency.read_conf_scores(path2)








def test_current_schema_rejects_an_empty_answer_bucket(tmp_path):
    row = _row("eval", 0, [None, "8", "8", "8"])
    row["counts"][""] = row["counts"].pop(consistency.FAILURE)
    row["first_seen"][""] = row["first_seen"].pop(consistency.FAILURE)
    path = _payload(tmp_path, [row])

    with pytest.raises(ValueError, match="invalid counts"):
        consistency.read_conf_scores(path, expected_splits=("eval",))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(first_seen={"8": 0, "9": 0}), "first_seen"),
        (lambda row: row.update(cap_hits=1), "finish_reasons"),
        (lambda row: row.update(predictor_text=""), "gold/predictor text"),
        (lambda row: row.update(majority_trace=""), "majority trace"),
    ],
)
def test_conf_scores_rejects_internally_inconsistent_rows(tmp_path, mutation, message):
    row = _row("eval", 0, ["8", "8", "9", "8"])
    mutation(row)
    path = _payload(tmp_path, [row])
    with pytest.raises(ValueError, match=message):
        consistency.read_conf_scores(path, expected_splits=("eval",))


def test_mmlu_pro_confidence_requires_matching_trace_and_selected_option(tmp_path):
    row = _row("eval", 0, ["B", "B", "A", "B"], gold="A", task="mmlu_pro")
    row["majority_trace"] = "Reasoning. the answer is (A)."
    path = _payload(tmp_path, [row], task="mmlu_pro")
    with pytest.raises(ValueError, match="trace extracts as"):
        consistency.read_conf_scores(path, expected_splits=("eval",))

    row["majority_trace"] = "Reasoning. the answer is (B)."
    row.pop("majority_option")
    option_path = str(tmp_path / "option" / "conf_scores.json")
    os.makedirs(os.path.dirname(option_path))
    with open(path) as handle:
        meta = json.load(handle)["meta"]
    consistency.write_conf_scores(option_path, meta, [row])
    with pytest.raises(ValueError, match="majority option"):
        consistency.read_conf_scores(option_path, expected_splits=("eval",))


def test_conf_scores_rejects_nonfinite_json(tmp_path):
    path = tmp_path / "conf_scores.json"
    path.write_text('{"meta": {"temperature": NaN}, "rows": []}')
    with pytest.raises(ValueError, match="non-finite JSON"):
        consistency.read_conf_scores(path)


def test_conf_scores_writer_rejects_nonfinite_values(tmp_path):
    with pytest.raises(ValueError, match="JSON compliant"):
        consistency.write_conf_scores(
            str(tmp_path / "conf_scores.json"), {"value": float("nan")}, []
        )


def test_conf_scores_rejects_wrong_k(tmp_path):
    path = _payload(tmp_path, [_row("eval", 0, ["8"] * 4), _row("test", 0, ["8"] * 4)])
    with pytest.raises(ValueError, match="frozen k/temperature"):
        consistency.read_conf_scores(path, expected_k=64)


def test_conf_scores_rejects_wrong_frozen_protocol_or_label(tmp_path):
    rows = [_row("eval", 0, ["8"] * 4), _row("test", 0, ["8"] * 4)]
    path = _payload(tmp_path, rows)
    with open(path) as handle:
        payload = json.load(handle)

    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    protocol_path = str(protocol_dir / "conf_scores.json")
    consistency.write_conf_scores(
        protocol_path, dict(payload["meta"], temperature=0.8), payload["rows"])
    with pytest.raises(ValueError, match="temperature"):
        consistency.read_conf_scores(protocol_path)
    with pytest.raises(ValueError, match="frozen k/temperature"):
        consistency.read_conf_scores(protocol_path, expected_temperature=0.8)

    label_dir = tmp_path / "label"
    label_dir.mkdir()
    label_path = str(label_dir / "conf_scores.json")
    bad_rows = [dict(payload["rows"][0], z=0), payload["rows"][1]]
    consistency.write_conf_scores(label_path, payload["meta"], bad_rows)
    with pytest.raises(ValueError, match="z disagrees"):
        consistency.read_conf_scores(label_path)


def test_confidence_schema_audits_finish_reasons_and_tie_break(tmp_path):
    rows = [_row("eval", 0, ["A", "B", "B", "A"], gold="A", task="mmlu_pro")]
    path = _payload(tmp_path, rows, task="mmlu_pro")
    consistency.read_conf_scores(path, expected_splits=("eval",))

    with open(path) as handle:
        payload = json.load(handle)
    payload["rows"][0]["majority"] = "B"
    payload["rows"][0]["z"] = 0
    wrong = tmp_path / "wrong" / "conf_scores.json"
    wrong.parent.mkdir()
    consistency.write_conf_scores(str(wrong), payload["meta"], payload["rows"])
    with pytest.raises(ValueError, match="tie break"):
        consistency.read_conf_scores(str(wrong), expected_splits=("eval",))


def test_build_consistency_texts_include_distribution_and_answer(tmp_path):
    rows = [_row("eval", 0, ["8", "8", "9", "8"]),
            _row("eval", 1, [None, None, None, None]),
            _row("test", 5, ["7", "7", "7", "7"], gold="7")]
    path = _payload(tmp_path, rows)
    built = build_consistency(consistency.read_conf_scores(path))
    (tr_text, tr_Z, tr_share, tr_ids) = built["eval"]
    (te_text, te_Z, te_share, te_ids) = built["test"]
    assert tr_ids == [0, 1] and te_ids == [5]
    assert list(tr_Z) == [1.0, 0.0] and list(te_Z) == [1.0]
    assert list(tr_share) == [0.75, 0.0] and list(te_share) == [1.0]
    assert tr_text[0].rstrip().endswith("Proposed answer: 8")
    assert "Sampled answers: 8=0.750 9=0.250" in tr_text[0]
    assert "Proposed solution:" in tr_text[0]
    # an all-failure GSM8K row yields no proposed answer or trace
    assert tr_text[1].rstrip().endswith("Proposed answer: <none>")
    assert "Proposed solution:" not in tr_text[1]


def test_mmlu_pro_predictor_separates_failures_and_includes_selected_option(tmp_path):
    prompt = "Question: Which one?\nOptions:\nA. Alpha\nB. Beta\nAnswer:"
    rows = [
        _row(
            "eval", 0, [None, None, None, "B"], gold="A", task="mmlu_pro",
            predictor_text=prompt, options=["Alpha", "Beta"],
        ),
        _row(
            "eval", 1, [None, None], gold="A", task="mmlu_pro",
            predictor_text=prompt, options=["Alpha", "Beta"],
        ),
    ]
    path = _payload(tmp_path, rows, task="mmlu_pro")

    texts = build_consistency(consistency.read_conf_scores(path, expected_splits=("eval",)))["eval"][0]
    text = texts[0]

    assert "Extraction failures: 96/128" in text
    assert "Sampled answers: B=32/128" in text
    assert consistency.FAILURE not in text
    assert text.endswith("Proposed answer: B. Beta")
    assert "Proposed solution:" not in texts[1]
    assert "Extraction failures: 128/128" in texts[1]
    assert "Sampled answers: <none>" in texts[1]
    assert texts[1].endswith("Proposed answer: <none>")


def test_build_consistency_target_only_is_an_explicit_verifier_without_demos(tmp_path):
    prompt = """The following are multiple choice questions (with answers).
Question: Misleading demonstration?
Options:
A. Yes
B. No
Answer: Let's think step by step. The answer is (A).

Question: Target question?
Options:
A. First
B. Second
Answer: Let's think step by step."""
    options = ["First", "Second"]
    rows = [_row("eval", 0, ["B", "B", "A", "B"], gold="A",
                 trace="Target reasoning. The answer is (B).", predictor_text=prompt,
                 task="mmlu_pro", options=options),
            _row("test", 1, ["A"] * 4, gold="A", predictor_text=prompt,
                 task="mmlu_pro", options=options)]
    path = _payload(tmp_path, rows, task="mmlu_pro")

    target = build_consistency(consistency.read_conf_scores(path))["eval"][0][0]
    assert target.startswith("Instruct: Predict whether the proposed answer")
    assert "\nQuery:\nQuestion: Target question?" in target
    assert "Misleading demonstration?" not in target
    assert "multiple choice questions (with answers)" not in target
    assert "Answer: Let's think step by step." not in target
    assert "Proposed solution:\nTarget reasoning. The answer is (B)." in target
    assert "Extraction failures: 0/128" in target
    assert "Sampled answers: B=96/128 A=32/128" in target
    assert target.endswith("Proposed answer: B. Second")
    assert "Gold" not in target and "correct answer" not in target


def test_build_consistency_gsm8k_target_only_matches_active_prompt_shape(tmp_path):
    prompt = """Solve the problem and put the final number after four hash marks.

Question: Misleading demonstration?
Answer: reasoning #### 1

Question: Janet has three apples and buys five more. How many apples?
Answer:"""
    rows = [_row(
        "eval", 17, ["8", "8", "9", "8"], gold="gold-is-not-input",
        trace="Three plus five is eight. #### 8", predictor_text=prompt,
    )]
    path = _payload(tmp_path, rows, task="gsm8k")

    target = build_consistency(consistency.read_conf_scores(path, expected_splits=("eval",)))["eval"][0][0]

    assert target == (
        "Instruct: Predict whether the proposed answer to this math problem is correct. "
        "Use the target problem, proposed solution, and sampled-answer frequencies as evidence.\n"
        "Query:\nQuestion: Janet has three apples and buys five more. How many apples?\n\n"
        "Proposed solution:\nThree plus five is eight. #### 8\n\n"
        "Sampled answers: 8=0.750 9=0.250\nProposed answer: 8"
    )
    assert "Misleading demonstration" not in target
    assert "gold-is-not-input" not in target


def test_predictor_input_length_diagnostics_detects_overflow():
    class Tokenizer:
        def __call__(self, texts, **_kwargs):
            return {"input_ids": [text.split() for text in texts]}

    diagnostics = input_length_diagnostics(
        Tokenizer(), ["one two", "one two three four"], max_len=3, batch_size=1)
    assert diagnostics == {"n": 2, "max_tokens": 4, "over_max_len": 1, "max_len": 3}


def test_predictor_rejects_invalid_microbatch_before_loading_model():
    with pytest.raises(ValueError, match="divisible by micro_bs"):
        train_verifier(
            "unused", ["train"], np.array([1.0]), ["test"], np.array([0.0]),
            emb_revision="a" * 40, tr_lfz=np.array([0.0]), te_lfz=np.array([0.0]),
            bs=16, micro_bs=3, fit_indices=[0], holdout_indices=[],
        )


def test_bucket_estimates_share_one_answer_distribution():
    """f_Z and L_Y come from one normalized distribution; the correctness loss cannot exceed L_Y."""
    cases = [  # counts, gold, majority, majority count, gold mass in counts
        ({"A": 128}, "A", "A", 128, 128),
        ({"A": 128}, "B", "A", 128, .5),                       # unanimous and wrong: 5.55 nats on both sides
        ({"A": 64, "B": 64}, "B", "A", 64, 64),
        ({consistency.FAILURE: 100, "A": 28}, "A", "A", 28, 28),
        ({consistency.FAILURE: 128}, "A", None, 0, .5),       # no valid answer: no prediction, f_Z = 0
    ]
    for counts, gold, majority, majority_count, gold_mass in cases:
        fz, loss = consistency.bucket_estimates(dict(counts=counts, gold=gold, majority=majority))
        assert fz == pytest.approx(majority_count / 128.5)
        assert loss == pytest.approx(-np.log(gold_mass / 128.5))
        assert (sum(counts.values()) + .5) / 128.5 == pytest.approx(1.0)
        correct = majority == gold
        assert (-np.log(fz) if correct else -np.log(1 - fz)) <= loss + 1e-12
    # integer gold labels match string count keys
    assert consistency.bucket_estimates(dict(counts={"7": 3, "8": 125}, gold=7, majority="8"))[1] == \
        pytest.approx(-np.log(3 / 128.5))
    assert consistency.anchor_logit(0.0) == pytest.approx(np.log(.5 / 128))
    assert consistency.anchor_logit(128 / 128.5) == pytest.approx(np.log(128 / .5))
