"""Resolving the gold option index from a dumped lm_eval sample.

The bug this guards is the second half of the winogrande failure. `render_options` (covered in
test_render_options.py) fixes the predictor's *features*; this file covers *which option was
correct*, which broke separately and harder: for `multiple_input` tasks lm_eval stores
`target = doc_to_target(doc)`, and that is the continuation shared by every option, not the answer.
The gold index comes from `doc_to_text(doc)` and is not written into the sample at all, so matching
`target` against the options raised ValueError and no winogrande number could be produced.

The fix has two halves, both tested here:
  * `pproute.evaluator` stamps the index while the task object is still in scope;
  * `pproute.scores` requires that stamp and refuses rather than guessing from serialized fields.

Everything here is offline: the samples are built by hand in exactly the shape lm_eval dumps.
"""
import os
import random

import joblib as jbl
import numpy as np
import pytest

from pproute import scores
from pproute.evaluator import (
    _gold_index as gold_from_task,
    build_predictor_contexts,
    stamp_gold_index,
)
from pproute.scores import GOLD_KEY, PREDICTOR_TEXT_KEY


# --------------------------------------------------------------------- sample builders
WINO_DOC = {"sentence": "Sarah was a much better surgeon than Maria so _ always got the hard cases.",
            "option1": "Sarah", "option2": "Maria", "answer": "1"}


def _wino_sample(stamp_gold=True, lls=(-3.2, -4.1)):
    """A winogrande sample exactly as lm_eval writes it: two contexts, one shared continuation,
    and `target` holding that continuation rather than the answer."""
    idx = WINO_DOC["sentence"].index("_")
    ctxs = [WINO_DOC["sentence"][:idx] + o
            for o in (WINO_DOC["option1"], WINO_DOC["option2"])]
    cont = WINO_DOC["sentence"][idx + 1:].strip()
    it = {"doc_id": 0, "doc": WINO_DOC, "target": cont,
          "arguments": [(c, " " + cont) for c in ctxs],
          "filtered_resps": [(ll, False) for ll in lls]}
    if stamp_gold:
        it[GOLD_KEY] = 0                                       # answer "1" -> index 0
    return it


def _csqa_sample(gold_letter="B", stamp_gold=None):
    """A commonsense_qa sample: one shared context, one continuation per option, letter target."""
    labels = ["A", "B", "C"]
    doc = {"question": "Where do you keep a fork?",
           "choices": {"label": labels, "text": ["garage", "kitchen", "ocean"]},
           "answerKey": gold_letter}
    ctx = "Q: Where do you keep a fork?\nA:"
    it = {"doc_id": 0, "doc": doc, "target": gold_letter,
          "arguments": [(ctx, " " + t) for t in doc["choices"]["text"]],
          "filtered_resps": [(-2.0, False), (-1.0, False), (-5.0, False)]}
    if stamp_gold is not None:
        it[GOLD_KEY] = stamp_gold
    return it


def _dump(tmp_path, task, samples):
    path = os.path.join(str(tmp_path), "%s_scores.jbl" % task)
    jbl.dump({"samples": {task: samples}}, path)
    return path


# --------------------------------------------------------------------- the stamp (eval side)
class _FakeWinograndeTask:
    """The three doc_to_* methods lm_eval's winogrande task defines, and multiple_input set."""
    multiple_input = 2

    @staticmethod
    def doc_to_text(doc):                       # the GOLD INDEX, per lm_eval's Winograd shape
        return {"1": 0, "2": 1}[doc["answer"]]

    @staticmethod
    def doc_to_target(doc):                     # the SHARED CONTINUATION, not the answer
        return doc["sentence"][doc["sentence"].index("_") + 1:].strip()

    @staticmethod
    def doc_to_choice(doc):
        i = doc["sentence"].index("_")
        return [doc["sentence"][:i] + o for o in (doc["option1"], doc["option2"])]


class _FakeStandardTask:
    """A standard MC task whose doc_to_target names the option."""
    multiple_input = 0

    @staticmethod
    def doc_to_text(doc):
        return doc["question"]

    @staticmethod
    def doc_to_target(doc):
        return doc["choices"]["text"][doc["choices"]["label"].index(doc["answerKey"])]

    @staticmethod
    def doc_to_choice(doc):
        return list(doc["choices"]["text"])


class _FakeGenerativeTask:
    docs = [(7, {"question": "target only"}), (9, {"question": "another target"})]

    def set_fewshot_seed(self, seed):
        self.rnd = random.Random(seed)

    def doc_iterator(self, rank=0, world_size=1):
        assert (rank, world_size) == (0, 1)
        return iter(self.docs)

    @staticmethod
    def get_config(key):
        return "generate_until" if key == "output_type" else None

    @staticmethod
    def doc_to_text(doc):
        return "Question: %s\nAnswer:" % doc["question"]

    @staticmethod
    def doc_to_prefix(_doc):
        return None

    def fewshot_context(
        self, doc, num_fewshot, system_instruction=None, apply_chat_template=False,
        fewshot_as_multiturn=False, chat_template=None, gen_prefix=None,
    ):
        assert system_instruction is None
        assert apply_chat_template is False
        assert fewshot_as_multiturn is False
        assert chat_template is None
        assert gen_prefix is None
        demonstrations = [
            "Question: demo %d\nAnswer: worked answer %d" % (value, value)
            for value in self.rnd.sample(range(100), num_fewshot)
        ]
        return "\n\n".join(demonstrations + [self.doc_to_text(doc)])


def test_stamp_uses_doc_to_text_for_multiple_input():
    """The whole point: for multiple_input the index must come from doc_to_text, and doc_to_target
    must NOT be treated as an option."""
    assert gold_from_task(_FakeWinograndeTask, WINO_DOC) == 0
    doc2 = dict(WINO_DOC, answer="2")
    assert gold_from_task(_FakeWinograndeTask, doc2) == 1
    assert _FakeWinograndeTask.doc_to_target(WINO_DOC) not in _FakeWinograndeTask.doc_to_choice(WINO_DOC)


def test_stamp_maps_named_option_for_standard_layout():
    doc = _csqa_sample()["doc"]
    assert gold_from_task(_FakeStandardTask, doc) == 1          # "kitchen"


def test_stamp_marks_out_of_range_gold():
    """Out of range is stamped -100 (lm_eval's own convention) rather than silently wrapping."""
    doc = dict(_csqa_sample()["doc"])
    doc["answerKey"] = "A"
    bad_task = type("T", (_FakeStandardTask,), {
        "doc_to_target": staticmethod(lambda d: "not an option")})
    assert gold_from_task(bad_task, doc) == -100


def test_stamp_preserves_full_normalized_fewshot_context_for_raw_and_chat():
    task = _FakeGenerativeTask()
    contexts = build_predictor_contexts({"fixture": task}, 5, 1234)
    assert build_predictor_contexts({"fixture": task}, 5, 1234) == contexts
    normalized = contexts["fixture"][7]
    assert normalized.count("Question:") == 6
    assert normalized != task.doc_to_text(task.docs[0][1])
    # The helper's final seed reset leaves lm-eval ready to reproduce the same first prompt.
    assert task.fewshot_context(task.docs[0][1], num_fewshot=5) == normalized

    for rendered in (normalized, "<chat>five demo turns + target</chat>"):
        sample = {
            "doc_id": 7,
            "doc": {"question": "target only"},
            "arguments": [(rendered, {})],
            "exact_match": 1.0,
        }
        other = {
            "doc_id": 9,
            "doc": {"question": "another target"},
            "arguments": [(rendered, {})],
            "exact_match": 1.0,
        }
        results = {"samples": {"fixture": [sample, other]}}
        stamp_gold_index(results, {"fixture": task}, contexts)
        assert sample[PREDICTOR_TEXT_KEY] == normalized


def test_stamp_rejects_inexact_predictor_context_id_map():
    task = _FakeGenerativeTask()
    contexts = build_predictor_contexts({"fixture": task}, 5, 1234)
    results = {"samples": {"fixture": [{"doc_id": 7, "doc": task.docs[0][1]}]}}
    with pytest.raises(ValueError, match="do not exactly match"):
        stamp_gold_index(results, {"fixture": task}, contexts)


def test_stamp_rejects_missing_generative_samples():
    task = _FakeGenerativeTask()
    contexts = build_predictor_contexts({"fixture": task}, 5, 1234)
    with pytest.raises(ValueError, match="no per-sample records"):
        stamp_gold_index({}, {"fixture": task}, contexts)


def test_stamp_accepts_two_filter_rows_for_the_same_doc():
    task = _FakeGenerativeTask()
    contexts = build_predictor_contexts({"fixture": task}, 5, 1234)
    samples = [
        {"doc_id": 7, "doc": task.docs[0][1], "filter": "flexible-extract"},
        {"doc_id": 7, "doc": task.docs[0][1], "filter": "strict-match"},
        {"doc_id": 9, "doc": task.docs[1][1], "filter": "flexible-extract"},
        {"doc_id": 9, "doc": task.docs[1][1], "filter": "strict-match"},
    ]
    stamp_gold_index({"samples": {"fixture": samples}}, {"fixture": task}, contexts)
    assert [row[PREDICTOR_TEXT_KEY] for row in samples] == [
        contexts["fixture"][7], contexts["fixture"][7],
        contexts["fixture"][9], contexts["fixture"][9],
    ]


def test_predictor_context_builder_rejects_target_only_five_shot():
    task = _FakeGenerativeTask()
    task.fewshot_context = lambda doc, **_kwargs: task.doc_to_text(doc)
    with pytest.raises(ValueError, match="dropped all 5 demonstrations"):
        build_predictor_contexts({"fixture": task}, 5, 1234)


def test_predictor_context_builder_respects_a_task_frozen_at_zero_shot():
    task = _FakeGenerativeTask()
    native_five_shot = "\n\n".join(
        ["Question: native demo %d\nAnswer: worked" % i for i in range(5)]
        + [task.doc_to_text(task.docs[0][1])]
    )
    task.docs = [(7, {"question": "target only"})]
    task.doc_to_text = lambda _doc: native_five_shot
    task.get_config = lambda key: {
        "output_type": "generate_until",
        "num_fewshot": 0,
    }.get(key)
    contexts = build_predictor_contexts({"fixture": task}, 5, 1234)
    assert contexts == {"fixture": {7: native_five_shot}}


# --------------------------------------------------------------------- reading it back (scores side)
def test_winogrande_scores_when_stamped(tmp_path):
    """With the stamp present, winogrande scores end to end -- Z, f_Z and L_Y all defined."""
    path = _dump(tmp_path, "winogrande", [_wino_sample(stamp_gold=True)])
    ex = scores.mc_examples(path)
    assert len(ex) == 1
    e = ex[0]
    assert e["layout"] == "multiple_input"
    assert e["conts"] == ["Sarah", "Maria"]
    assert e["gold"] == 0
    assert e["pred"] == 0                       # -3.2 > -4.1
    assert e["Z"] == 1.0
    assert 0.0 < e["fz"] < 1.0
    assert e["L"] == pytest.approx(-np.log(e["pr"][0]), abs=1e-9)


def test_winogrande_prediction_follows_likelihoods(tmp_path):
    """A wrong argmax must come out as Z=0, not as an exception or a silent 1."""
    path = _dump(tmp_path, "winogrande", [_wino_sample(stamp_gold=True, lls=(-5.0, -0.1))])
    e = scores.mc_examples(path)[0]
    assert e["pred"] == 1 and e["gold"] == 0 and e["Z"] == 0.0


def test_unstamped_multiple_input_refuses_instead_of_guessing(tmp_path):
    """This is the regression. Before the fix it raised 'substring not found' from a letter lookup;
    now it names the missing key and says what to re-run."""
    path = _dump(tmp_path, "winogrande", [_wino_sample(stamp_gold=False)])
    with pytest.raises(KeyError) as excinfo:
        scores.mc_examples(path)
    message = str(excinfo.value)
    assert GOLD_KEY in message
    assert "evaluate_model" in message


def test_stamp_wins_over_target(tmp_path):
    """When both are present the stamp is authoritative -- it is the one computed from the task."""
    path = _dump(tmp_path, "commonsense_qa", [_csqa_sample(stamp_gold=2)])
    assert scores.mc_examples(path)[0]["gold"] == 2


def test_out_of_range_gold_is_refused_with_an_actionable_message(tmp_path):
    """`stamp_gold_index` writes -100 when the task's gold is not among its own choices. Reading
    it back used to be a bare numpy IndexError naming neither the sample nor the cause."""
    path = _dump(tmp_path, "commonsense_qa", [_csqa_sample(stamp_gold=-100)])
    with pytest.raises(ValueError) as excinfo:
        scores.mc_examples(path)
    message = str(excinfo.value)
    assert "-100" in message
    assert "evaluate_model" in message


def test_out_of_range_gold_is_refused_on_a_task_with_over_100_options(tmp_path):
    """The half that was silent rather than loud: pr[-100] is a perfectly valid index once a task
    has more than 100 options, so the sample was scored against option 50 and produced a plausible
    Z and L_Y from a gold that does not exist."""
    it = {"doc_id": 0, "doc": {}, "target": 0,
          "arguments": [("Q", " c%d" % i) for i in range(150)],
          "filtered_resps": [(-float(i), False) for i in range(150)],
          GOLD_KEY: -100}
    path = _dump(tmp_path, "some_task", [it])
    with pytest.raises(ValueError, match="outside the 150 option"):
        scores.mc_examples(path)


def test_gold_past_the_last_option_is_refused_too(tmp_path):
    path = _dump(tmp_path, "commonsense_qa", [_csqa_sample(stamp_gold=3)])
    with pytest.raises(ValueError, match="outside the 3 option"):
        scores.mc_examples(path)


def test_multiple_target_is_refused(tmp_path):
    """A set of acceptable answers needs a deliberate decision about Z, f_Z and L_Y, so it raises
    rather than quietly scoring against the first one."""
    path = _dump(tmp_path, "some_task", [_csqa_sample(stamp_gold=[0, 1])])
    with pytest.raises(NotImplementedError):
        scores.mc_examples(path)
