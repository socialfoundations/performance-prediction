"""Length normalisation for acc_norm tasks.

lm_eval's acc_norm is `argmax(lls / len(doc_to_choice(doc)[i]))` -- the length of the *choice*.
The string it actually scored is `target_delimiter + choice`, so deriving the denominator from
`arguments[i][1]` is off by the delimiter, and the delimiter is not recoverable from the score
file either: lm_eval silently drops it to "" under a chat template without changing the recorded
task config. On a real 6,000-sample score file the two rules disagree on ~5% of per-prompt labels,
which would move Z, f_Z, L_Y and every curve built on them away from the benchmark's own metric.

Same shape of problem as the winogrande gold index (test_gold_index.py): a per-option fact only
the lm-eval task object knows. So it is stamped at eval time and refused when absent, and both
halves are covered here. Everything is offline; samples are built in the shape lm_eval dumps.
"""
import os

import joblib as jbl
import numpy as np
import pytest

from pproute import scores
from pproute.evaluator import stamp_gold_index
from pproute.scores import CHOICE_LENS_KEY, GOLD_KEY


# hellaswag is in ACC_NORM_TASKS; commonsense_qa is not.
ACC_NORM_TASK = "hellaswag"
PLAIN_TASK = "commonsense_qa"
DELIMITER = " "
# Chosen so the two rules genuinely disagree, with room to spare on both sides:
#   lls / len(choice)             = [-5.0, -6.0, -4.0]   -> option 2, lm_eval's answer
#   lls / len(" " + choice)       = [-2.5, -4.0, -3.2]   -> option 0, the old behaviour
# Gold is option 2, so the difference is not cosmetic: the same sample scores Z=1 under the
# benchmark's own metric and Z=0 under the delimiter-inflated one.
CHOICES = ["a", "bb", "cccc"]
LLS = (-5.0, -12.0, -16.0)
GOLD = 2


def _sample(choices=CHOICES, lls=LLS, stamp_lens=True, gold=GOLD, doc_id=0):
    it = {"doc_id": doc_id,
          "doc": {"query": "the question"},
          "arguments": [("the question", DELIMITER + c) for c in choices],
          "filtered_resps": [(ll, False) for ll in lls],
          GOLD_KEY: gold}
    if stamp_lens:
        it[CHOICE_LENS_KEY] = [len(c) for c in choices]
    return it


def _dump(tmp_path, task, samples):
    path = os.path.join(str(tmp_path), "%s_scores.jbl" % task)
    jbl.dump({"samples": {task: samples}}, path)
    return path


def _lm_eval_acc_norm_pred(choices, lls):
    """lm_eval's own rule, transcribed from ConfigurableTask.process_results."""
    completion_len = np.array([float(len(c)) for c in choices])
    return int(np.argmax(np.array(lls) / completion_len))


# --------------------------------------------------------------- the stamp (eval side)
class _FakeTask:
    multiple_input = 0

    @staticmethod
    def get_config(key):
        return "multiple_choice" if key == "output_type" else None

    @staticmethod
    def doc_to_text(doc):
        return doc["query"]

    @staticmethod
    def doc_to_target(_doc):
        return CHOICES[0]

    @staticmethod
    def doc_to_choice(_doc):
        return list(CHOICES)


def test_stamp_records_choice_lengths_not_scored_request_lengths():
    samples = [{"doc_id": 0, "doc": {"query": "the question"}}]
    stamp_gold_index({"samples": {ACC_NORM_TASK: samples}}, {ACC_NORM_TASK: _FakeTask})
    assert samples[0][CHOICE_LENS_KEY] == [len(c) for c in CHOICES]
    # The point of the fix: NOT the length of the string lm_eval scored.
    assert samples[0][CHOICE_LENS_KEY] != [len(DELIMITER + c) for c in CHOICES]


def test_stamp_records_choice_lengths_for_plain_acc_tasks_too():
    """Stamped for every MC task, so adding one to ACC_NORM_TASKS later does not invalidate
    scores that have already been paid for."""
    samples = [{"doc_id": 0, "doc": {"query": "the question"}}]
    stamp_gold_index({"samples": {PLAIN_TASK: samples}}, {PLAIN_TASK: _FakeTask})
    assert samples[0][CHOICE_LENS_KEY] == [len(c) for c in CHOICES]


# --------------------------------------------------------------- reading it back (scores side)
def test_acc_norm_matches_lm_evals_own_rule(tmp_path):
    """The regression. The two normalizers order these options differently, and the stamped one
    is the one that agrees with lm_eval."""
    path = _dump(tmp_path, ACC_NORM_TASK, [_sample()])
    got = scores.mc_examples(path)[0]

    assert got["pred"] == _lm_eval_acc_norm_pred(CHOICES, LLS)
    wrong = int(np.argmax(np.array(LLS) / np.array([len(DELIMITER + c) for c in CHOICES])))
    assert got["pred"] != wrong, "fixture must actually separate the two rules"
    # The label itself moves, not just the argmax: this is what would have desynced Z from the
    # benchmark's reported acc_norm.
    assert got["Z"] == 1.0 and got["gold"] == GOLD
    assert float(wrong == GOLD) == 0.0


def test_acc_norm_probabilities_use_the_stamped_lengths(tmp_path):
    path = _dump(tmp_path, ACC_NORM_TASK, [_sample()])
    got = scores.mc_examples(path)[0]
    assert got["pr"] == pytest.approx(
        scores.softmax(np.array(LLS) / np.array([float(len(c)) for c in CHOICES])))


def test_plain_acc_task_ignores_the_stamp_entirely(tmp_path):
    """A task whose official metric is raw acc must keep using raw log-likelihoods."""
    path = _dump(tmp_path, PLAIN_TASK, [_sample()])
    got = scores.mc_examples(path)[0]
    assert got["pr"] == pytest.approx(scores.softmax(np.array(LLS)))
    assert got["pred"] == 0


def test_unstamped_acc_norm_refuses_instead_of_approximating(tmp_path):
    """No correct guess exists here, so it names the key and says what to re-run rather than
    normalising by the wrong string."""
    path = _dump(tmp_path, ACC_NORM_TASK, [_sample(stamp_lens=False)])
    with pytest.raises(KeyError) as excinfo:
        scores.mc_examples(path)
    message = str(excinfo.value)
    assert CHOICE_LENS_KEY in message
    assert "evaluate_model" in message


def test_unstamped_plain_task_still_reads(tmp_path):
    """Backward compatibility: the 132 pre-stamp commonsense_qa files must keep working."""
    path = _dump(tmp_path, PLAIN_TASK, [_sample(stamp_lens=False)])
    assert scores.mc_examples(path)[0]["pred"] == 0


def test_stamped_length_count_must_match_the_scored_options(tmp_path):
    it = _sample()
    it[CHOICE_LENS_KEY] = [1, 2]
    path = _dump(tmp_path, ACC_NORM_TASK, [it])
    with pytest.raises(ValueError, match="choice length"):
        scores.mc_examples(path)
