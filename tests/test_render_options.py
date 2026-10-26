"""Layout detection for lm_eval multiple-choice requests (pproute.scores.render_options).

Run with pytest; add ``--live`` to also check real lm_eval tasks.

The bug this guards: winogrande is `multiple_input`, so its two candidates are two *contexts*
sharing one continuation. Reading it as standard multiple choice yields option A's context plus
the same continuation twice -- the predictor then cannot distinguish the options, its edge
collapses to ~0, and nothing raises.
"""
import pytest

from pproute.scores import fill_option, render_options


def test_standard():
    """CommonsenseQA shape: shared context, one continuation per option."""
    args = [("Q: where do you keep a fork?\nA:", " kitchen"),
            ("Q: where do you keep a fork?\nA:", " garage"),
            ("Q: where do you keep a fork?\nA:", " ocean")]
    q, opts, layout = render_options(args)
    assert layout == "standard"
    assert q == args[0][0]
    assert opts == [" kitchen", " garage", " ocean"]
    assert len(set(opts)) == 3
    assert fill_option(q, opts[0], layout) == args[0][0] + " kitchen"


def test_multiple_input():
    """WinoGrande shape: one continuation, one context per option."""
    args = [("Sarah was a much better surgeon than Maria so Sarah", " always got the easier cases."),
            ("Sarah was a much better surgeon than Maria so Maria", " always got the easier cases.")]
    q, opts, layout = render_options(args)
    assert layout == "multiple_input"
    assert opts == ["Sarah", "Maria"]
    assert len(set(opts)) == 2
    assert "___" in q
    assert q.endswith("always got the easier cases.")
    assert q == "Sarah was a much better surgeon than Maria so ___ always got the easier cases."
    filled = fill_option(q, opts[0], layout)
    assert filled == "Sarah was a much better surgeon than Maria so Sarah always got the easier cases."
    assert "___" not in filled


def test_mid_sentence_blank():
    """The differing span is the trailing one because doc_to_choice cuts at the blank."""
    args = [("They were worried the wine would ruin the bed and the blanket, but the blanket",
             " was't ruined."),
            ("They were worried the wine would ruin the bed and the blanket, but the bed",
             " was't ruined.")]
    _, opts, _ = render_options(args)
    assert opts == ["blanket", "bed"]


def test_rejects_unknown_layout():
    """Anything that is neither layout must raise rather than silently corrupt features."""
    with pytest.raises(ValueError):
        render_options([("ctx a", " cont a"), ("ctx b", " cont b")])   # both vary
    with pytest.raises(ValueError):
        render_options([("same ctx", " same cont"), ("same ctx", " same cont")])  # neither varies


@pytest.mark.live
def test_live():
    """Check against instances lm_eval actually builds."""
    from lm_eval.tasks import TaskManager, get_task_dict
    for name, expect in [("winogrande", "multiple_input"), ("commonsense_qa", "standard")]:
        t = get_task_dict([name], TaskManager())[name]
        t.build_all_requests(apply_chat_template=False)
        k = len(t.doc_to_choice(list(t.eval_docs)[0]))
        args = [i.arguments for i in t.instances[:k]]
        q, opts, layout = render_options(args)
        assert layout == expect
        assert len(set(opts)) == k
        assert q.strip()
