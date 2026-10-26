"""A reasoning model must not be scored with its thinking block left open.

`Qwen/Qwen3.5-9B` ends its generation prompt `<|im_start|>assistant\\n<think>\\n`. lm_eval scores a
multiple-choice option as the continuation of that prompt, so `f_Z` would be the model's confidence
in the first token of its *thoughts* rather than in its answer -- the quantity the whole experiment
is about. Qwen's thinking switch must be disabled for multiple-choice scoring.

The switch cannot be passed at a call site: lm_eval calls `apply_chat_template` itself, inside
request construction, so the entry script never gets to. The template is rewritten instead, which
also means a checkpoint saved by finetune_model carries it and training cannot disagree with eval.
"""
import pytest

from pproute.models import chat_scaffolding, disable_chat_template_thinking


class _Tok:
    """Stands in for a Qwen-shaped template: opens a thought unless enable_thinking is false."""

    all_special_tokens = []
    chat_template = "<tmpl>"          # only its presence is checked; rendering is done below

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=None, **kw):
        # The prefixed `{%- set enable_thinking = false %}` is modelled by honouring the flag the
        # template itself carries once it has been rewritten.
        if "set enable_thinking = false" in self.chat_template:
            enable_thinking = False
        tail = "<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n"
        return "user: " + messages[0]["content"] + "\nassistant\n" + tail


class _Plain:
    """No thinking branch at all: the flag makes no difference."""

    all_special_tokens = []
    chat_template = "<tmpl>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return "user: " + messages[0]["content"] + "\nassistant\n"


class _Stubborn(_Tok):
    """A template that advertises the switch but ignores the rewrite -- must raise, not pass."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=None, **kw):
        tail = "<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n"
        return "user: " + messages[0]["content"] + "\nassistant\n" + tail


def test_thinking_branch_is_closed():
    tok = _Tok()
    assert disable_chat_template_thinking(tok) is True
    _, suffix = chat_scaffolding(tok)
    assert suffix.endswith("<think>\n\n</think>\n\n")


def test_is_idempotent():
    """chat_scaffolding also calls this, so it runs more than once on the same tokenizer."""
    tok = _Tok()
    assert disable_chat_template_thinking(tok) is True
    assert disable_chat_template_thinking(tok) is False
    assert tok.chat_template.count("set enable_thinking = false") == 1


def test_template_without_thinking_is_untouched():
    tok = _Plain()
    before = tok.chat_template
    assert disable_chat_template_thinking(tok) is False
    assert tok.chat_template == before


def test_no_template_at_all_is_untouched():
    class _Bare:
        chat_template = None
    assert disable_chat_template_thinking(_Bare()) is False


def test_a_rewrite_that_does_not_take_raises():
    """Silently leaving the thought open is the failure this exists to prevent, so it must not be
    reachable by a template whose guard the rewrite does not reach."""
    with pytest.raises(ValueError, match="did not take effect"):
        disable_chat_template_thinking(_Stubborn())


def test_scaffolding_closes_thinking_without_an_explicit_call():
    """The predictor loads the scored model's tokenizer bare, so the wrapper it derives has to
    match a prompt scored with thinking off -- otherwise stripping fails on every example."""
    tok = _Tok()
    _, suffix = chat_scaffolding(tok)          # no disable_chat_template_thinking() call first
    assert suffix.endswith("<think>\n\n</think>\n\n")
