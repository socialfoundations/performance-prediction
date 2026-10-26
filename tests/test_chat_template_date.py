"""A chat template must not render "today" into the prompt.

`swiss-ai/Apertus-*-Instruct` emits "Current date: <today>" via jinja's `strftime_now`. Both axes of
this experiment are before-minus-after differences within one model, so the protocol has to be held
fixed across a model's own two endpoints -- and those endpoints are scored on different days. Left
alone, the before and after prompts differ by a string the model can read, and nothing in the score
file records it.

`pin_chat_template_date` rewrites the call to a literal so the substitution is visible on the
tokenizer afterwards, rather than patching `datetime.now` and leaving no trace.
"""
import pytest

from pproute.models import PINNED_DATE, pin_chat_template_date


class _Tok:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


# The shape Apertus actually ships, reduced to the part that matters.
APERTUS = ("{{ 'You are Apertus.\\nKnowledge cutoff: 2024-04\\nCurrent date: ' "
           "+ strftime_now('%Y-%m-%d') + end_system_token }}")


def test_pins_the_apertus_shape():
    t = _Tok(APERTUS)
    assert pin_chat_template_date(t) == 1
    assert "strftime_now" not in t.chat_template
    assert "'2026-01-01'" in t.chat_template
    assert PINNED_DATE == "2026-01-01", "the pinned date is part of the prompt; it must not move"


def test_is_idempotent():
    t = _Tok(APERTUS)
    pin_chat_template_date(t)
    once = t.chat_template
    assert pin_chat_template_date(t) == 0, "a second pass must find nothing left to pin"
    assert t.chat_template == once


def test_leaves_everything_else_alone():
    t = _Tok(APERTUS)
    pin_chat_template_date(t)
    assert t.chat_template == APERTUS.replace("strftime_now('%Y-%m-%d')", "'2026-01-01'")


def test_no_template_and_no_strftime_are_both_no_ops():
    assert pin_chat_template_date(_Tok(None)) == 0
    plain = "{{ messages[0].content }}"
    t = _Tok(plain)
    assert pin_chat_template_date(t) == 0
    assert t.chat_template == plain


def test_honours_the_templates_own_format_string():
    """A template asking for "%B %d, %Y" gets that, not an ISO date it never requested."""
    t = _Tok("{{ strftime_now('%B %d, %Y') }}")
    pin_chat_template_date(t)
    assert t.chat_template == "{{ 'January 01, 2026' }}"


def test_handles_double_quotes_and_several_call_sites():
    t = _Tok('{{ strftime_now("%Y") }} and {{ strftime_now(\'%Y-%m-%d\') }}')
    assert pin_chat_template_date(t) == 2
    assert t.chat_template == "{{ '2026' }} and {{ '2026-01-01' }}"


def test_an_unmatched_call_raises_rather_than_passing_through():
    """A form the regex misses must fail loudly -- silently leaving it in is the original bug."""
    with pytest.raises(ValueError, match="still calls strftime_now"):
        pin_chat_template_date(_Tok("{{ strftime_now(fmt_from_a_variable) }}"))


def test_an_alternate_date_is_respected():
    t = _Tok(APERTUS)
    pin_chat_template_date(t, date="2020-07-04")
    assert "'2020-07-04'" in t.chat_template
