"""Finetuning and evaluation use each model's released prompt format.

A shipped chat template enables chat formatting, except for explicit base models.
The CLI can request chat formatting but cannot silently disable a model's default.
"""
import pytest

from pproute.models import (declares_base_model, default_format_is_chat,
                            is_instruct_model, resolve_chat_template)


class _Tok:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


TEMPLATED = _Tok("{% for m in messages %}{{ m.content }}{% endfor %}")
BARE = _Tok(None)


def _tok(has_template):
    return TEMPLATED if has_template else BARE


# Released prompt format and presence of a tokenizer chat template are independent.
COHORT = [
    # (model id,                                   chat,  has_template)
    ("Qwen/Qwen3.5-9B-Base",                       False, True),   # base, ships ChatML anyway
    ("google/gemma-3-12b-pt",                     False, False),
    ("meta-llama/Llama-3.1-8B",                    False, False),
    ("swiss-ai/Apertus-8B-2509",                   False, False),
    ("allenai/Olmo-3-1025-7B",                     False, False),
    ("tiiuae/Falcon3-7B-Base",                     False, False),
    ("mistralai/Mistral-7B-v0.3",                  False, False),
    ("nvidia/Mistral-NeMo-Minitron-8B-Base",       False, False),
    ("Qwen/Qwen3.5-9B",                            True,  True),   # instruct, says so nowhere
    ("google/gemma-3-12b-it",                  True,  True),
    ("meta-llama/Llama-3.1-8B-Instruct",           True,  True),
    ("swiss-ai/Apertus-8B-Instruct-2509",          True,  True),
    ("allenai/Olmo-3-7B-Instruct",                 True,  True),
    ("tiiuae/Falcon3-7B-Instruct",                 True,  True),
    ("mistralai/Mistral-7B-Instruct-v0.3",         True,  True),
    ("nvidia/Mistral-NeMo-Minitron-8B-Instruct",   True,  True),
]


# --- the rule --------------------------------------------------------------------------------

@pytest.mark.parametrize("name,chat,has", COHORT)
def test_default_format_matches_the_release_for_every_cohort_model(name, chat, has):
    assert default_format_is_chat(name, _tok(has)) is chat, name


@pytest.mark.parametrize("name,chat,has", COHORT)
def test_derivation_alone_is_correct_without_any_flag(name, chat, has):
    """The flag exists to be forgotten. Derivation with `requested=False` must still be right."""
    apply, _ = resolve_chat_template(name, _tok(has), requested=False)
    assert apply is chat, name


def test_the_name_alone_would_get_qwen_wrong():
    """Regression: the previous name-only rule trained and scored this one on raw completions."""
    assert is_instruct_model("Qwen/Qwen3.5-9B") is False, "no instruct token in the name"
    assert default_format_is_chat("Qwen/Qwen3.5-9B", TEMPLATED) is True, "but it is chat-formatted"


def test_a_base_release_shipping_a_template_is_still_raw():
    assert declares_base_model("Qwen/Qwen3.5-9B-Base")
    assert default_format_is_chat("Qwen/Qwen3.5-9B-Base", TEMPLATED) is False
    apply, info = resolve_chat_template("Qwen/Qwen3.5-9B-Base", TEMPLATED, requested=False)
    assert apply is False
    assert info["tokenizer_has_template"] and info["name_declares_base"]


def test_tokens_are_whole_words_not_substrings():
    for name in ("mistralai/Mistral-7B-v0.3", "nvidia/Mistral-NeMo-Minitron-8B-Base",
                 "ibm-granite/granite-4.1-8b"):
        assert not is_instruct_model(name), "'it' inside a word must not match: %s" % name
    for name in ("tiiuae/Falcon3-7B-Base", "databricks/dbrx-base"):
        assert declares_base_model(name), name
    assert not declares_base_model("meta-llama/Llama-3.1-8B-Instruct")


# --- the flag and its guards -----------------------------------------------------------------

def test_the_flag_can_only_turn_the_template_on():
    apply, _ = resolve_chat_template("meta-llama/Llama-3.1-8B", TEMPLATED, requested=True)
    assert apply is True, "an explicit flag must still force the template on"


def test_template_requested_but_unavailable_raises():
    with pytest.raises(ValueError, match="tokenizer has none"):
        resolve_chat_template("meta-llama/Llama-3.1-8B", BARE, requested=True)


def test_instruct_name_without_a_template_raises():
    """The name and the tokenizer disagree and neither can be trusted -- refuse rather than guess."""
    with pytest.raises(ValueError, match="no chat template"):
        resolve_chat_template("some-org/Mystery-7B-Instruct", BARE, requested=False)


def test_info_records_both_signals_for_later_audit():
    _, info = resolve_chat_template("Qwen/Qwen3.5-9B", TEMPLATED, requested=False)
    assert info["default_format_is_chat"] is True
    assert info["name_is_instruct"] is False        # the disagreement is on the record
    assert info["tokenizer_has_template"] is True
    assert info["name_declares_base"] is False
