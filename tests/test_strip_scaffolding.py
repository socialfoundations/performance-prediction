"""The predictor `g` must see the same prompt text whether the scored model was templated or not.

An instruct checkpoint is scored through its chat template, so the prompt lm_eval stores carries
role headers, a system preamble and the model's special tokens. That wrapper is *constant across
every example of a given model*, so it predicts nothing about correctness -- it only shifts the
embedding distribution, eats `g`'s token budget (it more than doubled the text on
Llama-3.1-8B-Instruct: 501 chars vs 270 for the base model), and makes `g`'s input a different
shape for templated models than for untemplated ones. Stripping it makes the cohort comparable.

Two things here are easy to get wrong and are what these tests pin down:

  1. Whether to strip is decided from the *text*, not from whether the tokenizer has a template.
     Several base checkpoints ship a chat template they were never evaluated with
     (`Qwen/Qwen3.5-9B-Base` ships ChatML), so keying off the tokenizer would try to strip a
     wrapper that was never applied.

  2. Under the Winograd (`multiple_input`) layout the template wraps *each option's context*
     separately, so the wrapper lands inside the per-option strings:

         conts = ['tree<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n',
                  'bush<|eot_id|>...']

     Trimming the assembled question cannot reach that. Stripping therefore happens on the raw
     request context, before the layout is resolved.
"""
import pytest

from pproute.models import PINNED_DATE, chat_scaffolding, strip_chat_scaffolding
from pproute.scores import render_options


PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
SUFFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


class _Tok:
    """A tokenizer stand-in whose template is exactly PREFIX + content + SUFFIX."""

    all_special_tokens = ["<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>",
                          "<|end_header_id|>"]

    def __init__(self, has_template=True):
        self.chat_template = "irrelevant" if has_template else None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return PREFIX + messages[0]["content"] + SUFFIX


def test_scaffolding_is_recovered_exactly():
    assert chat_scaffolding(_Tok()) == (PREFIX, SUFFIX)
    assert chat_scaffolding(_Tok(has_template=False)) == ("", "")


def test_wrapper_present_is_stripped():
    assert strip_chat_scaffolding(PREFIX + "Question: what" + SUFFIX, _Tok()) == "Question: what"


def test_wrapper_absent_is_left_alone():
    """A base checkpoint that ships a template but was scored raw: nothing to strip.

    This is the Qwen3.5-9B-Base case. Raising here (or stripping anyway) would break a model whose
    prompt was never templated in the first place."""
    raw = "Question: what\nAnswer:"
    assert strip_chat_scaffolding(raw, _Tok()) == raw


def test_half_a_wrapper_raises():
    """Matching one end but not the other means the text is not what this template produces.

    Cutting it would corrupt the prompt silently, so refuse. This is also the guard that caught the
    real bug: whitespace-trimming the text first removed the template's trailing "\\n\\n", leaving a
    prefix-only match."""
    with pytest.raises(ValueError, match="only part"):
        strip_chat_scaffolding(PREFIX + "Question: what", _Tok())
    with pytest.raises(ValueError, match="only part"):
        strip_chat_scaffolding("Question: what" + SUFFIX, _Tok())


def test_trimming_before_stripping_is_what_breaks_it():
    """Documents the ordering constraint: strip first, trim after.

    Templates commonly end in "\\n\\n"; trimming removes part of the suffix so the match fails."""
    text = PREFIX + "Question: what" + SUFFIX
    assert strip_chat_scaffolding(text, _Tok()).strip() == "Question: what"
    with pytest.raises(ValueError):
        strip_chat_scaffolding(text.strip(), _Tok())


def test_inner_markup_raises():
    """fewshot_as_multiturn makes each in-context example its own turn, leaving role markers that
    an outer strip cannot reach. Half-stripped text would train g on a different distribution from
    the rest of the cohort, silently."""
    inner = PREFIX + "A<|eot_id|>more" + SUFFIX
    with pytest.raises(ValueError, match="special tokens survived"):
        strip_chat_scaffolding(inner, _Tok())


def test_no_template_is_a_noop():
    raw = PREFIX + "not really wrapped" + SUFFIX
    assert strip_chat_scaffolding(raw, _Tok(has_template=False)) == raw


def test_scaffolding_pins_the_template_date():
    """A template rendering "today" would match on the day of the eval and stop matching the day
    after -- the wrapper has to reproduce a prompt rendered on some earlier day.

    `load_model_and_tokenizer` pins, but the predictor loads the *scored* model's tokenizer
    bare, just to strip with it, so chat_scaffolding cannot assume a pinned tokenizer arrives.
    swiss-ai/Apertus-8B-Instruct-2509 is the cohort model this applies to."""

    class _DatedTok:
        all_special_tokens = []
        chat_template = "Current date: {{ strftime_now('%Y-%m-%d') }}\n{{ messages[0].content }}"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            import jinja2
            return jinja2.Template(self.chat_template).render(messages=messages)

    tok = _DatedTok()
    prefix, suffix = chat_scaffolding(tok)
    assert PINNED_DATE in prefix
    assert "strftime_now" not in tok.chat_template      # rewritten in place, and visibly so
    assert chat_scaffolding(tok) == (prefix, suffix)    # idempotent


# --------------------------------------------------------------- layout interaction
def _strip(ctx):
    return strip_chat_scaffolding(ctx, _Tok())


def test_standard_layout_matches_untemplated():
    """Standard MC: one shared context, one continuation per option."""
    q = "Question: what\nAnswer:"
    plain = [(q, " A"), (q, " B")]
    templated = [(PREFIX + q + SUFFIX, " A"), (PREFIX + q + SUFFIX, " B")]

    got = render_options([(_strip(a[0]), a[1]) for a in templated])
    assert got == render_options(plain)


def test_multiple_input_options_come_out_clean():
    """Winograd: the wrapper is inside each option's context, which is why stripping happens on the
    raw context rather than on the assembled question."""
    base = "More people sat near the tree than the bush because the "
    plain = [(base + "tree", " had bees."), (base + "bush", " had bees.")]
    templated = [(PREFIX + base + "tree" + SUFFIX, "had bees."),
                 (PREFIX + base + "bush" + SUFFIX, "had bees.")]

    q_t, opts_t, layout_t = render_options([(_strip(a[0]), a[1]) for a in templated])
    q_p, opts_p, layout_p = render_options(plain)

    assert layout_t == layout_p == "multiple_input"
    assert opts_t == opts_p == ["tree", "bush"]
    for o in opts_t:
        assert "<|" not in o
    # lm_eval keeps the continuation's leading space normally but drops it under a chat template;
    # render_options normalises so the two agree rather than differing by one char.
    assert q_t == q_p
    assert q_t.endswith("the ___ had bees.")
