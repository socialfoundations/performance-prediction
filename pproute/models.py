"""Loading the model under test, for all three entry scripts that need one.

Two things live here because getting them wrong is silent or fatal and they were previously
copy-pasted three ways:

  * **which auto class.** `AutoModelForCausalLM` does not cover the whole model list -- several
    current releases are `...ForConditionalGeneration` and are simply not in the causal-LM
    mapping, so loading them raises. `load_auto_class` asks transformers' own registries which
    auto class claims the checkpoint's config instead of assuming.
  * **the pad token.** Models without one need `tokenizer.pad_token` set before any batched
    forward pass, or padding silently becomes some real token.

`dtype=` replaced `torch_dtype=` in transformers 5; `dtype_kwargs` picks whichever the installed
version accepts so neither a deprecation warning nor a TypeError depends on the env.
"""
import logging
import re
from datetime import datetime

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers import __version__ as transformers_version
from peft import PeftConfig, PeftModel

logger = logging.getLogger(__name__)

# Auto classes that can host a generative decoder, in the order we prefer them. The first whose
# mapping claims the config wins, so a plain decoder keeps loading through AutoModelForCausalLM
# exactly as before and only the models that need something else get something else.
CANDIDATE_AUTO_CLASSES = (
    "AutoModelForCausalLM",
    "AutoModelForMultimodalLM",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModelForSeq2SeqLM",
)

_FIX_MISTRAL_REGEX_MODELS = ("mistral-nemo-minitron",)


def tokenizer_load_kwargs(model_name):
    """Tokenizer kwargs required by specific checkpoints in the experiment cohort."""
    kw = {"trust_remote_code": True}
    name = str(model_name).lower()
    if any(token in name for token in _FIX_MISTRAL_REGEX_MODELS):
        # Transformers detects the affected tokenizer and warns that leaving the legacy Mistral
        # regex in place produces incorrect tokenization. Keep this in the shared loader so SFT,
        # evaluation and predictor prompt normalization cannot disagree.
        kw["fix_mistral_regex"] = True
    return kw


def dtype_kwargs(dtype="auto"):
    """`{"dtype": ...}` on transformers >= 5, `{"torch_dtype": ...}` before it."""
    major = int(str(transformers_version).split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def requested_dtype(model_dtype="auto"):
    """Resolve an explicit compute dtype without guessing when checkpoint metadata is absent."""
    if model_dtype == "auto":
        return "auto"
    if model_dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("model_dtype must be auto, bfloat16, float16 or float32")
    return getattr(torch, model_dtype)


def config_dtype(cfg):
    """The dtype a config declares, as a torch dtype, or None if it declares none anywhere.

    Three ways this returns None by accident, all of which mean a silent fp32 run:
      * transformers 5 renamed `config.torch_dtype` to `config.dtype`, so reading one name misses
        checkpoints released under the other;
      * the multimodal/`...ForConditionalGeneration` configs nest it -- Qwen/Qwen3.5-9B declares
        bfloat16 only in `text_config` and has no top-level dtype at all;
      * either may hold the name as a string ("bfloat16") rather than a torch dtype.
    The language tower is the only part this project trains, so `text_config` is the right place
    to look when the top level is silent.
    """
    for holder in (cfg, getattr(cfg, "text_config", None)):
        if holder is None:
            continue
        for attr in ("dtype", "torch_dtype"):
            dt = getattr(holder, attr, None)
            if isinstance(dt, str):
                dt = getattr(torch, dt, None)
            if isinstance(dt, torch.dtype):
                return dt
    return None


def config_max_length(cfg):
    """The context window declared by a model config, including a nested language config."""
    for holder in (cfg, getattr(cfg, "text_config", None)):
        if holder is None:
            continue
        for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
            value = getattr(holder, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def load_auto_class(model_name, revision="main"):
    """The auto class that can load `model_name`, resolved from its config.

    Order of attempts: an explicit `auto_map` in the config (custom code on the Hub), then
    transformers' own config -> model mappings, then the architecture name's suffix as a last
    resort. Raises with the architectures listed if nothing matches, because loading a model
    through the wrong auto class is not a recoverable situation.
    """
    cfg = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    archs = list(getattr(cfg, "architectures", None) or [])

    # The class the checkpoint itself names is the most direct answer, and the only one that is
    # right when a config is in no auto mapping at all.
    for arch in archs:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            logger.info("%s: %s (config.architectures)", model_name, arch)
            return cls

    auto_map = getattr(cfg, "auto_map", None) or {}
    for name in CANDIDATE_AUTO_CLASSES:
        if name in auto_map and hasattr(transformers, name):
            logger.info("%s: %s (from config.auto_map)", model_name, name)
            return getattr(transformers, name)

    for name in CANDIDATE_AUTO_CLASSES:
        cls = getattr(transformers, name, None)
        mapping = getattr(cls, "_model_mapping", None)
        if mapping is not None and type(cfg) in mapping:
            logger.info("%s: %s (architectures=%s)", model_name, name, archs)
            return cls

    for arch in archs:                      # config not in any mapping (e.g. remote code)
        for suffix, name in (("ForCausalLM", "AutoModelForCausalLM"),
                             ("ForConditionalGeneration", "AutoModelForMultimodalLM"),
                             ("ForImageTextToText", "AutoModelForImageTextToText")):
            if arch.endswith(suffix) and hasattr(transformers, name):
                logger.warning("%s: falling back to %s from architecture name %s",
                               model_name, name, arch)
                return getattr(transformers, name)

    raise ValueError(
        "no auto class for %s (architectures=%s, config=%s). Tried config.auto_map, the "
        "transformers config->model mappings, and the architecture suffix. Add the right class to "
        "CANDIDATE_AUTO_CLASSES rather than forcing AutoModelForCausalLM, which would raise or "
        "load the wrong head." % (model_name, archs, type(cfg).__name__))


# Date pinned into any chat template that renders "today". Arbitrary, but it must never change:
# it is part of the prompt, so moving it makes new scores incomparable with old ones.
PINNED_DATE = "2026-01-01"
_STRFTIME_NOW = re.compile(r"strftime_now\(\s*(['\"])(.*?)\1\s*\)")


def pin_chat_template_date(tokenizer, date=PINNED_DATE):
    """Freeze `strftime_now(...)` in a chat template to a fixed date. No-op if it is not used.

    Some templates render the current date into the system prompt -- `swiss-ai/Apertus-*-Instruct`
    emits "Current date: <today>". Left alone, a model's before and after endpoints see *different
    prompts*, because they are scored on different days, and both axes of this experiment are
    before-minus-after differences within one model. The protocol only has to be fixed across a
    model's own two endpoints, and this is the one thing that silently breaks that.

    Rewrites the template rather than patching jinja's globals or `datetime.now`: the substitution
    is visible in `tokenizer.chat_template` afterwards, so what the model was actually shown can be
    read off the object instead of inferred from which day the job ran.

    Returns the number of call sites pinned.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template or "strftime_now" not in template:
        return 0

    stamp = datetime.strptime(date, "%Y-%m-%d")

    def literal(m):
        # Render the pinned date through the template's OWN format string, so a template asking for
        # "%B %d, %Y" still gets a sensible string rather than an ISO date it did not ask for.
        return "'%s'" % stamp.strftime(m.group(2)).replace("'", "\\'")

    pinned, n = _STRFTIME_NOW.subn(literal, template)
    if "strftime_now" in pinned:
        raise ValueError(
            "chat template still calls strftime_now after pinning -- the call is written in a form "
            "this does not match, so the prompt would still carry the current date. Template:\n%s"
            % pinned[:2000])
    tokenizer.chat_template = pinned
    logger.info("pinned %d strftime_now call(s) in the chat template to %s", n, date)
    return n


# Prepended to a chat template to force its thinking branch off. A jinja `set` at the top shadows
# the variable the caller passes, so it works whatever guard the template writes around it -- which
# matters because nothing downstream can pass the kwarg (see disable_chat_template_thinking).
_THINKING_OFF = "{%- set enable_thinking = false %}"


def disable_chat_template_thinking(tokenizer):
    """Close the thinking block in the generation prompt. No-op if the template has no such switch.

    A reasoning model's generation prompt ends *inside an open thinking block*: `Qwen/Qwen3.5-9B`
    renders `<|im_start|>assistant\\n<think>\\n`. That is right for generation and wrong for
    everything this experiment measures on a multiple-choice task, because lm_eval scores each
    option as the continuation of that prompt -- so `f_Z` becomes the model's confidence in the
    first token of its *thoughts*, not in its answer.

    With the switch off the prompt ends `<think>\\n\\n</think>\\n\\n`, an empty closed block, so the
    scored continuation sits where the answer goes.

    This rewrites the template rather than passing `enable_thinking=False` at the call site because
    there is no call site to pass it at: lm_eval calls `apply_chat_template` itself, deep inside
    request construction, and would render the thinking prompt no matter what the entry script
    wanted. Rewriting also keeps the substitution visible on the tokenizer afterwards and travels
    with a checkpoint saved by finetune_model, so training and scoring cannot disagree.

    Verified by rendering rather than by pattern-matching the template: the rewrite is accepted only
    if the default rendering then equals what `enable_thinking=False` produced. Returns whether the
    template was changed.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return False

    probe = [{"role": "user", "content": "X"}]

    def render(**kw):
        return tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, **kw)

    try:
        default, off = render(), render(enable_thinking=False)
    except Exception:                      # template rejects the kwarg -> it has no such switch
        return False
    if default == off:
        # No thinking switch, or it is already off -- including a template this has already
        # rewritten, which is what makes repeated calls idempotent.
        return False

    tokenizer.chat_template = _THINKING_OFF + template
    if render() != off:
        raise ValueError(
            "disabling the thinking branch did not take effect: the template still renders its "
            "thinking prompt by default. Scoring options against an open thinking block measures "
            "confidence in the first thought token rather than in the answer, so this cannot be "
            "left to pass through.")
    logger.info("disabled the chat template's thinking branch; generation prompt now ends %r",
                off[-40:])
    return True


def set_pad_token(tokenizer, model=None):
    """Give the tokenizer a pad token if it has none, and tell the model about it."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer


def configure_chat_sft_tokenizer(tokenizer, apply_chat_template):
    """Prevent a rendered chat string from receiving a second automatic BOS during SFT.

    lm-eval gives us already-rendered chat text. TRL treats that text as a plain string and calls
    the tokenizer with its default special-token behavior, while canonical chat tokenization uses
    ``add_special_tokens=False`` because the template owns the special tokens. Explicitly assigning
    ``add_bos_token=False`` updates the fast-tokenizer post-processor even on tokenizers whose
    property already reports False.
    """
    if apply_chat_template and hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
        logger.info("disabled automatic BOS insertion for already-rendered chat SFT text")
        return True
    return False


# Whole `-`/`_`/`.`-separated tokens of the model id's last path segment. Never substring matches:
# "it" appears inside `granite`, `mistral` and `minitron`, none of which are instruct checkpoints.
INSTRUCT_TOKENS = frozenset(
    ("instruct", "instruction", "instructed", "it", "chat", "sft", "dpo", "rlhf", "tulu")
)
BASE_TOKENS = frozenset(("base", "pretrained"))


def _tokens(model_name):
    return set(re.split(r"[-_.\s]+", str(model_name).rstrip("/").split("/")[-1].lower()))


def is_instruct_model(model_name):
    """Whether a model id *names* an instruction-tuned checkpoint.

    A weak signal, kept only as a cross-check -- plenty of instruct releases say nothing in their
    name (`Qwen/Qwen3.5-9B`, `zai-org/GLM-4-9B-0414`). It is NOT how the prompt format is decided;
    see `default_format_is_chat`."""
    return bool(_tokens(model_name) & INSTRUCT_TOKENS)


def declares_base_model(model_name):
    """Whether a model id explicitly says it is a base checkpoint (`...-Base`)."""
    return bool(_tokens(model_name) & BASE_TOKENS)


def default_format_is_chat(model_name, tokenizer):
    """The format the model was built for: its own chat template if it ships one.

    Every model is evaluated and finetuned in its default format, and a model's default format is
    the one its tokenizer carries -- so the template, not the name, decides. The single exception
    is a release that names itself a base checkpoint while still shipping a generic ChatML template
    as tokenizer boilerplate (`Qwen/Qwen3.5-9B-Base`); a base checkpoint's default format is raw
    completion, so an explicit `...-Base` in the name vetoes the template.

    Checked against every tokenizer in the cohort: 16 of 16 agree with the format the model was
    released for. The name alone does not -- it misses `Qwen/Qwen3.5-9B`, an instruct checkpoint
    whose name carries no instruct token at all, and would have trained and scored it on raw
    completions.
    """
    has_template = getattr(tokenizer, "chat_template", None) is not None
    return has_template and not declares_base_model(model_name)


# A sentinel that will not occur in a prompt. Deliberately plain alphanumerics with no leading or
# trailing punctuation: several templates (Llama-3.1 among them) apply `| trim` to the message
# content, which silently ate an earlier sentinel that was wrapped in control characters and made
# the wrapper undetectable.
_SCAFFOLD_SENTINEL = "PPROUTECONTENTSENTINEL9F3A"


def chat_scaffolding(tokenizer):
    """The (prefix, suffix) this tokenizer's chat template wraps a single user turn in.

    Obtained by rendering the template around a sentinel and splitting on it, so the result is the
    exact wrapper this model produces rather than a guess. ("", "") when there is no template.

    Applies the same two template rewrites `load_model_and_tokenizer` does, and does so here rather
    than trusting the caller, because the predictor loads the scored model's tokenizer bare
    solely to remove its wrapper.
    Both rewrites are idempotent, so a tokenizer that arrives already treated is unaffected.

      - the date, or a template calling `strftime_now` (`swiss-ai/Apertus-*-Instruct` renders
        "Current date: <today>") yields a prefix that matches on the day of the eval and stops
        matching the day after.
      - the thinking branch, or a reasoning model's suffix comes back as the *open* `<think>\\n`
        while the scored prompt carries the closed `<think>\\n\\n</think>\\n\\n`, and the wrapper
        fails to match at all.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        return "", ""
    pin_chat_template_date(tokenizer)
    disable_chat_template_thinking(tokenizer)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": _SCAFFOLD_SENTINEL}],
        tokenize=False, add_generation_prompt=True)
    if rendered.count(_SCAFFOLD_SENTINEL) != 1:
        raise ValueError(
            "chat template did not reproduce the content exactly once (%d occurrences), so its "
            "wrapper cannot be determined" % rendered.count(_SCAFFOLD_SENTINEL))
    prefix, suffix = rendered.split(_SCAFFOLD_SENTINEL)
    return prefix, suffix


def strip_chat_scaffolding(text, tokenizer, strict=True):
    """`text` with this model's chat wrapper removed, leaving the prompt a base model would see.

    Why this exists: the predictor `g` is trained on the scored prompt, and for an instruct
    checkpoint that prompt arrives wrapped in chat markup -- role headers, a system preamble, the
    model's own special tokens. That wrapper is *constant across every example of a given model*,
    so it carries nothing about whether an answer is correct. It only shifts the embedding
    distribution, eats `g`'s token budget (it roughly doubled the input length on Llama-3.1-8B-
    Instruct), and makes `g`'s input format differ systematically between templated and untemplated
    models. Stripping it hands `g` the same shape of text for every model.

    Inverting the template is exact rather than heuristic: the same template produced the string, so
    its prefix and suffix are known rather than pattern-matched.

    Whether to strip is decided from the text itself, not from whether the tokenizer *has* a
    template, because those are different questions: several base checkpoints ship a chat template
    they were never evaluated with (Qwen3.5-9B-Base is one), so keying off the tokenizer alone would
    try to strip a wrapper that was never applied. The three cases:

      wrapper present   -> stripped.
      wrapper absent    -> returned unchanged; the run scored the raw prompt.
      half present      -> raises. Matching one end but not the other means the text is not what
                           this template produces, and cutting it would corrupt the prompt.

    Note `text` must not be whitespace-trimmed first: templates commonly end in "\\n\\n", and
    trimming removes part of the suffix so the match then fails. Trim the result instead.

    Also raises when special tokens survive stripping. That is what `fewshot_as_multiturn`
    produces -- in-context examples become their own turns and leave inner role markers an outer
    strip cannot reach. Failing loudly matters: half-stripped text would train `g` on a different
    distribution from the rest of the cohort, silently.
    """
    prefix, suffix = chat_scaffolding(tokenizer)
    if not prefix and not suffix:
        return text

    # Only non-empty ends carry evidence: `text.endswith("")` is vacuously true, so an empty suffix
    # must not count as a match or every untemplated text would look half-wrapped.
    ends = [text.startswith(prefix)] if prefix else []
    if suffix:
        ends.append(text.endswith(suffix))

    if not any(ends):
        return text                      # scored without the template -- nothing to strip
    if not all(ends):
        if strict:
            raise ValueError(
                "text carries only part of this model's chat wrapper, so it cannot be stripped "
                "safely (prefix %s, suffix %s).\n"
                "  expected prefix: %r\n  expected suffix: %r\n  text starts: %r\n  text ends: %r"
                % ("matched" if text.startswith(prefix) else "no",
                   "matched" if text.endswith(suffix) else "no",
                   prefix[:80], suffix[:80], text[:80], text[-80:]))
        return text

    out = text[len(prefix):len(text) - len(suffix)] if suffix else text[len(prefix):]

    leftover = [t for t in (getattr(tokenizer, "all_special_tokens", None) or []) if t and t in out]
    if leftover and strict:
        raise ValueError(
            "special tokens survived stripping: %s. The outer wrapper came off but inner markup "
            "remains -- this is what fewshot_as_multiturn leaves behind, and the result would not "
            "be comparable with an untemplated model's text." % sorted(set(leftover))[:5])
    return out


def resolve_chat_template(model_name, tokenizer, requested=False):
    """The `apply_chat_template` actually used for `model_name`, plus a record of how it was set.

    Every model runs in its default format, and `default_format_is_chat` derives that from the
    tokenizer. `requested` is the job's `--dataset_param.apply_chat_template` flag; it can only
    turn the template *on*, never off. Omitting it retains the model's default.

    Raises rather than warns where the run would be meaningless: a template asked for on a model
    that has none, and a model whose name says instruct while carrying no template at all.

    Returns `(apply, info)`; `info` is recorded alongside the scores so the choice is auditable
    later, when the only surviving evidence is the score file.
    """
    has_template = getattr(tokenizer, "chat_template", None) is not None
    named_instruct = is_instruct_model(model_name)
    named_base = declares_base_model(model_name)
    default_chat = default_format_is_chat(model_name, tokenizer)
    apply = bool(requested) or default_chat

    if apply and not has_template:
        raise ValueError(
            "%s: chat template requested but the tokenizer has none, so there is no chat format to "
            "run this model in. Check the model id." % model_name)

    # The name and the tokenizer disagreeing is worth saying out loud either way: one of the two is
    # wrong about what this checkpoint is, and which one decides the prompt the model sees.
    if named_instruct and not has_template:
        raise ValueError(
            "%s: the name says instruction-tuned but the tokenizer carries no chat template. Its "
            "default format cannot be determined -- pass --dataset_param.apply_chat_template if it "
            "should be templated, or rename/replace the checkpoint." % model_name)
    if named_base and has_template:
        logger.info(
            "%s: names itself a base checkpoint but ships a chat template; running it on raw "
            "completions, which is a base checkpoint's default format.", model_name)
    if default_chat and not named_instruct:
        logger.info(
            "%s: no instruct token in the name, but it ships a chat template and does not claim to "
            "be a base checkpoint -- running it in chat format.", model_name)
    logger.info("%s: apply_chat_template=%s (%s)", model_name, apply,
                "requested" if requested else "the model's default format")
    return apply, {"apply_chat_template": apply, "requested": bool(requested),
                   "default_format_is_chat": default_chat, "name_is_instruct": named_instruct,
                   "name_declares_base": named_base, "tokenizer_has_template": has_template}


def load_model_and_tokenizer(model_name, revision="main", dir_checkpoint="",
                             device_map=None, eval_mode=False, model_dtype="auto"):
    """Load either a base model (`model_name`) or a LoRA checkpoint on top of one
    (`dir_checkpoint`, which also supplies the tokenizer), plus its tokenizer.

    Returns (model, tokenizer, tokenizer_name), where tokenizer_name is what lm_eval wants for
    its request cache -- the model id for a base model, the checkpoint path with separators
    flattened for a checkpoint.
    """
    from_pretrained_kw = dict(
        trust_remote_code=True, **dtype_kwargs(requested_dtype(model_dtype))
    )
    if device_map is not None:
        from_pretrained_kw["device_map"] = device_map

    if dir_checkpoint:
        # Deliberately NOT AutoPeftModelForCausalLM: it hardcodes the causal-LM auto class, so it
        # fails on exactly the models load_auto_class exists to handle. Load the base model by its
        # own declared class -- the adapter config names it -- then attach the adapter.
        pcfg = PeftConfig.from_pretrained(dir_checkpoint)
        base_name = pcfg.base_model_name_or_path
        if str(base_name) != str(model_name):
            raise ValueError(
                "adapter %s declares base model %s but the run requested %s"
                % (dir_checkpoint, base_name, model_name)
            )
        adapter_revision = getattr(pcfg, "revision", None)
        if adapter_revision is not None and str(adapter_revision) != str(revision):
            raise ValueError(
                "adapter %s pins base revision %s but the run requested %s"
                % (dir_checkpoint, adapter_revision, revision)
            )
        tokenizer = AutoTokenizer.from_pretrained(
            dir_checkpoint, **tokenizer_load_kwargs(base_name)
        )
        tokenizer_name = dir_checkpoint.replace("/", "--")
        base_class = load_auto_class(base_name, revision=revision)
        base = base_class.from_pretrained(
            base_name, revision=revision, **from_pretrained_kw
        )
        model = PeftModel.from_pretrained(base, dir_checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, **tokenizer_load_kwargs(model_name)
        )
        tokenizer_name = model_name
        auto_class = load_auto_class(model_name, revision=revision)
        model = auto_class.from_pretrained(model_name, revision=revision, **from_pretrained_kw)

    # Before anything renders a prompt, and for every entry script, so a checkpoint saved by
    # finetune_model carries the pinned template too and its eval cannot drift back.
    pin_chat_template_date(tokenizer)
    disable_chat_template_thinking(tokenizer)
    set_pad_token(tokenizer, model)
    if eval_mode:
        model.eval()
    return model, tokenizer, tokenizer_name


def get_mixed_precision(model_name, revision="main", model_dtype="auto"):
    """Trainer flags from the explicit compute dtype, or the checkpoint metadata in auto mode.

    fp32 checkpoints train in fp32 -- neither flag is set. bf16 is dropped when the GPU cannot do
    it, so a bf16 model on older hardware falls back rather than failing at the first matmul."""
    cfg = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    explicit_dtype = requested_dtype(model_dtype)
    model_dtype = config_dtype(cfg) if explicit_dtype == "auto" else explicit_dtype

    if model_dtype is None:
        raise ValueError(
            "%s declares no dtype in its config or text_config. Inspect the checkpoint and "
            "set --model_dtype explicitly; silently choosing fp32 can double memory use."
            % model_name)

    bf16 = model_dtype == torch.bfloat16
    fp16 = model_dtype == torch.float16
    if not (bf16 or fp16 or model_dtype == torch.float32):
        raise ValueError(
            "unsupported dtype %s for %s: expected bfloat16, float16 or float32. Pick the "
            "precision explicitly rather than letting the trainer guess." % (model_dtype, model_name))

    if bf16 and not torch.cuda.is_bf16_supported():
        logger.warning("%s is bf16 but this device does not support bf16; training in fp32", model_name)
        bf16 = False

    logger.info("Model dtype: %s, bf16: %s, fp16: %s", model_dtype, bf16, fp16)
    return bf16, fp16
