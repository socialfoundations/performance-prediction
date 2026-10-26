import hashlib
import json
import logging
import jinja2
import random
import torch
import numpy as np

from functools import partial
from copy import deepcopy
from datasets import disable_caching
from zarth_utils.nn_utils import random_split, set_random_seed
from datasets import Dataset, concatenate_datasets

from lm_eval.api.samplers import FirstNSampler
from lm_eval.tasks import TaskManager, get_task_dict

from pproute import mmlu_pro
from pproute.evaluator import _gold_index

eval_logger = logging.getLogger(__name__)

disable_caching()
# NOTE: datasets 5 removed both `config.HF_DATASETS_TRUST_REMOTE_CODE` and the `trust_remote_code`
# argument to load_dataset -- script-based datasets are gone entirely. Every benchmark in this
# project loads from parquet, so nothing here needs it; a task added later that does will fail at
# load time rather than being silently trusted.

# For tasks WITHOUT a labeled test split, get_dataset carves the predictor-training ("valid")
# set out of training_docs (the else-branch below); the rest is the finetune-train pool and the
# lm-eval eval split becomes the held-out test. This fraction is how much of training_docs goes
# to valid. commonsense_qa's labeled pool is small (9741 train): finetuning consumes only ~6400
# examples (max_steps 800 x batch 8), so we hand the remaining ~34% (~3342) to valid to give the
# predictor g more training data, while keeping the full official-validation (1221) as test.
# finetune and eval both call get_dataset, so keeping this here guarantees an identical split.
DEFAULT_VALID_FRAC = 0.2
TASK_VALID_FRAC = {"commonsense_qa": 0.343, "gsm8k": 0.30}
# GSM8K validation is carved from its training split, which is also lm-eval's default few-shot
# source. Sampling demonstrations from that unsplit pool leaks validation gold solutions into
# other validation prompts. Reserve a fixed five-row demonstration pool before the carve instead;
# neither SFT targets nor validation queries can then appear as a demonstration.
TASK_FEWSHOT_RESERVE = {"gsm8k": 5}
FEWSHOT_RESERVE_SEED = 1234

def resolve_task_names(task_name):
    """The lm_eval task(s) one benchmark name refers to.

    Every benchmark is a single task except MMLU-Pro, which is fourteen per-category subtasks --
    lm_eval applies a description and a few-shot sampler per task, and MMLU-Pro's are per category.
    Expanding here means callers and recorded configs all keep saying `mmlu_pro`.
    """
    if task_name == mmlu_pro.TASK_NAME:
        return list(mmlu_pro.SUBTASKS)
    return [task_name]


def three_way_carve(names):
    """The benchmark module for a task set with NO training split, or None.

    Such a benchmark's labelled pool IS its test split and has to be carved three ways. MMLU-Pro is
    the only one: it ships test=12,032 and validation=70, and the 70 are lm_eval's demonstrations
    rather than labelled training data, so they stay untouched and the carve comes out of test. It
    also arrives as fourteen per-category subtasks, and the carve is defined once over their pooled
    documents so the split sizes are the pre-registered 6,000 / 3,000 / 3,032 rather than fourteen
    independent roundings.
    """
    if names and set(names) <= set(mmlu_pro.SUBTASKS):
        return mmlu_pro
    return None


def get_tasks(tasks, num_fewshot=0, extra_until=()):
    """Load tasks with the frozen RNG and few-shot settings used by the experiments."""
    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if not tasks:
        raise ValueError("No tasks specified")
    task_manager = TaskManager("INFO")
    task_dict = get_task_dict(tasks, task_manager)

    def adjust(task_dict):
        for task in task_dict.values():
            if isinstance(task, dict):
                adjust(task)
                continue
            if task.get_config("output_type") == "generate_until" and extra_until:
                # Preserve the task's stops and append model-specific turn delimiters.
                until = list(task.get_config("generation_kwargs")["until"])
                until += [stop for stop in extra_until if stop not in until]
                task.set_config(key="generation_kwargs", value={"until": until}, update=True)
            # A task explicitly configured for zero shots must stay zero-shot.
            if task.get_config("num_fewshot") != 0:
                task.set_config(key="num_fewshot", value=num_fewshot)
            task.set_fewshot_seed(seed=1234)

    adjust(task_dict)
    return task_dict, task_manager


def apply_chat_template_fn(chat_history, tokenizer, add_generation_prompt=True):
    """
    Method to apply a chat template to a list of chat history between user and model.
    """
    try:
        chat_templated = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
    except jinja2.exceptions.TemplateError:
        eval_logger.warning(
            "Failed to apply chat template. removing the system role in chat history."
        )
        chat_history = [msg for msg in chat_history if msg["role"] != "system"]
        chat_templated = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
    return chat_templated


_ASSISTANT_BOUNDARY_SENTINEL = "PPROUTEASSISTANTBOUNDARY9F3A"


def assistant_message_bridge(tokenizer):
    """Literal bytes between this tokenizer's generation prompt and assistant content.

    Most chat templates end their generation prompt exactly where assistant content starts. The
    Mistral ``[INST]`` template instead renders ``[/INST] answer`` for a completed assistant turn
    while its generation prompt ends at ``[/INST]``. lm-eval removes the task delimiter from
    chat-templated multiple-choice continuations, so that model was scored and SFT-trained on the
    noncanonical ``[/INST]answer`` boundary.

    Derive the bridge from the template rather than naming a model or guessing whitespace. The
    sentinel is plain alphanumeric text because some templates trim message content.
    """
    user = [{"role": "user", "content": "PPROUTEUSERBOUNDARY9F3A"}]
    prompt = apply_chat_template_fn(user, tokenizer, add_generation_prompt=True)
    completed = apply_chat_template_fn(
        user + [{"role": "assistant", "content": _ASSISTANT_BOUNDARY_SENTINEL}],
        tokenizer,
        add_generation_prompt=False,
    )
    if not completed.startswith(prompt):
        raise ValueError(
            "canonical assistant rendering does not preserve the generation-prompt prefix"
        )
    tail = completed[len(prompt):]
    if tail.count(_ASSISTANT_BOUNDARY_SENTINEL) != 1:
        raise ValueError(
            "chat template did not reproduce the assistant boundary sentinel exactly once"
        )
    return tail.split(_ASSISTANT_BOUNDARY_SENTINEL, 1)[0]


def canonicalize_mc_chat_requests(task, tokenizer, apply_chat_template):
    """Put every chat-templated MC continuation at its canonical assistant boundary.

    This is deliberately a no-op for raw MC and for every generative task. For chat MC, the
    tokenizer's own assistant bridge is prepended to each continuation. Check that appending it
    preserves the context tokens and gives a stable continuation encoding. lm-eval subsequently
    moves trailing context whitespace into the scored continuation; these checks do not emulate
    that split or assume its probability cancels across options with different contexts.
    """
    if not apply_chat_template or task.get_config("output_type") != "multiple_choice":
        return {"requests": 0, "changed": 0, "bridge": ""}
    if tokenizer is None:
        raise ValueError("chat-templated multiple-choice requests require a tokenizer")

    bridge = assistant_message_bridge(tokenizer)
    changed = 0
    for instance in task.instances:
        context, continuation, *rest = instance.arguments
        if not isinstance(context, str) or not isinstance(continuation, str):
            raise ValueError("multiple-choice request context and continuation must be text")
        canonical_continuation = bridge + continuation
        context_ids = list(tokenizer.encode(context, add_special_tokens=False))
        combined_ids = list(tokenizer.encode(
            context + canonical_continuation, add_special_tokens=False
        ))
        if combined_ids[:len(context_ids)] != context_ids:
            raise ValueError(
                "canonical assistant boundary changes the multiple-choice context token IDs"
            )
        suffix_ids = combined_ids[len(context_ids):]
        standalone_ids = list(tokenizer.encode(
            canonical_continuation, add_special_tokens=False
        ))
        if not suffix_ids or suffix_ids != standalone_ids:
            raise ValueError(
                "multiple-choice continuation token IDs differ at the canonical assistant boundary"
            )
        if canonical_continuation != continuation:
            changed += 1
        instance.arguments = (context, canonical_continuation, *rest)

    diagnostics = {"requests": len(task.instances), "changed": changed, "bridge": bridge}
    task._pproute_mc_chat_boundary = diagnostics
    eval_logger.info(
        "%s: canonical assistant boundary %r on %d MC requests (%d changed)",
        task.get_config("task"), bridge, len(task.instances), changed,
    )
    return diagnostics


def _install_split(task_dict, task_name, data, max_num, random_seed, override_cls):
    """Point one task copy at `data` as its eval docs, subsampling to `max_num` first.

    The RNG is reseeded per split, exactly as the three inlined copies of this did, so the
    subsample a given (seed, split) produces does not depend on the other splits."""
    if max_num != -1 and len(data) > max_num:
        np.random.seed(random_seed)
        data = data.select(np.random.permutation(len(data))[:max_num])
    task = task_dict[task_name]
    task.__class__ = override_cls
    task._eval_docs = data
    return task


def _subsample_pooled(rows_by_task, max_num, random_seed):
    """Cap a multi-task split at `max_num` documents in total, not per task.

    The pool is walked in sorted task order so the draw cannot depend on dictionary ordering, and
    the RNG is reseeded exactly as `_install_split` does, so a given (seed, split) selects the same
    documents whether or not the other splits were built.
    """
    order = [(name, i) for name in sorted(rows_by_task) for i in range(len(rows_by_task[name]))]
    if len(order) <= max_num:
        return rows_by_task
    np.random.seed(random_seed)
    keep = {}
    for position in np.random.permutation(len(order))[:max_num]:
        name, index = order[position]
        keep.setdefault(name, []).append(index)
    return {name: rows.select(sorted(keep.get(name, [])))
            for name, rows in rows_by_task.items()}


def _build_requests(task, apply_chat_template, fewshot_as_multiturn, tokenizer, tokenizer_name):
    task.build_all_requests(
        system_instruction=None,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        chat_template=partial(apply_chat_template_fn, tokenizer=tokenizer),
        tokenizer_name=tokenizer_name,
    )
    return task


def _gold_idx_for_sft(task, inst):
    """Index of the option that is correct for this instance's doc.

    Resolved by the same rule that stamps the gold index into score files, so SFT trains on
    exactly the option evaluation scores as gold. A doc whose gold is not one of its options
    (-100) or a multiple-target doc has no single training row and is refused.
    """
    gold = _gold_index(task, inst.doc)
    if not isinstance(gold, int) or gold < 0:
        raise ValueError("no single gold option for an SFT row of %s: %r" % (task, gold))
    return gold


def _empty_sft_dataset():
    return Dataset.from_dict({"prompt": [], "completion": []})


def _build_sft_dataset(task):
    """Completion-only MC training rows, using the exact gold scoring request.

    Generative benchmarks use the frozen teacher traces through trace_sft instead.
    """
    if task.get_config("output_type") != "multiple_choice":
        raise ValueError("Generative SFT requires teacher traces")
    rows = {"prompt": [], "completion": []}
    for inst in task.instances:
        if inst.idx == _gold_idx_for_sft(task, inst):
            rows["prompt"].append(inst.arguments[0])
            rows["completion"].append(inst.arguments[1])
    return Dataset.from_dict(rows)


def split_fingerprint(task):
    """A short hash of a split's doc ids, for asserting that two runs used the same split.

    `get_dataset` re-derives the train/valid/test carve in every process from
    (task, seed, TASK_VALID_FRAC, len(training_docs)). That is deterministic, but it also means
    changing TASK_VALID_FRAC silently makes new scores incomparable with old ones -- the predictor
    would train on a different set than the one the checkpoint was selected on, with nothing
    raising. Recording this alongside each set of scores makes the mismatch detectable."""
    docs = task.eval_docs
    h = hashlib.sha1()
    for d in docs:
        h.update(json.dumps(d, sort_keys=True, default=str).encode())
    result = "%s:%d:%s" % (task.config.task, len(docs), h.hexdigest()[:12])
    fewshot_docs = getattr(task, "_pproute_fewshot_docs", None)
    if fewshot_docs is not None:
        fewshot = hashlib.sha1()
        for d in fewshot_docs:
            fewshot.update(json.dumps(d, sort_keys=True, default=str).encode())
        result += ":fewshot:%d:%s" % (len(fewshot_docs), fewshot.hexdigest()[:12])
        if getattr(task, "_pproute_fixed_fewshot_order", False):
            result += ":order:fixed"
    return result


def reserve_fewshot_data(data, size, seed=FEWSHOT_RESERVE_SEED):
    """Return (query pool, fixed demonstrations), disjoint and deterministic."""
    if size <= 0 or size >= len(data):
        raise ValueError("few-shot reserve size must be between 1 and len(data)-1")
    rng = np.random.RandomState(seed)
    reserved = list(rng.permutation(len(data))[:size])
    reserved_set = set(reserved)
    remaining = [index for index in range(len(data)) if index not in reserved_set]
    return data.select(remaining), data.select(reserved)


def get_dataset(
    task_name,
    random_seed=0,
    max_num_train=-1,
    max_num_valid=-1,
    max_num_test=-1,
    apply_chat_template=False,
    num_fewshot=0,
    fewshot_as_multiturn=False,
    tokenizer_name="",
    tokenizer=None,
    completion_only=True,
    extra_until=(),
    build_sft=False,
):
    # `task_name` may be one task or several. A benchmark that ships as several -- MMLU-Pro's
    # fourteen per-category subtasks -- is carved, built and returned as a set, and every split's
    # finetuning rows are the concatenation across its tasks.
    if build_sft and not completion_only:
        raise ValueError("SFT requires completion_only=True")
    names = resolve_task_names(task_name) if isinstance(task_name, str) else list(task_name)
    test_task_dict, task_manager = get_tasks(tasks=names, num_fewshot=num_fewshot,
                                             extra_until=tuple(extra_until))
    train_task_dict = deepcopy(test_task_dict)
    valid_task_dict = deepcopy(test_task_dict)

    set_random_seed(random_seed)

    carve_module = three_way_carve(names)
    task = test_task_dict[names[0]]

    # split the data into train/valid/test, as {split: {task_name: docs}}
    fewshot_data = None
    if carve_module is not None:
        # No training split at all: the labelled pool IS the test split. The carve is taken once
        # over every subtask's documents pooled together, then each subtask keeps its own rows from
        # each split -- so the partition is exactly the one a single pooled task would have given.
        # Unlike the branches below there is nothing to withhold for a max_num=0 caller: the pool is
        # already-materialized test docs, and request building is skipped downstream anyway.
        docs = {name: test_task_dict[name].test_docs() for name in names}
        id_sets = carve_module.carve_example_ids(docs, random_seed)
        data = {}
        for split, ids in zip(("train", "valid", "test"), id_sets):
            data[split] = {
                name: rows.select([i for i, doc in enumerate(rows)
                                   if carve_module.example_id(doc) in ids])
                for name, rows in docs.items()
            }
            eval_logger.info(
                "%s: three-way carve, %s split has %d docs over %d task(s) (seed %d, stratified "
                "on %s)", carve_module.TASK_NAME, split,
                sum(len(v) for v in data[split].values()), len(names), random_seed,
                list(carve_module.STRATA_FIELDS),
            )
    else:
        # Every released benchmark with a training split ships exactly one labelled eval split
        # (GSM8K: test; CommonsenseQA and WinoGrande: validation), which becomes the held-out test
        # set while development is carved from train. A task with both would need its own rule.
        if not task.has_training_docs() or (task.has_validation_docs() and task.has_test_docs()):
            raise ValueError("unsupported split layout for %s" % task_name)
        train_data = task.training_docs()
        reserve = TASK_FEWSHOT_RESERVE.get(task_name, 0)
        if reserve:
            train_data, fewshot_data = reserve_fewshot_data(train_data, reserve)
        valid_frac = TASK_VALID_FRAC.get(task_name, DEFAULT_VALID_FRAC)
        train_indices, valid_indices = random_split(
            len(train_data), (1 - valid_frac, valid_frac), random_seed
        )
        # Preserve the carve (and therefore the training split) even when validation is not
        # requested, but do not materialize held-out rows for a train-only caller.
        valid_data = (
            train_data.select([])
            if max_num_valid == 0
            else train_data.select(valid_indices)
        )
        train_data = train_data.select(train_indices)
        test_data = train_data.select([]) if max_num_test == 0 else task.eval_docs

    # Reserve MC demonstrations only after the fixed carve, so adding shots cannot change
    # development/test membership. Zero-shot runs and the generative reservation stay unchanged.
    fixed_mc_fewshot = task_name in ("commonsense_qa", "winogrande") and num_fewshot > 0
    if fixed_mc_fewshot:
        train_data, fewshot_data = reserve_fewshot_data(train_data, num_fewshot)

    if carve_module is None:
        data = {"train": {names[0]: train_data}, "valid": {names[0]: valid_data},
                "test": {names[0]: test_data}}

    class _OverrideTask(type(test_task_dict[names[0]])):
        @property
        def eval_docs(self):
            return self._eval_docs

        def build_all_requests(self, **kwargs):
            super().build_all_requests(**kwargs)
            canonicalize_mc_chat_requests(
                self, tokenizer, kwargs.get("apply_chat_template", False)
            )

    # Build scoring requests for requested splits. Materialize gold MC SFT rows only
    # when training needs them; evaluation and teacher-trace SFT use the requests directly.
    task_dicts = {"train": train_task_dict, "valid": valid_task_dict, "test": test_task_dict}
    datasets_by_split = {}
    for split, max_num in (("train", max_num_train), ("valid", max_num_valid),
                           ("test", max_num_test)):
        rows = data[split]
        # A cap counts DOCUMENTS, not documents per task, so for a multi-task benchmark it is
        # applied to the pool before the rows are handed back to their subtasks. Letting
        # _install_split cap each task separately would multiply the cap by fourteen.
        per_task_max = max_num
        if len(names) > 1 and max_num not in (-1, 0):
            rows = _subsample_pooled(rows, max_num, random_seed)
            per_task_max = -1
        frames = []
        for name in names:
            split_task = _install_split(
                task_dicts[split], name, rows[name], per_task_max, random_seed, _OverrideTask
            )
            if fewshot_data is not None:
                # random.Random.sample requires a collections.abc.Sequence; HF Dataset supports
                # indexing but is not registered as one. The reserve is exactly five rows, so the
                # explicit list is both cheap and the correct sampler boundary.
                fewshot_rows = list(fewshot_data)
                split_task.sampler.replace_df(fewshot_rows)
                split_task._pproute_fewshot_docs = fewshot_rows
                if fixed_mc_fewshot:
                    # Every query/seed/model sees the reserved rows in their selection order;
                    # a stateful random sampler would otherwise reorder them per query.
                    split_task.sampler = FirstNSampler(fewshot_rows)
                    split_task._pproute_fixed_fewshot_order = True
            # Skip request building for an empty split rather than asking lm_eval to build requests
            # for zero docs, which raises. Two ways a split can be empty: the caller asked for none
            # (evaluate_model passes 0 for splits it does not need), or a pooled cap left one
            # subtask with no rows -- a small category can draw zero when max_num is much smaller
            # than the split.
            if max_num == 0 or len(rows[name]) == 0:
                eval_logger.info("%s/%s: no docs, building no requests", split, name)
                frames.append(_empty_sft_dataset())
                continue
            _build_requests(
                split_task, apply_chat_template, fewshot_as_multiturn, tokenizer, tokenizer_name
            )
            frames.append(_build_sft_dataset(split_task) if build_sft else _empty_sft_dataset())
        datasets_by_split[split] = (
            frames[0] if len(frames) == 1 else concatenate_datasets(frames)
        )

    train_dataset = datasets_by_split["train"]
    valid_dataset = datasets_by_split["valid"]
    test_dataset = datasets_by_split["test"]

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        train_task_dict,
        valid_task_dict,
        test_task_dict,
        task_manager,
    )
