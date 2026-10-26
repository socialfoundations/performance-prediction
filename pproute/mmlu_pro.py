"""MMLU-Pro's exact answer contract, its three-way carve, and the generated-trace SFT boundary.

The benchmark's reported Z is lm-eval's ``custom-extract`` filter: the FIRST match of
``answer is \\(?([A-J])\\)?`` in the response, compared case-insensitively against the gold letter.
Trace validation, confidence resampling, and evaluation all call the functions here rather than
keeping their own copies, for the same reason :mod:`pproute.gsm8k` does: every stage must use the
extractor that defines the benchmark metric.

Two things make MMLU-Pro unlike the other benchmarks.

**It is fourteen lm-eval tasks, not one.** ``mmlu_pro`` ships as a group of per-category subtasks
because both the description and the five demonstrations are per category, while lm-eval applies
each per *task*. They are run as fourteen tasks and their per-prompt records are pooled afterwards.


**There is no train split.** It ships ``test`` = 12,032 and ``validation`` = 70, and the 70 are the
demonstrations. So the labelled pool *is* the test split, carved 6,000 / 3,000 / 3,032, stratified
on ``(category, src)`` -- categories run 1,351 to 381 and ``ori_mmlu-professional_law`` alone is
1,003 rows, so an unstratified draw could leave SFT and test differing in composition, which would
show up as delta-accuracy and be misread as a finetuning effect.
"""

import random
import re

from pproute import trace_sft

TASK_NAME = "mmlu_pro"
DATASET_PATH = "TIGER-Lab/MMLU-Pro"
FILTER_NAME = "custom-extract"

# The fourteen shipped leaf subtasks. Named explicitly rather than resolved through the `mmlu_pro`
# group: a group hands lm_eval a different object, and this pipeline wants the leaves.
CATEGORIES = ("biology", "business", "chemistry", "computer science", "economics", "engineering",
              "health", "history", "law", "math", "other", "philosophy", "physics", "psychology")
SUBTASKS = tuple("mmlu_pro_" + c.replace(" ", "_") for c in CATEGORIES)

# Fixed split sizes; test is the remainder, 12,032 - 6,000 - 3,000 = 3,032.
SFT_SIZE = 6000
PREDICTOR_TRAIN_SIZE = 3000

# Fields the carve is stratified on, most significant first.
STRATA_FIELDS = ("category", "src")

# lm_eval owns the demonstrations now, so this is the real shot count, and passing it is what makes
# `fewshot_as_multiturn` meaningful for chat models -- the whole reason the flat task was dropped.
EVAL_SHOTS = trace_sft.EVAL_SHOTS
EVAL_MAX_NEW_TOKENS = trace_sft.EVAL_MAX_NEW_TOKENS

# Byte-identical to the `custom-extract` filter's pattern in the shipped mmlu_pro YAML, and
# deliberately NOT case-insensitive: lm_eval compiles it without re.IGNORECASE, so a trace ending
# "The Answer is (G)" does not match and scores zero. `ignore_case` on the metric applies to the
# comparison of the *extracted* letter against gold, which is a different step.
ANSWER_PATTERN = r"answer is \(?([ABCDEFGHIJ])\)?"
ANSWER_RE = re.compile(ANSWER_PATTERN)

# What lm_eval's RegexFilter emits when nothing matches. It then fails exact_match, so Z = 0.
INVALID = "[invalid]"
UNAVAILABLE_OPTION = "<not an available option>"


def example_id(doc):
    """Stable identity for an MMLU-Pro row.

    Unlike GSM8K, the dataset ships one: ``question_id`` is unique over the 12,032 test rows and
    independent of both the row's split and which subtask carries it, so there is nothing to hash.
    """
    return str(doc["question_id"])


def extract_answer(text):
    """The letter lm_eval's filter would extract, or None if extraction fails.

    ``take_first`` follows the regex, so this is the FIRST match in the response, not the last -- a
    model that says "the answer is (A)" mid-reasoning and "the answer is (C)" at the end is scored
    on A. Mirroring that here is the point: the teacher's rejection sampling has to accept exactly
    the traces the evaluation would mark correct.
    """
    match = ANSWER_RE.search(text or "")
    return match.group(1) if match else None


def gold_answer(gold):
    """Canonical gold letter. The comparison is case-insensitive (`ignore_case: true`)."""
    return str(gold).strip().upper()


def option_contents(options):
    """Validate and copy the target's ordered answer-option strings."""
    if not isinstance(options, (list, tuple)) or not options or len(options) > 10:
        raise ValueError("MMLU-Pro options must be a non-empty sequence of at most ten strings")
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("MMLU-Pro options must contain non-empty strings")
    return list(options)


def selected_option_content(options, answer):
    """Exact content selected by an extracted A--J answer, or an explicit unavailable marker."""
    answer = gold_answer(answer)
    if len(answer) != 1 or answer not in "ABCDEFGHIJ":
        raise ValueError("MMLU-Pro selected answer must be one letter from A through J")
    choices = option_contents(options)
    index = ord(answer) - ord("A")
    return choices[index] if index < len(choices) else UNAVAILABLE_OPTION


def is_correct(completion, gold):
    """Whether lm_eval would score this completion correct against `gold`.

    Judged on the completion ALONE, never on prompt + completion: the prompt carries five
    demonstrations that each end "The answer is (X)", and the filter is ``take_first``, so a
    whole-text check reads the first demonstration's letter instead of the model's answer.
    """
    extracted = extract_answer(completion)
    return extracted is not None and gold_answer(extracted) == gold_answer(gold)


TRACE_SFT_ADAPTER = trace_sft.TaskAdapter(
    task_name=TASK_NAME,
    label="MMLU-Pro",
    example_id=example_id,
    is_correct=is_correct,
    extract_answer=extract_answer,
    filter_name=FILTER_NAME,
    split_description="MMLU-Pro's SFT split",
    gold_answer=gold_answer,
    # A letter is already canonical, so comparing an extracted answer and normalising a gold answer
    # are the same operation here -- unlike GSM8K, where gold is a full solution.
    normalize=gold_answer,
)


def _column(docs, field):
    """`field` for every doc, whether `docs` is a HF Dataset or a list of dicts."""
    try:
        return list(docs[field])           # Dataset: one columnar read
    except (KeyError, TypeError):
        return [d[field] for d in docs]    # list of dicts


def stratified_three_way_split(docs, n_train, n_valid, seed=0, fields=STRATA_FIELDS):
    """Carve `docs` into (train, valid, test) index lists, stratified on `fields`.

    Every stratum contributes to all three splits in proportion to its own size, so the three
    splits match the pool's composition and each other's.

    Method is randomised systematic sampling. Each stratum is shuffled, and its j-th item gets the
    key `(j + u) / m` -- its quantile within the stratum, offset by one uniform draw `u` shared by
    that stratum. Sorting every item by that key and slicing at `n_train` and `n_train + n_valid`
    gives each stratum a proportional share of each split, while the slice points fix the global
    totals exactly.

    The `u` offset is what makes small strata behave. Without it a singleton stratum always keys to
    0.5 and every singleton lands in the same split; with it, singletons distribute across the three
    splits in proportion to the split sizes, which is the best that can be done when a stratum
    cannot be divided at all.

    Deterministic in `seed` alone: `random.Random` rather than numpy, so the carve does not move
    when numpy changes its bit generator.
    """
    n = len(docs)
    if n_train + n_valid >= n:
        raise ValueError(
            "n_train + n_valid = %d leaves no test rows out of %d" % (n_train + n_valid, n)
        )

    columns = {f: _column(docs, f) for f in fields}
    strata = {}
    for i in range(n):
        strata.setdefault(tuple(str(columns[f][i]) for f in fields), []).append(i)

    rng = random.Random(seed)
    keyed = []
    for key in sorted(strata):             # sorted: the shuffle order cannot depend on dict order
        members = strata[key]
        rng.shuffle(members)
        offset = rng.random()
        size = len(members)
        for j, i in enumerate(members):
            keyed.append(((j + offset) / size, key, i))
    keyed.sort()

    order = [i for _, _, i in keyed]
    return (order[:n_train],
            order[n_train:n_train + n_valid],
            order[n_train + n_valid:])


def carve_example_ids(docs_by_subtask, seed=0,
                      n_train=SFT_SIZE, n_valid=PREDICTOR_TRAIN_SIZE):
    """Three sets of `example_id`, carved once over the POOLED docs of every subtask.

    Defining the carve on the pool and then letting each subtask keep its own category's slice
    gives exactly the partition a single flat task would have produced, so the split sizes are the
    pre-registered 6,000 / 3,000 / 3,032 rather than fourteen independent roundings, and the
    stratification still sees whole categories. Subtasks are visited in sorted order so the pooled
    index a document receives cannot depend on dictionary ordering.
    """
    pool = []
    for name in sorted(docs_by_subtask):
        for doc in docs_by_subtask[name]:
            pool.append({"question_id": example_id(doc),
                         **{f: doc[f] for f in STRATA_FIELDS}})
    idx_train, idx_valid, idx_test = stratified_three_way_split(pool, n_train, n_valid, seed)
    return tuple({pool[i]["question_id"] for i in group}
                 for group in (idx_train, idx_valid, idx_test))


def _extraction_failed(sample):
    """Whether extraction failed for one lm_eval sample.

    Reads `filtered_resps` -- the filter's own output -- rather than re-running the regex, so this
    reports what actually scored rather than what would score under a reimplementation.
    """
    filtered = sample.get("filtered_resps")
    if not filtered:
        return True
    first = filtered[0]
    if isinstance(first, (list, tuple)):
        first = first[0] if first else INVALID
    return str(first).strip() == INVALID


def pooled_samples(results):
    """Every subtask's samples, in sorted subtask order, as (subtask, sample) pairs."""
    samples = (results or {}).get("samples", {})
    present = [name for name in SUBTASKS if samples.get(name)]
    if not present:
        raise ValueError(
            "results hold no MMLU-Pro subtask samples (found %s)" % sorted(samples))
    return [(name, row) for name in present for row in samples[name]
            if row.get("filter") in (None, FILTER_NAME)]


def validate_sft_protocol(task_name, dataset_param, apply_chat_template):
    trace_sft.validate_sft_protocol(TRACE_SFT_ADAPTER, task_name, dataset_param, apply_chat_template)


def validate_eval_protocol(task_name, dataset_param, max_new_tokens, apply_chat_template):
    # The reason the fourteen subtasks are run instead of one flattened task: lm_eval owns the
    # demonstrations, so a chat model can receive them as user/assistant turns exactly as it does
    # on GSM8K.
    trace_sft.validate_eval_protocol(
        TRACE_SFT_ADAPTER, task_name, dataset_param, max_new_tokens, apply_chat_template
    )


def evaluation_diagnostics(results, tokenizer, max_new_tokens, expected_shots=EVAL_SHOTS):
    """Accuracy and failure modes over the pooled samples of all fourteen subtasks.

    The protocol check flags only prompts with **too few** demonstrations. That asymmetry is
    deliberate: dropping the shots is the failure, and a question whose own text happens to contain
    the answer phrase can only push the count up, never down. Counting exactly would fail such a
    prompt, and evaluate_model raises on a protocol failure after generation has been paid for.
    """
    selected = pooled_samples(results)
    if not selected:
        raise ValueError(
            "MMLU-Pro results hold subtask samples but none under filter %r" % FILTER_NAME
        )
    report = trace_sft.generation_diagnostics(
        [row for _, row in selected], tokenizer, max_new_tokens,
        extraction_failed=lambda row, response: _extraction_failed(row),
        protocol_failed=lambda context: len(ANSWER_RE.findall(context)) < expected_shots,
    )
    by_subtask = {}
    for subtask, row in selected:
        cell = by_subtask.setdefault(subtask, {"n": 0, "correct": 0.0, "extraction_failures": 0})
        cell["n"] += 1
        cell["correct"] += float(row.get("exact_match", row.get("acc", 0.0)) or 0.0)
        cell["extraction_failures"] += _extraction_failed(row)
    report["subtasks"] = len(by_subtask)
    # Reported per subtask because teacher acceptance is category-dependent, so the SFT mixture
    # is tilted relative to the test split and per-category movement is what shows it.
    report["by_subtask"] = {name: {"n": c["n"], "acc": c["correct"] / c["n"],
                                   "extraction_failure_rate": c["extraction_failures"] / c["n"]}
                            for name, c in sorted(by_subtask.items())}
    return report
