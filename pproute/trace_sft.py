"""Task-agnostic generated-trace SFT contracts.

Benchmark modules own their split, stable example identity, prompt rendering, and exact
correctness extractor.  This module owns the reusable boundary after those choices have been
made: immutable prompt/completion rows, SFT-split allowlisting, uniform provenance, and
completion-only truncation diagnostics.

The canonical trace schema is ``{example_id, gold, completion, provenance}``.  It deliberately
does not consume a prejoined ``text`` field: prompts are rendered by the current task/tokenizer and
kept separate from completions so TRL can mask every prompt token.
"""

from dataclasses import dataclass
import json
from typing import Any, Callable, Optional

from pproute.provenance import sha256_file as file_sha256

# The frozen generative evaluation protocol shared by GSM8K and MMLU-Pro. evaluate_model's
# --max_new_tokens is passed into gen_kwargs unconditionally, which OVERRIDES the task YAML's own
# max_gen_toks; leaving it at the default truncates the chain of thought before it can reach the
# final answer, turning capability into an extraction failure on every model.
EVAL_SHOTS = 5
EVAL_MAX_NEW_TOKENS = 2048


@dataclass(frozen=True)
class TaskAdapter:
    """Task-specific operations shared by trace SFT and confidence resampling."""

    task_name: str
    example_id: Callable[[dict], str]
    is_correct: Callable[[str, str], bool]
    extract_answer: Optional[Callable[[str], Any]] = None
    split_description: str = "the task's SFT split"

    # Canonical gold string for a dataset row and the normaliser for an extracted sample.
    gold_answer: Optional[Callable[[Any], str]] = None
    normalize: Optional[Callable[[str], Optional[str]]] = None
    # Which lm_eval filter defines Z. The sampling-consistency arm must select the same filter the
    # greedy run scored under, and it differs per benchmark ("flexible-extract" for GSM8K,
    # "custom-extract" for MMLU-Pro) -- hardcoding either silently drops every sample of the other.
    filter_name: str = ""
    # Display name used in protocol error messages.
    label: str = ""


def adapter_for(task_name):
    """Return the one active generative benchmark adapter."""
    from pproute import gsm8k, mmlu_pro

    adapters = {
        gsm8k.TASK_NAME: gsm8k.TRACE_SFT_ADAPTER,
        mmlu_pro.TASK_NAME: mmlu_pro.TRACE_SFT_ADAPTER,
    }
    try:
        return adapters[task_name]
    except KeyError as exc:
        raise ValueError("no generated-trace adapter for %r; expected one of %s" %
                         (task_name, sorted(adapters))) from exc


def validate_sft_protocol(adapter, task_name, dataset_param, apply_chat_template):
    """Fail before loading weights when generated-trace SFT would train off the eval format.

    A checkpoint trained in one prompt format and scored in another measures the format change
    rather than the finetuning, and nothing downstream can notice afterwards.
    """
    if task_name != adapter.task_name:
        return
    if dataset_param["num_fewshot"] not in {0, EVAL_SHOTS}:
        raise ValueError("%s trace SFT prompt must be 0-shot or 5-shot" % adapter.label)
    # 0-shot SFT against a 5-shot evaluation is a deliberate arm, so the shot count is allowed to
    # differ; what may not differ is whether demonstrations become turns given that a chat template
    # is in play, since that changes the rendering of the demonstrations that ARE present.
    should_multiturn = bool(apply_chat_template and dataset_param["num_fewshot"])
    if dataset_param["fewshot_as_multiturn"] != should_multiturn:
        raise ValueError(
            "%s trace SFT requires fewshot_as_multiturn exactly for 5-shot chat SFT" % adapter.label
        )


def validate_eval_protocol(adapter, task_name, dataset_param, max_new_tokens, apply_chat_template):
    """Fail before loading weights when a generative evaluation would silently be off protocol.

    Chat models must receive the demonstrations as user/assistant turns: scoring a chat model
    without `fewshot_as_multiturn` would reintroduce the prompt-format difference this design
    exists to remove.
    """
    if task_name != adapter.task_name:
        return
    if dataset_param["num_fewshot"] != EVAL_SHOTS:
        raise ValueError("%s evaluation is frozen at 5-shot" % adapter.label)
    if max_new_tokens != EVAL_MAX_NEW_TOKENS:
        raise ValueError("%s evaluation is frozen at max_new_tokens=2048" % adapter.label)
    if apply_chat_template and not dataset_param["fewshot_as_multiturn"]:
        raise ValueError("5-shot %s chat evaluation requires fewshot_as_multiturn" % adapter.label)
    if not apply_chat_template and dataset_param["fewshot_as_multiturn"]:
        raise ValueError("raw-completion %s cannot use fewshot_as_multiturn" % adapter.label)


def response_text(sample):
    """The first generated response of one lm_eval sample, or an empty string."""
    responses = sample.get("resps") or []
    if not responses:
        return ""
    first = responses[0]
    if isinstance(first, (list, tuple)):
        first = first[0] if first else ""
    return str(first)


def generation_diagnostics(rows, tokenizer, max_new_tokens, *, extraction_failed, protocol_failed):
    """Accuracy and failure modes over saved generative samples.

    Z here is one regex, so the two failure modes have to be told apart from being wrong: a
    response that never emits an answer scores zero whether the model knew the answer or spent its
    whole budget reasoning. `extraction_failures` and `generation_cap_hits` separate them, and only
    the second is fixed by a larger budget.

    ``extraction_failed(row, response)`` and ``protocol_failed(context)`` are the task's own rules.
    The protocol check reads the saved prompt, because the saved prompts and not the requested
    config are the evidence that the shots survived request construction.
    """
    correct = extraction_failures = cap_hits = protocol_failures = 0
    prompt_lengths, response_lengths = [], []
    for row in rows:
        response = response_text(row)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        context = row["arguments"][0][0]
        correct += float(row.get("exact_match", row.get("acc", 0.0)) or 0.0)
        extraction_failures += bool(extraction_failed(row, response))
        cap_hits += len(response_tokens) >= max_new_tokens
        protocol_failures += bool(protocol_failed(context))
        prompt_lengths.append(len(tokenizer.encode(context, add_special_tokens=False)))
        response_lengths.append(len(response_tokens))
    n = len(rows)
    return {
        "n": n,
        "accuracy": correct / n,
        "extraction_failures": extraction_failures,
        "extraction_failure_rate": extraction_failures / n,
        "generation_cap_hits": cap_hits,
        "generation_cap_hit_rate": cap_hits / n,
        "prompt_protocol_failures": protocol_failures,
        "prompt_tokens_max": max(prompt_lengths),
        "response_tokens_max": max(response_lengths),
    }


def read_jsonl(path):
    rows = []
    with open(path) as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "%s:%d is not valid JSON: %s" % (path, lineno, exc)
                ) from exc
    return rows


def train_only_dataset_params(trace_path, dataset_param):
    """Generated traces may construct only the task's registered SFT split."""
    params = dict(dataset_param)
    if trace_path:
        params["max_num_valid"] = 0
        params["max_num_test"] = 0
    return params


def load_accepted_traces(path, adapter, allowed_ids=None):
    """Load exact-correct traces while enforcing identity, split, and provenance."""
    allowed_ids = None if allowed_ids is None else {str(value) for value in allowed_ids}
    traces = {}
    provenance = None
    for lineno, row in enumerate(read_jsonl(path), 1):
        qid = str(row.get("example_id", ""))
        if not qid:
            raise ValueError("%s:%d has no example_id" % (path, lineno))
        if qid in traces:
            raise ValueError("%s contains duplicate example_id %s" % (path, qid))
        if allowed_ids is not None and qid not in allowed_ids:
            raise ValueError(
                "%s:%d trains on %s, which is NOT in %s"
                % (path, lineno, qid, adapter.split_description)
            )

        completion, gold = row.get("completion", ""), row.get("gold", "")
        if not adapter.is_correct(completion, gold):
            extracted = (
                adapter.extract_answer(completion)
                if adapter.extract_answer is not None
                else "<task extractor rejected completion>"
            )
            raise ValueError(
                "%s:%d completion extracts to %r, not gold %r"
                % (path, lineno, extracted, gold)
            )

        row_provenance = row.get("provenance")
        if not isinstance(row_provenance, dict):
            raise ValueError("%s:%d has no provenance object" % (path, lineno))
        if row_provenance.get("task") != adapter.task_name:
            raise ValueError(
                "%s:%d trace provenance task %r does not match %r"
                % (path, lineno, row_provenance.get("task"), adapter.task_name)
            )
        if provenance is None:
            provenance = row_provenance
        elif row_provenance != provenance:
            raise ValueError("%s mixes trace provenance at line %d" % (path, lineno))
        traces[qid] = row

    if not traces:
        raise ValueError("%s contains no accepted traces" % path)
    return traces, {
        "path": str(path),
        "sha256": file_sha256(path),
        "n": len(traces),
        "task_name": adapter.task_name,
        "id_field": "example_id",
        "provenance": provenance,
    }


def build_trace_sft_dataset(tasks, trace_path, completion_source, adapter):
    """Build prompt/completion rows from the current rendered SFT requests.

    ``tasks`` is one lm_eval task or several. A benchmark that ships as several -- MMLU-Pro's
    fourteen per-category subtasks -- holds its SFT split across all of them, so the allowlist, the
    duplicate check and the "every accepted trace was used" check all span the whole set. Doing
    this per task instead would report almost every trace as missing, since each subtask carries
    only its own category.
    """
    from datasets import Dataset

    if completion_source != "teacher":
        raise ValueError("completion_source must be 'teacher'")

    tasks = [tasks] if not isinstance(tasks, (list, tuple)) else list(tasks)
    allowed = {str(adapter.example_id(doc)) for task in tasks for doc in task.eval_docs}
    traces, trace_info = load_accepted_traces(
        trace_path, adapter=adapter, allowed_ids=allowed
    )
    rows = {"prompt": [], "completion": [], "example_id": []}
    seen = set()
    for task in tasks:
        for instance in task.instances:
            qid = str(adapter.example_id(instance.doc))
            if qid not in traces:
                continue
            if qid in seen:
                raise ValueError(
                    "%s produced multiple SFT requests for %s" % (adapter.task_name, qid)
                )

            gold = str(task.doc_to_target(instance.doc))
            trace_completion = traces[qid]["completion"]
            if not adapter.is_correct(trace_completion, gold):
                raise ValueError(
                    "trace completion for %s does not match the current task gold" % qid
                )
            completion = trace_completion

            prompt = instance.arguments[0]
            if not isinstance(prompt, str):
                raise ValueError("rendered SFT prompt for %s is not text" % qid)
            rows["prompt"].append(prompt)
            # Keep lm-eval's task-owned target delimiter in the completion. This makes
            # prompt+completion byte-identical while masking every prompt token.
            delimiter = task.get_config("target_delimiter")
            if delimiter is None:
                delimiter = " "
            if not isinstance(delimiter, str):
                raise ValueError("target_delimiter for %s is not text" % adapter.task_name)
            rows["completion"].append(delimiter + str(completion).strip())
            rows["example_id"].append(qid)
            seen.add(qid)

    missing = set(traces) - seen
    if missing:
        raise ValueError(
            "%d accepted traces are absent from the current SFT split (first: %s)"
            % (len(missing), sorted(missing)[:3])
        )
    trace_info = dict(
        trace_info,
        completion_source=completion_source,
        sft_split_n=len(allowed),
    )
    return Dataset.from_dict(rows), trace_info


def training_diagnostics(dataset, tokenizer, max_length):
    """Measure completion masking and the truncation TRL's keep-start path will apply."""
    truncated = fully_masked = completion_tokens_lost = 0
    maximum = 0
    for row in dataset:
        completion = row["completion"]
        if tokenizer.eos_token and not completion.endswith(tokenizer.eos_token):
            completion += tokenizer.eos_token
        prompt_ids = tokenizer(row["prompt"])["input_ids"]
        full_ids = tokenizer(row["prompt"] + completion)["input_ids"]
        maximum = max(maximum, len(full_ids))
        truncated += len(full_ids) > max_length
        fully_masked += len(prompt_ids) >= max_length
        completion_tokens_lost += max(0, len(full_ids) - max(max_length, len(prompt_ids)))
    n = len(dataset)
    return {
        "n": n,
        "max_length": max_length,
        "max_untruncated_tokens": maximum,
        "truncated_examples": truncated,
        "truncation_rate": truncated / n if n else 0.0,
        "fully_masked_examples": fully_masked,
        "completion_tokens_lost": completion_tokens_lost,
    }
