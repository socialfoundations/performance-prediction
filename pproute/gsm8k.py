"""GSM8K's exact answer contract and the generated-trace SFT boundary.

The benchmark's reported Z is lm-eval's ``flexible-extract`` filter: take the last
number-like match, then exact-match after removing commas, dollar signs, the gold
``####`` prefix, and a final period.

Trace validation, confidence resampling, and evaluation all call the functions here rather than
keeping their own copies. That is a correctness requirement: accepted SFT completions and sampled
answers must be judged by the same extractor that defines the benchmark metric.
"""

import hashlib
import json
import re

from pproute import trace_sft

TASK_NAME = "gsm8k"
FILTER_NAME = "flexible-extract"
ANSWER_PATTERN = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
ANSWER_RE = re.compile(ANSWER_PATTERN)
IGNORE_PATTERNS = (r",", r"\$", r"(?s).*#### ", r"\.$")
INVALID = "[invalid]"
EVAL_SHOTS = trace_sft.EVAL_SHOTS
EVAL_MAX_NEW_TOKENS = trace_sft.EVAL_MAX_NEW_TOKENS


def example_id(doc):
    """Stable identity for a GSM8K row; independent of its position in a split."""
    payload = json.dumps(
        {"question": doc["question"], "answer": doc["answer"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def extract_answer(text):
    """Exactly what lm-eval's flexible RegexFilter extracts, or None on failure."""
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return None
    match = matches[-1]  # group_select=-1
    if isinstance(match, tuple):
        match = next((part for part in match if part), INVALID)
    match = str(match).strip()
    return None if match == INVALID else match


def normalize_answer(text):
    value = str(text)
    for pattern in IGNORE_PATTERNS:
        value = re.sub(pattern, "", value)
    return value.lower()


def normalize_extracted_answer(text):
    """Normalize one extracted sample, returning None when normalization erases it."""
    value = normalize_answer(text)
    return value or None


def gold_answer(gold):
    """Canonical gold string from either a full GSM8K solution or its final answer."""
    return normalize_answer(gold)


def is_correct(completion, gold):
    extracted = extract_answer(completion)
    return extracted is not None and normalize_extracted_answer(extracted) == gold_answer(gold)


TRACE_SFT_ADAPTER = trace_sft.TaskAdapter(
    task_name=TASK_NAME,
    label="GSM8K",
    example_id=example_id,
    is_correct=is_correct,
    extract_answer=extract_answer,
    filter_name=FILTER_NAME,
    split_description="GSM8K's SFT split",
    gold_answer=gold_answer,
    normalize=normalize_extracted_answer,
)


def validate_sft_protocol(task_name, dataset_param, apply_chat_template):
    trace_sft.validate_sft_protocol(TRACE_SFT_ADAPTER, task_name, dataset_param, apply_chat_template)


def validate_eval_protocol(task_name, dataset_param, max_new_tokens, apply_chat_template):
    trace_sft.validate_eval_protocol(
        TRACE_SFT_ADAPTER, task_name, dataset_param, max_new_tokens, apply_chat_template
    )


def evaluation_diagnostics(results, tokenizer, max_new_tokens, expected_shots=EVAL_SHOTS):
    """Accuracy and failure modes from the exact saved GSM8K samples."""
    samples = (results or {}).get("samples", {}).get(TASK_NAME, [])
    selected = [row for row in samples if row.get("filter") == FILTER_NAME]
    if not selected:
        raise ValueError("GSM8K results contain no flexible-extract samples")
    expected_questions = expected_shots + 1
    return trace_sft.generation_diagnostics(
        selected, tokenizer, max_new_tokens,
        extraction_failed=lambda row, response: extract_answer(response) is None,
        # Question text may itself quote this marker. This is a missing-demonstration
        # diagnostic, not a parser for an exact number of question blocks.
        protocol_failed=lambda context: context.count("Question:") < expected_questions,
    )
