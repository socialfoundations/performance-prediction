"""Cascade metrics and sampled-answer records shared by sampling and analysis.

Routing replaces the weak answer, including when the strong model makes it wrong. Exact
score ties are randomized in expectation so input order cannot change the measured AUC.
"""
import json
from pathlib import Path

import numpy as np

from pproute.consistency import FAILURE, aggregate
from pproute.provenance import atomic_json_dump, identified, validate_identity


def publish_once(payload, path):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Existing cascade artifact differs: {path}")
    else:
        atomic_json_dump(payload, path)


def curve(weak, strong, confidence):
    weak, strong, confidence = (np.asarray(x, dtype=float) for x in
                                (weak, strong, confidence))
    if weak.ndim != 1 or weak.size == 0 or strong.shape != weak.shape or confidence.shape != weak.shape:
        raise ValueError("Cascade inputs must be nonempty aligned vectors")
    if not np.isin(weak, (0, 1)).all() or not np.isin(strong, (0, 1)).all():
        raise ValueError("Correctness labels must be binary")
    if not np.isfinite(confidence).all():
        raise ValueError("Routing scores must be finite")
    order = np.argsort(confidence, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(confidence[order])) + 1, weak.size]
    rho = np.r_[0, ends] / weak.size
    accuracy = np.r_[weak.mean(), weak.mean() + np.cumsum((strong-weak)[order])[ends-1]/weak.size]
    auc = float(np.sum(np.diff(rho) * (accuracy[:-1]+accuracy[1:])/2))
    random_auc = float((weak.mean()+strong.mean())/2)
    return dict(rho=rho.tolist(), accuracy=accuracy.tolist(), auc=auc,
                random_auc=random_auc, excess_auc=auc-random_auc,
                weak_accuracy=float(weak.mean()), strong_accuracy=float(strong.mean()),
                repairs=int(((weak == 0) & (strong == 1)).sum()),
                regressions=int(((weak == 1) & (strong == 0)).sum()))


def extract_answer(task, completion):
    # Keep the registered generative-task extractors exactly, including MMLU's first match.
    if task == "gsm8k":
        from pproute.gsm8k import extract_answer as extract, normalize_extracted_answer
        answer = extract(completion)
        return normalize_extracted_answer(answer) if answer is not None else None
    if task == "mmlu_pro":
        from pproute.mmlu_pro import extract_answer as extract
        return extract(completion)
    raise ValueError(f"Unsupported cascade generation task: {task}")


def summarize_answers(task, doc, completions):
    if not completions:
        raise ValueError("Cannot aggregate no completions")
    answers = [extract_answer(task, row["text"]) for row in completions]
    majority, share, counts = aggregate(answers, task=task)
    # Cascade records store failures separately; weak-model records include a failure bucket.
    n_fail = counts.pop(FAILURE, 0)
    return dict(id=doc["id"], gold=doc["gold"], majority=majority,
                z=int(majority is not None and majority == doc["gold"]),
                k=len(answers), share=share,
                counts=dict(counts), answers=answers, n_fail=n_fail,
                cap_hits=sum(row["finish_reason"] == "length" for row in completions),
                output_tokens=sum(row["n_tokens"] for row in completions))


def load_record(path):
    payload = json.loads(Path(path).read_text())
    return validate_identity(payload, str(path))


def prompt_record(task, split, rows, sources):
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Cascade prompts require unique nonempty document IDs")
    if any(not row["prompt"] or not row["gold"] for row in rows):
        raise ValueError("Cascade prompts require question text and gold labels")
    return identified(dict(schema="pproute-cascade-prompts-v1", task=task, split=split,
                           rows=rows, sources=sources))
