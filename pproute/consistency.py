"""The final sampling-consistency confidence protocol.

The model's confidence is its own sampled answer distribution: k samples per query, prediction =
the task-registered empirical majority, f_Z = that answer's share, Z = majority answer vs gold.
This module owns the frozen aggregation rules and the ``conf_scores.json``
contract. ``scripts.conf_sample`` produces the files through vLLM and the predictor
pipeline consumes them. Keeping the rules in one place lets every stage enforce one contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile

from pproute.provenance import require_pinned_revision

# Frozen sampling protocol.
K = 128
TEMPERATURE = 0.9
TOP_P = 1.0  # deliberately no nucleus truncation: hard-zeroing tail continuations prunes
             # minority answers and biases the majority share upward (see the spec).
MAX_NEW_TOKENS = 2048

FAILURE = "<extraction-failure>"
# One smoothed answer distribution per question gives both f_Z and the answer loss. Recorded by
# predictor artifacts so every stage uses this estimator.
ESTIMATOR = "bucket-v1"
UNSEEN_PSEUDOCOUNT = 0.5
CONF_SCHEMA = "pproute-consistency-v4"
PROMPTS_SCHEMA = "pproute-confidence-prompts-v2"
GSM8K_PROMPTS_SCHEMA = "pproute-confidence-prompts-v1"


def supports_prompts_schema(task, schema):
    """Whether prepared prompts carry every field needed by the task's current sampler."""
    return schema == prompts_schema_for(task)


def prompts_schema_for(task):
    """Schema emitted for a task; GSM8K's portable prompt contract did not change."""
    if task == "gsm8k":
        return GSM8K_PROMPTS_SCHEMA
    if task == "mmlu_pro":
        return PROMPTS_SCHEMA
    raise ValueError("confidence prompts task must be gsm8k or mmlu_pro")


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def ids_sha256(rows):
    """Hash an ordered confidence-document identity vector."""
    encoded = json.dumps([row["id"] for row in rows], separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shard_documents(docs, num_shards, shard_index):
    """Balanced ordered partition; interleaving shard outputs restores source order."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return list(docs)[shard_index::num_shards]


def request_seed(endpoint_identity, split, doc_id):
    """Stable seed from endpoint plus a typed, split-namespaced document identity."""
    if split not in ("eval", "test"):
        raise ValueError("request seed split must be eval or test")
    source = json.dumps(
        [str(endpoint_identity), split, doc_id], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _summary(counts, first_seen, *, total):
    candidates = [answer for answer in counts if answer != FAILURE]
    if not candidates:
        return None, 0.0
    majority = min(
        candidates, key=lambda answer: (-counts[answer], first_seen.get(answer, total))
    )
    return majority, counts[majority] / total


def bucket_estimates(row, k=K):
    """Confidence f_Z and answer loss L_Y of one row, both from a single answer distribution.

    Every observed outcome, failures included, has mass ``count / (k + 0.5)`` and one unseen-answer
    bucket has mass ``0.5 / (k + 0.5)``, so the masses sum to one. f_Z is the mass of the returned
    majority answer (zero when no sample was valid). L_Y is ``-log`` of the gold answer's mass, or
    of the unseen bucket when gold was never sampled: a bucket loss, not a full-answer loss.
    Sharing one distribution is a correctness requirement, not a convenience: it makes f_Z the
    probability the answer distribution gives its own answer, so the correctness loss of f_Z
    never exceeds L_Y on the same row. The earlier pair (clipped ``count / k`` against
    ``(count + 0.5) / (k + 1)``) charged a unanimous wrong answer 13.8 and 5.55 nats.
    """
    counts = row["counts"]
    total = k + UNSEEN_PSEUDOCOUNT
    majority = row.get("majority")
    fz = counts[majority] / total if majority is not None and majority != FAILURE else 0.0
    gold = counts.get(str(row["gold"]), counts.get(row["gold"], 0))
    return fz, -math.log((gold if gold else UNSEEN_PSEUDOCOUNT) / total)


def anchor_logit(fz, k=K):
    """Log-odds of f_Z for the predictor anchor; a no-answer row (f_Z = 0) uses the unseen mass."""
    floor = UNSEEN_PSEUDOCOUNT / (k + UNSEEN_PSEUDOCOUNT)
    p = max(float(fz), floor)
    return math.log(p / (1 - p))


def aggregate(answers, *, task):
    """The pre-registered reduction of k extracted answers to (majority, share, Z-ready dist).

    ``answers`` holds one normalized answer string per sample, or None for a sample whose
    generation contained no extractable answer. Rules:
    - extraction failures are pooled and stay in the k denominator;
    - the empirical majority is selected only among successfully extracted answers; failures never
      become a prediction, but still reduce the selected answer's share;
    - an all-failure row has no prediction and confidence zero;
    - argmax ties break to the answer whose first occurrence is earliest in seeded sample
      order, which makes the reduction deterministic given the seeds.
    Returns (majority, share, counts) where counts maps answer -> count over the k samples,
    with all failures pooled under FAILURE.
    """
    if task not in ("gsm8k", "mmlu_pro"):
        raise ValueError("aggregate() task must be gsm8k or mmlu_pro")
    if not answers:
        raise ValueError("aggregate() needs at least one sample")
    counts, first_seen = {}, {}
    for position, answer in enumerate(answers):
        if answer == "":
            raise ValueError("normalized answers must not be empty")
        if answer is None:
            answer = FAILURE
        counts[answer] = counts.get(answer, 0) + 1
        first_seen.setdefault(answer, position)
    majority, share = _summary(counts, first_seen, total=len(answers))
    return majority, share, counts




def write_conf_scores(path, meta, rows):
    """Atomic create-once write of one endpoint's conf_scores.json."""
    payload = {"meta": meta, "rows": rows}
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".partial",
                                     dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)          # fails if the file already exists: create-once
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _reject_nonfinite_json(value):
    raise ValueError("confidence artifact contains non-finite JSON value %s" % value)


def read_conf_scores(path, expected_k=K, expected_temperature=TEMPERATURE,
                     expected_splits=("eval", "test")):
    """Load and validate one canonical confidence artifact."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_nonfinite_json)
    if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict) or \
            not isinstance(payload.get("rows"), list):
        raise ValueError("%s is not a confidence artifact" % path)
    meta, rows = payload["meta"], payload["rows"]
    schema = meta.get("schema")
    task = meta.get("task")
    if schema != CONF_SCHEMA:
        raise ValueError("%s has unknown confidence schema %r" % (path, schema))
    if expected_k != K or expected_temperature != TEMPERATURE:
        raise ValueError("confidence checks must use frozen k/temperature")
    for key in ("k", "temperature", "top_p", "max_new_tokens", "stop",
                "task", "scores_dir", "model", "revision", "endpoint", "model_path",
                "endpoint_manifest", "endpoint_identity", "endpoint_manifest_sha256",
                "scores_sha256", "prompts_file", "prompts_sha256", "fingerprints",
                "splits", "n_eval", "n_test", "subset"):
        if key not in meta:
            raise ValueError("%s meta lacks %r" % (path, key))
    for key, expected in (("k", K), ("temperature", TEMPERATURE), ("top_p", TOP_P),
                          ("max_new_tokens", MAX_NEW_TOKENS)):
        if meta[key] != expected:
            raise ValueError("%s sampled %s=%r, expected %r" %
                             (path, key, meta[key], expected))
    require_pinned_revision(meta["revision"], "%s confidence model revision" % path)
    if not isinstance(meta["task"], str) or not meta["task"] or \
            not isinstance(meta["model"], str) or not meta["model"]:
        raise ValueError("%s has an empty task/model identity" % path)
    if meta["endpoint"] not in ("before", "after"):
        raise ValueError("%s has invalid endpoint %r" % (path, meta["endpoint"]))
    for key in ("scores_dir", "model_path", "endpoint_manifest", "prompts_file"):
        if not isinstance(meta[key], str) or not os.path.isabs(meta[key]):
            raise ValueError("%s meta %s is not an absolute path" % (path, key))
    if not isinstance(meta["stop"], list) or any(
            not isinstance(value, str) or not value for value in meta["stop"]):
        raise ValueError("%s has invalid stop strings" % path)
    expected_splits = tuple(expected_splits)
    if not expected_splits or len(set(expected_splits)) != len(expected_splits) or \
            not set(expected_splits) <= {"eval", "test"}:
        raise ValueError("expected_splits must be a non-empty subset of eval/test")
    declared_splits = meta.get("splits")
    if not isinstance(declared_splits, list) or \
            any(split not in ("eval", "test") for split in declared_splits) or \
            len(set(declared_splits)) != len(declared_splits) or \
            set(declared_splits) != set(expected_splits):
        raise ValueError("%s split declaration disagrees" % path)
    fingerprints = meta["fingerprints"]
    if not isinstance(fingerprints, dict) or not all(
            fingerprints.get(split + "_split") for split in expected_splits):
        raise ValueError("%s lacks fingerprints for %s" % (path, expected_splits))
    if not _is_sha256(meta.get("endpoint_identity")) or \
            not _is_sha256(meta.get("endpoint_manifest_sha256")):
        raise ValueError("%s has invalid endpoint hashes" % path)
    score_hashes = meta.get("scores_sha256")
    if not isinstance(score_hashes, dict) or set(score_hashes) != set(expected_splits) or \
            not all(_is_sha256(value) for value in score_hashes.values()):
        raise ValueError("%s has invalid score hashes/splits" % path)
    if not _is_sha256(meta.get("prompts_sha256")):
        raise ValueError("%s has invalid prepared-prompt hash" % path)
    if any(isinstance(meta.get("n_" + split), bool) or
           not isinstance(meta.get("n_" + split), int) or meta["n_" + split] < 0
           for split in ("eval", "test")):
        raise ValueError("%s has invalid split counts" % path)

    from pproute import trace_sft

    trace_adapter = trace_sft.adapter_for(meta["task"])
    seen = {"eval": set(), "test": set()}
    for row in rows:
        split = row.get("split")
        if split not in seen:
            raise ValueError("%s has unknown split %r" % (path, split))
        doc_id = row.get("id")
        if isinstance(doc_id, bool) or not isinstance(doc_id, (int, str)):
            raise ValueError("%s has invalid %s id %r" % (path, split, doc_id))
        if doc_id in seen[split]:
            raise ValueError("%s duplicates %s id %r" % (path, split, row.get("id")))
        seen[split].add(doc_id)
        if not isinstance(row.get("gold"), str) or not row["gold"] or \
                not isinstance(row.get("predictor_text"), str) or not row["predictor_text"].strip():
            raise ValueError("%s row %r lacks gold/predictor text" % (path, doc_id))
        counts = row.get("counts")
        if not isinstance(counts, dict) or not counts or any(
                not isinstance(answer, str) or not answer or
                not isinstance(count, int) or count <= 0
                for answer, count in counts.items()):
            raise ValueError("%s row %r has invalid counts" % (path, row.get("id")))
        if sum(counts.values()) != K:
            raise ValueError("%s row %r counts do not sum to k" % (path, row.get("id")))
        if meta["task"] == "mmlu_pro" and meta["schema"] == CONF_SCHEMA and any(
                answer != FAILURE and answer not in "ABCDEFGHIJ" for answer in counts):
            raise ValueError("%s row %r has a non-MMLU-Pro answer category" %
                             (path, row.get("id")))
        majority = row.get("majority")
        if majority is not None and majority not in counts:
            raise ValueError("%s row %r majority is absent from counts" %
                             (path, row.get("id")))
        share = float(row.get("share", float("nan")))
        expected_majority, expected_share = _summary(
            counts, row.get("first_seen", {}), total=K
        )
        if not math.isfinite(share) or abs(share - expected_share) > 1e-9:
            raise ValueError("%s row %r share disagrees with counts" % (path, row.get("id")))
        expected_z = int(majority is not None and majority != FAILURE and
                         majority == row.get("gold"))
        if row.get("z") not in (0, 1) or row.get("z") != expected_z:
            raise ValueError("%s row %r z disagrees with majority and gold" %
                             (path, row.get("id")))
        if row.get("n_fail") != counts.get(FAILURE, 0):
            raise ValueError("%s row %r n_fail disagrees with counts" %
                             (path, row.get("id")))
        if not isinstance(row.get("cap_hits"), int) or not 0 <= row["cap_hits"] <= K:
            raise ValueError("%s row %r has invalid cap_hits" % (path, row.get("id")))
        first_seen = row.get("first_seen")
        if not isinstance(first_seen, dict) or set(first_seen) != set(counts) or any(
                not isinstance(position, int) or not 0 <= position < K
                for position in first_seen.values()) or \
                len(set(first_seen.values())) != len(first_seen):
            raise ValueError("%s row %r has invalid first_seen" % (path, row.get("id")))
        if majority != expected_majority:
            raise ValueError("%s row %r violates earliest-occurrence tie break" %
                             (path, row.get("id")))
        reasons = row.get("finish_reasons")
        if not isinstance(reasons, dict) or not reasons or any(
                not isinstance(reason, str) or not reason or
                not isinstance(count, int) or count <= 0
                for reason, count in reasons.items()
        ) or sum(reasons.values()) != K or row["cap_hits"] != reasons.get("length", 0):
            raise ValueError("%s row %r has invalid finish_reasons" %
                             (path, row.get("id")))
        trace = row.get("majority_trace")
        if (majority in (None, FAILURE) and trace is not None) or \
                (majority not in (None, FAILURE) and
                 (not isinstance(trace, str) or not trace.strip())):
            raise ValueError("%s row %r has invalid majority trace" % (path, row.get("id")))
        if majority not in (None, FAILURE):
            extracted = trace_adapter.extract_answer(trace)
            normalized = trace_adapter.normalize(extracted) if extracted is not None else None
            if normalized != majority:
                raise ValueError("%s row %r majority trace extracts as %r, not %r" %
                                 (path, row.get("id"), normalized, majority))
        if meta["task"] == "mmlu_pro" and meta["schema"] == CONF_SCHEMA:
            option = row.get("majority_option")
            if (majority is None and option is not None) or \
                    (majority is not None and
                     (not isinstance(option, str) or not option.strip())):
                raise ValueError("%s row %r has invalid majority option" %
                                 (path, row.get("id")))

    actual_splits = {split for split, values in seen.items() if values}
    if actual_splits != set(expected_splits):
        raise ValueError("%s has splits %s, expected %s" %
                         (path, sorted(actual_splits), sorted(expected_splits)))
    for split in ("eval", "test"):
        if len(seen[split]) != meta["n_" + split]:
            raise ValueError("%s %s row count %d != meta %d" %
                             (path, split, len(seen[split]), meta["n_" + split]))
    subset = meta.get("subset")
    if not isinstance(subset, dict) or subset.get("method") != "all":
        raise ValueError("%s has invalid subset declaration" % path)
    for split in expected_splits:
        members = [row for row in rows if row["split"] == split]
        if subset.get(split + "_ids_sha256") != ids_sha256(members):
            raise ValueError("%s %s ID hash disagrees with rows" % (path, split))
    return payload

def extract_prompts(scores_path, adapter):
    """(docs, stop_strings) for one split of a greedy score file: the prompts the sampler replays.

    MMLU-Pro score payloads import ``lm_eval`` task objects; the sampler consumes this portable
    plain-JSON projection instead of unpickling or re-rendering them. The protocol requires the
    task-native predictor-text stamp used by its target-only predictor.

    Every lm_eval task key in the file is pooled. Ids are namespaced ``subtask:doc_id`` when the
    file holds more than one task, matching pproute.scores._samples: doc_id restarts at zero inside
    each subtask, so a bare id collides across categories AND still resolves, which downstream would
    join silently wrong rather than fail.

    Nothing is rendered here -- prompts are copied out of the score file as stored.
    """
    import joblib

    payload = joblib.load(scores_path)
    task_keys = sorted(payload["samples"])
    single = len(task_keys) == 1
    untils = {tuple(payload["configs"][key]["generation_kwargs"]["until"]) for key in task_keys}
    if len(untils) != 1:
        raise ValueError("%s: task keys disagree on stop strings: %s"
                         % (scores_path, sorted(untils)))
    docs = []
    for task_key in task_keys:
        for sample in payload["samples"][task_key]:
            if sample.get("filter") not in (None, adapter.filter_name):
                continue
            if "pproute_predictor_text" not in sample:
                raise ValueError(
                    "%s sample %s lacks the task-native predictor-text stamp"
                    % (scores_path, sample.get("doc_id"))
                )
            doc_id = sample["doc_id"] if single else "%s:%s" % (task_key, sample["doc_id"])
            doc = {
                "id": doc_id,
                "prompt": sample["arguments"][0][0],
                "predictor_text": sample["pproute_predictor_text"],
                "gold": adapter.gold_answer(sample["doc"]["answer"]),
            }
            if adapter.task_name == "mmlu_pro":
                from pproute import mmlu_pro

                doc["options"] = mmlu_pro.option_contents(sample["doc"].get("options"))
            docs.append(doc)
    ids = [doc["id"] for doc in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("%s has duplicate doc ids after filter dedup" % scores_path)
    if not docs:
        raise ValueError("%s yields no %s documents" % (scores_path, adapter.filter_name))
    return docs, list(next(iter(untils)))
