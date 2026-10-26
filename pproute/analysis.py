"""Development-only selection rules that must not be revisited when reporting results.

Both are documented in README.md and used by pproute.selection:

  * `select_model_candidate` -- which seed/checkpoint becomes ``f_after``. Selected on the
    development split using the task's fixed checkpoint-selection metric. Never on test.
  * `select_predictor_candidate` -- the lowest-loss predictor fit on a common validation split.

Model selection reads development scores through pproute.scores. Predictor selection takes
the already computed validation losses; neither selection rule reads test metrics.
"""
import os

import numpy as np

from pproute import scores

EVAL_FILE = "eval_scores.jbl"        # the only split model selection may read


def validation_metric(scores_dir, task_type):
    """The selection objective for one checkpoint, read from its validation scores.

    MC uses option-distribution L_Y (lower is better). Generative checkpoint validation uses
    greedy accuracy (higher is better); sampled-answer L_Y is computed only after endpoint
    selection, so it does not enter the checkpoint-selection rule.
    """
    path = os.path.join(scores_dir, EVAL_FILE)
    if task_type == "mc":
        m = scores.self_metrics(path)
        return {"n": m["n"], "acc": m["acc"], "L_f": m["L_f"],
                "objective": m["L_f"], "higher_is_better": False}
    if task_type == "generative":
        Z = np.array([e["Z"] for e in scores.gen_examples(path)], float)
        return {"n": len(Z), "acc": float(Z.mean()), "L_f": None,
                "objective": float(Z.mean()), "higher_is_better": True}
    raise ValueError("task_type must be 'mc' or 'generative', got %r" % (task_type,))


def select_model_candidate(candidates, task_type):
    """Select one model fit from seed/checkpoint candidates using development only.

    Each candidate must provide ``scores_dir``, ``step`` and ``seed``.  The protocol tie break is
    earliest checkpoint first, then lower training seed; it is explicit here rather than relying
    on whatever order a caller happened to construct its table in.
    """
    if not candidates:
        raise ValueError("no model candidates given")
    table = []
    identities = set()
    for candidate in candidates:
        try:
            scores_dir = candidate["scores_dir"]
            step = int(candidate["step"])
            seed = int(candidate["seed"])
        except KeyError as exc:
            raise ValueError("candidate lacks %s" % exc.args[0]) from exc
        identity = (seed, step)
        if identity in identities:
            raise ValueError("duplicate model candidate seed=%d step=%d" % identity)
        identities.add(identity)
        row = validation_metric(scores_dir, task_type)
        if int(row["n"]) <= 0 or not np.isfinite(float(row["objective"])):
            raise ValueError("model candidate seed=%d step=%d has invalid validation metric" %
                             identity)
        row.update(scores_dir=scores_dir, step=step, seed=seed)
        table.append(row)
    higher = table[0]["higher_is_better"]
    if any(row["higher_is_better"] != higher for row in table):
        raise ValueError("candidate validation objectives disagree on direction")
    best = min(
        table,
        key=lambda row: (
            -row["objective"] if higher else row["objective"],
            row["step"],
            row["seed"],
        ),
    )
    return best, table


def select_predictor_candidate(candidates):
    """Choose minimum validation L_Z(g), breaking an exact tie toward the lower seed."""
    if not candidates:
        raise ValueError("no predictor candidates given")
    rows = []
    seeds = set()
    for candidate in candidates:
        seed = int(candidate["predictor_seed"])
        loss = float(candidate["LZ_pred"])
        if seed in seeds:
            raise ValueError("duplicate predictor seed %d" % seed)
        if not np.isfinite(loss):
            raise ValueError("predictor seed %d has non-finite validation loss" % seed)
        seeds.add(seed)
        rows.append(candidate)
    return min(rows, key=lambda row: (float(row["LZ_pred"]), int(row["predictor_seed"])))
