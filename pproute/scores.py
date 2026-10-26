"""Read lm_eval score files and extract per-prompt correctness signals.

Shared by the predictor (features + labels) and the analysis notebooks (loss curves and
uncertainty bands). All quantities are in nats. For multiple-choice tasks the option
distribution is length-normalized for acc_norm tasks and raw otherwise, matching each
task's official metric -- the normalizer is the stamped `doc_to_choice` length, which is what
lm_eval divides by, and NOT the length of the scored request string. See `_choice_lengths`.
"""
import numpy as np
import joblib as jbl
import os

from pproute.provenance import require_pinned_revision

EPS = 1e-6
# tasks whose official lm_eval metric is acc_norm (length-normalized); others use raw acc
ACC_NORM_TASKS = {"hellaswag", "medmcqa", "arc_challenge", "arc_easy",
                  "piqa", "sciq", "openbookqa", "mathqa"}
# placeholder for the varying span in a multiple_input (Winograd) question
BLANK = "___"
# key under which pproute.evaluator stamps the authoritative gold option index into each sample
GOLD_KEY = "gold_idx"
# Task-native, pre-chat text stamped while the lm-eval task object is still available. Generative
# predictors read this instead of trying to invert vendor chat templates (which is not possible
# once few-shot examples have been rendered as separate turns).
PREDICTOR_TEXT_KEY = "pproute_predictor_text"
# Five-shot MC inputs must be reconstructed from target fields, never from a rendered dialogue.
MC_TARGET_KEY = "pproute_mc_target"
# key under which pproute.evaluator stamps len(doc_to_choice(doc)[i]) for each option. Only the
# task object knows the choices, and they are what lm_eval length-normalizes by; see
# _choice_lengths for why the score file cannot supply them on its own.
CHOICE_LENS_KEY = "pproute_choice_lens"
# Which lm_eval filter defines Z on a generative task that ships more than one. Pre-registered in
# See gen_examples for why strict-match cannot be used with base checkpoints.
GEN_FILTER = "flexible-extract"


def _recorder_values(config_path):
    """Read the finalized Recorder paired with a config, or ``None`` for a failed attempt."""
    import json

    path = os.path.splitext(config_path)[0] + ".result"
    if not os.path.isfile(path):
        return None
    values = {}
    finalized = False
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line == "$END$":
                finalized = True
                continue
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                values.update(item)
    if not finalized:
        return None
    return values


def recorded_split_fingerprint(scores_dir, split):
    """Return the unique finalized dataset fingerprint for one scored split."""
    import glob
    import json

    if split not in ("train", "eval", "test"):
        raise ValueError("split must be train, eval, or test")
    fingerprints = set()
    for path in sorted(glob.glob(os.path.join(scores_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception:
            continue
        if not config.get("base_model_name"):
            continue
        values = _recorder_values(path)
        if values is not None and values.get(split + "_split"):
            fingerprints.add(str(values[split + "_split"]))
    if len(fingerprints) != 1:
        raise ValueError("%s has %d finalized %s split fingerprints; expected one" %
                         (scores_dir, len(fingerprints), split))
    return next(iter(fingerprints))


def scored_model_provenance(scores_dir, required=True, require_git=True):
    """Immutable task/model/checkpoint identity recorded by every score run.

    evaluate_model dumps its config and finalized resolved-template record beside the scores, so
    the directory is self-describing; nothing is threaded through or parsed out of a path.
    Validation and delayed-test invocations may leave multiple configs in one endpoint directory;
    they are accepted only when task, model, pinned Hub revision, and checkpoint all agree.
    """
    import glob
    import json
    import os

    identities = set()
    for path in sorted(glob.glob(os.path.join(scores_dir, "*.json"))):
        try:
            with open(path) as fh:
                cfg = json.load(fh)
        except Exception:
            continue
        name = cfg.get("base_model_name")
        if not name:
            continue
        recorder_values = _recorder_values(path)
        if recorder_values is None:
            continue
        task_name = cfg.get("task_name")
        revision = str(cfg.get("revision") or "")
        if not task_name:
            raise ValueError("%s score config has no task_name" % path)
        require_pinned_revision(revision, "%s score config revision" % path)
        checkpoint = str(cfg.get("dir_checkpoint") or "")
        if checkpoint and not os.path.isabs(checkpoint):
            run_dir = str(recorder_values.get("meta_data.run_dir") or "")
            if not os.path.isabs(run_dir):
                raise ValueError(
                    "%s records relative checkpoint %r without an absolute Recorder run_dir"
                    % (path, checkpoint)
                )
            checkpoint = os.path.normpath(os.path.join(run_dir, checkpoint))
        dataset = cfg.get("dataset_param") or {}
        recorder_commit = recorder_values.get("meta_data.git_commit")
        if recorder_commit is not None:
            require_pinned_revision(recorder_commit, "%s Recorder git commit" % path)
        elif require_git:
            raise ValueError("%s Recorder has no git commit" % path)
        chat_info = recorder_values.get("chat_template")
        if not isinstance(chat_info, dict) or "apply_chat_template" not in chat_info:
            raise ValueError("%s has no resolved chat-template record" % path)
        protocol = (
            int(cfg.get("random_seed", 0)),
            int(cfg.get("dataset_seed", cfg.get("random_seed", 0))),
            int(cfg.get("fewshot_random_seed", 1234)),
            int(dataset.get("num_fewshot", 0)),
            bool(dataset.get("fewshot_as_multiturn", False)),
            bool(chat_info["apply_chat_template"]),
            bool(dataset.get("apply_chat_template", False)),
            bool(dataset.get("allow_untemplated_instruct", False)),
            int(cfg.get("max_new_tokens", -1)),
            bool(cfg.get("eval_use_auto_batch_size", False)),
            int(cfg.get("eval_max_batch_size", 32)),
        )
        identities.add(
            (str(task_name), str(name), revision, checkpoint, protocol,
             str(cfg.get("model_dtype", "auto")))
        )
    if len(identities) > 1:
        raise ValueError(
            "%s mixes scored model identities: %s"
            % (scores_dir, sorted(repr(identity) for identity in identities))
        )
    if not identities:
        if required:
            raise ValueError("%s has no recorded scored-model provenance" % scores_dir)
        return None
    task_name, name, revision, checkpoint, protocol, model_dtype = next(iter(identities))
    (seed, dataset_seed, fewshot_seed, shots, multiturn, apply_chat, requested_chat, allow_untemplated,
     max_new_tokens, auto_batch, max_batch_size) = protocol
    return {
        "task_name": task_name,
        "base_model_name": name,
        "revision": revision,
        "dir_checkpoint": checkpoint,
        # Preserve existing auto-dtype records while binding explicit compute overrides.
        **({"model_dtype": model_dtype} if model_dtype != "auto" else {}),
        "prompt_protocol": {
            "random_seed": seed,
            "dataset_seed": dataset_seed,
            "fewshot_random_seed": fewshot_seed,
            "num_fewshot": shots,
            "fewshot_as_multiturn": multiturn,
            "apply_chat_template": apply_chat,
            "requested_apply_chat_template": requested_chat,
            "allow_untemplated_instruct": allow_untemplated,
            "max_new_tokens": max_new_tokens,
            "eval_use_auto_batch_size": auto_batch,
            "eval_max_batch_size": max_batch_size,
        },
    }


def require_requested_model(provenance, base_model_name, revision, dir_checkpoint):
    """Refuse a downstream command that does not name the endpoint which produced its scores."""
    requested = {
        "base_model_name": str(base_model_name),
        "revision": str(revision),
        "dir_checkpoint": str(dir_checkpoint or ""),
    }
    for key, value in requested.items():
        actual = provenance.get(key)
        equal = (
            os.path.realpath(actual) == os.path.realpath(value)
            if key == "dir_checkpoint" and actual and value
            else actual == value
        )
        if not equal:
            raise ValueError(
                "requested %s %r does not match score provenance %r"
                % (key, value, actual)
            )


def softmax(x):
    x = np.asarray(x, float); x = x - x.max(); e = np.exp(x); return e / e.sum()


def bce(z, p):
    """Mean binary cross-entropy of predictions p against labels z, in nats.

    Lives here rather than in predictor.py because it is a metric over score arrays; importing
    predictor.py would pull in torch and peft just to average some logs."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(z * np.log(p) + (1 - z) * np.log(1 - p)).mean())


def _gold_index(it, n_options=None):
    """The gold option index for one lm_eval multiple-choice sample, checked against `n_options`.

    `n_options` is how many options were actually scored. An index outside `[0, n_options)` is
    refused rather than returned. -100 is what `pproute.evaluator` stamps when the task's gold is
    not among its own choices (lm_eval's convention, see `stamp_gold_index`), and Z, f_Z and L_Y
    are all undefined for such a sample -- but `pr[-100]` is not an error on a task with more than
    100 options: it silently selects a real one and yields a perfectly plausible Z and L_Y. Below
    100 options it raises `IndexError` from numpy, which is safe but says nothing about what is
    wrong or what to re-run. Both are handled here, once, with a message.
    """
    gold = _resolve_gold_index(it)
    if n_options is not None and not 0 <= gold < n_options:
        raise ValueError(
            "sample %s carries gold option index %d, outside the %d option(s) that were scored. "
            "-100 is what pproute.evaluator stamps when the task's gold is not among its own "
            "choices at eval time, so Z, f_Z and L_Y are undefined for this sample. It is refused "
            "rather than indexed: pr[%d] selects a real option on any task with more than 100 of "
            "them. Re-run evaluate_model.py for this checkpoint, or exclude the affected task."
            % (it.get("doc_id", "<unknown>"), gold, n_options, gold))
    return gold


def _resolve_gold_index(it):
    """The evaluator-stamped gold option index, before any range check."""
    if GOLD_KEY not in it:
        raise KeyError(
            "multiple-choice sample carries no '%s'. The task object is the only generally "
            "correct source of the option index; re-run evaluate_model.py for this endpoint."
            % GOLD_KEY
        )
    gold = it[GOLD_KEY]
    if isinstance(gold, (list, tuple)):                    # multiple_target task
        raise NotImplementedError(
            "multiple_target task: gold is a set of option indices %s. Z, f_Z and L_Y are "
            "all defined against a single gold option here, so this needs a deliberate "
            "decision rather than silently taking the first." % (list(gold),))
    return int(gold)


def _samples(scores_path, include_config=False):
    """Per-sample records from one score file, pooling every task the file contains.

    A benchmark can ship as several lm_eval tasks -- MMLU-Pro is fourteen per-category subtasks,
    because both its description and its demonstrations are per category while lm_eval applies each
    per task. One evaluation writes them all into one score file and every reported quantity is a
    micro-average over the pool, which is the estimand `g`, the selection rule and the validity
    filter already assume.

    `doc_id` restarts at zero inside each task, so pooled ids are namespaced by task name. A bare
    `doc_id` would collide across subtasks and silently misalign the predictor split -- the failure
    is invisible because every id still resolves.
    """
    d = jbl.load(scores_path)
    keys = sorted(d.get("samples") or {})
    if not keys:
        raise ValueError("%s holds no per-sample records" % scores_path)
    if len(keys) == 1:
        result = (keys[0], d["samples"][keys[0]])
        return (*result, d.get("configs", {})) if include_config else result
    pooled = []
    for tk in keys:
        for it in d["samples"][tk]:
            it["doc_id"] = "%s:%s" % (tk, it.get("doc_id"))
            pooled.append(it)
    # e.g. the fourteen mmlu_pro_* subtasks report as "mmlu_pro".
    result = (os.path.commonprefix(keys).rstrip("_") or keys[0], pooled)
    return (*result, d.get("configs", {})) if include_config else result


def _common_prefix_len(strs):
    n = min(len(s) for s in strs)
    i = 0
    while i < n and len({s[i] for s in strs}) == 1:
        i += 1
    return i


def fill_option(question, option, layout):
    """The question with one option realised: substituted into the blank under multiple_input,
    appended otherwise. Standard-layout questions end where the continuation begins, so appending
    completes them; Winograd-layout questions carry the blank mid-sentence."""
    if layout == "multiple_input":
        return question.replace(BLANK, option.strip(), 1)
    return question + option


def render_options(args):
    """Split an lm_eval multiple-choice request into (question, options, layout).

    lm_eval builds these requests in two layouts, and which one applies is self-describing:

      standard        one shared context, one continuation per option.  args[i][0] is constant.
      multiple_input  one shared continuation, one context per option.  args[i][1] is constant.
                      This is the Winograd shape (winogrande, xwinograd, ...): lm_eval sets it
                      whenever doc_to_text returns an index rather than a string, so `choices`
                      are the contexts. doc_to_choice truncates the sentence at the blank, so
                      the contexts share a prefix and differ only in the trailing option.

    Assuming the standard layout unconditionally silently destroys the multiple_input case: the
    options collapse to one repeated string and the predictor cannot tell them apart. So detect,
    and refuse anything that is neither.

    Returns the displayed options only. Length normalisation must NOT use their lengths: lm_eval
    scores args[i][1] given args[i][0], so the scored string is args[i][1] in both layouts, and
    under multiple_input it is identical across options while the displayed tails differ.
    """
    ctx_const = len({a[0] for a in args}) == 1
    cont_const = len({a[1] for a in args}) == 1

    if ctx_const and not cont_const:                       # standard multiple choice
        return args[0][0], [a[1] for a in args], "standard"

    if cont_const and not ctx_const:                       # multiple_input / Winograd
        ctxs = [a[0] for a in args]
        p = _common_prefix_len(ctxs)
        # Back off to a word boundary: options sharing a leading letter ("blanket"/"bed") would
        # otherwise be split mid-word into "lanket"/"ed".
        sp = ctxs[0].rfind(" ", 0, p)
        p = sp + 1 if sp >= 0 else 0
        # Exactly one space after the blank. lm_eval scores the continuation with its leading space
        # ("' had bees.'") normally but without it under a chat template, which would otherwise
        # render as "the ___had bees." for instruct checkpoints and "the ___ had bees." for base
        # ones -- a systematic difference in g's input that has nothing to do with the model.
        tail = args[0][1]
        question = ctxs[0][:p].rstrip() + " " + BLANK + (tail if tail[:1].isspace() else " " + tail)
        return question, [c[p:] for c in ctxs], "multiple_input"

    raise ValueError(
        "unrecognised multiple-choice request layout: contexts %s, continuations %s "
        "(n=%d). render_options handles exactly one of the two lm_eval layouts; a third "
        "would silently corrupt the predictor's features, so it is refused instead."
        % ("constant" if ctx_const else "vary",
           "constant" if cont_const else "vary", len(args))
    )


def _choice_lengths(it, task_name, n_options):
    """Per-option normalizers for a length-normalized (acc_norm) task.

    lm_eval computes acc_norm as `argmax(lls / completion_len)` with
    `completion_len = [len(c) for c in doc_to_choice(doc)]` (lm_eval/api/task.py,
    ConfigurableTask.process_results). That is the length of the *choice*, not of the string that
    was scored: requests are built as `(ctx, target_delimiter + choice)`, so `arguments[i][1]`
    carries the delimiter as well and normalizing by it disagrees with the benchmark's own metric.
    Measured on a 6,000-sample score file with a one-character delimiter, ~5% of per-prompt labels
    flip -- and the delimiter is not recoverable from the score file either, because lm_eval drops
    it to "" under a chat template without changing the recorded task config.

    So only the task object knows the denominator, exactly as with the gold index, and
    pproute.evaluator stamps it at eval time. A file written before that stamp existed is refused
    rather than approximated: there is no correct guess to make here, and quietly using the wrong
    normalizer would move Z, f_Z, L_Y and every curve built on them.

    The one deliberate divergence: a zero-length choice is clamped to 1 rather than dividing by
    zero. lm_eval would produce -inf there, which can never be the argmax; no real benchmark ships
    an empty choice, and a NaN in `pr` would poison f_Z and L_Y silently.
    """
    lens = it.get(CHOICE_LENS_KEY)
    if lens is None:
        raise KeyError(
            "%s is length-normalized (acc_norm) but sample %s carries no '%s'. The per-option "
            "normalizer is len(doc_to_choice(doc)[i]), which cannot be recovered from the score "
            "file -- arguments[i][1] additionally carries the task's target_delimiter, and "
            "normalizing by it disagrees with lm_eval's own acc_norm. Re-run evaluate_model.py "
            "on this checkpoint: pproute.evaluator now stamps '%s' from the task object."
            % (task_name, it.get("doc_id", "<unknown>"), CHOICE_LENS_KEY, CHOICE_LENS_KEY))
    if len(lens) != n_options:
        raise ValueError(
            "%s sample %s stamped %d choice length(s) for %d scored option(s)"
            % (task_name, it.get("doc_id", "<unknown>"), len(lens), n_options))
    return np.array([max(int(v), 1) for v in lens], float)


def mc_examples(scores_path, strip_ctx=None):
    """Per-example dicts for a multiple-choice task: question, the per-option text + probs,
    predicted/gold index, confidence fz (=max prob) and its logit lfz, correctness Z, and
    y-loss L (=-log p_gold). The single source of truth for the MC predictor and analyses.

    `conts` holds whichever part of the request varies across options -- the continuations in
    the standard layout, the differing tails of the contexts under multiple_input. See
    render_options.

    New five-shot CommonsenseQA/WinoGrande samples supply task-native target fields instead.
    These contain the question and answer texts without demonstrations or chat scaffolding;
    old zero-shot artifacts retain their original request-based rendering.

    `strip_ctx` is an optional callable applied to each request *context* before the layout is
    resolved -- used to remove an instruct checkpoint's chat wrapper (see
    pproute.models.strip_chat_scaffolding). It has to happen here, on the raw context, rather than
    on the assembled question: under multiple_input the template wraps every option's context
    separately, so the wrapper ends up *inside* each option string and no amount of trimming the
    question can reach it. Length normalisation is unaffected -- it uses the stamped choice
    lengths, which the template does not touch.
    """
    tk, S, configs = _samples(scores_path, include_config=True)
    require_target = (tk in ("commonsense_qa", "winogrande") and int(
        (configs.get(tk) or {}).get("num_fewshot") or 0
    ) > 0)
    length_norm = tk in ACC_NORM_TASKS
    out = []
    for it in S:
        fr = it.get("filtered_resps")
        if not fr:
            continue
        ll = np.array([v[0] for v in fr], float); args = it["arguments"]
        target = it.get(MC_TARGET_KEY)
        if target is None:
            if require_target:
                raise ValueError("%s five-shot sample %s is missing %s" %
                                 (tk, it.get("doc_id"), MC_TARGET_KEY))
            if strip_ctx is not None:
                args = [(strip_ctx(a[0]),) + tuple(a[1:]) for a in args]
            question, options, layout = render_options(args)
        else:
            question, options, layout = target["question"], target["choices"], target["layout"]
            if not isinstance(question, str) or not question.strip() or \
                    not isinstance(options, list) or len(options) != len(ll) or \
                    not all(isinstance(option, str) and option.strip() for option in options) or \
                    layout not in ("standard", "multiple_input"):
                raise ValueError("%s sample %s has invalid MC target fields" %
                                 (tk, it.get("doc_id")))
        if length_norm:
            pr = softmax(ll / _choice_lengths(it, tk, len(ll)))
        else:
            pr = softmax(ll)
        gold = _gold_index(it, n_options=len(ll)); pred = int(pr.argmax())
        fz = float(np.clip(pr.max(), EPS, 1 - EPS))
        out.append(dict(question=question, conts=options, layout=layout, pr=pr,
                        pred=pred, gold=gold, fz=fz, lfz=float(np.log(fz / (1 - fz))),
                        Z=float(pred == gold), L=float(-np.log(max(pr[gold], EPS))),
                        id=it.get("doc_id", len(out))))
    return out


def gen_examples(scores_path, filter_name=GEN_FILTER):
    """Per-example dicts for a generative task: rendered question, predictor text, response, Z.

    ``question`` is the exact rendered request and ``predictor_text`` is the required task-native
    pre-chat stamp. Keeping them separate is essential for five-shot multiturn prompts.

    lm_eval writes one sample per (document, *filter*), and gsm8k ships two filters. Keeping both
    is not a harmless duplicate: `strict-match` and `flexible-extract` are two different
    definitions of correctness, so the same question would appear twice, sometimes with opposite
    labels, and `acc` would be their average -- a number that is neither metric.

    `flexible-extract` is pre-registered as Z for gsm8k. It takes the last
    number in the generation; `strict-match` requires the canonical "#### N" format, which base
    models do not produce -- they score 0.000-0.006, below the validity floor, so every gsm8k point
    would be excluded. It would also credit finetuning for teaching an output format rather than
    arithmetic, which is the confound the spec's grading section exists to avoid.
    """
    tk, S = _samples(scores_path)

    # Fail loudly rather than silently averaging two metrics: the whole reason this cost us
    # anything is that nothing checked. Single-filter tasks are left alone.
    present = {it.get("filter") for it in S if it.get("filter") is not None}
    if len(present) > 1:
        if filter_name not in present:
            raise ValueError(
                "%s holds samples under filters %s but not the requested %r. Z is only defined "
                "once a filter is chosen -- averaging them mixes two definitions of correctness."
                % (scores_path, sorted(present), filter_name))
        S = [it for it in S if it.get("filter") == filter_name]

    out = []
    for it in S:
        z = it.get("exact_match", it.get("acc"))
        if z is None:
            continue
        resp = ""
        if it.get("resps"):
            resp = it["resps"][0][0] if isinstance(it["resps"][0], (list, tuple)) else it["resps"][0]
        elif it.get("filtered_resps"):
            resp = it["filtered_resps"][0]
        question = it["arguments"][0][0]
        if PREDICTOR_TEXT_KEY not in it:
            raise KeyError(
                "%s sample %s lacks %s; re-run scripts.evaluate_model"
                % (tk, it.get("doc_id", len(out)), PREDICTOR_TEXT_KEY)
            )
        predictor_text = it[PREDICTOR_TEXT_KEY]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "%s sample %s has no usable rendered question"
                % (tk, it.get("doc_id", len(out)))
            )
        if not isinstance(predictor_text, str) or not predictor_text.strip():
            raise ValueError(
                "%s sample %s has no usable predictor text"
                % (tk, it.get("doc_id", len(out)))
            )
        out.append(dict(question=question, predictor_text=predictor_text,
                        resp=str(resp), Z=float(z),
                        id=it.get("doc_id", len(out))))
    return out


def self_metrics(scores_path):
    """Model-only metrics for one MC score file: acc, y-loss L(f), self-confidence z-loss L_Z(f)."""
    examples = mc_examples(scores_path)
    Z = np.array([e["Z"] for e in examples])
    fz = np.clip(np.array([e["fz"] for e in examples]), EPS, 1 - EPS)
    L = np.array([e["L"] for e in examples])
    return {"n": len(Z), "acc": float(Z.mean()), "L_f": float(L.mean()),
            "LZ_self": bce(Z, fz)}
