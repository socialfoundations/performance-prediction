"""Score one model (base or LoRA checkpoint) on a task, per prompt.

Writes one `<split>_scores.jbl` per split: the raw lm_eval results including per-sample option
log-likelihoods, from which pproute.scores derives Z, f_Z and L_Y. Also records each split's
fingerprint, so a later run that used a different carve cannot be mistaken for a comparable one.

Splits: "eval" is the predictor-training (validation) split and the one checkpoint selection reads;
"test" is held out; "train" is off by default -- it is only useful for diagnostics and costs a full
generation pass on generative tasks. Multi-task splits checkpoint each completed task in a hidden
resume directory and merge them into the same single canonical score file.
"""
import os
import logging
import subprocess
import tempfile

import joblib as jbl

from zarth_utils.config import Config
from zarth_utils.logger import get_logger
from zarth_utils.nn_utils import set_random_seed, get_all_paths
from zarth_utils.recorder import Recorder
from pproute.tasks import get_dataset, resolve_task_names, split_fingerprint
from pproute.evaluator import Evaluator, FEWSHOT_RANDOM_SEED
from pproute.models import load_model_and_tokenizer, resolve_chat_template
from pproute import gsm8k, mmlu_pro
from pproute.provenance import require_pinned_revision

# Benchmarks that own a frozen evaluation protocol and a diagnostics report. Each module refuses a
# run of its own task that is off protocol and returns immediately for any other task, so the whole
# registry can be walked. These are the generative benchmarks, where the shot count and the
# generation ceiling are part of the measurement rather than of the scheduling.
BENCHMARKS = (gsm8k, mmlu_pro)

SCORE_PART_SCHEMA = "pproute-evaluate-score-part-v1"
TASK_RESULT_FIELDS = (
    "configs", "higher_is_better", "n-samples", "n-shot", "results", "samples", "versions",
)


def validate_validation_only(dataset_param, validation_only):
    """The task-neutral development rail: score validation and build no train or test requests."""
    if not validation_only:
        return
    if dataset_param["max_num_train"] != 0 or dataset_param["max_num_test"] != 0:
        raise ValueError("validation_only requires max_num_train=max_num_test=0")
    if dataset_param["max_num_valid"] == 0:
        raise ValueError("validation_only requires a non-empty validation split")


def atomic_joblib_dump(value, path):
    """Write a score file atomically, including when duplicate jobs race on the same split."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".partial", dir=os.path.dirname(path)
    )
    os.close(fd)
    try:
        jbl.dump(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _validate_task_scores(scores, task_name, label):
    if not isinstance(scores, dict):
        raise ValueError("%s is not an evaluation score payload" % label)
    for key in TASK_RESULT_FIELDS:
        value = scores.get(key)
        if not isinstance(value, dict) or set(value) != {task_name}:
            raise ValueError("%s has unexpected %s tasks" % (label, key))


def merge_task_scores(parts):
    """Merge independently scored lm-eval tasks into one canonical score payload."""
    if not parts:
        raise ValueError("at least one task score part is required")
    for task_name, scores in parts:
        _validate_task_scores(scores, task_name, task_name)
    first = parts[0][1]
    expected_keys = set(first)
    # lm-eval reads the working tree's HEAD after each task. A commit made while the
    # process is running changes this passive stamp, not the already loaded evaluator.
    # Resume-part metadata separately pins the evaluation code revision and settings.
    varying_keys = {"date", "pretty_env_info", "git_hash"}
    shared_keys = expected_keys - set(TASK_RESULT_FIELDS) - varying_keys
    for task_name, scores in parts[1:]:
        if set(scores) != expected_keys:
            raise ValueError("%s score part has different top-level keys" % task_name)
        for key in shared_keys:
            if jbl.hash(scores[key]) != jbl.hash(first[key]):
                raise ValueError("%s score part differs in shared field %s" % (task_name, key))

    merged = {key: first[key] for key in shared_keys}
    if "date" in expected_keys:
        merged["date"] = min(scores["date"] for _, scores in parts)
    if "pretty_env_info" in expected_keys:
        merged["pretty_env_info"] = first["pretty_env_info"]
    if "git_hash" in expected_keys:
        hashes = {task_name: scores["git_hash"] for task_name, scores in parts}
        if len({jbl.hash(value) for value in hashes.values()}) == 1:
            merged["git_hash"] = first["git_hash"]
        else:
            merged["git_hash"] = None
            merged["pproute_git_hash_by_task"] = hashes
    for key in TASK_RESULT_FIELDS:
        merged[key] = {
            task_name: scores[key][task_name] for task_name, scores in parts
        }
    return merged


def evaluate_resumable(task_dict, task_manager, model, tokenizer, evaluator_kwargs,
                       parts_dir, part_metadata):
    """Score a multitask split one task at a time, reusing validated completed parts."""
    os.makedirs(parts_dir, exist_ok=True)
    parts = []
    for index, (task_name, task) in enumerate(task_dict.items()):
        path = os.path.join(parts_dir, "part-%03d.jbl" % index)
        expected_meta = part_metadata[task_name]
        if os.path.isfile(path):
            saved = jbl.load(path)
            if not isinstance(saved, dict) or saved.get("meta") != expected_meta or \
                    "scores" not in saved:
                raise ValueError("stale or malformed evaluation resume part: %s" % path)
            scores = saved["scores"]
            _validate_task_scores(scores, task_name, path)
            logging.info("%s already exists, resuming after it", path)
        else:
            evaluator = Evaluator(
                {task_name: task}, task_manager, **evaluator_kwargs
            )
            scores = evaluator.evaluate(model, tokenizer)
            _validate_task_scores(scores, task_name, path)
            atomic_joblib_dump({"meta": expected_meta, "scores": scores}, path)
        parts.append((task_name, scores))
    return merge_task_scores(parts)


def main():
    config = Config(
        default_config_dict={
            "exp_name": "",
            "task_name": "",
            "base_model_name": "",
            "revision": "main",
            "model_dtype": "auto",
            "dir_checkpoint": "",
            "dataset_param": {
                # 0 by default: the train split is a diagnostic, and scoring it means a full extra
                # generation pass on generative tasks. Set it explicitly if you want it.
                "max_num_train": 0,
                "max_num_valid": 2000,
                "max_num_test": 10000,
                # Can only turn the template ON. Leaving it off means "derive from the model
                # name", not "score by raw completion": it is a bare store_true, so a job that
                # simply forgot it used to be indistinguishable from a base model, and the
                # instruct checkpoint was then scored as a raw completion at near-chance accuracy.
                # See pproute.models.resolve_chat_template.
                "apply_chat_template": False,
                "num_fewshot": 5,
                # lm_eval requires apply_chat_template for this, and refuses otherwise: it turns
                # the in-context examples into user/assistant turns instead of concatenated text.
                # Right for an instruct checkpoint, meaningless for a base one -- which is why it
                # travels with the same per-model switch as the chat template itself.
                "fewshot_as_multiturn": False,
            },
            "eval_use_auto_batch_size": False,
            "eval_max_batch_size": 32,
            # -1 => keep the task's own max_gen_toks. Override per benchmark, never globally:
            # MMLU-Pro asks for 2048 (5-shot CoT) and GSM8K wants 2048 for reasoning models, so a
            # single default silently truncates one of them.
            "max_new_tokens": -1,
            # Score development data without constructing train or held-out test requests.
            "validation_only": False,
            "use_git": True,
            "dataset_seed": 0,
            "random_seed": 0,
            # lm-eval uses a separate stateful RNG for demonstration order. Keep it explicit so
            # the normalized predictor context can be reconstructed before raw/chat rendering.
            "fewshot_random_seed": FEWSHOT_RANDOM_SEED,
            # Extra stop strings appended to the task's `until` list for models whose chat
            # template ends a turn with something other than the tokenizer EOS (NeMo's
            # `<extra_id_1>`). The recorded generation_kwargs carry the full list, so
            # conf_prepare reuses it for resampling.
            "extra_until": [],
        },
        use_argparse=True,
    )
    config.show()
    for required in ("exp_name", "task_name", "base_model_name"):
        if not config[required]:
            raise SystemExit("--%s is required" % required)
    require_pinned_revision(config["revision"], "model revision")
    if config["dataset_seed"] < 0:
        raise ValueError("dataset_seed must be non-negative")

    dir_base = os.getcwd()
    dir_results = os.path.join(dir_base, config["exp_name"])
    all_paths = get_all_paths(dir_results)
    config.dump(all_paths["path_config"])
    recorder = Recorder(
        all_paths["path_record"], config=config, use_git=config["use_git"]
    )
    get_logger(all_paths["path_log"], force_add_handler=True)
    set_random_seed(config["random_seed"])

    ###################################################################################
    # Prepare the model
    ###################################################################################
    model, tokenizer, tokenizer_name = load_model_and_tokenizer(
        config["base_model_name"],
        revision=config["revision"],
        dir_checkpoint=config["dir_checkpoint"],
        device_map="auto",
        eval_mode=True,
        model_dtype=config["model_dtype"],
    )

    ###################################################################################
    # Prompt format -- resolved once, here, and used for both the dataset and the Evaluator
    ###################################################################################
    dataset_param = dict(config["dataset_param"])
    apply_chat_template, chat_info = resolve_chat_template(
        config["base_model_name"],
        tokenizer,
        requested=dataset_param.pop("apply_chat_template"),
    )
    dataset_param["apply_chat_template"] = apply_chat_template
    # Recorded because the score file is the only surviving evidence later: a run whose accuracy
    # sits at chance is a broken prompt or a weak model, and this is what tells the two apart.
    recorder.add_with_logging(key="chat_template", value=chat_info)

    # lm_eval refuses multiturn without a chat template, and the pairing is not automatic now that
    # the template can be derived: fail here with a reason rather than deep inside request building.
    assert not (dataset_param["fewshot_as_multiturn"] and not apply_chat_template), (
        "fewshot_as_multiturn requires a chat template, which %s is not being scored with"
        % config["base_model_name"])
    if apply_chat_template and dataset_param["num_fewshot"] and \
            not dataset_param["fewshot_as_multiturn"]:
        logging.warning(
            "%s: chat template on with %d in-context examples but fewshot_as_multiturn off -- the "
            "examples become concatenated text inside one user turn, not user/assistant turns.",
            config["base_model_name"], dataset_param["num_fewshot"])

    validate_validation_only(dataset_param, config["validation_only"])
    for benchmark in BENCHMARKS:
        benchmark.validate_eval_protocol(
            config.task_name, dataset_param, config["max_new_tokens"], apply_chat_template
        )

    ###################################################################################
    # Prepare the benchmark
    ###################################################################################
    task_names = resolve_task_names(config.task_name)
    code_revision = require_pinned_revision(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "evaluation code revision",
    )
    train_task_dict, valid_task_dict, test_task_dict, task_manager = get_dataset(
        config.task_name,
        config["dataset_seed"],
        **dataset_param,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        extra_until=config["extra_until"],
        build_sft=False,
    )[-4:]

    ###################################################################################
    # Evaluation -- identical for every split, so run them in a loop
    ###################################################################################
    splits = (
        ("train", train_task_dict, config["dataset_param"]["max_num_train"]),
        ("eval", valid_task_dict, config["dataset_param"]["max_num_valid"]),
        ("test", test_task_dict, config["dataset_param"]["max_num_test"]),
    )
    for split_name, task_dict, max_num in splits:
        if max_num == 0:
            logging.info("%s: max_num is 0, skipping", split_name)
            continue

        # Recorded whether or not the scores are recomputed: this is what makes a changed
        # TASK_VALID_FRAC (or seed, or subsample cap) detectable after the fact instead of
        # silently producing scores that cannot be compared with the rest of the run.
        # One fingerprint per lm_eval task. A benchmark that ships as several (MMLU-Pro's fourteen
        # subtasks) records all of them, so a changed carve is detectable per category rather than
        # only in aggregate.
        fingerprints = {name: split_fingerprint(task_dict[name]) for name in task_names}
        recorder.add_with_logging(
            key="%s_split" % split_name,
            value=" ".join(fingerprints[name] for name in task_names),
        )

        path_save = os.path.join(dir_results, "%s_scores.jbl" % split_name)
        if os.path.exists(path_save):
            logging.info("%s already exists, skipping evaluation", path_save)
            continue

        evaluator_kwargs = {
            "use_auto_batch_size": config["eval_use_auto_batch_size"],
            "max_batch_size": config["eval_max_batch_size"],
            "max_new_tokens": config["max_new_tokens"],
            "apply_chat_template": apply_chat_template,
            # These two must be handed to the Evaluator, not merely to get_dataset. Evaluator
            # calls lm_eval.simple_evaluate, which **rebuilds every request from the task config**
            # -- so whatever get_dataset built is discarded here, and the Evaluator's own defaults
            # (0 / False) silently win. Omitting them is why gsm8k kept producing 0-shot prompts
            # while its recorded config said num_fewshot=5: the parameter arrived, and was then
            # thrown away one layer down.
            "num_fewshot": dataset_param["num_fewshot"],
            "fewshot_as_multiturn": dataset_param["fewshot_as_multiturn"],
            "fewshot_random_seed": config["fewshot_random_seed"],
        }
        if len(task_dict) > 1:
            if set(task_dict) != set(task_names):
                raise ValueError("resolved tasks differ from the evaluation task dictionary")
            ordered_task_dict = {name: task_dict[name] for name in task_names}
            common_part_meta = {
                "schema": SCORE_PART_SCHEMA,
                "split": split_name,
                "task_group": config["task_name"],
                "task_names": list(task_names),
                "model": config["base_model_name"],
                "revision": config["revision"],
                **({"model_dtype": config["model_dtype"]}
                   if config["model_dtype"] != "auto" else {}),
                "checkpoint": (os.path.realpath(config["dir_checkpoint"])
                               if config["dir_checkpoint"] else ""),
                "code_revision": code_revision,
                "dataset_param": dataset_param,
                "evaluator": evaluator_kwargs,
                "random_seed": config["random_seed"],
                "dataset_seed": config["dataset_seed"],
            }
            part_metadata = {
                name: dict(common_part_meta, task=name, fingerprint=fingerprints[name])
                for name in task_names
            }
            eval_scores = evaluate_resumable(
                ordered_task_dict, task_manager, model, tokenizer, evaluator_kwargs,
                os.path.join(dir_results, ".%s_scores.parts" % split_name),
                part_metadata,
            )
        else:
            evaluator = Evaluator(task_dict, task_manager, **evaluator_kwargs)
            eval_scores = evaluator.evaluate(model, tokenizer)
        for benchmark in BENCHMARKS:
            if config.task_name != benchmark.TASK_NAME:
                continue
            diagnostics = benchmark.evaluation_diagnostics(
                eval_scores, tokenizer, config["max_new_tokens"]
            )
            # The saved prompts, not the requested config, are the protocol evidence. A previous
            # evaluator rebuild discarded num_fewshot while every config still said 5; refuse to
            # publish another plausible-looking off-protocol score file.
            if diagnostics["prompt_protocol_failures"]:
                raise ValueError(
                    "%d/%d saved %s prompts do not carry the protocol's demonstrations"
                    % (diagnostics["prompt_protocol_failures"], diagnostics["n"],
                       benchmark.TASK_NAME)
                )
            eval_scores.setdefault("pproute_diagnostics", {})[benchmark.TASK_NAME] = diagnostics
            recorder.add_with_logging(
                key="%s_%s_diagnostics" % (split_name, benchmark.TASK_NAME), value=diagnostics
            )
        atomic_joblib_dump(eval_scores, path_save)
        for name in task_names:
            prefix = split_name if len(task_names) == 1 else "%s_%s" % (split_name, name)
            for metric, value in eval_scores["results"][name].items():
                recorder.add_with_logging(key="%s_%s" % (prefix, metric), value=value)

    recorder.end_recording()


if __name__ == "__main__":
    main()
