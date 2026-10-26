"""Freeze task questions for a shared resampled strong model, from an existing raw endpoint."""
import os

from pproute import cascade, consistency, predictor_protocol, scores, trace_sft
from pproute.provenance import sha256_file, sha256_text


def prepare(task, scores_dir, split, split_manifest):
    if split != "test":
        raise ValueError("The final generative cascade evaluates the test split")
    adapter = trace_sft.adapter_for(task)
    source = os.path.join(scores_dir, "test_scores.jbl")
    fingerprint = scores.recorded_split_fingerprint(scores_dir, "test")
    provenance = scores.scored_model_provenance(scores_dir)
    if provenance["task_name"] != task or provenance["prompt_protocol"]["apply_chat_template"]:
        raise ValueError("Prepare from a task-matching raw, unwrapped endpoint")
    docs, _ = consistency.extract_prompts(source, adapter)
    rows = [{k: row[k] for k in ("id", "prompt", "gold", "options") if k in row}
            for row in docs]
    predictor_protocol.read_split_manifest(split_manifest, task=task)
    return cascade.prompt_record(task, split, rows, dict(
        scores=source, scores_sha256=sha256_file(source), fingerprint=fingerprint,
        fingerprint_sha256=sha256_text(fingerprint), split_manifest=split_manifest,
        split_manifest_sha256=sha256_file(split_manifest), model=provenance["base_model_name"],
        revision=provenance["revision"]))


def main():
    from zarth_utils.config import Config
    from zarth_utils.nn_utils import get_all_paths
    from zarth_utils.recorder import Recorder
    config = Config(default_config_dict=dict(exp_name="", task_name="", scores_dir="",
        split="test", split_manifest="", expected_n=0,
        expected_fingerprint_sha256="", use_git=True), use_argparse=True)
    if config["split"] != "test" or not config["exp_name"]:
        raise ValueError("Require exp_name and test split")
    paths = get_all_paths(os.path.abspath(config["exp_name"]))
    config.dump(paths["path_config"])
    recorder = Recorder(paths["path_record"], config=config, use_git=config["use_git"])
    payload = prepare(config["task_name"], config["scores_dir"], config["split"], config["split_manifest"])
    if len(payload["rows"]) != config["expected_n"] or payload["sources"]["fingerprint_sha256"] != config["expected_fingerprint_sha256"]:
        raise ValueError("Prompt count/fingerprint differs from frozen cohort")
    recorder.add("n_prompts", len(payload["rows"]))
    recorder.end_recording()
    cascade.publish_once(payload, os.path.join(config["exp_name"], "prompts.json"))
    print("PREPARED", config["task_name"], config["split"], len(payload["rows"]), flush=True)


if __name__ == "__main__":
    main()
