"""Sample a pinned strong endpoint, saving resumable per-question answers for cascade analysis."""
from collections import Counter
import gzip
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import time

# Spawned tensor-parallel workers re-import this module but never call main(). Apply the
# integrated Python 3.10 workaround here so they also take vLLM's ImportError fallback.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
try:
    _flashinfer_version = version("flashinfer-python")
except PackageNotFoundError:
    _flashinfer_version = None
if sys.version_info < (3, 12) and _flashinfer_version == "0.6.16.post3":
    sys.modules["flashinfer.comm"] = None

from pproute import cascade
from pproute.provenance import identified, require_pinned_revision, sha256_file, sha256_text


def document_key(qid):
    return sha256_text(str(qid))[:24]


def validate_result(row, doc, run_identity, k):
    if (row.get("id"), row.get("gold"), row.get("run_identity"), row.get("k")) != (
            doc["id"], doc["gold"], run_identity, k):
        raise ValueError("Saved strong answer differs from the requested document/protocol")
    if len(row["answers"]) != k or sum(row["counts"].values()) + row["n_fail"] != k:
        raise ValueError("Saved strong answer has incomplete sample counts")
    counts = Counter(answer for answer in row["answers"] if answer is not None)
    majority = max(counts, key=counts.get) if counts else None
    if dict(counts) != row["counts"] or row["majority"] != majority or row["z"] != int(
            majority is not None and majority == doc["gold"]):
        raise ValueError("Saved strong majority/correctness is inconsistent")
    if row["n_fail"] != row["answers"].count(None) or row["share"] != counts.get(majority, 0)/k:
        raise ValueError("Saved strong confidence/failure count is inconsistent")


def main():
    from zarth_utils.config import Config
    from zarth_utils.nn_utils import get_all_paths, set_random_seed
    from zarth_utils.recorder import Recorder
    config = Config(default_config_dict=dict(exp_name="", prompts="", model_name="", revision="",
        k=128, temperature=1.0, top_p=.95, top_k=20, presence_penalty=1.5,
        max_new_tokens=32768, max_model_len=40960, tensor_parallel_size=4, max_num_seqs=128,
        thinking=True, start=0, stop=-1, use_git=True, random_seed=0),
        use_argparse=True)
    for key in ("exp_name", "prompts", "model_name", "revision"):
        if not config[key]:
            raise ValueError(f"Missing --{key}")
    require_pinned_revision(config["revision"])
    prompts = cascade.load_record(config["prompts"])
    if prompts["schema"] != "pproute-cascade-prompts-v1":
        raise ValueError("Unexpected prompt schema")
    stop = len(prompts["rows"]) if config["stop"] < 0 else config["stop"]
    if not 0 <= config["start"] < stop <= len(prompts["rows"]) or config["k"] <= 0:
        raise ValueError("Invalid document range or sample count")
    docs = prompts["rows"][config["start"]:stop]
    snapshot = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + config["model_name"].replace("/", "--")) / "snapshots" / config["revision"]
    if not snapshot.is_dir():
        raise ValueError(f"Missing pinned snapshot: {snapshot}")
    policy = dict(model=config["model_name"], revision=config["revision"], dtype="bfloat16",
        k=config["k"], temperature=config["temperature"], top_p=config["top_p"],
        top_k=config["top_k"], min_p=0., presence_penalty=config["presence_penalty"], repetition_penalty=1.,
        max_new_tokens=config["max_new_tokens"], max_model_len=config["max_model_len"],
        thinking=config["thinking"],
        max_num_seqs=config["max_num_seqs"], seed=config["random_seed"],
        backend="vllm", backend_version=version("vllm"), sampler="torch",
        model_config_sha256=sha256_file(snapshot/"config.json"),
        tokenizer_config_sha256=sha256_file(snapshot/"tokenizer_config.json"))
    # Placement is execution provenance, not a different sampling policy. This keeps
    # document seeds fixed when the same endpoint runs on 2 large or 4 smaller GPUs.
    endpoint = identified(dict(schema="pproute-cascade-strong-v2", **policy))
    execution = dict(tensor_parallel_size=config["tensor_parallel_size"],
                     host=os.uname().nodename)
    run = identified(dict(schema="pproute-cascade-sampling-v2", strong=endpoint,
        task=prompts["task"], split=prompts["split"], prompts_identity=prompts["identity"]))
    output = Path(config["exp_name"])
    output.mkdir(parents=True, exist_ok=True)
    cascade.publish_once(run, output/"run.json")
    complete = output/"units"/f"{config['start']:05d}-{stop:05d}.json"
    pending = []
    for doc in docs:
        path = output/"answers"/(document_key(doc["id"])+".json")
        if path.exists():
            row = json.loads(path.read_text())
            validate_result(row, doc, run["identity"], config["k"])
            if not (output/row["completions_file"]).is_file() or sha256_file(output/row["completions_file"]) != row["completions_sha256"]:
                raise ValueError(f"Missing or changed completion evidence: {path}")
        else:
            pending.append(doc)
    if complete.exists():
        record = cascade.load_record(complete)
        if pending or record["run_identity"] != run["identity"] or record["ids"] != [d["id"] for d in docs]:
            raise ValueError("Completion marker differs from invocation")
        print("COMPLETE", complete, flush=True)
        return
    paths = get_all_paths(str(output/"records"/f"{config['start']:05d}-{stop:05d}"))
    config.dump(paths["path_config"])
    recorder = Recorder(paths["path_record"], config=config, use_git=config["use_git"])
    set_random_seed(config["random_seed"], no_torch=True)
    start_time = time.monotonic()
    if pending:
        from pproute.models import tokenizer_load_kwargs
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), **tokenizer_load_kwargs(config["model_name"]))
        rendered = [tokenizer.apply_chat_template([dict(role="user", content=doc["prompt"])],
            tokenize=False, add_generation_prompt=True, enable_thinking=config["thinking"]) for doc in pending]
        prompt_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in rendered]
        if max(prompt_lengths)+config["max_new_tokens"] > config["max_model_len"]:
            raise ValueError("Strong prompt plus answer budget exceeds context; truncation is forbidden")
        llm = LLM(model=str(snapshot), dtype="bfloat16", max_model_len=config["max_model_len"],
            tensor_parallel_size=config["tensor_parallel_size"], max_num_seqs=config["max_num_seqs"],
            gpu_memory_utilization=.90, seed=config["random_seed"], enable_prefix_caching=True,
            disable_custom_all_reduce=True, additional_config={"gdn_prefill_backend": "triton"})
        loaded_seconds = time.monotonic()-start_time
        recorder.add("load_seconds", loaded_seconds)
        for doc, prompt, prompt_length in zip(pending, rendered, prompt_lengths):
            seed = int(sha256_text(f"{endpoint['identity']}:{prompts['task']}:{prompts['split']}:{doc['id']}")[:8], 16)
            params = SamplingParams(n=config["k"], temperature=config["temperature"], top_p=config["top_p"],
                top_k=config["top_k"], min_p=0., presence_penalty=config["presence_penalty"],
                repetition_penalty=1., max_tokens=config["max_new_tokens"], seed=seed)
            tick = time.monotonic()
            generated = llm.generate([prompt], params, use_tqdm=False)[0]
            if len(generated.outputs) != config["k"]:
                raise ValueError("vLLM returned the wrong sample count")
            completions = [dict(text=x.text, n_tokens=len(x.token_ids), finish_reason=x.finish_reason)
                           for x in sorted(generated.outputs, key=lambda x: x.index)]
            row = cascade.summarize_answers(prompts["task"], doc, completions)
            key = document_key(doc["id"])
            archive = output/"completions"/(key+".json.gz")
            archive.parent.mkdir(exist_ok=True)
            temporary = archive.with_suffix(".gz.partial")
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(dict(id=doc["id"], run_identity=run["identity"], seed=seed,
                               prompt=prompt, completions=completions), handle, ensure_ascii=False)
            os.replace(temporary, archive)
            row.update(run_identity=run["identity"], execution=execution, seed=seed, prompt_tokens=prompt_length,
                generation_seconds=time.monotonic()-tick, completions_file=str(archive.relative_to(output)),
                completions_sha256=sha256_file(archive))
            cascade.publish_once(row, output/"answers"/(key+".json"))
            print("SAMPLED", prompts["task"], prompts["split"], doc["id"],
                  "z", row["z"], "fail", row["n_fail"], "caps", row["cap_hits"],
                  "tokens", row["output_tokens"], "seconds", round(row["generation_seconds"], 1), flush=True)
    rows = [json.loads((output/"answers"/(document_key(doc["id"])+".json")).read_text()) for doc in docs]
    report = identified(dict(schema="pproute-cascade-unit-v1", run_identity=run["identity"],
        ids=[d["id"] for d in docs], execution=execution, n=len(docs), elapsed_seconds=time.monotonic()-start_time,
        accuracy=sum(r["z"] for r in rows)/len(rows), output_tokens=sum(r["output_tokens"] for r in rows),
        generation_seconds=sum(r["generation_seconds"] for r in rows),
        n_fail=sum(r["n_fail"] for r in rows), cap_hits=sum(r["cap_hits"] for r in rows)))
    recorder.add("cascade_unit", report)
    recorder.end_recording()
    cascade.publish_once(report, complete)
    print("DONE", complete, flush=True)


if __name__ == "__main__":
    main()
