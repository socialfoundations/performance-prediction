"""Merge a selected LoRA adapter into its pinned base model for vLLM sampling.

Loading uses pproute.models to support the cohort's architectures and tokenizers.
The merged weights and run metadata are saved under --exp_name. CPU-only.
"""
from __future__ import annotations

import json
import os
import sys

from pproute import predictor_protocol
from pproute.provenance import atomic_json_dump, sha256_file


# save_pretrained on the language model writes neither the (video) preprocessor configs nor the
# processor config that the ...ForConditionalGeneration architectures ship, and vLLM refuses to
# load a multimodal model without them; merges.txt/vocab.json are redundant with tokenizer.json
# but copied for completeness.
PROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json",
                   "processor_config.json", "merges.txt", "vocab.json")


def copy_processor_files(model_name, revision, out):
    import shutil

    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                       "hub")
    snapshot = os.path.join(hub, "models--" + model_name.replace("/", "--"),
                            "snapshots", revision)
    copied = []
    for name in PROCESSOR_FILES:
        source = os.path.join(snapshot, name)
        target = os.path.join(out, name)
        if os.path.exists(source) and not os.path.exists(target):
            shutil.copyfile(source, target)
            copied.append(name)
    if copied:
        print(f"copied from base snapshot: {copied}")


def main():
    from zarth_utils.config import Config
    from zarth_utils.logger import get_logger
    from zarth_utils.nn_utils import get_all_paths, set_random_seed
    from zarth_utils.recorder import Recorder

    config = Config(
        default_config_dict={
            "endpoint_manifest": "", "exp_name": "",
            "model_dtype": "auto",
            "use_git": True, "random_seed": 0,
        },
        use_argparse=True,
    )
    config.show()
    for required in ("endpoint_manifest", "exp_name"):
        if not config[required]:
            raise SystemExit("--%s is required" % required)

    endpoint = predictor_protocol.read_endpoint_manifest(
        config["endpoint_manifest"], endpoint="after",
    )
    model_name = endpoint["model"]
    revision = endpoint["revision"]
    checkpoint = endpoint["checkpoint"]
    for name, field in (("adapter_model.safetensors", "adapter_sha256"),
                        ("adapter_config.json", "adapter_config_sha256")):
        path = os.path.join(checkpoint, name)
        if sha256_file(path) != endpoint[field]:
            raise ValueError("endpoint manifest hash differs for %s" % path)
    endpoint_manifest_hash = sha256_file(config["endpoint_manifest"])
    result_dir = os.path.join(os.getcwd(), config["exp_name"])

    provenance_path = os.path.join(result_dir, "merge_provenance.json")
    if os.path.exists(provenance_path):
        with open(provenance_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        expected = {
            "base_model_name": model_name, "revision": revision,
            "dir_checkpoint": checkpoint,
            "endpoint_identity": endpoint["identity"],
            "endpoint_manifest_sha256": endpoint_manifest_hash,
        }
        if existing.get("model_dtype", "auto") != config["model_dtype"]:
            raise ValueError("existing merged endpoint model dtype mismatch")
        bad = {key: (existing.get(key), value) for key, value in expected.items()
               if existing.get(key) != value}
        if bad:
            raise ValueError("existing merged endpoint identity mismatch: %s" % bad)
        print(f"COMPLETE {result_dir}: provenance validated", flush=True)
        return 0

    os.makedirs(result_dir, exist_ok=True)
    all_paths = get_all_paths(result_dir)
    config.dump(all_paths["path_config"])
    recorder = Recorder(all_paths["path_record"], config=config, use_git=config["use_git"])
    get_logger(all_paths["path_log"], force_add_handler=True)
    set_random_seed(config["random_seed"])
    import torch

    from pproute.models import load_model_and_tokenizer

    model, tokenizer, _ = load_model_and_tokenizer(
        model_name, revision=revision, dir_checkpoint=checkpoint, device_map="cpu",
        model_dtype=config["model_dtype"])
    model = model.merge_and_unload().to(torch.bfloat16)
    model.save_pretrained(result_dir, safe_serialization=True)
    tokenizer.save_pretrained(result_dir)
    copy_processor_files(model_name, revision, result_dir)
    provenance = {"base_model_name": model_name, "revision": revision,
                  "dir_checkpoint": checkpoint,
                  "endpoint_identity": endpoint["identity"],
                  "endpoint_manifest_sha256": endpoint_manifest_hash}
    if config["model_dtype"] != "auto":
        provenance["model_dtype"] = config["model_dtype"]
    recorder.add_with_logging(key="merge_provenance", value=provenance)
    recorder.end_recording()
    # Published atomically and last: presence implies a complete model save and finalized record.
    atomic_json_dump(provenance, provenance_path)
    print(f"DONE merged {model_name} + {checkpoint} "
          f"-> {result_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
