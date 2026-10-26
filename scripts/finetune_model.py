"""LoRA-SFT one pinned model candidate and save its dense checkpoint sequence.

All configuration is inline and every real run is recorded under ``exp_name``. The dataset carve
is fixed by ``dataset_seed`` while ``random_seed`` independently controls optimization, allowing
three comparable model fits over byte-identical rendered SFT rows.
"""

import os
import json
import hashlib

from transformers import logging
from peft import LoraConfig
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from trl import SFTTrainer, SFTConfig

from zarth_utils.config import Config
from zarth_utils.logger import get_logger, logging_info
from zarth_utils.nn_utils import set_random_seed, get_all_paths
from zarth_utils.recorder import Recorder
from pproute.tasks import get_dataset, resolve_task_names
from pproute import gsm8k, mmlu_pro, trace_sft
from pproute.models import (
    configure_chat_sft_tokenizer,
    get_mixed_precision,
    load_model_and_tokenizer,
    resolve_chat_template,
)
from pproute.provenance import atomic_json_dump, require_pinned_revision

logging.set_verbosity_error()  # only errors, no warnings


# Hybrid attention stacks: q_proj/v_proj exist, but only in the minority of layers that use full
# attention -- the rest carry a fused in_proj_qkv. Targeting q_proj/v_proj alone therefore adapts a
# fraction of the depth (Qwen3.5-9B: 8 layers, against 32 for Llama-3.1-8B), so LoRA capacity would
# vary several-fold across a cohort whose whole point is comparing how much each model gains from
# finetuning. Extending the targets keeps the adapter comparable rather than the module names.
HYBRID_EXTRA_TARGETS = {
    "qwen3_5": ["in_proj_qkv"],
}


def trace_sft_dataset_params(task_name, trace_path, dataset_param):
    """Return dataset parameters that construct only the trace-SFT training split."""
    if trace_path:
        trace_sft.adapter_for(task_name)  # fail before constructing any benchmark split
    return trace_sft.train_only_dataset_params(trace_path, dataset_param)


def validate_trace_sft_args(trace_path, completion_source):
    """Require the trace path and its completion selector as one configuration pair."""
    if completion_source and completion_source != "teacher":
        raise ValueError("Trace SFT uses the frozen teacher completions")
    if bool(trace_path) != bool(completion_source):
        raise ValueError(
            "--sft_jsonl and --sft_completion_source must be supplied together"
        )


def dataset_content_sha256(dataset):
    """Hash the exact ordered SFT rows after rendering and trace substitution."""
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_clean_sft_diagnostics(diagnostics, completion_only):
    """Enforce the configured no-truncation/no-mask-loss contract."""
    failures = {"truncated_examples": int(diagnostics["truncated_examples"])}
    if completion_only:
        failures.update(
            fully_masked_examples=int(diagnostics["fully_masked_examples"]),
            completion_tokens_lost=int(diagnostics["completion_tokens_lost"]),
        )
    failures = {key: value for key, value in failures.items() if value}
    if failures:
        raise ValueError("SFT tokenization violates the frozen context/masking protocol: %s" %
                         failures)


# The generative benchmarks, each owning its trace adapter and its SFT prompt-format rule.
BENCHMARKS = (gsm8k, mmlu_pro)

def get_target_modules(model):
    """Use PEFT's LoRA targets, with text-only and hybrid-attention cohort overrides."""
    model_type = None
    if hasattr(model, "config") and hasattr(model.config, "model_type"):
        model_type = model.config.model_type

    if model_type == "gemma3":
        # Text-only SFT must not attach adapters to the SigLIP vision tower.
        return r".*language_model\..*\.(q_proj|v_proj)"

    if model_type == "qwen3_5_moe":
        # The 122B cascade model has full attention in only 12 of its 48 text layers.
        # Adapt the other 36 linear-attention layers too, leaving vision/expert weights frozen.
        return r".*language_model\..*\.(q_proj|v_proj|in_proj_qkv)"

    if model_type and model_type in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
        targets = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_type]
        # Some entries are a regex string rather than a list of module names (gemma4 is
        # `.*language_model\..*\.(q_proj|v_proj)`, which is how it reaches the language tower of a
        # multimodal checkpoint). peft accepts either, but list() over a string yields one target
        # per character and raises -- so only copy when it is genuinely a sequence.
        targets = targets if isinstance(targets, str) else list(targets)
    else:
        logging_info(
            "no LoRA target-module entry for model_type=%s; falling back to q_proj/v_proj"
            % model_type
        )
        targets = ["q_proj", "v_proj"]

    if model_type in HYBRID_EXTRA_TARGETS and not isinstance(targets, str):
        targets = targets + HYBRID_EXTRA_TARGETS[model_type]
        logging_info(
            "model_type=%s is a hybrid stack; extending targets to %s so LoRA reaches the "
            "linear-attention layers too" % (model_type, targets)
        )
    return targets


def main():
    config = Config(
        default_config_dict={
            "exp_name": "",
            "task_name": "",
            "model_name": "",
            "revision": "main",
            "model_dtype": "auto",  # explicit compute dtype for checkpoints missing metadata
            "device_map": "",  # "balanced" for single-process LoRA across multiple GPUs
            # Trace path and completion source are a pair. With neither, training uses the task's
            # ordinary gold answers; with both, training uses stored teacher completions.
            "sft_jsonl": "",
            "sft_completion_source": "",
            "dataset_param": {
                "max_num_train": -1,
                "max_num_valid": 0,
                "max_num_test": 0,
                "completion_only": True,
                # Can only turn the template ON; leaving it off derives from the model name.
                # See pproute.models.resolve_chat_template.
                "apply_chat_template": False,
                "num_fewshot": 5,
                "fewshot_as_multiturn": False,
            },
            "resume_from_checkpoint": False,
            "train_param": {
                "loss_type": "chunked_nll",  # use "nll" with Accelerate device_map hooks
                "max_length": 1024,     # raise per benchmark; long-context tasks need much more
                "per_device_train_batch_size": 8,
                # Left off deliberately: with auto_find_batch_size the effective batch is whatever
                # fits, so "max_steps x batch" no longer says how many examples were consumed.
                # Pin it here (and use gradient_accumulation_steps to hold the effective batch when
                # the per-device one has to shrink for a longer context).
                "auto_find_batch_size": False,
                "num_train_epochs": 1,
                "max_steps": -1,
                "logging_strategy": "steps",
                "save_strategy": "steps",
                "logging_steps": 0.2,
                "save_steps": 0.2,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False,
                "optim": "adamw_torch",
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_steps": 0,
                "report_to": "tensorboard",
                # NOTE: no save_safetensors -- transformers 5 dropped the argument (safetensors is
                # the only format now), and passing it raises before the model is even loaded.
                "packing": False,
            },
            "peft_config": {
                "task_type": "CAUSAL_LM",
                "inference_mode": False,
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            },
            "use_git": True,
            # Fixed independently of optimization seed so all three candidates see identical rows.
            "dataset_seed": 0,
            "expected_sft_rows": -1,
            "random_seed": 0,
        },
        use_argparse=True,
    )
    config.show()
    for required in ("exp_name", "task_name", "model_name"):
        if not config[required]:
            raise SystemExit("--%s is required" % required)
    if not config["dataset_param"]["completion_only"]:
        raise ValueError("The paper uses completion-only SFT")
    validate_trace_sft_args(config["sft_jsonl"], config["sft_completion_source"])
    require_pinned_revision(config["revision"], "model revision")

    dir_base = os.getcwd()
    dir_results = os.path.join(dir_base, config["exp_name"])
    all_paths = get_all_paths(dir_results)
    config.dump(all_paths["path_config"])
    recorder = Recorder(
        all_paths["path_record"], config=config, use_git=config["use_git"]
    )
    get_logger(all_paths["path_log"], force_add_handler=True)

    ###################################################################################
    # Prepare training arguments
    ###################################################################################
    bf16, fp16 = get_mixed_precision(
        config["model_name"], revision=config["revision"], model_dtype=config["model_dtype"]
    )
    if config["device_map"] and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Balanced device-map training requires one process")

    training_params = SFTConfig(
        output_dir=dir_results,
        **config["train_param"],
        # TrainingArguments otherwise defaults both to 42, which would make configured model
        # seeds share the same data order. These seeds govern optimization only; the task carve is
        # controlled independently below.
        seed=config["random_seed"],
        data_seed=config["random_seed"],
        bf16=bf16,
        bf16_full_eval=bf16,
        fp16=fp16,
        fp16_full_eval=fp16,
        completion_only_loss=config["dataset_param"]["completion_only"],
    )

    ###################################################################################
    # Prepare the pre-trained model
    ###################################################################################
    set_random_seed(config["random_seed"])
    model, tokenizer, _ = load_model_and_tokenizer(
        config["model_name"], revision=config["revision"], model_dtype=config["model_dtype"],
        device_map=config["device_map"] or None,
    )
    if config["device_map"]:
        placement = getattr(model, "hf_device_map", {})
        if any(str(device) in ("cpu", "disk") for device in placement.values()):
            raise ValueError("device_map LoRA training requires all weights on GPUs")
        recorder.add_with_logging(key="device_map", value=placement)

    ###################################################################################
    # Prepare the benchmark
    ###################################################################################
    # Same asymmetry as evaluate_model: the flag can only turn the template on, and an instruct
    # checkpoint is never trained on raw completions by accident. Finetuning and evaluation must
    # agree on the format -- a checkpoint trained templated and scored untemplated (or the reverse)
    # measures the format change, not the finetuning.
    dataset_param = trace_sft_dataset_params(
        config["task_name"], config["sft_jsonl"], config["dataset_param"]
    )
    apply_chat_template, chat_info = resolve_chat_template(
        config["model_name"],
        tokenizer,
        requested=dataset_param.pop("apply_chat_template"),
    )
    dataset_param["apply_chat_template"] = apply_chat_template
    recorder.add_with_logging(key="chat_template", value=chat_info)
    configure_chat_sft_tokenizer(tokenizer, apply_chat_template)

    if config["sft_jsonl"]:
        # Each generative benchmark owns its own prompt-format rule and ignores the other's task,
        # so the whole registry is walked. Without this MMLU-Pro had no SFT-side guard at all.
        for benchmark in BENCHMARKS:
            benchmark.validate_sft_protocol(
                config.task_name, dataset_param, apply_chat_template
            )

    data_seed = int(config["dataset_seed"])
    if data_seed < 0:
        raise ValueError("dataset_seed must be non-negative")
    recorder.add_with_logging(key="dataset_seed", value=data_seed)
    dataset_out = get_dataset(
        config.task_name,
        data_seed,
        build_sft=not bool(config["sft_jsonl"]),
        **dataset_param,
        tokenizer_name=config["model_name"],
        tokenizer=tokenizer,
    )
    train_dataset = dataset_out[0]
    if config["sft_jsonl"]:
        if not dataset_param["completion_only"]:
            raise ValueError("generated-trace SFT requires dataset_param.completion_only")
        # Every task of the benchmark: MMLU-Pro's SFT split is spread over fourteen subtasks, and
        # `dataset_out[3]` is keyed by lm_eval task name rather than by benchmark name.
        train_dataset, trace_info = trace_sft.build_trace_sft_dataset(
            [dataset_out[3][name] for name in resolve_task_names(config.task_name)],
            config["sft_jsonl"],
            config["sft_completion_source"],
            adapter=trace_sft.adapter_for(config.task_name),
        )
        recorder.add_with_logging(key="sft_trace", value=trace_info)
        recorder.add_with_logging(
            key="sft_condition",
            value={
                "completion_source": config["sft_completion_source"],
                "num_fewshot": dataset_param["num_fewshot"],
                "fewshot_as_multiturn": dataset_param["fewshot_as_multiturn"],
                "completion_only_loss": True,
                "max_num_valid": dataset_param["max_num_valid"],
                "max_num_test": dataset_param["max_num_test"],
            },
        )

    if config["expected_sft_rows"] != -1 and len(train_dataset) != config["expected_sft_rows"]:
        raise ValueError("SFT dataset has %d rows, expected %d" %
                         (len(train_dataset), config["expected_sft_rows"]))
    diagnostics = trace_sft.training_diagnostics(
        train_dataset, tokenizer, config["train_param"]["max_length"]
    )
    diagnostics["dataset_content_sha256"] = dataset_content_sha256(train_dataset)
    diagnostics["dataset_seed"] = data_seed
    recorder.add_with_logging(key="sft_diagnostics", value=diagnostics)
    atomic_json_dump(diagnostics, os.path.join(dir_results, "sft_diagnostics.json"))
    require_clean_sft_diagnostics(diagnostics, dataset_param["completion_only"])

    ###################################################################################
    # Training
    ###################################################################################
    # Dataset construction seeds NumPy/Python/Torch for its deterministic carve. Restore the
    # optimization seed here so LoRA initialization and training order vary when the protocol fixes
    # dataset_seed=0 across all three fits.
    set_random_seed(config["random_seed"])
    target_modules = get_target_modules(model)
    peft_config = (
        LoraConfig(
            **config["peft_config"],
            target_modules=target_modules,
            revision=config["revision"],
        )
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_params,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=config["resume_from_checkpoint"])

    recorder.end_recording()


if __name__ == "__main__":
    main()
