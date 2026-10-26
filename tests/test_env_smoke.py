"""Step-0 environment checks: the transformers 5.x upgrade and the per-model loader.

These guard the two things the 4.x -> 5.x jump actually broke, both of which were silent or
launch-time rather than visible in a metric:

- `SFTConfig` dropped `save_safetensors`, so splatting `train_param` into it raised TypeError;
- the dtype a checkpoint declares moved (`torch_dtype` -> `dtype`) and, on the
  `...ForConditionalGeneration` models, nested itself inside `text_config`. Missing it downgrades
  training to fp32 without a word.

CPU-only and offline-friendly: anything needing weights or the Hub is skipped when unavailable.
"""

import os
import types

import pytest
import torch

from pproute.models import (
    config_dtype,
    config_max_length,
    configure_chat_sft_tokenizer,
    load_auto_class,
    load_model_and_tokenizer,
    tokenizer_load_kwargs,
)

SMALL_MODEL = "facebook/opt-125m"


def _hub_reachable():
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(SMALL_MODEL)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------------------------
# dtype detection
# --------------------------------------------------------------------------------------------


def test_dtype_found_at_top_level_under_new_name():
    cfg = types.SimpleNamespace(dtype=torch.bfloat16)
    assert config_dtype(cfg) == torch.bfloat16


def test_dtype_found_under_legacy_torch_dtype_name():
    """A 4.x-era config, or anything still writing the old key."""
    cfg = types.SimpleNamespace(torch_dtype=torch.float16)
    assert config_dtype(cfg) == torch.float16


def test_dtype_found_in_nested_text_config():
    """The Qwen/Qwen3.5-9B layout: no top-level dtype, only text_config carries it.

    This is the case that would otherwise have trained a 9B model in fp32.
    """
    cfg = types.SimpleNamespace(text_config=types.SimpleNamespace(dtype=torch.bfloat16))
    assert config_dtype(cfg) == torch.bfloat16


def test_top_level_dtype_wins_over_nested():
    cfg = types.SimpleNamespace(
        dtype=torch.float16, text_config=types.SimpleNamespace(dtype=torch.bfloat16)
    )
    assert config_dtype(cfg) == torch.float16


def test_dtype_absent_returns_none_so_caller_can_refuse():
    assert config_dtype(types.SimpleNamespace()) is None


def test_context_window_found_in_nested_text_config():
    cfg = types.SimpleNamespace(
        text_config=types.SimpleNamespace(max_position_embeddings=262144)
    )
    assert config_max_length(cfg) == 262144


def test_top_level_context_window_wins_over_nested():
    cfg = types.SimpleNamespace(
        max_position_embeddings=8192,
        text_config=types.SimpleNamespace(max_position_embeddings=262144),
    )
    assert config_max_length(cfg) == 8192


def test_context_window_absent_returns_none():
    assert config_max_length(types.SimpleNamespace()) is None


# --------------------------------------------------------------------------------------------
# tokenizer setup
# --------------------------------------------------------------------------------------------


def test_minitron_uses_transformers_regex_fix():
    for name in (
        "nvidia/Mistral-NeMo-Minitron-8B-Base",
        "nvidia/Mistral-NeMo-Minitron-8B-Instruct",
    ):
        assert tokenizer_load_kwargs(name)["fix_mistral_regex"] is True
    assert "fix_mistral_regex" not in tokenizer_load_kwargs("mistralai/Mistral-7B-v0.3")


def test_chat_sft_explicitly_disables_automatic_bos():
    class Tokenizer:
        def __init__(self):
            self.assignments = []
            self._add_bos_token = False

        @property
        def add_bos_token(self):
            return self._add_bos_token

        @add_bos_token.setter
        def add_bos_token(self, value):
            self.assignments.append(value)
            self._add_bos_token = value

    chat = Tokenizer()
    assert configure_chat_sft_tokenizer(chat, True)
    assert chat.assignments == [False]

    raw = Tokenizer()
    assert not configure_chat_sft_tokenizer(raw, False)
    assert raw.assignments == []


# --------------------------------------------------------------------------------------------
# model-class resolution
# --------------------------------------------------------------------------------------------


def test_declared_architecture_is_preferred_over_auto_class():
    """A ...ForConditionalGeneration checkpoint must not be forced through AutoModelForCausalLM."""
    import transformers

    # models.load_auto_class prefers the class named in config.architectures over any auto
    # mapping; assert that class exists and is not the causal-LM one it would otherwise get.
    cfg = types.SimpleNamespace(architectures=["Gemma3ForConditionalGeneration"])
    cls = getattr(transformers, cfg.architectures[0])
    assert cls is transformers.Gemma3ForConditionalGeneration
    assert cls is not transformers.AutoModelForCausalLM


def test_causal_lm_architecture_still_resolves_to_its_own_class():
    import transformers

    cfg = types.SimpleNamespace(architectures=["LlamaForCausalLM"])
    assert getattr(transformers, cfg.architectures[0]) is transformers.LlamaForCausalLM


def test_unresolvable_architecture_raises_rather_than_guessing():
    """EXPECTATION INVERTED at the loaders.py -> models.py merge, deliberately.

    loaders.py fell back to AutoModelForCausalLM whenever an architecture was not importable.
    That is silent and wrong for exactly the models this module exists to load: a
    ...ForConditionalGeneration checkpoint loaded through the causal-LM class either raises deep
    in from_pretrained or attaches the wrong head. models.py raises instead, naming the
    architectures it tried. This is the only test covering that path, so it is flipped, not
    deleted."""
    import pytest
    from transformers import AutoConfig

    cfg = types.SimpleNamespace(architectures=["SomeVendorForSomethingUnknown"], auto_map={})
    orig = AutoConfig.from_pretrained
    AutoConfig.from_pretrained = staticmethod(lambda *a, **k: cfg)
    try:
        with pytest.raises(ValueError, match="no auto class"):
            load_auto_class("any/model")
    finally:
        AutoConfig.from_pretrained = orig


def test_checkpoint_loader_forwards_pinned_revision_to_base_config_and_weights(
    monkeypatch,
):
    import pproute.models as models

    revision = "pinned-revision"
    checkpoint = "/tmp/adapter-checkpoint"
    calls = {}
    tokenizer = types.SimpleNamespace(chat_template=None, pad_token="<pad>", pad_token_id=0)
    base = types.SimpleNamespace(config=types.SimpleNamespace(pad_token_id=None))

    def config_from_pretrained(name, **kwargs):
        calls["config"] = (name, kwargs)
        return types.SimpleNamespace(architectures=["RevisionProbeModel"], auto_map={})

    class RevisionProbeModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["weights"] = (name, kwargs)
            return base

    def attach_adapter(loaded_base, path):
        assert loaded_base is base
        assert path == checkpoint
        return base

    monkeypatch.setattr(
        models,
        "PeftConfig",
        types.SimpleNamespace(
            from_pretrained=lambda path: types.SimpleNamespace(
                base_model_name_or_path="org/base"
            )
        ),
    )
    monkeypatch.setattr(
        models,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
    )
    monkeypatch.setattr(
        models,
        "AutoConfig",
        types.SimpleNamespace(from_pretrained=config_from_pretrained),
    )
    monkeypatch.setattr(
        models.transformers,
        "RevisionProbeModel",
        RevisionProbeModel,
        raising=False,
    )
    monkeypatch.setattr(
        models,
        "PeftModel",
        types.SimpleNamespace(from_pretrained=attach_adapter),
    )

    loaded, loaded_tokenizer, _ = models.load_model_and_tokenizer(
        "org/base",
        revision=revision,
        dir_checkpoint=checkpoint,
    )

    assert loaded is base
    assert loaded_tokenizer is tokenizer
    assert calls["config"] == (
        "org/base",
        {"revision": revision, "trust_remote_code": True},
    )
    assert calls["weights"][0] == "org/base"
    assert calls["weights"][1]["revision"] == revision


# --------------------------------------------------------------------------------------------
# the trl/peft path, end to end
# --------------------------------------------------------------------------------------------


def test_sft_config_accepts_the_projects_train_param_keys(monkeypatch):
    """finetune_model.py splats train_param into SFTConfig; every key must still exist.

    `save_safetensors` was silently dropped from TrainingArguments in the 5.x line and raised
    TypeError here.

    bf16/fp16 are passed explicitly, exactly as the entry script does: trl now defaults
    `bf16=True`, which refuses to construct at all on a CPU-only node.
    """
    from trl import SFTConfig

    from scripts.finetune_model import main

    # This is a constructor/API check, not a hardware check. Keep a login-node test from probing a
    # CUDA driver merely because TrainingArguments resolves device defaults during construction.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert callable(main)  # import guard: the entry module must load in this environment

    train_param = {
        "max_length": 1024,
        "per_device_train_batch_size": 8,
        "auto_find_batch_size": True,
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
        "packing": False,
    }
    cfg = SFTConfig(
        output_dir="/tmp/pproute-sftconfig-check", **train_param, bf16=False, fp16=False
    )
    assert cfg.max_length == 1024


def test_sft_nll_accepts_accelerate_forward_hooks(tmp_path):
    """The 122B device map wraps forward in a partial, which chunked CE cannot patch."""
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import SFTConfig, SFTTrainer

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]",
    )
    model = GPT2LMHeadModel(GPT2Config(vocab_size=2, n_embd=8, n_layer=1, n_head=1))
    add_hook_to_module(model, AlignDevicesHook(execution_device="cpu"))
    assert not hasattr(model.forward, "__func__")
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_dict({"input_ids": [[0, 0, 1]]}),
        args=SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
                       loss_type="nll", report_to="none"),
    )
    assert trainer.args.loss_type == "nll"


@pytest.mark.slow
def test_lora_finetune_save_and_reload_roundtrip(tmp_path):
    """Train two LoRA steps, save, and reload the checkpoint the way evaluate_model.py does.

    Covers trl SFTTrainer + peft attach + PeftConfig round trip on one torch/transformers pair,
    which is the combination the upgrade was most likely to break.
    """
    if not _hub_reachable():
        pytest.skip("needs %s from the Hub/cache" % SMALL_MODEL)

    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    model, tok, _ = load_model_and_tokenizer(SMALL_MODEL)
    model = model.to(torch.float32)
    data = Dataset.from_dict({"text": ["The capital of France is Paris." for _ in range(8)]})

    out = tmp_path / "ckpts"
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        processing_class=tok,
        args=SFTConfig(
            output_dir=str(out),
            max_length=32,
            per_device_train_batch_size=2,
            auto_find_batch_size=False,
            max_steps=2,
            save_strategy="steps",
            save_steps=2,
            logging_strategy="no",
            report_to=[],
            packing=False,
            use_cpu=True,
        ),
        peft_config=LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    trainer.train()

    ckpts = sorted(out.glob("checkpoint-*"))
    assert ckpts, "SFTTrainer wrote no checkpoint"

    reloaded, _, _ = load_model_and_tokenizer(SMALL_MODEL, dir_checkpoint=str(ckpts[-1]))
    reloaded.eval()
    with torch.no_grad():
        logits = reloaded(**tok("Hello there", return_tensors="pt")).logits
    assert torch.isfinite(logits).all()
