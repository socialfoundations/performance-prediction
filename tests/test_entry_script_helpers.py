"""Entry-script helpers must be callable the way their own main() calls them.

These are cheap checks on preprocessing gates called by real experiment entry points.
"""
import re
from types import SimpleNamespace

import pytest

from scripts.finetune_model import (dataset_content_sha256,
                                    get_target_modules,
                                    require_clean_sft_diagnostics,
                                    validate_trace_sft_args)


def test_gemma3_lora_reaches_language_attention_only():
    pattern = get_target_modules(SimpleNamespace(config=SimpleNamespace(model_type="gemma3")))
    names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.47.self_attn.v_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
    ]
    assert [bool(re.fullmatch(pattern, name)) for name in names] == [True, True, False, False]


def test_qwen35_moe_lora_reaches_both_language_attention_types():
    pattern = get_target_modules(SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_moe")))
    names = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.v_proj",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj",
        "model.visual.blocks.0.attn.q_proj",
    ]
    assert [bool(re.fullmatch(pattern, name)) for name in names] == [True, True, True, False, False]


@pytest.mark.parametrize(
    ("trace_path", "completion_source"),
    [("", "teacher"), ("traces.jsonl", "")],
)
def test_trace_sft_path_and_source_must_be_supplied_together(trace_path, completion_source):
    with pytest.raises(ValueError, match="must be supplied together"):
        validate_trace_sft_args(trace_path, completion_source)


@pytest.mark.parametrize(
    ("trace_path", "completion_source"),
    [("", ""), ("traces.jsonl", "teacher")],
)
def test_trace_sft_path_and_source_accept_valid_pairs(trace_path, completion_source):
    validate_trace_sft_args(trace_path, completion_source)


def test_sft_content_hash_binds_values_and_order():
    rows = [{"text": "first"}, {"text": "second"}]
    assert dataset_content_sha256(rows) == dataset_content_sha256(list(rows))
    assert dataset_content_sha256(rows) != dataset_content_sha256(list(reversed(rows)))
    assert dataset_content_sha256(rows) != dataset_content_sha256(
        [{"text": "first"}, {"text": "changed"}]
    )
