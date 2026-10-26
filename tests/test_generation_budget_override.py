"""Check shipped generation ceilings against pinned models in the local HF cache.

The offline evaluator tests verify that these ceilings are cleared for scoring.
"""
import json
import os

import pytest

pytestmark = pytest.mark.live

# (name, revision, expected generation_config.max_new_tokens)
COHORT_CEILINGS = [
    ("allenai/Olmo-3-7B-Instruct", "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc", 32768),
    ("allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f", None),
    ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", None),
    ("mistralai/Mistral-7B-v0.3", "caa1feb0e54d415e2df31207e5f4e273e33509b1", None),
    ("tiiuae/Falcon3-7B-Instruct", "1e57a0ecd176c7c139f289c60a74e57f887c3dfb", None),
]


@pytest.mark.parametrize("name, revision, expected", COHORT_CEILINGS)
def test_only_olmo_instruct_ships_a_generation_ceiling(name, revision, expected):
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(name, "generation_config.json", revision=revision)
    if not isinstance(path, str) or not os.path.exists(path):
        pytest.skip("generation_config.json for %s is not in the local cache" % name)
    assert json.load(open(path)).get("max_new_tokens") == expected
