"""Canonical assistant boundaries for forced multiple-choice continuations."""
from types import SimpleNamespace

import pytest

from pproute.tasks import assistant_message_bridge, canonicalize_mc_chat_requests


class _Tokenizer:
    def __init__(self, bridge=" ", unstable_prefix=False):
        self.bridge = bridge
        self.unstable_prefix = unstable_prefix

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            continue_final_message=False):
        assert tokenize is False
        prompt = "<s>[INST]" + messages[0]["content"] + "[/INST]"
        if add_generation_prompt:
            return prompt
        assert continue_final_message is True
        return prompt + self.bridge + messages[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids = list(text.encode())
        if self.unstable_prefix and text.endswith(" A"):
            ids[0] += 1
        return ids


class _Task:
    def __init__(self, output_type="multiple_choice", arguments=()):
        self.output_type = output_type
        self.instances = [SimpleNamespace(arguments=value) for value in arguments]

    def get_config(self, key):
        return {"output_type": self.output_type, "task": "test"}[key]


def test_mistral_like_bridge_is_added_to_every_option():
    tokenizer = _Tokenizer(bridge=" ")
    context = "<s>[INST]Question[/INST]"
    task = _Task(arguments=[(context, "A"), (context, "B")])

    diagnostics = canonicalize_mc_chat_requests(task, tokenizer, True)

    assert assistant_message_bridge(tokenizer) == " "
    assert [instance.arguments for instance in task.instances] == [
        (context, " A"), (context, " B")
    ]
    assert diagnostics == {"requests": 2, "changed": 2, "bridge": " "}


def test_template_with_no_bridge_keeps_options_byte_identical():
    tokenizer = _Tokenizer(bridge="")
    context = "<s>[INST]Question[/INST]"
    task = _Task(arguments=[(context, "A")])

    diagnostics = canonicalize_mc_chat_requests(task, tokenizer, True)

    assert task.instances[0].arguments == (context, "A")
    assert diagnostics == {"requests": 1, "changed": 0, "bridge": ""}


@pytest.mark.parametrize("output_type,apply", [
    ("multiple_choice", False),
    ("generate_until", True),
])
def test_raw_mc_and_generation_are_untouched(output_type, apply):
    task = _Task(output_type=output_type, arguments=[("prompt", "answer")])

    diagnostics = canonicalize_mc_chat_requests(task, None, apply)

    assert task.instances[0].arguments == ("prompt", "answer")
    assert diagnostics == {"requests": 0, "changed": 0, "bridge": ""}


def test_context_retokenization_is_rejected():
    context = "<s>[INST]Question[/INST]"
    task = _Task(arguments=[(context, "A")])
    with pytest.raises(ValueError, match="changes the multiple-choice context token IDs"):
        canonicalize_mc_chat_requests(task, _Tokenizer(unstable_prefix=True), True)
