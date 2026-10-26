import logging

import lm_eval
# Imported explicitly: lm_eval's top-level module is lazy, so `lm_eval.models.huggingface.HFLM`
# raises AttributeError ("module 'lm_eval' has no attribute 'models'") even though the import works.
from lm_eval.models.huggingface import HFLM as HarnessHFLM

from pproute.models import config_max_length
from pproute.scores import BLANK, CHOICE_LENS_KEY, GOLD_KEY, MC_TARGET_KEY, PREDICTOR_TEXT_KEY

eval_logger = logging.getLogger(__name__)
FEWSHOT_RANDOM_SEED = 1234


class HFLM(HarnessHFLM):
    """Refuse the harness's implicit prompt truncation when reserving generation tokens."""

    def tok_batch_encode(self, strings, padding_side="left", left_truncate_len=None,
                         truncation=False):
        if truncation:
            raise ValueError("evaluation forbids tokenizer truncation")
        encoded, mask = super().tok_batch_encode(strings, padding_side=padding_side,
                                                 truncation=False)
        if left_truncate_len is not None and encoded.shape[1] > left_truncate_len:
            raise ValueError(
                "generation prompt has %d tokens but only %d fit after reserving the answer "
                "budget; refusing to truncate the prompt" % (encoded.shape[1], left_truncate_len))
        return encoded, mask


def _gold_index(task, doc):
    """The gold option index for one doc, resolved exactly as lm_eval resolves it internally
    (lm_eval/api/task.py, ConfigurableTask.process_results). Returns -100 when out of range,
    matching lm_eval, and a list for multiple_target tasks.

    The distinction that matters: under `multiple_input` the options live in the *context*, so the
    gold index comes from `doc_to_text`, while `doc_to_target` returns the continuation shared by
    every option. Only the task object knows this, which is why the index has to be recorded here
    rather than reconstructed later from the score file."""
    gold = task.doc_to_text(doc) if getattr(task, "multiple_input", 0) else task.doc_to_target(doc)
    choices = task.doc_to_choice(doc)
    if isinstance(gold, list):                                  # multiple_target
        return [i if i < len(choices) else -100 for i in gold]
    if isinstance(gold, int):
        return gold if gold < len(choices) else -100
    if isinstance(gold, str):
        return choices.index(gold) if gold in choices else -100
    return -100


def build_predictor_contexts(
    task_dict, num_fewshot, fewshot_random_seed, rank=0, world_size=1
):
    """Rebuild full task-native prompts before vendor chat rendering.

    lm-eval's few-shot sampler is stateful: prompt construction consumes one sample per document.
    Reset it to the explicit evaluation seed, walk the same stable doc iterator, and let
    ``simple_evaluate`` reset it to that seed again before building the actual raw/chat requests.
    This gives predictor ``g`` the same demonstrations and target question without vendor role
    markers, independently of the scored model's prompt format.
    """
    contexts = {}
    for task_name, task in task_dict.items():
        if isinstance(task, dict) or task.get_config("output_type") != "generate_until":
            continue
        configured_num_fewshot = task.get_config("num_fewshot")
        # Mirror lm-eval's override rule exactly: a task that explicitly freezes zero-shot
        # ignores a caller's nonzero override. This matters for task-native prompts (for example,
        # MMLU-Pro) that already carry their demonstrations in doc_to_text.
        effective_num_fewshot = (
            0 if configured_num_fewshot == 0 else num_fewshot
        )
        if effective_num_fewshot is None:
            effective_num_fewshot = configured_num_fewshot or 0
        task.set_fewshot_seed(seed=fewshot_random_seed)
        by_doc_id = {}
        try:
            for doc_id, doc in task.doc_iterator(rank=rank, world_size=world_size):
                text = task.fewshot_context(
                    doc,
                    num_fewshot=effective_num_fewshot,
                    system_instruction=None,
                    apply_chat_template=False,
                    fewshot_as_multiturn=False,
                    chat_template=None,
                    gen_prefix=task.doc_to_prefix(doc),
                )
                target_only = task.fewshot_context(
                    doc,
                    num_fewshot=0,
                    system_instruction=None,
                    apply_chat_template=False,
                    fewshot_as_multiturn=False,
                    chat_template=None,
                    gen_prefix=task.doc_to_prefix(doc),
                )
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        "%s produced no normalized predictor context for doc_id %s"
                        % (task_name, doc_id)
                    )
                if effective_num_fewshot and text == target_only:
                    raise ValueError(
                        "%s normalized predictor context dropped all %d demonstrations for "
                        "doc_id %s" % (task_name, effective_num_fewshot, doc_id)
                    )
                if doc_id in by_doc_id:
                    raise ValueError(
                        "%s predictor context duplicated doc_id %s" % (task_name, doc_id)
                    )
                by_doc_id[doc_id] = text
        finally:
            # Leave the task in the same state simple_evaluate expects, including on a failed
            # precompute. simple_evaluate explicitly performs the same reset once more.
            task.set_fewshot_seed(seed=fewshot_random_seed)
        contexts[task_name] = by_doc_id
    return contexts


def _mc_predictor_target(task_name, doc):
    """Only the target question and ordered choices; never consult its gold answer."""
    if task_name == "commonsense_qa":
        choices = dict(zip(doc["choices"]["label"], doc["choices"]["text"]))
        return {"question": doc["question"].strip(),
                "choices": [choices[label] for label in "ABCDE"], "layout": "standard"}
    if task_name == "winogrande":
        return {"question": doc["sentence"].replace("_", BLANK, 1),
                "choices": [doc["option1"], doc["option2"]], "layout": "multiple_input"}
    raise ValueError("no native MC target renderer for %s" % task_name)


def stamp_gold_index(results, task_dict, predictor_contexts=None):
    """Record task-owned metadata needed after lm-eval releases the task object.

    Multiple-choice samples receive the exact gold option index and the per-option
    ``len(doc_to_choice(doc))``, which is what lm_eval length-normalizes by for acc_norm tasks.
    Neither survives in the score file otherwise: the gold index comes from ``doc_to_text`` under
    ``multiple_input``, and the scored request string carries the task's ``target_delimiter`` on
    top of each choice, so its length is the wrong denominator (see ``scores._choice_lengths``).
    The lengths are stamped for every multiple-choice task, not only the acc_norm ones, so that
    adding a task to ``scores.ACC_NORM_TASKS`` later does not invalidate scores already paid for.

    Generative samples receive the full task-native, pre-chat few-shot prompt used by predictor
    ``g``. It is precomputed before lm-eval turns demonstrations into vendor-specific chat turns,
    then joined here by stable ``doc_id``.

    Out-of-range golds are stamped as -100 and warned about rather than raised on: the evaluation
    has already been paid for, so the score file is still written, and `scores._gold_index` then
    refuses to score the affected samples -- naming them -- instead of quietly mis-scoring them."""
    for task_name, task in task_dict.items():
        if isinstance(task, dict):
            continue
        output_type = task.get_config("output_type")
        samples = results.get("samples", {}).get(task_name)
        if output_type == "generate_until":
            if samples is None:
                raise ValueError("%s results have no per-sample records" % task_name)
            if predictor_contexts is None or task_name not in predictor_contexts:
                raise ValueError("%s has no precomputed predictor context map" % task_name)
            by_doc_id = predictor_contexts[task_name]
            observed_doc_ids = [it.get("doc_id") for it in samples]
            if set(observed_doc_ids) != set(by_doc_id):
                raise ValueError(
                    "%s predictor context IDs %s do not exactly match result IDs %s"
                    % (task_name, sorted(by_doc_id), sorted(observed_doc_ids))
                )
            for it in samples:
                it[PREDICTOR_TEXT_KEY] = by_doc_id[it["doc_id"]]
            eval_logger.info(
                "%s: stamped %s on %d samples",
                task_name, PREDICTOR_TEXT_KEY, len(samples),
            )
            continue
        if not samples:
            continue
        if output_type != "multiple_choice":
            continue
        bad = []
        for it in samples:
            it[GOLD_KEY] = _gold_index(task, it["doc"])
            it[CHOICE_LENS_KEY] = [len(choice) for choice in task.doc_to_choice(it["doc"])]
            if (task_name in ("commonsense_qa", "winogrande")
                    and (task.get_config("num_fewshot") or 0) > 0):
                it[MC_TARGET_KEY] = _mc_predictor_target(task_name, it["doc"])
            if it[GOLD_KEY] == -100:
                bad.append(it.get("doc_id"))
        if bad:
            eval_logger.warning(
                "%s: gold option out of range for %d/%d samples (doc_ids %s%s). Stamped as -100; "
                "pproute.scores will refuse them.", task_name, len(bad), len(samples),
                bad[:10], " ..." if len(bad) > 10 else "")
        else:
            eval_logger.info("%s: stamped %s on %d samples", task_name, GOLD_KEY, len(samples))
    return results


class Evaluator:
    def __init__(
        self,
        task_dict,
        task_manager,
        use_auto_batch_size=True,
        max_batch_size=8,
        max_new_tokens=-1,
        add_bos_token=True,
        apply_chat_template=False,
        num_fewshot=0,
        fewshot_as_multiturn=False,
        fewshot_random_seed=FEWSHOT_RANDOM_SEED,
    ):
        self.task_manager = task_manager
        self.task_dict = task_dict
        self.max_batch_size = max_batch_size
        self.add_bos_token = add_bos_token
        self.max_new_tokens = max_new_tokens
        self.apply_chat_template = apply_chat_template
        self.num_fewshot = num_fewshot
        self.fewshot_as_multiturn = fewshot_as_multiturn
        self.fewshot_random_seed = fewshot_random_seed
        self.use_auto_batch_size = use_auto_batch_size

    def evaluate(self, model, tokenizer):
        # HFLM checks only top-level config fields before falling back to 2,048. Several cohort
        # models keep the language tower's real context window in text_config; the fallback can
        # either left-truncate silently or reject a 2,048-token generation budget outright.
        max_length = config_max_length(getattr(model, "config", None)) \
            if model is not None else None
        # A checkpoint can ship its own generation ceiling: transformers gives
        # generation_config.max_new_tokens precedence over the max_length that HFLM passes to
        # generate(), so a model carrying one escapes the frozen task budget without an error
        # (Olmo-3-7B-Instruct ships 32768 and ran five GSM8K validation units at that ceiling
        # instead of 2,048 on 16 Aug 2026, changing Z for every answer past the budget). Clear
        # it so the task budget is the only ceiling.
        gen_config = getattr(model, "generation_config", None) if model is not None else None
        if gen_config is not None and getattr(gen_config, "max_new_tokens", None) is not None:
            eval_logger.warning(
                "clearing generation_config.max_new_tokens=%s so the task budget governs",
                gen_config.max_new_tokens)
            gen_config.max_new_tokens = None
        lm_obj = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            truncation=False,
            trust_remote_code=True,
            add_bos_token=self.add_bos_token,
            max_length=max_length,
            batch_size="auto" if self.use_auto_batch_size else self.max_batch_size,
            max_batch_size=self.max_batch_size,
            enable_thinking=False,
        )
        rank = getattr(lm_obj, "rank", 0)
        world_size = getattr(lm_obj, "world_size", 1)
        if rank != 0 or world_size != 1:
            raise ValueError(
                "normalized predictor contexts currently require one-process evaluation"
            )
        predictor_contexts = build_predictor_contexts(
            self.task_dict,
            self.num_fewshot,
            self.fewshot_random_seed,
            rank=rank,
            world_size=world_size,
        )
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=[self.task_dict[k] for k in self.task_dict.keys()],
            num_fewshot=self.num_fewshot,
            fewshot_as_multiturn=self.fewshot_as_multiturn,
            fewshot_random_seed=self.fewshot_random_seed,
            task_manager=self.task_manager,
            batch_size="auto" if self.use_auto_batch_size else self.max_batch_size,
            max_batch_size=self.max_batch_size,
            apply_chat_template=self.apply_chat_template,
            # -1 => do not override; the task's own max_gen_toks applies. Passing a value here is
            # unconditional in lm_eval, so a default of 256 silently truncated any task that asks
            # for more: MMLU-Pro requests 2048 for its 5-shot chain of thought and would have been
            # cut off long before the answer, on every model, driving Z toward zero for reasons that
            # have nothing to do with the model. Set it explicitly per benchmark instead.
            **({} if self.max_new_tokens is None or self.max_new_tokens < 0
               else {"gen_kwargs": f"max_gen_toks={self.max_new_tokens}"}),
        )
        return stamp_gold_index(results, self.task_dict, predictor_contexts)
