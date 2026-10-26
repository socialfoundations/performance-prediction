"""MMLU-Pro: the carve, the answer regex, the trace contract, the protocol, and pooling.

MMLU-Pro is the one benchmark that arrives as several lm-eval tasks. The offline tests here run on
synthetic docs and cost nothing; the `live` ones need the MMLU-Pro dataset from the HF cache and
check the properties that only the real fourteen subtasks can show -- that the pooled carve is an
exact 6,000 / 3,000 / 3,032 partition, that the demonstrations never enter it, and that the shipped
tasks really do own the demonstrations so `fewshot_as_multiturn` has something to convert.
"""
import joblib as jbl
import pytest

from pproute import gsm8k, mmlu_pro, scores, trace_sft

CATEGORIES = list(mmlu_pro.CATEGORIES)


def fake_docs(n=400, seed=7):
    """A pool shaped like the real one: 14 categories, and within each a dominant src plus two
    smaller ones -- MMLU-Pro has 91 srcs over 12,032 rows, with `ori_mmlu-professional_law` alone
    at 1,003, so strata are large and singletons are rare. Singletons get their own test below,
    because no stratifier can divide a stratum of one and pretending otherwise would only test
    the fixture."""
    import random

    rng = random.Random(seed)
    docs = []
    for i in range(n):
        category = CATEGORIES[i % len(CATEGORIES)]
        src = rng.choices(["main", "mid", "rare"], weights=[70, 25, 5])[0]
        docs.append({"question_id": i, "category": category,
                     "src": "%s-%s" % (category, src), "answer": "ABCDEFGHIJ"[i % 10]})
    return docs


# ------------------------------------------------------------------ Z, as lm_eval defines it
def test_extract_answer_matches_the_eval_filter():
    assert mmlu_pro.extract_answer("blah blah. The answer is (G).") == "G"
    assert mmlu_pro.extract_answer("so the answer is D") == "D"
    assert mmlu_pro.extract_answer("I think it is G.") is None    # no required phrase -> invalid
    assert mmlu_pro.extract_answer("") is None


def test_extraction_is_case_sensitive_but_comparison_is_not():
    """lm_eval compiles the pattern without re.IGNORECASE, while the metric sets ignore_case.
    Those are two different steps and conflating them would silently change Z."""
    assert mmlu_pro.extract_answer("The Answer Is (G).") is None   # regex misses it entirely
    assert mmlu_pro.extract_answer("the answer is (g)") is None    # only A-J, upper case
    assert mmlu_pro.is_correct("the answer is (G)", "g")           # comparison ignores case


def test_take_first_not_take_last():
    """`take_first` follows the regex, so a model that names a letter mid-reasoning is scored on
    that one. The teacher's acceptance rule has to agree with this or it would keep traces the
    evaluation marks wrong."""
    text = "the answer is (A), no wait, the answer is (C)."
    assert mmlu_pro.extract_answer(text) == "A"
    assert not mmlu_pro.is_correct(text, "C")


def test_selected_option_content_uses_structured_options_and_marks_unavailable_letters():
    options = ["first choice", "second choice"]
    assert mmlu_pro.selected_option_content(options, "B") == "second choice"
    assert mmlu_pro.selected_option_content(options, "J") == mmlu_pro.UNAVAILABLE_OPTION
    with pytest.raises(ValueError, match="A through J"):
        mmlu_pro.selected_option_content(options, "K")
    with pytest.raises(ValueError, match="non-empty"):
        mmlu_pro.selected_option_content([], "A")


def test_example_id_is_the_shipped_question_id():
    """MMLU-Pro ships a stable id, so unlike GSM8K there is nothing to hash. It is also what makes
    an id meaningful across subtasks, which the pooled carve depends on."""
    assert mmlu_pro.example_id({"question_id": 1000}) == "1000"
    assert mmlu_pro.example_id({"question_id": "1000"}) == "1000"


def test_the_fourteen_subtask_names_are_the_shipped_ones():
    assert len(mmlu_pro.SUBTASKS) == 14
    assert "mmlu_pro_computer_science" in mmlu_pro.SUBTASKS   # the space becomes an underscore
    assert "mmlu_pro_flat" not in mmlu_pro.SUBTASKS


# ------------------------------------------------------------------ the carve
def test_carve_is_an_exact_disjoint_partition():
    docs = fake_docs(400)
    tr, va, te = mmlu_pro.stratified_three_way_split(docs, 200, 100, seed=0)
    assert (len(tr), len(va), len(te)) == (200, 100, 100)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert set(tr) | set(va) | set(te) == set(range(len(docs)))


def test_carve_is_deterministic_in_the_seed_and_moves_with_it():
    docs = fake_docs(400)
    a = mmlu_pro.stratified_three_way_split(docs, 200, 100, seed=0)
    b = mmlu_pro.stratified_three_way_split(docs, 200, 100, seed=0)
    c = mmlu_pro.stratified_three_way_split(docs, 200, 100, seed=1)
    assert a == b
    assert a != c


def test_carve_keeps_every_category_proportional():
    """The reason the carve is stratified at all: an unstratified draw can leave SFT and test
    differing in composition, and that difference would be read as delta-accuracy."""
    docs = fake_docs(1400)
    n_train, n_valid = 700, 350
    tr, va, te = mmlu_pro.stratified_three_way_split(docs, n_train, n_valid, seed=0)
    shares = (n_train / len(docs), n_valid / len(docs), (len(docs) - n_train - n_valid) / len(docs))
    for category in CATEGORIES:
        members = {i for i, d in enumerate(docs) if d["category"] == category}
        assert members
        for split, want in zip((tr, va, te), shares):
            got = len(members & set(split)) / len(members)
            assert abs(got - want) < 0.05, (category, got, want)


def test_carve_beats_an_unstratified_draw_on_the_worst_stratum():
    """Not a tautology: it is the comparison that justifies the extra machinery.

    Measured over strata of at least 10 rows. A stratum of one cannot be divided proportionally by
    any method, so including singletons would floor both numbers at 0.5 and compare nothing.
    """
    import random

    docs = fake_docs(1400)
    n_train = 700
    want = n_train / len(docs)
    by_src = {}
    for i, d in enumerate(docs):
        by_src.setdefault(d["src"], set()).add(i)
    divisible = [m for m in by_src.values() if len(m) >= 10]
    assert len(divisible) > 10, "fixture has too few usable strata to compare"

    def worst(train_idx):
        train = set(train_idx)
        return max(abs(len(m & train) / len(m) - want) for m in divisible)

    tr, _, _ = mmlu_pro.stratified_three_way_split(docs, n_train, 350, seed=0)
    order = list(range(len(docs)))
    random.Random(0).shuffle(order)
    assert worst(tr) < worst(order[:n_train])


def test_singletons_do_not_all_land_in_one_split():
    """The per-stratum uniform offset exists for this. Without it every size-1 stratum keys to
    exactly 0.5 and they all land in whichever split covers that point."""
    docs = [{"question_id": i, "category": "c", "src": "only-%d" % i, "answer": "A"}
            for i in range(300)]
    tr, va, te = mmlu_pro.stratified_three_way_split(docs, 150, 75, seed=0)
    assert len(tr) and len(va) and len(te)
    assert 0.4 < len(tr) / len(docs) < 0.6


def test_carve_refuses_to_leave_no_test_rows():
    with pytest.raises(ValueError, match="no test rows"):
        mmlu_pro.stratified_three_way_split(fake_docs(100), 60, 40, seed=0)


def test_pooled_carve_partitions_across_subtasks_by_id():
    """The carve is taken once over every subtask's documents pooled together, so the totals are
    the pre-registered sizes rather than fourteen independent roundings, and no id can land in two
    splits."""
    docs = fake_docs(420)
    by_subtask = {}
    for d in docs:
        by_subtask.setdefault("mmlu_pro_" + d["category"].replace(" ", "_"), []).append(d)
    assert len(by_subtask) == 14

    train, valid, test = mmlu_pro.carve_example_ids(by_subtask, seed=0, n_train=210, n_valid=105)
    assert (len(train), len(valid), len(test)) == (210, 105, 105)
    assert not (train & valid) and not (train & test) and not (valid & test)
    assert train | valid | test == {mmlu_pro.example_id(d) for d in docs}


def test_pooled_carve_does_not_depend_on_subtask_ordering():
    docs = fake_docs(420)
    by_subtask = {}
    for d in docs:
        by_subtask.setdefault("mmlu_pro_" + d["category"].replace(" ", "_"), []).append(d)
    reversed_order = dict(reversed(list(by_subtask.items())))
    assert mmlu_pro.carve_example_ids(by_subtask, 0, 210, 105) == \
        mmlu_pro.carve_example_ids(reversed_order, 0, 210, 105)


# ------------------------------------------------------------------ the trace contract
def _trace(qid="1", gold="C", completion=None):
    return {"example_id": qid, "gold": gold,
            "completion": completion or "reasoning. the answer is (%s)." % gold,
            "provenance": {"task": mmlu_pro.TASK_NAME, "temperature": 0.7}}


def _write(tmp_path, rows):
    import json

    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_accepted_traces_load_under_the_shared_canonical_schema(tmp_path):
    path = _write(tmp_path, [_trace("1", "C"), _trace("2", "A")])
    traces, info = trace_sft.load_accepted_traces(
        path, adapter=mmlu_pro.TRACE_SFT_ADAPTER, allowed_ids={"1", "2"}
    )
    assert set(traces) == {"1", "2"}
    assert info["task_name"] == mmlu_pro.TASK_NAME


def test_traces_outside_the_sft_split_are_refused(tmp_path):
    path = _write(tmp_path, [_trace("1", "C")])
    with pytest.raises(ValueError, match="NOT in MMLU-Pro's SFT split"):
        trace_sft.load_accepted_traces(
            path, adapter=mmlu_pro.TRACE_SFT_ADAPTER, allowed_ids={"someone-else"}
        )


def test_a_trace_the_evaluation_would_mark_wrong_is_refused(tmp_path):
    path = _write(tmp_path, [_trace("1", "C", "reasoning. the answer is (D).")])
    with pytest.raises(ValueError, match="not gold"):
        trace_sft.load_accepted_traces(
            path, adapter=mmlu_pro.TRACE_SFT_ADAPTER, allowed_ids={"1"}
        )


def test_trace_validity_is_judged_on_the_completion_not_the_whole_text():
    """The bug worth remembering. The evaluation prompt carries five demonstrations that each end
    "The answer is (X)", and the filter is `take_first` -- so a check over prompt + completion
    reads the FIRST demonstration's letter and has nothing to do with the model's answer. On the
    real 6,000 answer-only rows only 617 would have passed a whole-text check; judged correctly,
    all 6,000 pass."""
    prompt = "...\nAnswer: The answer is (B).\n\nQuestion:\n...\nAnswer: Let's think step by step."
    completion = " The answer is (C)."
    assert mmlu_pro.is_correct(completion, "C")
    assert not mmlu_pro.is_correct(prompt + completion, "C")
    assert mmlu_pro.is_correct(prompt + completion, "B")   # the demonstration's letter


class FakeInstance:
    def __init__(self, doc, prompt):
        self.doc = doc
        self.arguments = (prompt, {})


class FakeSubtask:
    """One per-category lm_eval task, holding only its own slice of the SFT split."""

    def __init__(self, docs):
        self._docs = docs
        self.instances = [FakeInstance(d, "prompt for %s" % d["question_id"]) for d in docs]

    @property
    def eval_docs(self):
        return self._docs

    def doc_to_target(self, doc):
        return doc["answer"]

    def get_config(self, key):
        return " " if key == "target_delimiter" else None


def test_trace_sft_spans_every_subtask(tmp_path):
    """Regression. MMLU-Pro's SFT split is spread over fourteen tasks, so the allowlist and the
    "every accepted trace was used" check have to span the set. Building per task instead reports
    almost every trace as missing, because each subtask carries only its own category."""
    law = FakeSubtask([{"question_id": 1, "answer": "C"}, {"question_id": 2, "answer": "A"}])
    bio = FakeSubtask([{"question_id": 3, "answer": "B"}])
    path = _write(tmp_path, [_trace("1", "C"), _trace("2", "A"), _trace("3", "B")])

    dataset, info = trace_sft.build_trace_sft_dataset(
        [law, bio], path, "teacher", adapter=mmlu_pro.TRACE_SFT_ADAPTER
    )
    assert sorted(dataset["example_id"]) == ["1", "2", "3"]
    assert info["sft_split_n"] == 3
    assert all(c.startswith(" ") for c in dataset["completion"])


def test_trace_sft_still_accepts_a_single_task(tmp_path):
    """GSM8K passes one task and must keep working unchanged."""
    only = FakeSubtask([{"question_id": 1, "answer": "C"}])
    path = _write(tmp_path, [_trace("1", "C")])
    dataset, _ = trace_sft.build_trace_sft_dataset(
        only, path, "teacher", adapter=mmlu_pro.TRACE_SFT_ADAPTER
    )
    assert dataset["example_id"] == ["1"]


# ------------------------------------------------------------------ the frozen eval protocol
def _params(**changes):
    params = dict(max_num_train=0, max_num_valid=8, max_num_test=0,
                  num_fewshot=5, fewshot_as_multiturn=True)
    params.update(changes)
    return params


@pytest.mark.parametrize(
    "params, max_new_tokens, chat, message",
    [
        (_params(num_fewshot=0), 2048, True, "5-shot"),
        (_params(fewshot_as_multiturn=False), 2048, True, "multiturn"),
        (_params(), 256, True, "2048"),
        (_params(), 2048, False, "cannot use fewshot_as_multiturn"),
    ],
)
def test_eval_protocol_refuses_silent_misconfiguration(params, max_new_tokens, chat, message):
    with pytest.raises(ValueError, match=message):
        mmlu_pro.validate_eval_protocol(
            mmlu_pro.TASK_NAME, params, max_new_tokens, apply_chat_template=chat
        )


@pytest.mark.parametrize("benchmark, task", [(mmlu_pro, mmlu_pro.TASK_NAME), (gsm8k, "gsm8k")])
def test_sft_protocol_refuses_a_format_that_disagrees_with_evaluation(benchmark, task):
    """SFT and evaluation must use matching shot counts and chat formats."""
    ok = dict(num_fewshot=5, fewshot_as_multiturn=True)
    benchmark.validate_sft_protocol(task, ok, apply_chat_template=True)
    benchmark.validate_sft_protocol(
        task, dict(num_fewshot=0, fewshot_as_multiturn=False), apply_chat_template=True
    )
    with pytest.raises(ValueError, match="0-shot or 5-shot"):
        benchmark.validate_sft_protocol(
            task, dict(num_fewshot=3, fewshot_as_multiturn=False), apply_chat_template=True
        )
    with pytest.raises(ValueError, match="multiturn"):
        benchmark.validate_sft_protocol(
            task, dict(num_fewshot=5, fewshot_as_multiturn=False), apply_chat_template=True
        )
    with pytest.raises(ValueError, match="multiturn"):
        benchmark.validate_sft_protocol(task, ok, apply_chat_template=False)
    # another benchmark's task passes straight through
    benchmark.validate_sft_protocol("some_other_task", dict(num_fewshot=3), True)


def test_eval_protocol_requires_multiturn_for_chat_models():
    """The whole reason the fourteen subtasks replaced one flattened task: lm_eval owns the
    demonstrations again, so a chat model can receive them as turns exactly as it does on GSM8K.
    Scoring a chat model without it would reintroduce the difference this design removes."""
    mmlu_pro.validate_eval_protocol(mmlu_pro.TASK_NAME, _params(), 2048, apply_chat_template=True)
    mmlu_pro.validate_eval_protocol(
        mmlu_pro.TASK_NAME, _params(fewshot_as_multiturn=False), 2048, apply_chat_template=False
    )
    mmlu_pro.validate_eval_protocol("gsm8k", _params(num_fewshot=0), 256, False)  # other task


# ------------------------------------------------------------------ diagnostics over the pool
class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split()


def _context(shots=mmlu_pro.EVAL_SHOTS, question=""):
    return "".join("Q%d the answer is (A). " % i for i in range(shots)) + question


def _sample(response, filtered, z, shots=mmlu_pro.EVAL_SHOTS, question=""):
    return {"filter": mmlu_pro.FILTER_NAME, "arguments": [[_context(shots, question), {}]],
            "resps": [[response]], "filtered_resps": [filtered], "exact_match": z}


def _results(by_subtask):
    return {"samples": {name: rows for name, rows in by_subtask.items()}}


def test_diagnostics_pool_every_subtask():
    results = _results({
        "mmlu_pro_biology": [_sample("the answer is (A).", "A", 1.0)],
        "mmlu_pro_law": [_sample("the answer is (B).", "B", 0.0),
                         _sample("no idea", mmlu_pro.INVALID, 0.0)],
    })
    d = mmlu_pro.evaluation_diagnostics(results, FakeTokenizer(), 2048)
    assert d["n"] == 3 and d["subtasks"] == 2
    assert d["accuracy"] == pytest.approx(1 / 3)
    assert d["extraction_failures"] == 1
    assert d["by_subtask"]["mmlu_pro_biology"]["acc"] == 1.0
    assert d["by_subtask"]["mmlu_pro_law"]["extraction_failure_rate"] == 0.5


def test_diagnostics_separate_extraction_failure_from_being_wrong():
    """Z is one regex, so a response that never emits the phrase scores zero whether the model knew
    the answer or not. Only the cap-hit half is fixed by a larger budget."""
    results = _results({"mmlu_pro_biology": [
        _sample("the answer is (A).", "A", 1.0),
        _sample("the answer is (B).", "B", 0.0),     # extracted, simply wrong
        _sample("I think it is A.", mmlu_pro.INVALID, 0.0),
    ]})
    d = mmlu_pro.evaluation_diagnostics(results, FakeTokenizer(), 2048)
    assert d["extraction_failures"] == 1 and d["generation_cap_hits"] == 0
    assert d["prompt_protocol_failures"] == 0


def test_diagnostics_count_a_generation_that_hit_the_ceiling():
    results = _results({"mmlu_pro_biology": [_sample("a b c d", mmlu_pro.INVALID, 0.0)]})
    d = mmlu_pro.evaluation_diagnostics(results, FakeTokenizer(), max_new_tokens=4)
    assert d["generation_cap_hits"] == 1


def test_diagnostics_flag_a_prompt_whose_demonstrations_were_dropped():
    """The saved prompts, not the requested config, are the protocol evidence: a silently
    discarded num_fewshot is a bug this project has already had once, and it shows up here."""
    results = _results({"mmlu_pro_biology": [_sample("the answer is (A).", "A", 1.0, shots=0)]})
    d = mmlu_pro.evaluation_diagnostics(results, FakeTokenizer(), 2048)
    assert d["prompt_protocol_failures"] == 1


def test_a_question_containing_the_answer_phrase_is_not_a_protocol_failure():
    """Regression, and the reason the check is one-sided. evaluate_model RAISES on a protocol
    failure after generation and before the score file is written, so a false positive discards a
    completed evaluation. Only too FEW demonstrations can indicate dropped shots; question text can
    only ever push the count up."""
    results = _results({"mmlu_pro_biology": [
        _sample("the answer is (A).", "A", 1.0,
                question="which option states the answer is (B) correctly?")]})
    d = mmlu_pro.evaluation_diagnostics(results, FakeTokenizer(), 2048)
    assert d["prompt_protocol_failures"] == 0


def test_diagnostics_refuse_a_score_file_with_no_subtask_samples():
    with pytest.raises(ValueError, match="no MMLU-Pro subtask samples"):
        mmlu_pro.evaluation_diagnostics({"samples": {"gsm8k": []}}, FakeTokenizer(), 2048)


# ------------------------------------------------------------------ pooling in scores.py
def _score_file(tmp_path, samples):
    path = tmp_path / "eval_scores.jbl"
    jbl.dump({"samples": samples}, str(path))
    return str(path)


def test_samples_namespaces_doc_ids_when_pooling_subtasks(tmp_path):
    """The correctness-critical part of running fourteen tasks. `doc_id` restarts at zero inside
    each subtask, so a bare id collides across categories -- and every id would still resolve, so
    the predictor split would misalign silently rather than fail."""
    path = _score_file(tmp_path, {
        "mmlu_pro_law": [{"doc_id": 0, "x": "law0"}, {"doc_id": 1, "x": "law1"}],
        "mmlu_pro_biology": [{"doc_id": 0, "x": "bio0"}],
    })
    task, pooled = scores._samples(path)
    assert task == "mmlu_pro"
    assert len(pooled) == 3
    ids = [it["doc_id"] for it in pooled]
    assert len(set(ids)) == 3, "doc_ids collided across subtasks"
    assert "mmlu_pro_biology:0" in ids and "mmlu_pro_law:0" in ids


def test_samples_leaves_a_single_task_file_untouched(tmp_path):
    """Every other benchmark is one task, and its ids must not acquire a prefix -- existing score
    files and predictor joins already written against them would stop matching."""
    path = _score_file(tmp_path, {"gsm8k": [{"doc_id": 0}, {"doc_id": 1}]})
    task, pooled = scores._samples(path)
    assert task == "gsm8k"
    assert [it["doc_id"] for it in pooled] == [0, 1]


def test_samples_refuses_a_file_with_no_records(tmp_path):
    with pytest.raises(ValueError, match="no per-sample records"):
        scores._samples(_score_file(tmp_path, {}))


# ------------------------------------------------------------------ live: the real dataset
@pytest.mark.live
def test_dataset_shape_is_what_the_carve_assumes():
    from datasets import load_dataset

    data = load_dataset(mmlu_pro.DATASET_PATH)
    assert len(data["test"]) == mmlu_pro.SFT_SIZE + mmlu_pro.PREDICTOR_TRAIN_SIZE + 3032
    assert len(data["validation"]) == 70
    # the whole reason traces must be generated: no reasoning exists on the labelled pool
    assert all(not c.strip() for c in data["test"]["cot_content"])
    assert all(c.strip() for c in data["validation"]["cot_content"])


@pytest.mark.live
def test_the_shipped_subtasks_own_their_demonstrations():
    """The property the flat task lost, and the reason for this whole design: lm_eval draws the
    five demonstrations itself, from that category's own validation rows, deterministically. Only
    then does `fewshot_as_multiturn` have anything to turn into turns."""
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    for name in (mmlu_pro.SUBTASKS[0], mmlu_pro.SUBTASKS[-1]):
        task = manager.load_task_or_group([name])[name]
        assert task.get_config("num_fewshot") == mmlu_pro.EVAL_SHOTS
        assert task.get_config("fewshot_split") == "validation"
        assert task.get_config("output_type") == "generate_until"
        assert mmlu_pro.ANSWER_PATTERN == \
            task.get_config("filter_list")[0]["filter"][0]["regex_pattern"]
        assert mmlu_pro.FILTER_NAME == task.get_config("filter_list")[0]["name"]


@pytest.mark.live
def test_get_dataset_carves_the_pool_into_disjoint_fingerprinted_splits():
    from pproute.tasks import get_dataset, resolve_task_names, split_fingerprint

    names = resolve_task_names(mmlu_pro.TASK_NAME)
    assert len(names) == 14

    out = get_dataset(mmlu_pro.TASK_NAME, random_seed=0)
    ids, prints = [], []
    for task_dict in (out[3], out[4], out[5]):
        split_ids = set()
        for name in names:
            task = task_dict[name]
            split_ids |= {mmlu_pro.example_id(d) for d in task.eval_docs}
            prints.append(split_fingerprint(task))
        ids.append(split_ids)

    assert [len(s) for s in ids] == [mmlu_pro.SFT_SIZE, mmlu_pro.PREDICTOR_TRAIN_SIZE, 3032]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(set(prints)) == len(prints), "two splits or subtasks share a fingerprint"


@pytest.mark.live
def test_the_70_demonstrations_are_never_carved_into_a_split():
    """The demonstrations are the `validation` split and the carve comes out of `test`, so they are
    disjoint by construction -- asserted rather than trusted, because a demonstration that also
    appeared as an SFT target or a test question would leak its own answer into the prompt."""
    from datasets import load_dataset

    from pproute.tasks import get_dataset, resolve_task_names

    demos = {mmlu_pro.example_id(d)
             for d in load_dataset(mmlu_pro.DATASET_PATH, split="validation")}
    assert len(demos) == 70

    out = get_dataset(mmlu_pro.TASK_NAME, random_seed=0)
    carved = set()
    for task_dict in (out[3], out[4], out[5]):
        for name in resolve_task_names(mmlu_pro.TASK_NAME):
            carved |= {mmlu_pro.example_id(d) for d in task_dict[name].eval_docs}
    assert len(carved) == 12032
    assert not (demos & carved)
