"""The per-prompt correctness predictor g: a LoRA-fine-tuned embedding model with a
last-token-pooled linear head. Active task types:

  - mc:         f_Z comes from the option confidence.
  - consistency: f_Z comes from the sampled-majority share.

In both cases ``g = sigmoid(a + b*logit(f_Z) + head(...))``. With ``(a,b)=(0,1)`` and a
zero-initialized head, step 0 is exactly the development-fitted Platt anchor.

Trained with BCE(Z) on the eval split, evaluated on the test split. Feature/label builders
read the score files via pproute.scores.
"""
import os
import tempfile

import numpy as np
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

from pproute import consistency, models, predictor_protocol, scores
from pproute.provenance import require_pinned_revision


# --------------------------------------------------------------- feature / label builders
def _stripper(scored_tokenizer):
    """A callable removing `scored_tokenizer`'s chat wrapper from a request context, or None.

    `scored_tokenizer` is the tokenizer of the model that was *scored*, not the embedding backbone.
    An instruct checkpoint's prompt arrives wrapped in its own chat markup, which is identical for
    every example of that model and therefore predicts nothing -- while making `g`'s input a
    different shape than an untemplated model's. Passing None keeps the raw text, which is what a
    base checkpoint already has.

    Handed to pproute.scores rather than applied to the assembled question, because under the
    Winograd (multiple_input) layout the wrapper sits inside the per-option contexts; see
    scores.mc_examples.
    """
    if scored_tokenizer is None:
        return None
    return lambda ctx: models.strip_chat_scaffolding(ctx, scored_tokenizer)


def load_scored_tokenizer(provenance):
    """Load the tokenizer that produced an endpoint's score files."""
    source = provenance["dir_checkpoint"] or provenance["base_model_name"]
    kwargs = models.tokenizer_load_kwargs(provenance["base_model_name"])
    if not provenance["dir_checkpoint"]:
        kwargs["revision"] = provenance["revision"]
    return AutoTokenizer.from_pretrained(source, **kwargs)


def build_mc(scores_path, scored_tokenizer=None):
    """Build target-only MC verifier inputs, labels, logits, losses, and IDs."""
    texts, Z, lfz, L, ids = [], [], [], [], []
    for e in scores.mc_examples(scores_path, strip_ctx=_stripper(scored_tokenizer)):
        conts = [c.strip() for c in e["conts"]]
        q = e["question"].strip()
        labels = [chr(ord("A") + index) for index in range(len(conts))]
        choices = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, conts))
        dist = " ".join(
            f"{label}={float(probability):.6f}"
            for label, probability in zip(labels, e["pr"])
        )
        proposed = f"{labels[e['pred']]}. {conts[e['pred']]}"
        text = (
            "Instruct: Predict whether the proposed answer to this multiple-choice question "
            "is correct. Use the target question, answer choices, and the model's answer "
            "distribution as evidence.\n"
            "Query:\n" + q + "\n\nAnswer choices:\n" + choices
            + "\n\nModel probabilities: " + dist
            + "\nProposed answer: " + proposed
        )
        texts.append(text)
        Z.append(e["Z"]); lfz.append(e["lfz"]); L.append(e["L"])
        ids.append(e["id"])
    return (texts, np.array(Z, np.float32), np.array(lfz, np.float32), np.array(L), ids)


def _target_question(prompt):
    """Remove few-shot demonstrations and the target's generation cue from a task prompt."""
    answer = prompt.rfind("\nAnswer:")
    if answer < 0:
        raise ValueError("target-only consistency input requires a final Answer cue")
    prefix = prompt[:answer]
    # Active GSM8K and MMLU-Pro prompts render ``Question: {text}`` on one line.  Prefer the
    # benchmark block delimiter so a literal "Question:" inside the target body cannot become a
    # false boundary; retain a start-of-string fallback for synthetic/zero-shot inputs.
    start = prefix.rfind("\n\nQuestion:")
    if start >= 0:
        start += 2
    elif prefix.startswith("Question:"):
        start = 0
    else:
        start = prefix.rfind("\nQuestion:")
        if start >= 0:
            start += 1
    if start < 0:
        raise ValueError("target-only consistency input requires a Question block")
    return prefix[start:].strip()


def build_consistency(payload):
    """Consistency predictor input: question + majority trace + the sampled
    answer distribution + the majority answer, from a validated confidence payload.

    Solved demonstrations are removed and Qwen's instruction-aware embedding format states the
    verifier task explicitly. The input-length preflight refuses truncation. Returns
    ``{split: (texts, Z, shares, ids)}`` with raw majority shares for calibration downstream.
    """
    out = {}
    for split in ("eval", "test"):
        texts, Z, shares, ids = [], [], [], []
        for row in payload["rows"]:
            if row["split"] != split:
                continue
            first_seen = row.get("first_seen", {})
            ordered = sorted(
                row["counts"].items(),
                key=lambda kv: (-kv[1], first_seen.get(kv[0], payload["meta"]["k"]), kv[0]),
            )
            trace = (row.get("majority_trace") or "").strip()
            if payload["meta"].get("task") == "gsm8k":
                # Limit the open-ended numeric distribution to five entries. Using the seeded
                # first-occurrence order for ties keeps the proposed answer in this list.
                dist = " ".join(
                    f"{answer}={count / payload['meta']['k']:.3f}"
                    for answer, count in ordered[:5]
                )
                answer = row["majority"] if row["majority"] is not None else "<none>"
                task = (
                    "Predict whether the proposed answer to this math problem is correct. "
                    "Use the target problem, proposed solution, and sampled-answer "
                    "frequencies as evidence."
                )
                evidence = "\n\nSampled answers: " + dist + "\nProposed answer: " + answer
            else:
                valid = [(answer, count) for answer, count in ordered
                         if answer != consistency.FAILURE]
                dist = " ".join(
                    f"{answer}={count}/{payload['meta']['k']}" for answer, count in valid
                ) or "<none>"
                answer = ("<none>" if row["majority"] is None else
                          f"{row['majority']}. {row['majority_option']}")
                task = (
                    "Predict whether the proposed answer to this multiple-choice question is "
                    "correct. Use the target question, answer choices, proposed solution, and "
                    "sampled-answer frequencies as evidence."
                )
                evidence = (
                    f"\n\nExtraction failures: {row['n_fail']}/{payload['meta']['k']}"
                    + "\nSampled answers: " + dist
                    + "\nProposed answer: " + answer
                )
            text = (
                "Instruct: " + task
                + "\nQuery:\n" + _target_question(row["predictor_text"])
                + ("\n\nProposed solution:\n" + trace if trace else "")
                + evidence)
            texts.append(text)
            Z.append(float(row["z"])); shares.append(float(row["share"])); ids.append(row["id"])
        out[split] = (texts, np.array(Z, np.float32), np.array(shares, np.float32), ids)
    return out


def build_endpoint_data(task_type, scores_dir, split, scored_tokenizer, *, conf_path=""):
    """One endpoint split in the common predictor representation."""
    if split not in ("eval", "test"):
        raise ValueError("split must be eval or test")
    if task_type == "mc":
        name = "eval_scores.jbl" if split == "eval" else "test_scores.jbl"
        text, z, raw_lfz, y_loss, ids = build_mc(
            os.path.join(scores_dir, name), scored_tokenizer
        )
        return {"text": text, "Z": z, "raw_lfz": raw_lfz, "L": y_loss, "ids": ids}
    if task_type != "consistency":
        raise ValueError("task_type must be mc or consistency")
    if not conf_path:
        raise ValueError("consistency endpoint data requires conf_path")
    payload = consistency.read_conf_scores(conf_path, expected_splits=(split,))
    text, z, _, ids = build_consistency(payload)[split]
    rows = [row for row in payload["rows"] if row["split"] == split]
    estimates = [consistency.bucket_estimates(row) for row in rows]
    raw_lfz = np.asarray([consistency.anchor_logit(fz) for fz, _ in estimates], dtype=np.float32)
    y_loss = np.asarray([loss for _, loss in estimates], dtype=np.float64)
    return {"text": text, "Z": z, "raw_lfz": raw_lfz, "L": y_loss, "ids": ids}


# --------------------------------------------------------------- model
class Verifier(nn.Module):
    def __init__(self, base, lora_r=16):
        super().__init__()
        self.enc = get_peft_model(base, LoraConfig(
            r=lora_r, lora_alpha=predictor_protocol.LORA_ALPHA_FACTOR * lora_r,
            lora_dropout=predictor_protocol.LORA_DROPOUT,
            target_modules=["q_proj", "v_proj"], task_type="FEATURE_EXTRACTION"))
        self.head = nn.Linear(base.config.hidden_size, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.a = nn.Parameter(torch.zeros(1)); self.b = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attn, logit_fz=None):
        h = self.enc(input_ids=input_ids, attention_mask=attn).last_hidden_state
        s = self.head(h[:, -1, :].float()).squeeze(-1)   # left-padded -> last token
        if logit_fz is None:
            raise ValueError("verifier requires the f_Z anchor")
        return self.a + self.b * logit_fz + s


def param_groups(model):
    """Optimizer groups: only the head is weight-decayed.

    (a, b) calibrate logit f_Z rather than acting as features. b starts at 1 because g == f_Z at
    initialisation, so decaying it toward 0 would pull the anchor apart -- its meaningful prior is
    1, not 0. The LoRA parameters are left undecayed as well, as they were before.

    """
    lora = [p for n, p in model.named_parameters()
            if p.requires_grad and "head" not in n and n not in ("a", "b")]
    groups = [{"params": lora, "weight_decay": 0.0},
              {"params": list(model.head.parameters()), "weight_decay": predictor_protocol.HEAD_WEIGHT_DECAY}]
    anchor = {"params": [model.a, model.b], "weight_decay": 0.0}
    groups.append(anchor)
    return groups


def load_embedding_backbone(emb_model, revision):
    """Load both halves of predictor ``g`` from the same pinned Hub revision."""
    require_pinned_revision(revision, "embedding revision")
    tok = AutoTokenizer.from_pretrained(
        emb_model,
        revision=revision,
        **models.tokenizer_load_kwargs(emb_model),
    )
    base = AutoModel.from_pretrained(
        emb_model,
        revision=revision,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return tok, base


def input_length_diagnostics(tokenizer, texts, max_len, batch_size=256):
    """Measure predictor inputs with the exact tokenizer and refuse silent truncation upstream.

    Tokenization is batched only to keep the preflight cheap; no padding or truncation is applied.
    The caller runs this independently on predictor-train and test so both maxima are recorded.
    """
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start:start + batch_size]), add_special_tokens=True,
            padding=False, truncation=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    over = sum(length > max_len for length in lengths)
    return {
        "n": len(lengths),
        "max_tokens": max(lengths, default=0),
        "over_max_len": over,
        "max_len": int(max_len),
    }


# --------------------------------------------------------------- training
def train_verifier(emb_model, tr_text, tr_Z, te_text, te_Z, *, emb_revision,
                   tr_lfz, te_lfz, lora_r=16, epochs=3, lr=1e-4, bs=16,
                   micro_bs=None, max_len=384, device="cuda", return_info=False,
                   fit_indices, holdout_indices, checkpoint_path=None,
                   checkpoint_metadata=None):
    """Fit on the frozen development partition and select on its fixed holdout.

    Return probabilities on te_text using the calibrated confidence logits supplied by the caller.

    return_info=True also returns a dict with the holdout loss at initialisation, the selected
    holdout loss, and which epoch was selected (0 = initialisation, i.e. g == f_Z was kept). That
    last field is the only direct evidence that the step-0 anchor did its job, so it is worth
    logging on real runs as well as in tests."""
    micro_bs = bs if micro_bs in (None, 0) else int(micro_bs)
    if bs <= 0 or micro_bs <= 0 or micro_bs > bs or bs % micro_bs:
        raise ValueError("bs must be positive and divisible by micro_bs: bs=%r micro_bs=%r"
                         % (bs, micro_bs))
    if max_len <= 0:
        raise ValueError("max_len must be positive: %r" % max_len)
    if len(tr_text) != len(tr_Z) or len(te_text) != len(te_Z):
        raise ValueError("text/label lengths differ: train %d/%d test %d/%d"
                         % (len(tr_text), len(tr_Z), len(te_text), len(te_Z)))
    if len(tr_lfz) != len(tr_text) or len(te_lfz) != len(te_text):
        raise ValueError("confidence/text lengths differ: train %d/%d test %d/%d"
                         % (len(tr_lfz), len(tr_text), len(te_lfz), len(te_text)))
    tri = np.asarray(fit_indices, dtype=np.int64)
    hoi = np.asarray(holdout_indices, dtype=np.int64)
    if tri.ndim != 1 or hoi.ndim != 1 or not len(tri) or not len(hoi):
        raise ValueError("fixed predictor partitions must be non-empty vectors")
    joined = np.concatenate((tri, hoi))
    if joined.min() < 0 or joined.max() >= len(tr_text) or \
            len(np.unique(joined)) != len(joined) or len(joined) != len(tr_text):
        raise ValueError("fixed predictor partitions must be disjoint and exhaustive")
    tok, base = load_embedding_backbone(emb_model, revision=emb_revision)
    tok.padding_side = "left"; tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tr_lengths = input_length_diagnostics(tok, tr_text, max_len)
    te_lengths = input_length_diagnostics(tok, te_text, max_len)
    if tr_lengths["over_max_len"] or te_lengths["over_max_len"]:
        raise ValueError(
            "predictor input exceeds max_len=%d: train max=%d (%d over), test max=%d (%d over); "
            "truncation is disabled because it can remove the ICL examples or target"
            % (max_len, tr_lengths["max_tokens"], tr_lengths["over_max_len"],
               te_lengths["max_tokens"], te_lengths["over_max_len"])
        )
    if micro_bs < bs and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = False
    model = Verifier(base, lora_r).to(device)

    def enc(texts):
        return tok(texts, return_tensors="pt", padding=True, truncation=False).to(device)

    def predict(texts, lfz, order):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(order), micro_bs):
                b = order[i:i + micro_bs]
                e = enc([texts[k] for k in b])
                lz = torch.tensor(lfz[b], device=device)
                out.append(torch.sigmoid(model(e.input_ids, e.attention_mask, lz)).float().cpu().numpy())
        return np.concatenate(out) if len(order) else np.array([])

    tr_lfz = np.asarray(tr_lfz, dtype=np.float32)
    te_lfz = np.asarray(te_lfz, dtype=np.float32)
    if not np.all(np.isfinite(tr_lfz)) or not np.all(np.isfinite(te_lfz)):
        raise ValueError("confidence logits must be finite")
    opt = torch.optim.AdamW(param_groups(model), lr=lr)
    lossfn = nn.BCEWithLogitsLoss()

    def snapshot():
        """Copy only the trainable tensors. `state_dict()` holds live CUDA tensors for the frozen
        backbone too, so deep-copying it would duplicate the whole embedding model in VRAM (~8 GB
        for a 4B backbone) to preserve a few megabytes of LoRA weights."""
        return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    # Step 0 is a candidate for early stopping. In residual mode g == f_Z at initialisation, so
    # without this the model's own confidence is merely *feasible*, never *selected*: a run that
    # ends up worse than f_Z is still returned and the edge comes out negative for no reason worth
    # reporting. Evaluating here makes "no worse than f_Z on the holdout" an actual guarantee.
    # (It is a holdout guarantee, not a test one -- a small negative edge on test remains possible.)
    best = scores.bce(tr_Z[hoi], predict(tr_text, tr_lfz, hoi))
    hl_init, sel_epoch = best, 0
    best_state, bad = snapshot(), 0
    for ep in range(1, epochs + 1):
        model.train(); perm = np.random.permutation(tri)
        for i in range(0, len(perm), bs):
            logical_batch = perm[i:i + bs]
            opt.zero_grad()
            for j in range(0, len(logical_batch), micro_bs):
                b = logical_batch[j:j + micro_bs]
                e = enc([tr_text[k] for k in b]); y = torch.tensor(tr_Z[b], device=device)
                lz = torch.tensor(tr_lfz[b], device=device)
                loss = lossfn(model(e.input_ids, e.attention_mask, lz), y)
                (loss * (len(b) / len(logical_batch))).backward()
            opt.step()
        hl = scores.bce(tr_Z[hoi], predict(tr_text, tr_lfz, hoi))
        if hl < best - 1e-4:
            best, bad, best_state = hl, 0, snapshot()
            sel_epoch = ep
        else:
            bad += 1
            if bad >= 2:
                break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)   # trainable tensors only, see snapshot()
    if checkpoint_path:
        state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
        payload = {
            "schema": predictor_protocol.VERIFIER_CHECKPOINT_SCHEMA,
            "emb_model": emb_model,
            "emb_revision": emb_revision,
            "lora_r": int(lora_r),
            "max_len": int(max_len),
            "state_dict": state,
            "metadata": {
                **dict(checkpoint_metadata or {}),
                "selected_epoch": int(sel_epoch),
                "holdout_init": float(hl_init),
                "holdout_best": float(best),
            },
        }
        directory = os.path.dirname(checkpoint_path)
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=os.path.basename(checkpoint_path) + ".", suffix=".partial", dir=directory
        )
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    p_te = predict(te_text, te_lfz, np.arange(len(te_text)))
    if return_info:
        return p_te, {
            "holdout_init": hl_init, "holdout_best": best, "selected_epoch": sel_epoch,
            "effective_batch_size": int(bs), "micro_batch_size": int(micro_bs),
            "gradient_accumulation_steps": int(bs // micro_bs),
            "input_max_len": int(max_len),
            "input_train_max_tokens": tr_lengths["max_tokens"],
            "input_test_max_tokens": te_lengths["max_tokens"],
            "input_train_over_max_len": tr_lengths["over_max_len"],
            "input_test_over_max_len": te_lengths["over_max_len"],
            "anchor": {"a": float(model.a), "b": float(model.b)},   # what the fit made of (a, b)
        }
    return p_te


def predict_verifier_checkpoint(emb_model, texts, lfz, checkpoint_path, *, emb_revision,
                                micro_bs=0, max_len=None, device="cuda"):
    """Restore a validation-selected verifier and predict one untouched split."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema") != predictor_protocol.VERIFIER_CHECKPOINT_SCHEMA:
        raise ValueError("unknown verifier checkpoint schema %r" % payload.get("schema"))
    expected = {"emb_model": emb_model, "emb_revision": emb_revision}
    bad = {key: (payload.get(key), value) for key, value in expected.items()
           if payload.get(key) != value}
    if bad:
        raise ValueError("verifier checkpoint identity mismatch: %s" % bad)
    max_len = int(payload["max_len"] if max_len is None else max_len)
    if max_len != int(payload["max_len"]):
        raise ValueError("verifier checkpoint max_len %d != requested %d"
                         % (payload["max_len"], max_len))
    micro_bs = 1 if micro_bs in (None, 0) else int(micro_bs)
    if micro_bs <= 0:
        raise ValueError("micro_bs must be positive")
    if lfz is None or len(lfz) != len(texts):
        raise ValueError("confidence/text lengths differ: %d/%d" %
                         (0 if lfz is None else len(lfz), len(texts)))

    tok, base = load_embedding_backbone(emb_model, revision=emb_revision)
    tok.padding_side = "left"; tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lengths = input_length_diagnostics(tok, texts, max_len)
    if lengths["over_max_len"]:
        raise ValueError(
            "predictor test input exceeds max_len=%d: max=%d (%d over)"
            % (max_len, lengths["max_tokens"], lengths["over_max_len"])
        )
    model = Verifier(base, int(payload["lora_r"])).to(device)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    saved = set(payload["state_dict"])
    if saved != trainable:
        raise ValueError("verifier checkpoint trainable keys differ: missing=%s extra=%s"
                         % (sorted(trainable - saved), sorted(saved - trainable)))
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(texts), micro_bs):
            batch_text = list(texts[start:start + micro_bs])
            encoded = tok(batch_text, return_tensors="pt", padding=True,
                          truncation=False).to(device)
            batch_lfz = torch.tensor(
                np.asarray(lfz[start:start + micro_bs]), device=device
            )
            logits = model(encoded.input_ids, encoded.attention_mask, batch_lfz)
            probabilities.extend(torch.sigmoid(logits).float().cpu().tolist())
    return np.asarray(probabilities, dtype=np.float64), {
        "checkpoint_metadata": payload.get("metadata", {}),
        "input_max_len": max_len,
        "input_test_max_tokens": lengths["max_tokens"],
        "input_test_over_max_len": lengths["over_max_len"],
        "micro_batch_size": micro_bs,
    }
