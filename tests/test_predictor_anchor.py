"""The predictor must not manufacture an edge, and must fall back to f_Z when training hurts.

Run:
    pytest tests/test_predictor_anchor.py --slow    # downloads a tiny model, trains on CPU

WHAT IS AND IS NOT GUARANTEED
-----------------------------
`train_verifier` evaluates its early-stopping holdout at step 0, so the initialisation -- where
g == f_Z under the residual construction -- is a *selected* candidate rather than merely a feasible
one. That makes "no worse than f_Z" a guarantee **on the internal holdout**.

It is NOT a guarantee that g == f_Z exactly, nor that the edge is >= 0 on test:

  * (a, b) in sigma(a + b*logit f_Z + head) are trainable, so even with no signal in the text the
    model has two free calibration parameters that can fit sampling noise;
  * the acceptance rule is `holdout < best - 1e-4`, so any chance improvement above that tolerance
    is taken;
  * the holdout is 15% of the training split, and test is a disjoint sample.

So there are two things worth testing, and they are different:

  1. Behaviour -- on data where nothing is learnable, the edge must be ~0 in either direction. A
     large positive edge would mean the predictor sees the label through a path it should not.
  2. The anchor itself -- when training strictly degrades the holdout, `selected_epoch` must be 0
     and g must be exactly f_Z. Forced here with a divergent learning rate, which is the only
     reliable way to make training worse every epoch.

Train and test are disjoint samples throughout. Sharing them would let a higher-capacity backbone
memorise random labels and fail (1) for a reason unrelated to the property being tested.
"""
import numpy as np
import pytest

from pproute import predictor, scores

# Every test here downloads a tiny model and trains it: opt in with ``pytest --slow``.
pytestmark = pytest.mark.slow

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_REVISION = "4b10ebee6e13a2669155516652960f50984399fd"
TOL = 0.01          # nats
SIGNAL_RATIO = 10   # the control edge must exceed the noise edges by this factor

# NOTE on thresholds: the backbone here is *randomly initialised*, so it learns very little in 3
# epochs -- a planted, perfectly predictive label yields ~0.01 nats against 0.14-0.49 for a real
# backbone on real data. Absolute thresholds tuned for the real setting fail here for reasons
# unrelated to correctness, so the control is expressed as a ratio to the noise runs.


def _split(texts, Z, lfz):
    """Disjoint halves: (train, test)."""
    h = len(Z) // 2
    return (texts[:h], Z[:h], lfz[:h]), (texts[h:], Z[h:], lfz[h:])


def _fit(texts, Z, lfz, seed=0, lr=1e-4):
    """Train on the first half, predict the second. Returns (p_test, test_Z, test_fz, info)."""
    import torch
    (tr_t, tr_Z, tr_l), (te_t, te_Z, te_l) = _split(texts, Z, lfz)
    np.random.seed(seed); torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    order = np.random.default_rng(0).permutation(len(tr_t))
    cut = int(.85 * len(order))
    p_te, info = predictor.train_verifier(TINY, tr_t, tr_Z, te_t, te_Z,
                                          emb_revision=TINY_REVISION,
                                          tr_lfz=tr_l, te_lfz=te_l,
                                          epochs=3, bs=32, max_len=24, lr=lr, device=dev,
                                          return_info=True, fit_indices=order[:cut],
                                          holdout_indices=order[cut:])
    return p_te, te_Z, 1.0 / (1.0 + np.exp(-te_l)), info


def _report(name, ok, detail):
    """Keep slow-test diagnostics visible without using return values as test results."""
    print("  %s  %s: %s" % ("PASS" if ok else "FAIL", name, detail))
    assert ok, "%s: %s" % (name, detail)


@pytest.fixture(scope="module")
def edge_controls():
    """Fit the two null controls and the planted-signal control once each."""
    rng = np.random.RandomState(0); n = 800
    Z = rng.binomial(1, 0.5, n).astype(np.float32)
    fz = np.full(n, 0.5, np.float32)
    p_te, te_Z, te_fz, _ = _fit(["neutral filler %d" % i for i in range(n)], Z,
                                np.log(fz / (1 - fz)).astype(np.float32))
    noise = scores.bce(te_Z, te_fz) - scores.bce(te_Z, p_te)

    rng = np.random.RandomState(0); n = 800
    p = rng.uniform(0.05, 0.95, n).astype(np.float32)
    Z = rng.binomial(1, p).astype(np.float32)
    p_te, te_Z, te_fz, _ = _fit(["neutral filler %d" % i for i in range(n)], Z,
                                np.log(p / (1 - p)).astype(np.float32))
    calibrated = scores.bce(te_Z, te_fz) - scores.bce(te_Z, p_te)

    rng = np.random.RandomState(0); n = 800
    Z = rng.binomial(1, 0.5, n).astype(np.float32)
    fz = np.full(n, 0.5, np.float32)
    texts = ["the answer is correct" if z else "the answer is wrong" for z in Z]
    p_te, te_Z, te_fz, _ = _fit(
        texts, Z, np.log(fz / (1 - fz)).astype(np.float32)
    )
    signal = scores.bce(te_Z, te_fz) - scores.bce(te_Z, p_te)
    return {"noise": noise, "calibrated": calibrated, "signal": signal}


def test_no_edge_from_noise(edge_controls):
    """Labels independent of everything; f_Z is the base rate. No edge should appear."""
    edge = edge_controls["noise"]
    _report("unlearnable labels", abs(edge) < TOL,
            "edge = %+.6f (|edge| < %.2f)" % (edge, TOL))


def test_no_edge_over_calibrated_confidence(edge_controls):
    """f_Z already perfectly calibrated, text uninformative. Nothing is left for g to add."""
    edge = edge_controls["calibrated"]
    _report("calibrated f_Z, no signal in text", abs(edge) < TOL,
            "edge = %+.6f (|edge| < %.2f)" % (edge, TOL))


def test_signal_is_learnable(edge_controls):
    """When text carries the label, its edge must dominate both null controls."""
    signal = edge_controls["signal"]
    noise = max(abs(edge_controls["noise"]), abs(edge_controls["calibrated"]), 1e-9)
    _report("label present in text", signal > SIGNAL_RATIO * noise,
            "edge = %+.6f (>= %dx the largest noise edge %.6f)" %
            (signal, SIGNAL_RATIO, noise))


def test_anchor_holds_when_training_degrades():
    """The anchor itself: a divergent learning rate makes every epoch worse than initialisation,
    so step 0 must be selected and g must come back exactly equal to f_Z."""
    rng = np.random.RandomState(0); n = 800
    p = rng.uniform(0.05, 0.95, n).astype(np.float32)
    Z = rng.binomial(1, p).astype(np.float32)
    p_te, te_Z, te_fz, info = _fit(["neutral filler %d" % i for i in range(n)], Z,
                                   np.log(p / (1 - p)).astype(np.float32), lr=5.0)
    ok = info["selected_epoch"] == 0 and np.allclose(p_te, te_fz, atol=1e-4)
    _report("divergent lr -> initialisation kept", ok,
            "selected_epoch=%d, g==f_Z: %s, holdout %.6f -> %.6f"
            % (info["selected_epoch"], np.allclose(p_te, te_fz, atol=1e-4),
               info["holdout_init"], info["holdout_best"]))


def test_saved_candidate_checkpoint_roundtrips_exact_selected_epoch(tmp_path):
    """The seed is selected before test, so its validation-selected weights must be restorable."""
    import torch

    rng = np.random.RandomState(4)
    n = 64
    texts = ["candidate verifier example %d" % index for index in range(n)]
    labels = rng.binomial(1, 0.5, n).astype(np.float32)
    logits = np.zeros(n, dtype=np.float32)
    fit = np.arange(48)
    holdout = np.arange(48, n)
    checkpoint = tmp_path / "model.pt"
    np.random.seed(7); torch.manual_seed(7)

    expected, _ = predictor.train_verifier(
        TINY, texts, labels, [texts[index] for index in holdout], labels[holdout],
        emb_revision=TINY_REVISION, tr_lfz=logits, te_lfz=logits[holdout],
        epochs=1, bs=16, max_len=24, device="cpu", return_info=True,
        fit_indices=fit, holdout_indices=holdout, checkpoint_path=str(checkpoint),
        checkpoint_metadata={"candidate": "roundtrip"},
    )
    restored, info = predictor.predict_verifier_checkpoint(
        TINY, [texts[index] for index in holdout], logits[holdout], str(checkpoint),
        emb_revision=TINY_REVISION, micro_bs=16, max_len=24, device="cpu",
    )

    assert restored == pytest.approx(expected, abs=1e-7)
    assert info["checkpoint_metadata"]["candidate"] == "roundtrip"
