"""Unit tests for Experiment 79 (ConceptBiGRUStagingNet, EMA, smoothing, conformal)."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_staging_bigru_concept as concept_run  # noqa: E402
import run_staging_feature_context as sequence  # noqa: E402
from physionet_sleep.experiments.concept_bigru_staging import (  # noqa: E402
    CONCEPT_NAMES,
    CONCEPT_SLICES,
    ConceptBiGRUStagingNet,
    ConceptTerms,
    DEFAULT_CONCEPT_DIMS,
    EMAWeights,
    FEATURE_NAMES_79,
    TOTAL_CONCEPT_DIM,
    conformal_prediction_sets,
    evaluate_conformal_oof,
    fit_mondrian_conformal_thresholds,
    fit_split_conformal_threshold,
    smoothed_stage_loss,
)


def test_layout_dimensions_and_parameter_budget() -> None:
    assert TOTAL_CONCEPT_DIM == 81
    assert DEFAULT_CONCEPT_DIMS == (15, 14, 32, 20)
    assert len(FEATURE_NAMES_79) == TOTAL_CONCEPT_DIM
    assert len(set(FEATURE_NAMES_79)) == TOTAL_CONCEPT_DIM

    model = ConceptBiGRUStagingNet()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 70_000, f"Expected <= 70k params, got {n_params}"


def test_concept_additivity_isolation_and_init_identity() -> None:
    torch.manual_seed(7)
    model = ConceptBiGRUStagingNet(dropout=0.0).eval()
    x = torch.randn(2, 11, TOTAL_CONCEPT_DIM)
    recorded = torch.ones(2, 11, dtype=torch.bool)
    recorded[1, 7:] = False
    x[1, 7:] = float("nan")

    with torch.inference_mode():
        terms = model(x, recorded, return_components=True)

    assert isinstance(terms, ConceptTerms)
    assert terms.logits.shape == (2, 11, 5)
    assert terms.concept_logits.shape == (2, 11, 4, 5)
    assert torch.isfinite(terms.logits).all()

    # Zero-init past/future heads make initial logits equal local logits.
    torch.testing.assert_close(terms.logits, terms.local, rtol=0, atol=1e-6)
    expected_local = (
        terms.concept_logits.sum(dim=-2) + terms.local_bias.view(1, 1, 5)
    ).masked_fill(~recorded.unsqueeze(-1), 0)
    torch.testing.assert_close(terms.local, expected_local, rtol=0, atol=1e-6)
    assert (terms.logits[1, 7:] == 0).all()

    # Strict additive concept isolation: perturbing each slice alters only its own logit term.
    with torch.inference_mode():
        for idx, name in enumerate(CONCEPT_NAMES):
            x_p = x.clone()
            x_p[:, :, CONCEPT_SLICES[name]] += 2.5
            p_terms = model(x_p, recorded, return_components=True)
            for other_idx in range(len(CONCEPT_NAMES)):
                if other_idx == idx:
                    assert (
                        p_terms.concept_logits[:, :7, other_idx]
                        - terms.concept_logits[:, :7, other_idx]
                    ).abs().max() > 1e-3
                else:
                    torch.testing.assert_close(
                        p_terms.concept_logits[:, :, other_idx],
                        terms.concept_logits[:, :, other_idx],
                        rtol=0,
                        atol=1e-6,
                    )


def test_causality_anti_causality_and_padding_invariance() -> None:
    torch.manual_seed(13)
    model = ConceptBiGRUStagingNet(width_per_concept=8, dropout=0.0)
    for head in (model.past_classifier, model.future_classifier):
        torch.nn.init.normal_(head[-1].weight, std=0.2)
    torch.nn.init.normal_(model.gates.weight, std=0.2)
    concept_run.verify_concept_model(model)


def test_smoothed_stage_loss_matches_weighted_ce_when_zero() -> None:
    torch.manual_seed(5)
    logits = torch.randn(3, 10, 5)
    targets = torch.randint(0, 5, (3, 10))
    targets[2, 6:] = -1
    weights = torch.tensor([1.0, 2.0, 0.8, 1.5, 1.2])

    valid = targets >= 0
    expected = F.cross_entropy(logits[valid], targets[valid], weight=weights)
    actual = smoothed_stage_loss(logits, targets, weights, label_smoothing=0.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    smoothed = smoothed_stage_loss(logits, targets, weights, label_smoothing=0.05)
    assert torch.isfinite(smoothed) and smoothed > 0


def test_ema_weights_update_and_restore() -> None:
    torch.manual_seed(9)
    model = ConceptBiGRUStagingNet(width_per_concept=8)
    ema = EMAWeights(model, decay=0.9)
    orig_w = model.local_bias.detach().clone()

    with torch.no_grad():
        model.local_bias.add_(1.0)
    ema.update(model)
    assert not torch.allclose(ema.shadow["local_bias"], model.local_bias)
    assert not torch.allclose(ema.shadow["local_bias"], orig_w)

    backup = ema.copy_to(model)
    torch.testing.assert_close(model.local_bias, ema.shadow["local_bias"])
    model.load_state_dict(backup)
    torch.testing.assert_close(model.local_bias, orig_w + 1.0)


def test_conformal_split_and_mondrian_coverage() -> None:
    rng = np.random.default_rng(42)
    n = 5000
    folds = np.repeat(np.arange(5), n // 5)
    # Class 1 is intentionally noisier so marginal vs Mondrian coverage differs cleanly.
    y = rng.choice(5, size=n, p=[0.25, 0.12, 0.35, 0.13, 0.15])
    probs = np.zeros((n, 5), dtype=np.float64)
    for i, label in enumerate(y):
        conc = np.ones(5) * 0.4
        conc[label] = 2.0 if label == 1 else 6.0
        probs[i] = rng.dirichlet(conc)

    thr = fit_split_conformal_threshold(probs[:2500], y[:2500], alpha=0.10)
    sets, _ = conformal_prediction_sets(probs[2500:], thr)
    cov = sets[np.arange(2500), y[2500:]].mean()
    assert cov >= 0.88

    m_thr = fit_mondrian_conformal_thresholds(probs[:2500], y[:2500], alpha=0.10)
    m_sets, _ = conformal_prediction_sets(probs[2500:], m_thr)
    for c in range(5):
        mask = y[2500:] == c
        assert m_sets[mask, c].mean() >= 0.86

    is_trans = np.arange(n) % 10 == 0
    summary = evaluate_conformal_oof(probs, y, folds, is_trans, alphas=(0.10, 0.05))
    assert summary["coverage_90"]["marginal"]["empirical_coverage"] >= 0.885
    assert summary["coverage_95"]["mondrian"]["per_stage"]["N1"]["coverage"] >= 0.92


def test_fit_concept_bigru_exact_midpass_resume() -> None:
    rng = np.random.default_rng(7)
    recs = {}
    fit_subs = []
    val_subs = []
    for i in range(6):
        s = f"sub-{i:02d}"
        x = torch.from_numpy(rng.normal(size=(24, TOTAL_CONCEPT_DIM)).astype(np.float32))
        y = torch.from_numpy(rng.integers(0, 5, size=24, dtype=np.int64))
        recs[s] = sequence.Recording(x, y)
        if i < 4:
            fit_subs.append(s)
        else:
            val_subs.append(s)

    cfg = concept_run.TrainConfig(max_passes=2, min_passes=1, batch_size=2, patience=5)
    with TemporaryDirectory() as tmp:
        d1 = Path(tmp) / "full"
        d2 = Path(tmp) / "resumed"
        m1, _ = concept_run.fit_concept_bigru(
            recs, fit_subs, val_subs, 7, d1, d1 / "PAUSE", 1e12, cfg
        )
        assert m1 is not None
        p_mid, _ = concept_run.fit_concept_bigru(
            recs,
            fit_subs,
            val_subs,
            7,
            d2,
            d2 / "PAUSE",
            1e12,
            cfg,
            smoke_pause_updates=1,
        )
        assert p_mid is None
        m2, _ = concept_run.fit_concept_bigru(
            recs, fit_subs, val_subs, 7, d2, d2 / "PAUSE", 1e12, cfg
        )
        assert m2 is not None
        for k, v in m1.state_dict().items():
            torch.testing.assert_close(v, m2.state_dict()[k], rtol=0, atol=1e-6)
