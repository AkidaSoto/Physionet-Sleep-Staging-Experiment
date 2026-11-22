"""Unit tests for Experiment 80 cue extraction, F3/F4 guard, and 5-concept layouts."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_staging_bigru_concept as concept_run  # noqa: E402
from physionet_sleep.algorithms.eeg_tracks import FS  # noqa: E402
from physionet_sleep.experiments.concept_bigru_staging import ConceptBiGRUStagingNet  # noqa: E402
from physionet_sleep.experiments.cues_bigru_staging import (  # noqa: E402
    CUE_COLUMNS,
    compute_guarded_eog_burst,
    extract_missing_cues_for_subject,
    get_variant_layout,
)


def test_guarded_eog_burst_suppresses_frontal_slow_wave_leakage() -> None:
    n_sec = 40
    t = np.arange(n_sec * FS) / FS
    rng = np.random.default_rng(7)
    eog = rng.normal(0, 5.0, size=n_sec * FS).astype(np.float32)
    f3 = rng.normal(0, 5.0, size=n_sec * FS).astype(np.float32)
    f4 = rng.normal(0, 5.0, size=n_sec * FS).astype(np.float32)

    # Seconds 10..12: genuine rapid 3 Hz eye movement with quiet frontal leads
    s1 = slice(10 * FS, 13 * FS)
    eog[s1] += 60.0 * np.sin(2 * np.pi * 3.0 * t[s1])

    # Seconds 25..27: large 3 Hz/1 Hz ocular-frontal artifact + strong 1.2 Hz frontal slow wave
    s2 = slice(25 * FS, 28 * FS)
    eog[s2] += 60.0 * np.sin(2 * np.pi * 3.0 * t[s2])
    f3[s2] += 80.0 * np.sin(2 * np.pi * 1.2 * t[s2])
    f4[s2] += 80.0 * np.sin(2 * np.pi * 1.2 * t[s2])

    # Second 35 contains NaN -> must stay NaN at second 35
    eog[35 * FS + 10] = np.nan

    burst = compute_guarded_eog_burst(eog, f3, f4, fs=FS)
    assert burst.shape == (n_sec,)
    assert burst[10:13].mean() > 0.6
    assert burst[25:28].mean() == 0.0
    assert np.isnan(burst[35])


def test_variant_layouts_and_five_concept_isolation() -> None:
    expected = {
        "delta": (4, 91, (16, 16, 16, 16)),
        "rem_emg": (4, 96, (16, 16, 16, 16)),
        "autonomic": (5, 91, (16, 16, 16, 16, 8)),
        "rem_auto": (5, 106, (16, 16, 16, 16, 8)),
        "all": (5, 116, (16, 16, 16, 16, 8)),
    }
    for name, (n_concepts, dim, widths) in expected.items():
        layout = get_variant_layout(name)
        assert len(layout["concept_names"]) == n_concepts
        assert layout["input_dim"] == dim
        assert layout["concept_widths"] == widths
        assert len(set(layout["feature_names"])) == dim

    layout_all = get_variant_layout("all")
    torch.manual_seed(11)
    model = ConceptBiGRUStagingNet(
        concept_dims=layout_all["concept_dims"],
        concept_names=layout_all["concept_names"],
        width_per_concept=layout_all["concept_widths"],
        dropout=0.0,
    )
    for head in (model.past_classifier, model.future_classifier):
        torch.nn.init.normal_(head[-1].weight, std=0.2)
    torch.nn.init.normal_(model.gates.weight, std=0.2)
    concept_run.verify_concept_model(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 80_000

    # Verify autonomic slice affects only concept 4 (autonomic)
    model.eval()
    x = torch.randn(1, 10, layout_all["input_dim"])
    rec = torch.ones((1, 10), dtype=torch.bool)
    x_auto = x.clone()
    x_auto[:, :, layout_all["concept_slices"]["autonomic"]] += 3.0
    with torch.inference_mode():
        t1 = model(x, rec, return_components=True)
        t2 = model(x_auto, rec, return_components=True)
    torch.testing.assert_close(t1.concept_logits[:, :, :4], t2.concept_logits[:, :, :4], rtol=0, atol=1e-6)
    assert (t2.concept_logits[:, :, 4] - t1.concept_logits[:, :, 4]).abs().max() > 1e-3


def test_extract_missing_cues_enforces_manifest() -> None:
    with pytest.raises(ValueError, match="outside permitted manifest"):
        extract_missing_cues_for_subject(
            "forbidden-subject",
            REPO / "data/raw/cinc2018",
            None,
            permitted_subjects={"tr03-0005"},
        )
    assert len(CUE_COLUMNS) == 7
