from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from physionet_sleep.analysis.epoch_windows import WindowEncoder
from physionet_sleep.experiments.staging_recipe_features import ANCHORS, recipe_key, sample_recipe, validate_recipe


def test_recipe_fingerprint_ignores_key_order_and_rejects_inactive_options():
    anchor = ANCHORS["sigma"]
    assert recipe_key(anchor) == recipe_key(dict(reversed(list(anchor.items()))))
    with pytest.raises(ValueError, match="Inactive"):
        validate_recipe({**anchor, "reference_seconds": 180})


def test_sampling_only_enables_the_selected_reference_parameters():
    def prefer_local(name, options):
        return "local" if name == "sigma.reference" else options[0]

    recipe = sample_recipe("sigma", prefer_local)
    validate_recipe(recipe)
    assert recipe["reference_seconds"] == 60
    assert "threshold_mad" not in recipe


def test_extended_context_keeps_fine_core_and_pads_beyond_recording(monkeypatch):
    pytest.importorskip("optuna")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_staging_recipe_search import window_columns, window_matrix

    # Keep the values in the encoder's clipping range, with one value per epoch.
    a = np.repeat(np.arange(1, 7, dtype=np.float32), 30)[:, None]
    b = np.repeat(np.arange(6, 0, -1, dtype=np.float32), 30)[:, None]
    dense = {"first": a, "second": b}
    frame = pd.DataFrame({"subject_id": ["first", "second"], "epoch_index": [0, 5]})
    encoder = WindowEncoder(["signal"], context=2, mean=np.array([0.]), sd=np.array([1.]))
    core = window_matrix(encoder, 2, dense, frame)
    expanded = window_matrix(encoder, 4, dense, frame)

    np.testing.assert_array_equal(expanded[:, :25], core)
    np.testing.assert_array_equal(expanded[:, 25:], [[0, 0, 4, 5], [5, 4, 0, 0]])
    assert len(window_columns(encoder, 4)) == expanded.shape[1]
