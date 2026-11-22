import numpy as np
import pandas as pd
import pytest

from physionet_sleep.experiments.cached_staging_tuning import (
    CACHE_COLUMNS,
    SIGMA_CANDIDATES,
    THETA_CANDIDATES,
    cached_staging_candidates,
)


def make_caches():
    seconds = np.arange(307)
    tracks = pd.DataFrame({"second": seconds, "c_rel_theta": seconds / 1000})
    quality = pd.DataFrame(0.0, index=seconds, columns=CACHE_COLUMNS)
    quality["second"] = seconds
    quality[["O1_total", "C3_total", "C4_total"]] = 10.0
    quality["O2_total"] = 30.0
    quality[["O1_theta", "O2_theta"]] = [1.0, 15.0]
    quality[["C3_sigma", "C4_sigma"]] = 1.0
    return tracks, quality


def test_theta_candidates_match_six_second_grid_and_preserve_partial_slots():
    tracks, quality = make_caches()

    result = cached_staging_candidates(tracks, quality)

    assert result.index.equals(pd.RangeIndex(52, name="slot"))
    assert tuple(result.columns) == THETA_CANDIDATES + SIGMA_CANDIDATES
    np.testing.assert_allclose(result["c_theta_6s"].iloc[0], tracks["c_rel_theta"].iloc[:6].mean())
    np.testing.assert_allclose(result["c_theta_30s"].iloc[:5], tracks["c_rel_theta"].iloc[:30].mean())
    np.testing.assert_allclose(result["c_theta_6s"].iloc[-1], tracks["c_rel_theta"].iloc[-1])
    np.testing.assert_allclose(result[["o_theta_6s", "o_theta_30s"]], 16.0 / 40.0)


def test_sigma_candidates_count_elevated_bins_and_exclude_unusable_lead():
    tracks, quality = make_caches()
    quality.loc[[5, 10, 11, 35, 40, 41], "C3_sigma"] = 8.0
    quality.loc[[15, 45], "C4_sigma"] = 8.0
    quality.loc[:29, "C4_flat"] = 1.0
    quality.loc[60:89, ["C3_flat", "C4_flat"]] = 1.0

    result = cached_staging_candidates(tracks, quality)

    # The first epoch has only C3 available. In the second epoch a single
    # elevated C4 bin lowers the bilateral mean and fails persistence.
    np.testing.assert_allclose(result["sigma_1mad_1s_mean"].iloc[:10], np.repeat([3.0, 2.0], 5))
    np.testing.assert_allclose(result["sigma_1mad_2s_mean"].iloc[:10], np.repeat([2.0, 1.0], 5))
    np.testing.assert_allclose(result["sigma_1mad_2s_max"].iloc[:10], 2.0)
    assert result.loc[10:14, list(SIGMA_CANDIDATES)].isna().all().all()


def test_candidate_extraction_rejects_shifted_quality_grid():
    tracks, quality = make_caches()
    quality["second"] += 1

    with pytest.raises(ValueError, match="share the same 1-second grid"):
        cached_staging_candidates(tracks, quality)
