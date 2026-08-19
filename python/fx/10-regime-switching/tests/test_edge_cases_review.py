"""Edge-case and property tests added in the review pass (project 10).

Focus, per the regime-switching domain:

* single-regime (null) data — the model must not invent structure,
* label switching — economic labels must be invariant to state permutation
  and to the seed that produced them,
* peg-break style regime transitions — a sudden, persistent variance jump,
* tiny/empty samples, constant inputs, NaN/Inf rejection.
"""

import numpy as np
import pandas as pd
import pytest

from fx_regime.data.synthetic import (
    generate_null_gbm_panel,
    generate_roro_panel,
)
from fx_regime.detection import apply_hysteresis, label_states
from fx_regime.gmm import fit_gmm
from fx_regime.hmm import (
    HMMModel,
    expected_durations,
    filtered_probabilities,
    fit_hmm,
    hmm_bic,
    match_states,
    smoothed_probabilities,
    stationary_distribution,
    viterbi,
)


def _two_state_model(sig_lo=1.0, sig_hi=3.0, stay=0.95):
    return HMMModel(
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[stay, 1 - stay], [1 - stay, stay]]),
        means=np.array([[0.0], [0.0]]),
        covs=np.array([[[sig_lo**2]], [[sig_hi**2]]]),
    )


# ---------------------------------------------------------------------------
# Single-regime (null) data
# ---------------------------------------------------------------------------

def test_bic_prefers_one_state_on_single_regime_data():
    """The null GBM panel has no regimes: BIC must not reward k=2."""
    panel = generate_null_gbm_panel(n_periods=600, seed=3)
    X = panel.returns.to_numpy()
    b1 = hmm_bic(fit_hmm(X, 1, seed=0, n_init=1, max_iter=40), X)
    b2 = hmm_bic(fit_hmm(X, 2, seed=0, n_init=1, max_iter=40), X)
    assert b1 < b2


def test_single_state_hmm_is_degenerate_but_well_formed():
    panel = generate_null_gbm_panel(n_periods=300, seed=1)
    m = fit_hmm(panel.returns.to_numpy(), 1, seed=0, n_init=1, max_iter=30)
    assert m.transmat.shape == (1, 1)
    assert m.transmat[0, 0] == pytest.approx(1.0)
    assert stationary_distribution(m.transmat) == pytest.approx(np.array([1.0]))
    probs = filtered_probabilities(m, panel.returns.to_numpy())
    assert np.allclose(probs, 1.0)


def test_two_state_fit_on_null_data_yields_near_identical_states():
    """With no true regimes the two fitted covariances should be similar."""
    panel = generate_null_gbm_panel(n_periods=800, seed=7)
    m = fit_hmm(panel.returns.to_numpy(), 2, seed=0, n_init=2, max_iter=60)
    v0 = np.mean(np.diag(m.covs[0]))
    v1 = np.mean(np.diag(m.covs[1]))
    ratio = max(v0, v1) / min(v0, v1)
    # a genuine RORO panel separates by 4x+; the null must stay far below that
    assert ratio < 3.0


def test_null_panel_states_are_all_one_regime():
    panel = generate_null_gbm_panel(n_periods=200, seed=2)
    assert set(np.unique(panel.states)) == {0}
    assert panel.transition.shape == (1, 1)


# ---------------------------------------------------------------------------
# Label switching
# ---------------------------------------------------------------------------

def test_label_states_is_invariant_to_state_permutation():
    """Permuting the state order must permute labels identically, not change them."""
    cols = ["avg_vol", "haven_rs"]
    means = np.array([[-1.0, 0.0], [1.0, 0.8], [0.9, -0.7]])
    base = label_states(means, cols, 3)
    perm = np.array([2, 0, 1])
    permuted = label_states(means[perm], cols, 3)
    for new_idx, old_idx in enumerate(perm):
        assert permuted[new_idx] == base[old_idx]


def test_risk_on_is_always_the_lowest_vol_state():
    cols = ["avg_vol", "haven_rs"]
    for perm in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
        means = np.array([[-1.2, 0.1], [0.8, 0.9], [1.5, -0.6]])[perm]
        labels = label_states(means, cols, 3)
        lowest = int(np.argmin(means[:, 0]))
        assert labels[lowest] == "risk_on"


def test_haven_direction_separates_risk_off_from_usd_squeeze():
    """Both are high-vol; havens rally in risk_off and fall in a USD squeeze."""
    cols = ["avg_vol", "haven_rs"]
    means = np.array([[-1.0, 0.0], [1.0, +0.9], [1.1, -0.8]])
    labels = label_states(means, cols, 3)
    assert labels[1] == "risk_off"
    assert labels[2] == "usd_squeeze"
    flipped = label_states(np.array([[-1.0, 0.0], [1.0, -0.8], [1.1, +0.9]]), cols, 3)
    assert flipped[1] == "usd_squeeze"
    assert flipped[2] == "risk_off"


def test_match_states_recovers_a_known_permutation():
    true_means = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
    perm = np.array([2, 0, 1])
    est_means = true_means[perm]
    recovered = match_states(true_means, est_means)
    assert np.array_equal(recovered, perm)


def test_match_states_on_covariances_handles_indistinguishable_means():
    """FX drifts are noise; matching must work off covariances alone."""
    true_means = np.zeros((2, 1))
    est_means = np.zeros((2, 1))
    true_covs = np.array([[[1.0]], [[9.0]]])
    est_covs = np.array([[[9.0]], [[1.0]]])
    perm = match_states(true_means, est_means, true_covs, est_covs)
    assert np.array_equal(perm, np.array([1, 0]))


def test_label_states_input_validation():
    cols = ["avg_vol", "haven_rs"]
    means = np.array([[-1.0, 0.0], [1.0, 0.8]])
    with pytest.raises(ValueError, match="n_states must be 2 or 3"):
        label_states(means, cols, 4)
    with pytest.raises(ValueError, match="row count"):
        label_states(means, cols, 3)
    with pytest.raises(ValueError, match="avg_vol"):
        label_states(means, ["x", "haven_rs"], 2)
    with pytest.raises(ValueError, match="haven_rs"):
        label_states(np.zeros((3, 1)), ["avg_vol"], 3)


# ---------------------------------------------------------------------------
# Peg-break style regime transitions
# ---------------------------------------------------------------------------

def test_peg_break_variance_jump_is_detected_and_persists():
    """Calm peg then a sudden 10x variance jump: the filter must switch and stay."""
    rng = np.random.default_rng(11)
    calm = rng.normal(0.0, 0.0005, 400)
    broken = rng.normal(0.0, 0.005, 200)
    X = np.concatenate([calm, broken])[:, None]
    m = fit_hmm(X, 2, seed=0, n_init=3, max_iter=120)
    probs = filtered_probabilities(m, X)
    high_vol = int(np.argmax([np.mean(np.diag(c)) for c in m.covs]))
    # after the break the high-vol state dominates the back half
    assert probs[450:, high_vol].mean() > 0.8
    # before the break it does not
    assert probs[:350, high_vol].mean() < 0.2


def test_peg_regime_has_much_longer_expected_duration_than_the_break():
    """A peg is sticky; the post-break turmoil state need not be."""
    rng = np.random.default_rng(5)
    X = np.concatenate([rng.normal(0, 0.0005, 700), rng.normal(0, 0.005, 120)])[:, None]
    m = fit_hmm(X, 2, seed=0, n_init=3, max_iter=120)
    low_vol = int(np.argmin([np.mean(np.diag(c)) for c in m.covs]))
    durations = expected_durations(m.transmat)
    assert durations[low_vol] > 10.0
    assert np.all(durations > 1.0)


def test_planted_flip_is_recovered_by_viterbi():
    panel = generate_roro_panel(
        n_periods=800, n_states=2, seed=4, plant_flip_at=400,
        plant_flip_len=60, plant_flip_state=1,
    )
    X = panel.returns.to_numpy()
    m = fit_hmm(X, 2, seed=0, n_init=3, max_iter=100)
    path = viterbi(m, X)
    perm = match_states(panel.means, m.means, panel.covs, m.covs)
    mapped = perm[path]
    window = mapped[400:460]
    assert (window == 1).mean() > 0.6


def test_zero_vol_pegged_asset_column_is_rejected_by_the_fit():
    """A hard peg (identically zero returns) is a singular emission covariance."""
    rng = np.random.default_rng(9)
    X = np.column_stack([rng.normal(0, 0.01, 300), np.zeros(300)])
    m = fit_hmm(X, 2, seed=0, n_init=1, max_iter=20, reg_covar=1e-8)
    # the ridge keeps it invertible; the pegged column carries ~zero variance
    assert np.all(np.isfinite(m.covs))
    assert m.covs[0][1, 1] < 1e-6


# ---------------------------------------------------------------------------
# Hysteresis / committed regime path
# ---------------------------------------------------------------------------

def test_hysteresis_ignores_a_one_day_spike():
    idx = pd.RangeIndex(6)
    probs = pd.DataFrame(
        {"risk_on": [0.9, 0.9, 0.05, 0.9, 0.9, 0.9],
         "risk_off": [0.1, 0.1, 0.95, 0.1, 0.1, 0.1]},
        index=idx,
    )
    out = apply_hysteresis(probs, min_duration=2)
    assert (out == "risk_on").all()


def test_hysteresis_commits_after_the_confirmation_window():
    idx = pd.RangeIndex(6)
    probs = pd.DataFrame(
        {"risk_on": [0.9, 0.9, 0.05, 0.05, 0.05, 0.05],
         "risk_off": [0.1, 0.1, 0.95, 0.95, 0.95, 0.95]},
        index=idx,
    )
    out = apply_hysteresis(probs, min_duration=2)
    assert list(out) == ["risk_on", "risk_on", "risk_on", "risk_off", "risk_off", "risk_off"]


def test_hysteresis_is_causal_future_rows_cannot_change_the_past():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 60)
    probs = pd.DataFrame({"risk_on": p, "risk_off": 1 - p})
    full = apply_hysteresis(probs, min_duration=2)
    prefix = apply_hysteresis(probs.iloc[:30], min_duration=2)
    assert list(full.iloc[:30]) == list(prefix)


def test_hysteresis_parameter_validation():
    probs = pd.DataFrame({"a": [0.6, 0.6], "b": [0.4, 0.4]})
    with pytest.raises(ValueError, match="exit_threshold"):
        apply_hysteresis(probs, enter_threshold=0.3, exit_threshold=0.7)
    with pytest.raises(ValueError, match="min_duration"):
        apply_hysteresis(probs, min_duration=0)
    with pytest.raises(ValueError, match="not in columns"):
        apply_hysteresis(probs, initial="nope")


# ---------------------------------------------------------------------------
# Tiny samples, constant inputs, NaN/Inf
# ---------------------------------------------------------------------------

def test_hmm_rejects_series_shorter_than_the_minimum():
    X = np.random.default_rng(0).normal(0, 1, (5, 2))
    with pytest.raises(ValueError, match="too short"):
        fit_hmm(X, 2, seed=0)


def test_hmm_and_gmm_reject_nonpositive_state_counts():
    X = np.random.default_rng(0).normal(0, 1, (100, 2))
    with pytest.raises(ValueError, match="k must be >= 1"):
        fit_hmm(X, 0, seed=0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        fit_gmm(X, 0, seed=0)


def test_gmm_rejects_more_components_than_observations():
    X = np.random.default_rng(0).normal(0, 1, (3, 2))
    with pytest.raises(ValueError, match="at least"):
        fit_gmm(X, 5, seed=0)


def test_hmm_rejects_nan_input_with_an_actionable_message():
    """Previously failed later with the misleading 'transmat rows must sum to 1'."""
    X = np.random.default_rng(0).normal(0, 0.01, (200, 2))
    X[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        fit_hmm(X, 2, seed=0, n_init=1, max_iter=5)


def test_gmm_rejects_nan_input_instead_of_returning_nan_means():
    X = np.random.default_rng(0).normal(0, 0.01, (200, 2))
    X[7, 1] = np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        fit_gmm(X, 2, seed=0, n_init=1, max_iter=5)


def test_constant_input_fit_stays_finite():
    """A dead (constant) series must not produce NaN parameters."""
    X = np.zeros((200, 2))
    m = fit_hmm(X, 2, seed=0, n_init=1, max_iter=20)
    assert np.all(np.isfinite(m.means))
    assert np.all(np.isfinite(m.covs))
    assert np.isfinite(m.log_likelihood)


def test_filtered_and_smoothed_probabilities_are_valid_distributions():
    panel = generate_roro_panel(n_periods=300, n_states=2, seed=6)
    X = panel.returns.to_numpy()
    m = fit_hmm(X, 2, seed=0, n_init=1, max_iter=40)
    for probs in (filtered_probabilities(m, X), smoothed_probabilities(m, X)):
        assert probs.shape == (len(X), 2)
        assert np.all(probs >= -1e-12)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-10)


def test_stationary_distribution_is_a_fixed_point():
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    pi = stationary_distribution(P)
    assert np.allclose(pi @ P, pi, atol=1e-12)
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi > 0)


def test_stationary_distribution_rejects_non_stochastic_matrix():
    with pytest.raises(ValueError, match="rows must sum to 1"):
        stationary_distribution(np.array([[0.9, 0.2], [0.1, 0.9]]))


def test_expected_duration_matches_the_geometric_mean():
    P = np.array([[0.98, 0.02], [0.05, 0.95]])
    d = expected_durations(P)
    assert d[0] == pytest.approx(50.0)
    assert d[1] == pytest.approx(20.0)


def test_absorbing_state_duration_is_finite_not_infinite():
    """A_ii = 1 is clipped so the duration stays a representable number."""
    d = expected_durations(np.array([[1.0, 0.0], [0.05, 0.95]]))
    assert np.isfinite(d[0]) and d[0] > 1e9


def test_viterbi_path_is_in_range_and_full_length():
    panel = generate_roro_panel(n_periods=200, n_states=3, seed=8)
    X = panel.returns.to_numpy()
    m = fit_hmm(X, 3, seed=0, n_init=1, max_iter=30)
    path = viterbi(m, X)
    assert path.shape == (len(X),)
    assert set(np.unique(path)) <= {0, 1, 2}


def test_em_log_likelihood_path_is_monotone_nondecreasing():
    panel = generate_roro_panel(n_periods=300, n_states=2, seed=12)
    m = fit_hmm(panel.returns.to_numpy(), 2, seed=0, n_init=1, max_iter=50)
    ll = np.array(m.log_likelihood_path)
    assert np.all(np.diff(ll) >= -1e-8)


def test_panel_generator_rejects_degenerate_arguments():
    with pytest.raises(ValueError, match="n_periods"):
        generate_roro_panel(n_periods=10)
    with pytest.raises(ValueError, match="n_states"):
        generate_roro_panel(n_periods=100, n_states=5)
    with pytest.raises(ValueError, match="out of range"):
        generate_roro_panel(n_periods=100, plant_flip_at=500)


def test_same_seed_reproduces_the_panel_exactly():
    a = generate_roro_panel(n_periods=150, n_states=2, seed=21)
    b = generate_roro_panel(n_periods=150, n_states=2, seed=21)
    assert np.array_equal(a.states, b.states)
    pd.testing.assert_frame_equal(a.returns, b.returns)
