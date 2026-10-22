from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class HMMParams:
    trans: np.ndarray
    emis: np.ndarray
    state_labels: list


def learn_hmm_params(true_by_subject: list[np.ndarray], pred_by_subject: list[np.ndarray]) -> HMMParams:
    keep = [(np.asarray(t), np.asarray(p)) for t, p in zip(true_by_subject, pred_by_subject, strict=False) if len(t) and len(p)]
    if not keep:
        return HMMParams(trans=np.empty((0, 0)), emis=np.empty((0, 0)), state_labels=[])

    labels: list = []
    for y_true, y_pred in keep:
        for value in y_true.tolist() + y_pred.tolist():
            if value not in labels:
                labels.append(value)
    n_states = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    trans_counts = np.zeros((n_states, n_states), dtype=float)
    emis_counts = np.zeros((n_states, n_states), dtype=float)

    for y_true, y_pred in keep:
        n = min(y_true.size, y_pred.size)
        state_idx = np.array([index.get(v, -1) for v in y_true[:n].tolist()], dtype=int)
        obs_idx = np.array([index.get(v, -1) for v in y_pred[:n].tolist()], dtype=int)
        valid = (state_idx >= 0) & (obs_idx >= 0)
        state_idx = state_idx[valid]
        obs_idx = obs_idx[valid]
        for s, o in zip(state_idx, obs_idx, strict=False):
            emis_counts[s, o] += 1.0
        for a, b in zip(state_idx[:-1], state_idx[1:], strict=False):
            trans_counts[a, b] += 1.0

    return HMMParams(
        trans=_normalize_rows(trans_counts + 1.0),
        emis=_normalize_rows(emis_counts + 1.0),
        state_labels=labels,
    )


def apply_hmm_viterbi(pred_by_subject: list[np.ndarray], hmm_params: HMMParams) -> list[np.ndarray]:
    if hmm_params.trans.size == 0 or hmm_params.emis.size == 0 or not hmm_params.state_labels:
        return [np.asarray([]) for _ in pred_by_subject]

    index = {label: i for i, label in enumerate(hmm_params.state_labels)}
    decoded = []
    for y_pred in pred_by_subject:
        y_pred = np.asarray(y_pred)
        if y_pred.size == 0:
            decoded.append(np.asarray([]))
            continue
        obs_idx = np.array([index.get(v, -1) for v in y_pred.tolist()], dtype=int)
        valid = obs_idx[obs_idx >= 0]
        fill = int(np.bincount(valid).argmax()) if valid.size else 0
        obs_idx[obs_idx < 0] = fill
        path = viterbi_decode(obs_idx, hmm_params.trans, hmm_params.emis)
        decoded.append(np.asarray([hmm_params.state_labels[i] for i in path], dtype=object))
    return decoded


def viterbi_decode(obs_idx: np.ndarray, trans: np.ndarray, emis: np.ndarray) -> np.ndarray:
    obs_idx = np.asarray(obs_idx, dtype=int)
    n_steps = obs_idx.size
    n_states = trans.shape[0]
    if n_steps == 0 or n_states == 0:
        return np.asarray([], dtype=int)

    log_trans = np.log(np.maximum(trans, 1e-300))
    log_emis = np.log(np.maximum(emis, 1e-300))
    delta = np.full((n_steps, n_states), -np.inf, dtype=float)
    psi = np.zeros((n_steps, n_states), dtype=int)

    delta[0] = log_emis[:, obs_idx[0]]
    for t in range(1, n_steps):
        scores = delta[t - 1, :, None] + log_trans
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores[psi[t], np.arange(n_states)] + log_emis[:, obs_idx[t]]

    path = np.zeros(n_steps, dtype=int)
    path[-1] = int(delta[-1].argmax())
    for t in range(n_steps - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    row_sum = values.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return values / row_sum


# ---------------------------------------------------------------- probabilistic HMM over stage-1 probabilities
@dataclass(slots=True)
class TransitionModel:
    """Stage-to-stage transition matrix and initial distribution counted from labelled nights."""

    trans: np.ndarray
    initial: np.ndarray
    n_states: int


def learn_transition_matrix(label_nights: list[np.ndarray], n_states: int = 5, prior: float = 1.0) -> TransitionModel:
    """Count epoch-to-epoch transitions in time-ordered label sequences, with a Laplace prior."""
    trans = np.full((n_states, n_states), prior, dtype=float)
    initial = np.full(n_states, prior, dtype=float)
    for y in label_nights:
        y = np.asarray(y, dtype=int)
        if y.size == 0:
            continue
        initial[y[0]] += 1.0
        np.add.at(trans, (y[:-1], y[1:]), 1.0)
    return TransitionModel(trans=_normalize_rows(trans), initial=initial / initial.sum(), n_states=n_states)


def posterior_decode(proba: np.ndarray, model: TransitionModel, eps: float = 1e-6) -> np.ndarray:
    """Forward-backward state posteriors for one night, using class-balanced stage-1 probabilities as emissions."""
    emis = np.asarray(proba, dtype=float) + eps
    n_steps = len(emis)
    if n_steps == 0:
        return np.zeros((0, model.n_states))
    alpha = np.zeros_like(emis)
    beta = np.ones_like(emis)
    scale = np.zeros(n_steps)
    a = model.initial * emis[0]
    scale[0] = a.sum()
    alpha[0] = a / scale[0]
    for t in range(1, n_steps):
        a = (alpha[t - 1] @ model.trans) * emis[t]
        scale[t] = a.sum()
        alpha[t] = a / scale[t]
    for t in range(n_steps - 2, -1, -1):
        beta[t] = (model.trans @ (emis[t + 1] * beta[t + 1])) / scale[t + 1]
    gamma = alpha * beta
    return gamma / gamma.sum(axis=1, keepdims=True)


def bout_lengths(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(stage, length in epochs) of each run of identical labels."""
    y = np.asarray(y)
    if y.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    change = np.flatnonzero(np.r_[True, y[1:] != y[:-1]])
    return y[change], np.diff(np.r_[change, y.size])
