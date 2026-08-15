from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


@dataclass(slots=True)
class HSMMParams:
    trans: np.ndarray
    emis: np.ndarray
    dur_lambda: np.ndarray
    dur_d_max: int
    state_labels: list


def learn_hsmm_params(true_by_subject: list[np.ndarray], pred_by_subject: list[np.ndarray]) -> HSMMParams:
    keep = [(np.asarray(t), np.asarray(p)) for t, p in zip(true_by_subject, pred_by_subject, strict=False) if len(t) and len(p)]
    if not keep:
        return HSMMParams(
            trans=np.empty((0, 0)),
            emis=np.empty((0, 0)),
            dur_lambda=np.empty(0),
            dur_d_max=0,
            state_labels=[],
        )

    labels: list = []
    for y_true, y_pred in keep:
        for value in y_true.tolist() + y_pred.tolist():
            if value not in labels:
                labels.append(value)
    n_states = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    trans_counts = np.zeros((n_states, n_states), dtype=float)
    emis_counts = np.zeros((n_states, n_states), dtype=float)
    run_lengths: list[list[int]] = [[] for _ in range(n_states)]

    for y_true, y_pred in keep:
        n = min(y_true.size, y_pred.size)
        state_idx = np.array([index.get(v, -1) for v in y_true[:n].tolist()], dtype=int)
        obs_idx = np.array([index.get(v, -1) for v in y_pred[:n].tolist()], dtype=int)
        valid = (state_idx >= 0) & (obs_idx >= 0)
        state_idx = state_idx[valid]
        obs_idx = obs_idx[valid]
        for s, o in zip(state_idx, obs_idx, strict=False):
            emis_counts[s, o] += 1.0
        if state_idx.size < 2:
            if state_idx.size == 1:
                run_lengths[state_idx[0]].append(1)
            continue
        run_start = 0
        for t in range(1, state_idx.size):
            if state_idx[t] != state_idx[t - 1]:
                run_lengths[state_idx[t - 1]].append(t - run_start)
                trans_counts[state_idx[t - 1], state_idx[t]] += 1.0
                run_start = t
        run_lengths[state_idx[-1]].append(state_idx.size - run_start)

    dur_lambda = np.array(
        [max(1.0, float(np.mean(lengths))) if lengths else 1.0 for lengths in run_lengths],
        dtype=float,
    )
    dur_d_max = max(1, int(round(3.0 * float(np.max(dur_lambda)))))

    return HSMMParams(
        trans=_normalize_rows(trans_counts + 1.0),
        emis=_normalize_rows(emis_counts + 1.0),
        dur_lambda=dur_lambda,
        dur_d_max=dur_d_max,
        state_labels=labels,
    )


def apply_hsmm_viterbi(pred_by_subject: list[np.ndarray], hsmm_params: HSMMParams) -> list[np.ndarray]:
    if hsmm_params.trans.size == 0 or hsmm_params.emis.size == 0 or not hsmm_params.state_labels:
        return [np.asarray([]) for _ in pred_by_subject]

    trans = hsmm_params.trans
    emis = hsmm_params.emis
    dur_lambda = hsmm_params.dur_lambda
    d_max = hsmm_params.dur_d_max
    labels = hsmm_params.state_labels
    n_states = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    log_pois = _log_poisson_table(dur_lambda, d_max)
    log_emis = np.log(np.maximum(emis, 1e-300))
    log_trans = np.log(np.maximum(trans, 1e-300))
    np.fill_diagonal(log_trans, -np.inf)
    neg_inf = -1e18

    decoded: list[np.ndarray] = []
    for y_pred in pred_by_subject:
        y_pred = np.asarray(y_pred)
        if y_pred.size == 0:
            decoded.append(np.asarray([]))
            continue
        obs_idx = np.array([index.get(v, -1) for v in y_pred.tolist()], dtype=int)
        valid = obs_idx[obs_idx >= 0]
        fill = int(np.bincount(valid).argmax()) if valid.size else 0
        obs_idx[obs_idx < 0] = fill
        n_steps = obs_idx.size

        cum_emit = np.zeros((n_states, n_steps + 1), dtype=float)
        for t in range(n_steps):
            cum_emit[:, t + 1] = cum_emit[:, t] + log_emis[:, obs_idx[t]]

        delta = np.full((n_steps, n_states), neg_inf, dtype=float)
        psi_d = np.zeros((n_steps, n_states), dtype=int)
        psi_s = np.full((n_steps, n_states), -1, dtype=int)

        for state in range(n_states):
            for dur in range(1, min(d_max, n_steps) + 1):
                emit_score = cum_emit[state, dur] - cum_emit[state, 0]
                score = log_pois[state, dur - 1] + emit_score
                if score > delta[dur - 1, state]:
                    delta[dur - 1, state] = score
                    psi_d[dur - 1, state] = dur

        for t in range(1, n_steps):
            max_dur = min(d_max, t)
            for state in range(n_states):
                for dur in range(1, max_dur + 1):
                    t_prev = t - dur
                    emit_score = cum_emit[state, t + 1] - cum_emit[state, t_prev + 1]
                    dur_score = log_pois[state, dur - 1]
                    for prev_state in range(n_states):
                        if prev_state == state:
                            continue
                        prev_score = delta[t_prev, prev_state]
                        if prev_score <= neg_inf:
                            continue
                        score = prev_score + log_trans[prev_state, state] + dur_score + emit_score
                        if score > delta[t, state]:
                            delta[t, state] = score
                            psi_d[t, state] = dur
                            psi_s[t, state] = prev_state

        path = np.full(n_steps, -1, dtype=int)
        cur_state = int(np.argmax(delta[-1]))
        cur_t = n_steps - 1
        while cur_t >= 0:
            dur = psi_d[cur_t, cur_state]
            if dur <= 0:
                dur = cur_t + 1
            start = max(0, cur_t - dur + 1)
            path[start : cur_t + 1] = cur_state
            prev_state = psi_s[cur_t, cur_state]
            cur_t = start - 1
            if prev_state < 0:
                break
            cur_state = prev_state

        if np.any(path < 0):
            valid_states = path[path >= 0]
            fill_state = int(np.bincount(valid_states).argmax()) if valid_states.size else 0
            path[path < 0] = fill_state
        decoded.append(np.asarray([labels[idx] for idx in path.tolist()]))
    return decoded


def _log_poisson_table(lambdas: np.ndarray, d_max: int) -> np.ndarray:
    d_vec = np.arange(1, d_max + 1, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float).reshape(-1, 1)
    return -lambdas + d_vec.reshape(1, -1) * np.log(np.maximum(lambdas, 1e-300)) - gammaln(d_vec.reshape(1, -1) + 1.0)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    row_sum = values.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return values / row_sum
