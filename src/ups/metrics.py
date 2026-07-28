"""Function and representation preservation metrics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def action_kl(reference_logits: Array, candidate_logits: Array) -> float:
    def log_softmax(values: Array) -> Array:
        shifted = values - values.max(axis=-1, keepdims=True)
        return np.asarray(
            shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True)), dtype=np.float64
        )

    log_p = log_softmax(reference_logits)
    log_q = log_softmax(candidate_logits)
    p = np.exp(log_p)
    return float(np.mean(np.sum(p * (log_p - log_q), axis=-1)))


def action_agreement(reference_logits: Array, candidate_logits: Array) -> float:
    return float(np.mean(reference_logits.argmax(axis=-1) == candidate_logits.argmax(axis=-1)))


def linear_cka(left: Array, right: Array) -> float:
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(left.T @ right, ord="fro") ** 2
    denominator = np.linalg.norm(left.T @ left, ord="fro") * np.linalg.norm(
        right.T @ right, ord="fro"
    )
    return float(numerator / denominator) if denominator else 0.0


def normalized_value_rmse(reference: Array, candidate: Array) -> float:
    rmse = np.sqrt(np.mean((reference - candidate) ** 2))
    scale = np.sqrt(np.mean((reference - reference.mean()) ** 2))
    return float(rmse / scale) if scale else float(rmse)
