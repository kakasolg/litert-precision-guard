"""Numerical metrics and anomaly detection for tensor comparison."""

from typing import List, Tuple
import numpy as np

from litert_precision_guard.models import FailureReason, TensorComparison


def compute_tensor_metrics(
    tensor_name: str,
    baseline: np.ndarray,
    target: np.ndarray,
    min_cosine_similarity: float = 0.99,
    min_norm: float = 1e-4,
) -> TensorComparison:
    """Compares a target tensor output against the baseline (CPU) output.

    Identifies:
    1. Silent zero-vector / vanishing output (norm is ~0 while baseline is non-zero)
    2. Numerical instability (NaN or Inf values)
    3. Precision degradation (cosine similarity below threshold or high divergence)
    """
    baseline_flat = np.asarray(baseline, dtype=np.float64).ravel()
    target_flat = np.asarray(target, dtype=np.float64).ravel()

    # Check NaN / Inf
    has_nan = bool(np.isnan(target_flat).any())
    has_inf = bool(np.isinf(target_flat).any())

    # Calculate norms
    baseline_norm = float(np.linalg.norm(baseline_flat))
    target_norm = float(np.linalg.norm(target_flat))

    # Anomaly checks
    is_zero_vector = False
    if baseline_norm >= min_norm and target_norm < min_norm:
        is_zero_vector = True
    elif target_norm == 0.0 and baseline_norm > 0.0:
        is_zero_vector = True

    # Cosine Similarity
    if baseline_norm == 0.0 and target_norm == 0.0:
        cosine_sim = 1.0
    elif baseline_norm < 1e-12 or target_norm < 1e-12:
        cosine_sim = 0.0
    else:
        dot_product = float(np.dot(baseline_flat, target_flat))
        cosine_sim = dot_product / (baseline_norm * target_norm)
        # Clamp to [-1.0, 1.0] for precision float discrepancies
        cosine_sim = float(np.clip(cosine_sim, -1.0, 1.0))

    # Differences
    if has_nan or has_inf:
        max_abs_diff = float("inf")
        mean_abs_diff = float("inf")
    else:
        diff = np.abs(baseline_flat - target_flat)
        max_abs_diff = float(np.max(diff))
        mean_abs_diff = float(np.mean(diff))

    failure_reasons: List[FailureReason] = []
    if has_nan or has_inf:
        failure_reasons.append(FailureReason.NUMERICAL_INSTABILITY)
    if is_zero_vector:
        failure_reasons.append(FailureReason.SILENT_ZERO_VECTOR)
    if not (has_nan or has_inf or is_zero_vector) and cosine_sim < min_cosine_similarity:
        failure_reasons.append(FailureReason.PRECISION_DEGRADATION)

    passed = len(failure_reasons) == 0

    return TensorComparison(
        name=tensor_name,
        shape=list(target.shape),
        dtype=str(target.dtype),
        baseline_norm=baseline_norm,
        target_norm=target_norm,
        cosine_similarity=cosine_sim,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        has_nan=has_nan,
        has_inf=has_inf,
        is_zero_vector=is_zero_vector,
        passed=passed,
        failure_reasons=failure_reasons,
    )
