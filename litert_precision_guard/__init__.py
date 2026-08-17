"""LiteRT Precision Guard: Automated diagnostic tool for silent delegate failures."""

__version__ = "0.1.0"
__author__ = "LiteRT Precision Guard Contributors"
__license__ = "Apache-2.0"

from litert_precision_guard.models import (
    BackendConfig,
    BackendReport,
    FailureReason,
    TensorComparison,
    VerificationReport,
)
from litert_precision_guard.verifier import PrecisionVerifier

__all__ = [
    "PrecisionVerifier",
    "VerificationReport",
    "BackendReport",
    "BackendConfig",
    "FailureReason",
    "TensorComparison",
]
