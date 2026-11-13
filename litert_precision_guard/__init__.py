"""LiteRT Precision Guard: Automated diagnostic tool for silent delegate failures."""

__version__ = "0.2.0"
__author__ = "LiteRT Precision Guard Contributors"
__license__ = "Apache-2.0"

from litert_precision_guard.models import (
    BackendConfig,
    BackendReport,
    FailureReason,
    TensorComparison,
    VerificationReport,
)
from litert_precision_guard.verifier import PrecisionVerifier, compare_to_reference, verify_output_files

__all__ = [
    "PrecisionVerifier",
    "compare_to_reference",
    "verify_output_files",
    "VerificationReport",
    "BackendReport",
    "BackendConfig",
    "FailureReason",
    "TensorComparison",
]
