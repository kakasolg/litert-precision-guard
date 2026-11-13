"""Data models for precision and delegate verification."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FailureReason(str, Enum):
    NONE = "NONE"
    SILENT_ZERO_VECTOR = "SILENT_ZERO_VECTOR"
    NUMERICAL_INSTABILITY = "NUMERICAL_INSTABILITY"  # NaN or Inf
    PRECISION_DEGRADATION = "PRECISION_DEGRADATION"  # Cosine sim or error threshold exceeded
    EXECUTION_ERROR = "EXECUTION_ERROR"  # Delegate initialization or runtime crash
    REFERENCE_MISMATCH = "REFERENCE_MISMATCH"  # Output disagrees with an external reference (e.g. the original model)


@dataclass
class BackendConfig:
    name: str
    accelerator: str  # 'cpu', 'gpu'
    enforce_f32: bool = False
    description: str = ""

    @property
    def display_name(self) -> str:
        if self.accelerator == "gpu":
            return f"gpu (FP32 enforced)" if self.enforce_f32 else "gpu (default / FP16)"
        return self.accelerator


@dataclass
class TensorComparison:
    name: str
    shape: List[int]
    dtype: str
    baseline_norm: float
    target_norm: float
    cosine_similarity: float
    max_abs_diff: float
    mean_abs_diff: float
    has_nan: bool
    has_inf: bool
    is_zero_vector: bool
    passed: bool
    failure_reasons: List[FailureReason] = field(default_factory=list)
    min_row_cosine: Optional[float] = None  # set for [N, D] outputs with N > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "baseline_norm": float(self.baseline_norm),
            "target_norm": float(self.target_norm),
            "cosine_similarity": float(self.cosine_similarity),
            "max_abs_diff": float(self.max_abs_diff),
            "mean_abs_diff": float(self.mean_abs_diff),
            "has_nan": bool(self.has_nan),
            "has_inf": bool(self.has_inf),
            "is_zero_vector": bool(self.is_zero_vector),
            "passed": bool(self.passed),
            "failure_reasons": [r.value for r in self.failure_reasons],
            "min_row_cosine": None if self.min_row_cosine is None else float(self.min_row_cosine),
        }


@dataclass
class BackendReport:
    config: BackendConfig
    passed: bool
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    tensor_comparisons: List[TensorComparison] = field(default_factory=list)
    failure_reasons: List[FailureReason] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.config.name,
            "display_name": self.config.display_name,
            "accelerator": self.config.accelerator,
            "enforce_f32": self.config.enforce_f32,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 3),
            "error_message": self.error_message,
            "failure_reasons": [r.value for r in self.failure_reasons],
            "tensors": [t.to_dict() for t in self.tensor_comparisons],
        }


@dataclass
class VerificationReport:
    model_path: str
    signature_key: Optional[str]
    input_shapes: Dict[str, List[int]]
    min_cosine_similarity: float
    min_norm: float
    backend_reports: Dict[str, BackendReport] = field(default_factory=dict)
    all_passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "signature_key": self.signature_key,
            "input_shapes": self.input_shapes,
            "thresholds": {
                "min_cosine_similarity": self.min_cosine_similarity,
                "min_norm": self.min_norm,
            },
            "all_passed": self.all_passed,
            "backends": {
                name: report.to_dict() for name, report in self.backend_reports.items()
            },
        }
