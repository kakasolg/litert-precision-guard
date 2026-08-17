"""Orchestrator for differential precision verification across LiteRT backends."""

from typing import Dict, List, Optional, Sequence
import numpy as np

from litert_precision_guard.metrics import compute_tensor_metrics
from litert_precision_guard.models import (
    BackendConfig,
    BackendReport,
    FailureReason,
    TensorComparison,
    VerificationReport,
)
from litert_precision_guard.runners import LiteRTRunner


DEFAULT_BACKENDS = [
    BackendConfig(name="cpu", accelerator="cpu", enforce_f32=False, description="CPU Baseline (XNNPACK)"),
    BackendConfig(name="gpu", accelerator="gpu", enforce_f32=False, description="GPU Default (FP16 / Mixed)"),
    BackendConfig(name="gpu-fp32", accelerator="gpu", enforce_f32=True, description="GPU FP32 Enforced"),
]


class PrecisionVerifier:
    """Verifies numerical precision and detects silent failures across execution backends."""

    def __init__(
        self,
        model_path: str,
        backend_configs: Optional[List[BackendConfig]] = None,
        min_cosine_similarity: float = 0.99,
        min_norm: float = 1e-4,
        seed: int = 42,
        signature_key: Optional[str] = None,
    ):
        self.model_path = model_path
        self.backend_configs = backend_configs or DEFAULT_BACKENDS
        self.min_cosine_similarity = min_cosine_similarity
        self.min_norm = min_norm
        self.seed = seed
        self.signature_key = signature_key

    @classmethod
    def parse_backend_names(cls, backend_str: str) -> List[BackendConfig]:
        """Parses a comma-separated backend string into BackendConfig list."""
        names = [b.strip().lower() for b in backend_str.split(",") if b.strip()]
        configs: List[BackendConfig] = []
        for name in names:
            if name == "cpu":
                configs.append(BackendConfig(name="cpu", accelerator="cpu", enforce_f32=False, description="CPU Baseline"))
            elif name in ("gpu", "gpu-default", "gpu-fp16"):
                configs.append(BackendConfig(name=name, accelerator="gpu", enforce_f32=False, description="GPU Default (FP16)"))
            elif name in ("gpu-fp32", "gpu_fp32", "gpu-f32"):
                configs.append(BackendConfig(name=name, accelerator="gpu", enforce_f32=True, description="GPU FP32 Enforced"))
            else:
                raise ValueError(f"Unknown backend identifier: '{name}'. Supported: cpu, gpu, gpu-fp32")
        return configs

    def _generate_synthetic_inputs(
        self,
        input_specs: Dict[str, dict],
    ) -> Dict[str, np.ndarray]:
        """Generates deterministic synthetic inputs suitable for the model's expected dtypes and shapes."""
        rng = np.random.RandomState(self.seed)
        synthetic_inputs: Dict[str, np.ndarray] = {}

        for name, spec in input_specs.items():
            shape = spec.get("shape", [1, 256])
            dtype = spec.get("dtype", np.float32)

            # Resolve dynamic shapes (e.g. None, -1, 0)
            concrete_shape = []
            for dim in shape:
                if dim is None or dim <= 0:
                    concrete_shape.append(1)
                else:
                    concrete_shape.append(int(dim))

            # Handle types
            if np.issubdtype(dtype, np.integer):
                # Token IDs or integer inputs: generate reasonable integers (1 to 500)
                data = rng.randint(1, 500, size=concrete_shape).astype(dtype)
            elif np.issubdtype(dtype, np.floating):
                # Float inputs: standard normal distribution scaled
                data = rng.randn(*concrete_shape).astype(dtype) * 0.5
            elif dtype == np.bool_ or dtype == bool:
                data = rng.choice([True, False], size=concrete_shape)
            else:
                # Default to float32
                data = rng.randn(*concrete_shape).astype(np.float32)

            synthetic_inputs[name] = data

        return synthetic_inputs

    def verify(
        self,
        custom_inputs: Optional[Dict[str, np.ndarray]] = None,
        warmup: int = 1,
        iterations: int = 1,
    ) -> VerificationReport:
        """Executes differential verification against all configured backends."""
        # 1. Initialize CPU baseline runner to probe model structure and compute reference
        cpu_runner = LiteRTRunner(
            model_path=self.model_path,
            accelerator="cpu",
            enforce_f32=False,
            signature_key=self.signature_key,
        )
        cpu_runner.initialize()
        signature_key = cpu_runner.signature_key
        input_specs = cpu_runner.get_input_specs()

        # 2. Prepare inputs
        if custom_inputs is not None:
            inputs = custom_inputs
        else:
            inputs = self._generate_synthetic_inputs(input_specs)

        input_shapes = {k: list(v.shape) for k, v in inputs.items()}

        # 3. Execute CPU Baseline
        baseline_outputs, cpu_latency = cpu_runner.run(inputs, warmup=warmup, iterations=iterations)

        backend_reports: Dict[str, BackendReport] = {}
        all_passed = True

        # Register CPU baseline report
        cpu_config = BackendConfig(name="cpu", accelerator="cpu", enforce_f32=False, description="CPU Baseline")
        cpu_comparisons: List[TensorComparison] = []
        for out_name, arr in baseline_outputs.items():
            comp = compute_tensor_metrics(
                tensor_name=out_name,
                baseline=arr,
                target=arr,
                min_cosine_similarity=self.min_cosine_similarity,
                min_norm=self.min_norm,
            )
            cpu_comparisons.append(comp)

        cpu_passed = all(c.passed for c in cpu_comparisons)
        cpu_failures = []
        for c in cpu_comparisons:
            cpu_failures.extend(c.failure_reasons)

        backend_reports["cpu"] = BackendReport(
            config=cpu_config,
            passed=cpu_passed,
            latency_ms=cpu_latency,
            tensor_comparisons=cpu_comparisons,
            failure_reasons=cpu_failures,
        )
        if not cpu_passed:
            all_passed = False

        # 4. Execute target backends
        for cfg in self.backend_configs:
            if cfg.name == "cpu":
                continue  # already computed as baseline

            try:
                runner = LiteRTRunner(
                    model_path=self.model_path,
                    accelerator=cfg.accelerator,
                    enforce_f32=cfg.enforce_f32,
                    signature_key=signature_key,
                )
                runner.initialize()
                target_outputs, target_latency = runner.run(inputs, warmup=warmup, iterations=iterations)

                comparisons: List[TensorComparison] = []
                for out_name, baseline_arr in baseline_outputs.items():
                    target_arr = target_outputs.get(out_name)
                    if target_arr is None:
                        # Missing output
                        comp = TensorComparison(
                            name=out_name,
                            shape=list(baseline_arr.shape),
                            dtype=str(baseline_arr.dtype),
                            baseline_norm=float(np.linalg.norm(baseline_arr)),
                            target_norm=0.0,
                            cosine_similarity=0.0,
                            max_abs_diff=float("inf"),
                            mean_abs_diff=float("inf"),
                            has_nan=False,
                            has_inf=False,
                            is_zero_vector=True,
                            passed=False,
                            failure_reasons=[FailureReason.SILENT_ZERO_VECTOR],
                        )
                    else:
                        comp = compute_tensor_metrics(
                            tensor_name=out_name,
                            baseline=baseline_arr,
                            target=target_arr,
                            min_cosine_similarity=self.min_cosine_similarity,
                            min_norm=self.min_norm,
                        )
                    comparisons.append(comp)

                target_passed = all(c.passed for c in comparisons)
                target_failures: List[FailureReason] = []
                for c in comparisons:
                    for r in c.failure_reasons:
                        if r not in target_failures:
                            target_failures.append(r)

                backend_reports[cfg.name] = BackendReport(
                    config=cfg,
                    passed=target_passed,
                    latency_ms=target_latency,
                    tensor_comparisons=comparisons,
                    failure_reasons=target_failures,
                )
                if not target_passed:
                    all_passed = False

            except Exception as exc:
                backend_reports[cfg.name] = BackendReport(
                    config=cfg,
                    passed=False,
                    latency_ms=0.0,
                    error_message=str(exc),
                    failure_reasons=[FailureReason.EXECUTION_ERROR],
                )
                all_passed = False

        return VerificationReport(
            model_path=self.model_path,
            signature_key=signature_key,
            input_shapes=input_shapes,
            min_cosine_similarity=self.min_cosine_similarity,
            min_norm=self.min_norm,
            backend_reports=backend_reports,
            all_passed=all_passed,
        )
