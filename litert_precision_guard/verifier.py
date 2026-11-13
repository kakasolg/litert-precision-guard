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


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """Loads a .npz file into a {name: array} dict."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _pair_outputs(
    reference: Dict[str, np.ndarray],
    outputs: Dict[str, np.ndarray],
) -> List[tuple]:
    """Pairs reference arrays with outputs by name; a single array on each side is paired regardless of name."""
    if len(reference) == 1 and len(outputs) == 1:
        (rn, ra), (on, oa) = next(iter(reference.items())), next(iter(outputs.items()))
        return [(on, ra, oa)]
    pairs = []
    for name, ra in reference.items():
        if name not in outputs:
            raise ValueError(f"Reference output '{name}' not found among outputs {sorted(outputs)}")
        pairs.append((name, ra, outputs[name]))
    return pairs


def compare_to_reference(
    reference: Dict[str, np.ndarray],
    outputs: Dict[str, np.ndarray],
    min_cosine_similarity: float = 0.99,
    min_norm: float = 1e-4,
) -> List[TensorComparison]:
    """Compares outputs against an external reference (e.g. the original model's outputs for the same inputs).

    Unlike the CPU-vs-delegate comparison, this also catches a wrong CPU result, and outputs that keep a
    normal norm but point the wrong way. Arrays with the same number of elements but a different shape are
    reshaped to the output's shape.
    """
    comps = []
    for name, ref, out in _pair_outputs(reference, outputs):
        ref = np.asarray(ref)
        if ref.shape != out.shape:
            if ref.size != out.size:
                raise ValueError(f"Reference '{name}' has shape {list(ref.shape)}, output has {list(out.shape)}")
            ref = ref.reshape(out.shape)
        comps.append(compute_tensor_metrics(
            tensor_name=f"ref:{name}",
            baseline=ref,
            target=out,
            min_cosine_similarity=min_cosine_similarity,
            min_norm=min_norm,
            mismatch_reason=FailureReason.REFERENCE_MISMATCH,
        ))
    return comps


def verify_output_files(
    reference_path: str,
    outputs_path: str,
    min_cosine_similarity: float = 0.99,
    min_norm: float = 1e-4,
) -> VerificationReport:
    """Checks outputs produced elsewhere (a phone, Core AI, ONNX Runtime, a browser) against a reference.

    Both files are .npz. No model is run.
    """
    comps = compare_to_reference(load_npz(reference_path), load_npz(outputs_path), min_cosine_similarity, min_norm)
    reasons: List[FailureReason] = []
    for c in comps:
        for r in c.failure_reasons:
            if r not in reasons:
                reasons.append(r)
    passed = all(c.passed for c in comps)
    cfg = BackendConfig(name="outputs", accelerator="external", enforce_f32=False, description=outputs_path)
    return VerificationReport(
        model_path=outputs_path,
        signature_key=None,
        input_shapes={},
        min_cosine_similarity=min_cosine_similarity,
        min_norm=min_norm,
        backend_reports={"outputs": BackendReport(config=cfg, passed=passed, tensor_comparisons=comps,
                                                  failure_reasons=reasons)},
        all_passed=passed,
    )


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
        reference_outputs: Optional[Dict[str, np.ndarray]] = None,
    ) -> VerificationReport:
        """Executes differential verification against all configured backends.

        With `reference_outputs` (expected outputs for `custom_inputs`, e.g. from the original framework model),
        every backend, the CPU included, is also compared against that reference.
        """
        if reference_outputs is not None and custom_inputs is None:
            raise ValueError("reference_outputs needs the matching custom_inputs")
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

        if reference_outputs is not None:
            cpu_comparisons.extend(compare_to_reference(
                reference_outputs, baseline_outputs, self.min_cosine_similarity, self.min_norm))

        cpu_passed = all(c.passed for c in cpu_comparisons)
        cpu_failures: List[FailureReason] = []
        for c in cpu_comparisons:
            for r in c.failure_reasons:
                if r not in cpu_failures:
                    cpu_failures.append(r)

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

                if reference_outputs is not None:
                    comparisons.extend(compare_to_reference(
                        reference_outputs, target_outputs, self.min_cosine_similarity, self.min_norm))

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
