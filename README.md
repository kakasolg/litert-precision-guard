# LiteRT Precision Guard

[![Tests](https://github.com/kakasolg/litert-precision-guard/actions/workflows/test.yml/badge.svg)](https://github.com/kakasolg/litert-precision-guard/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

An automated diagnostic CLI tool to detect **silent numerical failures** (zero vectors, NaN/Inf poisoning, and precision underflow) when running quantized models across LiteRT hardware delegates.

---

## The Problem: The Silent Failure Gap

This tool is **not** about "how to run LLMs on mobile devices"—ecosystems like Apple's [MLX](https://github.com/ml-explore/mlx-swift-examples) already run open models on Apple Silicon and iOS natively and effectively.

Instead, the problem space is specific and critical:

> **When deploying vendor-distributed, pre-quantized model files (`.tflite`) across multiple hardware backends (CPU vs. GPU delegates), there is currently no automated way to verify whether a specific delegate backend produces silently corrupted outputs without raising an explicit runtime error.**

In production edge deployments, quantized models running on GPU delegates (Metal, OpenCL, WebGPU) frequently suffer from **silent shader underflow**:
- The model initializes successfully without errors.
- Inference returns exit code `0` and non-empty output tensors.
- **The output values are silently all zeros (`[0.0, 0.0, ...]`), corrupted with `NaN`s, or completely degraded in semantic representation.**

### Public Precedents & Cross-Platform Empirical Evidence

This failure pattern has now been confirmed across **at least five independent environments and GPU architectures** (spanning Apple Metal, Apple Core AI, WebGPU, and physical Android devices from multiple GPU vendors), each made correct again by running at FP32 (precision enforcement in three separate language bindings: Python `enforce_f32`, iOS Swift `kLiteRtDelegatePrecisionFp32`, and Android Kotlin `CompiledModel.GpuOptions.precision`, or an fp32 model/dtype):

1. **[google-ai-edge/LiteRT#7693](https://github.com/google-ai-edge/LiteRT/issues/7693)** (macOS / Metal): Quantized `EmbeddingGemma` with default GPU delegate precision outputs silent zero vectors on Metal backends, whereas enforcing FP32 precision (`enforce_f32=True`) restores correct embeddings.
2. **[huggingface/transformers.js#1728](https://github.com/huggingface/transformers.js/issues/1728)** (WebGPU / ONNX): `embeddinggemma-300m-ONNX` with `q4`/`q8` on WebGPU returns embeddings with normal-looking norms that are nearly orthogonal to the correct ones (cosine ≈ 0 to WASM and to the original model), with no error; `fp32` on WebGPU is correct. Zero/NaN checks alone do not catch this — a cosine comparison against a trusted output does (see [Checking against a reference](#checking-against-a-reference)).
3. **Android Physical Device Empirical Validation** *(confirmed independently via the public Android LiteRT Kotlin/Java API `com.google.ai.edge.litert`, outside this tool's Python scope)*:
   Testing pre-quantized `EmbeddingGemma` across different physical mobile SoC architectures demonstrated that two distinct GPU vendors exhibited two distinct failure modes under default precision—both completely resolved by forcing FP32 (`Precision.FP32`):

   | Hardware / SoC Class | GPU Architecture | Default Precision Symptom | FP32 Enforced Result | GPU FP32 speed vs. single-thread CPU (`Interpreter`) | GPU FP32 speed vs. 4-thread CPU (`CompiledModel`) |
   | :--- | :--- | :--- | :--- | :--- | :--- |
   | Tensor G5-based device | PowerVR | **Output norm `0.0000`** (Silent zero vector) | Norm `1.0000`, Cosine `0.999100` (Healthy) | **0.56x** *(slower)* | not measured *(expected to be slower still)* |
   | Snapdragon 8+ Gen 1 device | Adreno 730 | **Output norm `NaN`** (Numerical instability) | Norm `1.0000`, Cosine `0.9988–0.9993` (Healthy) | 1.53x *(faster)* | **0.73–0.80x** *(slower: 135 ms vs. 98–108 ms median)* |

   > **Correction (2026-11):** an earlier version of this table reported that FP32-enforced GPU on Adreno 730 was 1.53x
   > *faster* than CPU and concluded that GPU FP32 speed is vendor-dependent. That CPU baseline was the classic
   > `Interpreter` path, which on LiteRT 2.1.x runs XNNPACK on a single core regardless of the thread setting. Re-measured
   > against `CompiledModel` CPU with 4 threads (bit-identical vectors), FP32 GPU is **slower on both devices we tested**.
   > The correctness finding is unchanged: default precision silently breaks the output, FP32 fixes it.
   >
   > Re-checked on LiteRT **2.3.2** (`litert` + `litert-gpu`, Adreno 730, the official `seq256` mixed-precision
   > EmbeddingGemma file): default GPU precision still returns NaN for every output value (23,040/23,040), FP32 is
   > healthy (cosine 0.9988 to CPU) at 135 ms vs. 108 ms for 4-thread CPU.
   >
   > **Takeaway for deployment:** enforce FP32 if you use the GPU delegate at all, and benchmark it against a
   > multi-threaded CPU baseline (`CompiledModel` + `CpuOptions`), not against `Interpreter`. A slow single-thread
   > baseline can make the GPU look like a win when it is not.

4. **Apple Core AI (iOS 27 / macOS 27)**: an independent Core AI export of EmbeddingGemma 300M reports the same failure
   on Apple's newer runtime: its fp16 variants (both a full cast and mixed-precision autocast) "produce NaN embeddings
   on-device" because Gemma 3 activations overflow half precision, so only an fp32 build is shipped
   ([model card](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-CoreAI)). Not reproduced by us.

`litert-precision-guard` automates this verification on host/CI machines as a reproducible diagnostic tool and gatekeeper.

---

## How It Works

```
                        ┌───────────────────────────────┐
                        │   Input Model (.tflite)       │
                        └──────────────┬────────────────┘
                                       │
                    Generate Deterministic Inputs
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│  CPU Baseline   │           │   GPU Default   │           │ GPU FP32 Enforced│
│   (XNNPACK)     │           │ (FP16 / Mixed)  │           │ (enforce_f32)   │
└────────┬────────┘           └────────┬────────┘           └────────┬────────┘
         │                             │                             │
         │ (Ref Output)                │ (Target Output)             │ (Target Output)
         └──────────────┬──────────────┴──────────────┬──────────────┘
                        ▼                             ▼
                 [Metrics Check]               [Metrics Check]
                 - Cosine Sim vs Baseline      - Cosine Sim vs Baseline
                 - Silent Zero Check           - Silent Zero Check
                 - NaN / Inf Check             - NaN / Inf Check
```

1. **Deterministic Probing**: Inspects model input signatures and generates type-safe synthetic inputs (token IDs for integers, normalized distributions for floats).
2. **Differential Comparison**: Runs the CPU reference engine (XNNPACK) to establish ground truth, then executes target delegate configurations (`gpu`, `gpu-fp32`).
3. **Automated Anomaly Detection**:
   - `SILENT_ZERO_VECTOR`: Flags outputs with vanishing L2 norm ($\|\mathbf{y}\| < 10^{-4}$) where CPU baseline is healthy.
   - `NUMERICAL_INSTABILITY`: Flags `NaN` or `Inf` values.
   - `PRECISION_DEGRADATION`: Flags cosine similarity drop below threshold (e.g. $< 0.99$).
4. **CI/CD Exit Code**: Returns `0` if all evaluated backends pass, `1` if any silent failure is detected.

---

## Installation

`litert-precision-guard` maintains a minimal dependency footprint (`ai-edge-litert` and `numpy`):

```bash
pip install ai-edge-litert numpy
pip install -e .
```

---

## Usage

### Quick Start

Run verification against CPU, default GPU delegate, and FP32-enforced GPU delegate:

```bash
litert-precision-guard path/to/model.tflite
```

### Checking against a reference

The default run compares each GPU configuration with the CPU run. That cannot catch two cases:

- **The CPU result is wrong too** (a conversion or quantization error, or a runtime path that is silently wrong
  on every backend). Then CPU and GPU agree with each other and both are wrong.
- **Outputs from another runtime** (a phone, ONNX Runtime, Core ML / Core AI, a browser) that this tool cannot run.

For both, save expected outputs from a trusted source (for example the original PyTorch model) for a fixed set of
real inputs, and compare against them:

```bash
# run the .tflite on real inputs; every backend, the CPU included, is also compared with the reference
litert-precision-guard model.tflite --inputs inputs.npz --reference reference.npz

# compare outputs produced elsewhere; no model is run
litert-precision-guard --outputs device_outputs.npz --reference reference.npz --min-cosine-similarity 0.995
```

`.npz` keys are tensor names; with a single array on each side the names do not have to match. A failure here is
reported as `REFERENCE_MISMATCH`. For a batch output of shape `[N, D]` (e.g. N embeddings), the cosine of every row is
checked and the worst row is shown with `*`, so a few wrong rows cannot hide in a good overall cosine.

Example: EmbeddingGemma-300M seq64 int8 `.tflite` (run with the classic `Interpreter`) against the original PyTorch
model on 8 Korean/English texts:

```
Backend                  | Status | Cosine Sim  | Norm (Ref/Tgt)     | Max Diff   | Diagnostics
external                 | PASS   | 0.997503*   | 2.83 / 2.83        | 0.008726   | OK
* cosine of the worst row in a batch output
```

> **int64 inputs:** the LiteRT `CompiledModel` Python API (`TensorBuffer`) does not accept int64 tensors, so models
> with int64 inputs (common for token IDs) cannot be run by this tool. Re-export with int32 inputs, or run the model
> elsewhere and use `--outputs`.

### Example Output

Evaluating `EmbeddingGemma-300M` (`embeddinggemma-300M_seq256_mixed-precision.tflite`):

```text
=======================================================================
        LiteRT Delegate Precision Guard (Differential Verifier)        
=======================================================================

Model Path: embeddinggemma-300M_seq256_mixed-precision.tflite
Signature:  embed_256
Inputs:     {'text_batch': [1, 256]}
Thresholds: Min Cosine Sim = 0.9900, Min Norm = 0.0001

Backend                  | Status   | Cosine Sim  | Norm (Ref/Tgt)     | Max Diff   | Diagnostics
-------------------------+----------+-------------+--------------------+------------+---------------------
cpu                      | PASS     | 1.000000    | 1.00 / 1.00        | 0.000000   | OK (68.4 ms)
gpu (default / FP16)     | FAIL     | 0.000000    | 1.00 / 0.00        | 0.175859   | SILENT_ZERO_VECTOR
gpu (FP32 enforced)      | PASS     | 0.992228    | 1.00 / 1.00        | 0.015013   | OK (25.9 ms)

-----------------------------------------------------------------------

[DIAGNOSTIC FINDING]
  * Detected SILENT ZERO-VECTOR / VANISHING OUTPUT on default GPU delegate.
  * Root cause pattern: Shader precision underflow during quantized tensor computation.
  * Known public precedent: google-ai-edge/LiteRT#7693 & huggingface/transformers.js#1728
  * Remediation verified: Forcing FP32 on GPU delegate restored healthy cosine similarity (PASS).

>>> VERIFICATION FAILED: Silent errors or precision regressions detected. <<<
```

### CLI Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str` | *required* | Path to `.tflite` model file |
| `--backends` | `str` | `cpu,gpu,gpu-fp32` | Comma-separated backends to evaluate |
| `--min-cosine-similarity` | `float` | `0.99` | Cosine similarity threshold vs CPU |
| `--min-norm` | `float` | `1e-4` | L2 norm threshold for zero vector detection |
| `--seed` | `int` | `42` | Random seed for deterministic inputs |
| `--signature` | `str` | `None` | Specific signature key to execute |
| `--inputs` | `str` | `None` | `.npz` of real inputs instead of synthetic ones |
| `--reference` | `str` | `None` | `.npz` of expected outputs for `--inputs` (or for `--outputs`) |
| `--outputs` | `str` | `None` | `.npz` of outputs produced elsewhere; compared with `--reference`, no model is run |
| `--json-output` | `str` | `None` | Export structured JSON report |
| `-v`, `--verbose` | `flag` | `False` | Detailed tensor-level output |

### CI / CD Integration (GitHub Actions)

Use `litert-precision-guard` as a gatekeeper in your model build / release pipelines:

```yaml
name: Model Precision Verification Gate

on: [push, pull_request]

jobs:
  verify-model-precision:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install ai-edge-litert numpy
          pip install .
          
      - name: Run Precision Guard
        run: |
          # Fails pipeline if model produces silent zero-vectors or degraded embeddings
          litert-precision-guard models/embedding.tflite \
            --backends cpu,gpu-fp32 \
            --min-cosine-similarity 0.99 \
            --json-output artifacts/precision-report.json

      - name: Upload Report Artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: precision-report
          path: artifacts/precision-report.json
```

---

## Python API Usage

You can also use the verifier programmatically within Python test suites:

```python
from litert_precision_guard import PrecisionVerifier

verifier = PrecisionVerifier(
    model_path="path/to/model.tflite",
    min_cosine_similarity=0.99,
)

report = verifier.verify()

if not report.all_passed:
    for name, backend_report in report.backend_reports.items():
        if not backend_report.passed:
            print(f"Backend {name} failed: {backend_report.failure_reasons}")
```

---

## Future Roadmap

- **Multi-Runtime Extensibility**: Pluggable runner architecture for ExecuTorch (`.pte`), ONNX Runtime (CUDA / DirectML / WebGPU), and CoreML.
- **Dynamic Layer-by-Layer Probing**: Isolate the exact operator / sub-graph where numerical precision degrades to zero.
- **Automated Calibration Profile Suggestion**: Recommend per-operator FP32 fallback lists in `.tflite` quantization recipes.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
