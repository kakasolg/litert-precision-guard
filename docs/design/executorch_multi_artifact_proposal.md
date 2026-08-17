# Proposal: Multi-Artifact Differential Verification & ExecuTorch Support

## 1. Executive Summary & Feasibility Spike Results

This proposal outlines the architectural evolution of `litert-precision-guard` into a generalized **Precision Guard** capable of comparing both **single-artifact multi-delegate models** (LiteRT) and **multi-artifact pre-delegated models** (ExecuTorch `.pte`, ONNX Runtime, etc.).

A feasibility spike conducted on macOS (Apple Silicon) verified:
1. **XNNPACK vs. CoreML Verification in ExecuTorch**:
   - Successfully loaded and ran pre-delegated `.pte` files via ExecuTorch Python Runtime (`executorch.runtime.Runtime`).
   - Differential comparison using our existing `metrics.py` yielded **Cosine Similarity `0.99999968`** and confirmed zero-copy numpy interoperability.
2. **Key Architectural Difference**:
   - **LiteRT**: Delegate selection happens dynamically at *runtime* via flags (`cpu`, `gpu_enforce_f32`) on a single `.tflite` file.
   - **ExecuTorch**: Delegate partitioning is baked into the `.pte` artifact at *export time* (`to_backend(CoreMLPartitioner())`).
3. **Core Design Principle**:
   - **The tool NEVER exports or compiles models.** It strictly acts as a runtime verification engine for already-exported deployment artifacts. This prevents bloating the tool with massive compiler/torch dependencies.

---

## 2. Dependency Footprint Analysis

| Package Component | Download Size | Installed Footprint | Role |
| :--- | :--- | :--- | :--- |
| **`ai-edge-litert` (Core)** | ~18 MB | ~55 MB | Default LiteRT runtime |
| **`executorch` (Optional Extra)** | ~16.7 MB | ~65 MB (runtime only) | ExecuTorch runtime runner |
| **`torch` (Export-only dependency)** | ~111 MB | ~450 MB | **Excluded from Guard core** |

### Recommendation: Optional Extras
To preserve the ultra-lightweight installation (`pip install litert-precision-guard`), ExecuTorch runtime support will be provided as an optional extra:
```bash
pip install litert-precision-guard              # LiteRT only (default, lightweight)
pip install litert-precision-guard[executorch]  # Adds ExecuTorch runtime runner
```

---

## 3. CLI Interface Generalization Specification

The CLI must seamlessly support both single-file runtime delegation and multi-artifact differential comparison **without breaking existing LiteRT workflows**.

### Mode A: Single-Model Dynamic Delegate Mode (Backward-Compatible LiteRT)
```bash
# Existing syntax remains 100% functional
precision-guard model.tflite --backends cpu,gpu,gpu-fp32
```

### Mode B: Multi-Artifact Comparison Mode (ExecuTorch / Cross-Runtime)
Enables comparing multiple pre-exported `.pte` or `.tflite` artifacts where each file represents a distinct backend:

```bash
# Explicit baseline + target artifact flags
precision-guard \
  --baseline xnnpack=models/model_xnnpack.pte \
  --target coreml=models/model_coreml.pte \
  --min-cosine-similarity 0.99
```

### Mode C: Generalized Artifact Flag (Unified Key-Value Syntax)
```bash
# Key-Value artifact definition with role prefix
precision-guard \
  --artifact baseline:xnnpack=models/model_xnnpack.pte \
  --artifact target:coreml=models/model_coreml.pte \
  --artifact target:gpu_fp32=models/model.tflite:gpu-fp32
```

---

## 4. Internal Architecture & Runner Abstraction

```
                            ┌──────────────────────────────┐
                            │      PrecisionVerifier       │
                            └──────────────┬───────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
             ┌─────────────────────┐               ┌─────────────────────┐
             │    LiteRTRunner     │               │  ExecuTorchRunner   │
             │   (.tflite model)   │               │    (.pte model)     │
             └──────────┬──────────┘               └──────────┬──────────┘
                        │                                     │
                        ▼ (np.ndarray dict)                   ▼ (np.ndarray dict)
             ┌───────────────────────────────────────────────────────────┐
             │                   metrics.py (Core)                       │
             │  - Cosine Similarity Check (>= 0.99)                      │
             │  - Silent Zero-Vector Anomaly Check (||y|| < 1e-4)        │
             │  - Numerical Instability Check (NaN / Inf)                │
             └─────────────────────────────┬─────────────────────────────┘
                                           │
                                           ▼
             ┌───────────────────────────────────────────────────────────┐
             │                   report.py (Renderer)                    │
             │  - Unified Terminal Diagnostic Table                      │
             │  - CI JSON Export & Exit Code Management (0 / 1)          │
             └───────────────────────────────────────────────────────────┘
```

### Runner Base Interface (`runners/base.py`)
```python
class BaseRunner(ABC):
    @abstractmethod
    def initialize(self) -> None:
        ...

    @abstractmethod
    def get_input_specs(self) -> Dict[str, Dict[str, Any]]:
        ...

    @abstractmethod
    def run(self, inputs: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        ...
```

### ExecuTorch Runner (`runners/executorch_runner.py`)
```python
class ExecuTorchRunner(BaseRunner):
    def __init__(self, pte_path: str, method_name: str = "forward"):
        self.pte_path = pte_path
        self.method_name = method_name
        self._method = None

    def initialize(self) -> None:
        from executorch.runtime import Runtime
        runtime = Runtime.get()
        program = runtime.load_program(self.pte_path)
        self._method = program.load_method(self.method_name)

    def run(self, inputs: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        import torch
        # Convert numpy arrays to torch tensors (zero-copy when possible)
        tensor_inputs = [torch.from_numpy(arr) for arr in inputs.values()]
        
        t0 = time.perf_counter()
        outputs = self._method.execute(tensor_inputs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Convert back to numpy arrays
        out_dict = {
            f"output_{i}": out.detach().cpu().numpy()
            for i, out in enumerate(outputs)
        }
        return out_dict, latency_ms
```

---

## 5. Platform Limitations Discovered During Spike

| Backend | Platform Execution Status | Primary Empirical Evidence (Observed Error) | Secondary Context / Documentary Evidence |
| :--- | :--- | :--- | :--- |
| **XNNPACK** | **Verified (PASS)** | Model loaded and executed successfully (`output shape: [1, 8]`) | Built-in portable CPU backend in wheel |
| **CoreML** | **Verified (PASS)** | Model loaded and executed successfully via CoreML ANE/GPU (`Cosine Sim: 0.99999968` vs XNNPACK) | Built-in Apple backend library in macOS wheel |
| **Vulkan / VGF** | **Blocked on macOS Host** | • `[VGFBackend.cpp:827] Volk failed to initialize`<br>• `[VGFBackend.cpp:443] Failed to initialize the Vulkan device error 0xFFFFFFFD`<br>• `[method.cpp:94] Backend VgfBackend is not available.`<br>• `[method.cpp:89] Backend VulkanBackend is not registered.` | ExecuTorch RuntimeError 0x20은 `error.h` 정의상 `Error::NotFound` (0x10이 아니라 0x20)이며, "not supported"가 아니라 "리소스/드라이버를 찾지 못함"을 뜻한다. 즉 이 코드 자체가 "플랫폼 미지원"을 명시하는 게 아니라, VulkanBackend/VgfBackend가 런타임에 등록되지 못했다는 신호일 뿐이다. 실질적 원인은 macOS 시스템에 Vulkan 로더/ICD(MoltenVK 등)가 기본 존재하지 않아 Volk 초기화에 실패한 것으로 파악된다. |
| **QNN (Qualcomm)** | **Blocked on macOS Host** | • `ModuleNotFoundError: No module named 'executorch.backends.qualcomm.python'`<br>• `executorch/backends/qualcomm/__init__.py` 내 `is_linux_x86()` 하드코딩 분기 | ExecuTorch 공식 문서(`docs/source/backends-qualcomm.md`)는 정확히 이렇게 말하지 않는다 — 실제 원문은 "The QNN Backend is currently verified on the following Linux host operating systems: Ubuntu 22.04 LTS (x64), CentOS Stream 9, Windows Subsystem for Linux (WSL) with Ubuntu 22.04"이며, macOS/Darwin에 대한 명시적 비지원 문구는 없다. 다만 검증된 호스트 목록에 macOS가 없다는 사실 자체는 유효한 근거다. |

### Technical Analysis of 1st-Order Findings

1. **Vulkan / VGF Runtime**:
   - **Primary Evidence**: The Python wheel provides AoT partitioners (`VulkanPartitioner`, `VgfPartitioner`), and `.pte` export completes successfully. However, at runtime execution, the Volk meta-loader fails (`Volk failed to initialize`), causing device initialization error `0xFFFFFFFD`.
   - **Error Code Clarification**: ExecuTorch `RuntimeError 0x20` maps directly to `Error::NotFound = 0x20` in `runtime/core/error.h`. It indicates that the dynamic backend symbol or device entry point was not found in the runtime registry.

2. **Qualcomm QNN Backend**:
   - **Primary Evidence**: The Python extension `executorch.backends.qualcomm.python` is missing in macOS wheels, and setup scripts explicitly branch on `is_linux_x86()`.
   - **Secondary Context**: The verified host environments in the official documentation comprise x86_64 Linux distributions (Ubuntu, CentOS, WSL).

---

## 6. Implementation Roadmap (When Authorized)

1. **Phase 1**: Extract `BaseRunner` abstract class and refactor `LiteRTRunner`.
2. **Phase 2**: Add `ExecuTorchRunner` and make `executorch` an optional package dependency in `pyproject.toml`.
3. **Phase 3**: Update `cli.py` to parse `--baseline` and `--target` artifact pairs alongside positional single-model syntax.
4. **Phase 4**: Add test suite for multi-artifact verification and update documentation.
