"""LiteRT model execution runners for CPU and GPU delegates."""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.gpu_options import GpuOptions
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    from ai_edge_litert.options import Options
    LITERT_AVAILABLE = True
except ImportError:
    LITERT_AVAILABLE = False


class LiteRTRunner:
    """Wrapper for executing a LiteRT model under specific hardware accelerators and precision settings."""

    def __init__(
        self,
        model_path: str,
        accelerator: str = "cpu",
        enforce_f32: bool = False,
        signature_key: Optional[str] = None,
    ):
        if not LITERT_AVAILABLE:
            raise RuntimeError(
                "ai-edge-litert package is not installed. Please install it with `pip install ai-edge-litert`."
            )

        self.model_path = model_path
        self.accelerator_type = accelerator.lower()
        self.enforce_f32 = enforce_f32
        self.signature_key = signature_key
        self._model: Optional[CompiledModel] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._signature_index: int = 0

    def initialize(self) -> None:
        """Instantiates the LiteRT CompiledModel with the chosen accelerator configuration."""
        if self.accelerator_type == "cpu":
            opts = Options(hardware_accelerators=HardwareAccelerator.CPU)
        elif self.accelerator_type == "gpu":
            gpu_opts = GpuOptions(enforce_f32=self.enforce_f32)
            opts = Options(
                hardware_accelerators=HardwareAccelerator.GPU | HardwareAccelerator.CPU,
                gpu_options=gpu_opts,
            )
        else:
            raise ValueError(f"Unsupported accelerator: {self.accelerator_type}")

        self._model = CompiledModel.from_file(self.model_path, options=opts)

        # Inspect signatures
        sig_list = self._model.get_signature_list()
        if not sig_list:
            # Fallback to index 0 default signature
            self._signature_index = 0
            self._input_names = []
            self._output_names = []
        else:
            if self.signature_key and self.signature_key in sig_list:
                selected_key = self.signature_key
            else:
                selected_key = list(sig_list.keys())[0]
            self.signature_key = selected_key
            self._signature_index = self._model.get_signature_index(selected_key)
            self._input_names = sig_list[selected_key].get("inputs", [])
            self._output_names = sig_list[selected_key].get("outputs", [])

    def get_input_specs(self) -> Dict[str, Dict[str, Any]]:
        """Returns input tensor details (name, shape, dtype)."""
        if self._model is None:
            self.initialize()

        if self.signature_key:
            try:
                return self._model.get_input_tensor_details(self.signature_key)
            except Exception:
                pass
        return {}

    def get_output_specs(self) -> Dict[str, Dict[str, Any]]:
        """Returns output tensor details (name, shape, dtype)."""
        if self._model is None:
            self.initialize()

        if self.signature_key:
            try:
                return self._model.get_output_tensor_details(self.signature_key)
            except Exception:
                pass
        return {}

    def run(
        self,
        inputs: Dict[str, np.ndarray],
        warmup: int = 1,
        iterations: int = 1,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Runs model inference with given input arrays and returns (outputs_dict, avg_latency_ms)."""
        if self._model is None:
            self.initialize()

        assert self._model is not None

        # Allocate input and output buffers
        input_buffers = self._model.create_input_buffers(self._signature_index)
        output_buffers = self._model.create_output_buffers(self._signature_index)

        # Populate input buffers
        if self._input_names and len(self._input_names) == len(input_buffers):
            for i, name in enumerate(self._input_names):
                if name in inputs:
                    input_buffers[i].write(inputs[name])
                else:
                    # Try fallback by index
                    first_val = list(inputs.values())[i]
                    input_buffers[i].write(first_val)
        else:
            # Match by order
            for i, (k, arr) in enumerate(inputs.items()):
                if i < len(input_buffers):
                    input_buffers[i].write(arr)

        # Warmup
        for _ in range(warmup):
            self._model.run_by_index(self._signature_index, input_buffers, output_buffers)

        # Timed execution
        start_t = time.perf_counter()
        for _ in range(iterations):
            self._model.run_by_index(self._signature_index, input_buffers, output_buffers)
        end_t = time.perf_counter()

        avg_latency_ms = ((end_t - start_t) / max(1, iterations)) * 1000.0

        # Read output buffers
        outputs: Dict[str, np.ndarray] = {}
        for i, out_buf in enumerate(output_buffers):
            details = out_buf.get_tensor_details()
            shape = details.get("shape", [])
            dtype_str = details.get("dtype", "float32")

            # Map dtype string to numpy dtype
            if dtype_str == "float32":
                np_dtype = np.float32
            elif dtype_str == "float16":
                np_dtype = np.float16
            elif dtype_str == "int32":
                np_dtype = np.int32
            elif dtype_str == "int8":
                np_dtype = np.int8
            elif dtype_str == "uint8":
                np_dtype = np.uint8
            else:
                np_dtype = np.float32

            num_elements = int(np.prod(shape)) if shape else 1
            raw_arr = out_buf.read(num_elements, np_dtype)
            arr = raw_arr.reshape(shape) if shape else raw_arr

            out_name = self._output_names[i] if i < len(self._output_names) else f"output_{i}"
            outputs[out_name] = np.copy(arr)

        return outputs, avg_latency_ms
