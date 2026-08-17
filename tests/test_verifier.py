"""Unit tests for LiteRT Precision Guard metrics, models, and anomaly detection."""

import json
import os
import tempfile
import unittest
import numpy as np

from litert_precision_guard.cli import create_parser
from litert_precision_guard.metrics import compute_tensor_metrics
from litert_precision_guard.models import (
    BackendConfig,
    BackendReport,
    FailureReason,
    TensorComparison,
    VerificationReport,
)
from litert_precision_guard.report import export_report_to_json
from litert_precision_guard.verifier import PrecisionVerifier


class TestPrecisionMetrics(unittest.TestCase):

    def test_identical_tensors_pass(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        comp = compute_tensor_metrics("test_out", baseline=arr, target=arr)
        self.assertTrue(comp.passed)
        self.assertAlmostEqual(comp.cosine_similarity, 1.0, places=5)
        self.assertEqual(len(comp.failure_reasons), 0)
        self.assertFalse(comp.is_zero_vector)

    def test_silent_zero_vector_detected(self):
        baseline = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        comp = compute_tensor_metrics("test_out", baseline=baseline, target=target)
        self.assertFalse(comp.passed)
        self.assertTrue(comp.is_zero_vector)
        self.assertIn(FailureReason.SILENT_ZERO_VECTOR, comp.failure_reasons)
        self.assertEqual(comp.cosine_similarity, 0.0)

    def test_nan_instability_detected(self):
        baseline = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        target = np.array([1.0, np.nan, 3.0], dtype=np.float32)

        comp = compute_tensor_metrics("test_out", baseline=baseline, target=target)
        self.assertFalse(comp.passed)
        self.assertTrue(comp.has_nan)
        self.assertIn(FailureReason.NUMERICAL_INSTABILITY, comp.failure_reasons)

    def test_precision_degradation_below_threshold(self):
        baseline = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        target = np.array([0.5, 0.5, 0.0], dtype=np.float32)  # cos sim ~ 0.707

        comp = compute_tensor_metrics("test_out", baseline=baseline, target=target, min_cosine_similarity=0.99)
        self.assertFalse(comp.passed)
        self.assertIn(FailureReason.PRECISION_DEGRADATION, comp.failure_reasons)
        self.assertLess(comp.cosine_similarity, 0.99)


class TestVerifierCLIAndModels(unittest.TestCase):

    def test_parse_backend_names(self):
        configs = PrecisionVerifier.parse_backend_names("cpu,gpu,gpu-fp32")
        self.assertEqual(len(configs), 3)
        self.assertEqual(configs[0].accelerator, "cpu")
        self.assertEqual(configs[1].accelerator, "gpu")
        self.assertFalse(configs[1].enforce_f32)
        self.assertEqual(configs[2].accelerator, "gpu")
        self.assertTrue(configs[2].enforce_f32)

    def test_json_export(self):
        report = VerificationReport(
            model_path="/fake/model.tflite",
            signature_key="serving_default",
            input_shapes={"input:0": [1, 256]},
            min_cosine_similarity=0.99,
            min_norm=1e-4,
            all_passed=False,
            backend_reports={
                "cpu": BackendReport(
                    config=BackendConfig(name="cpu", accelerator="cpu"),
                    passed=True,
                    latency_ms=12.5,
                ),
                "gpu": BackendReport(
                    config=BackendConfig(name="gpu", accelerator="gpu", enforce_f32=False),
                    passed=False,
                    latency_ms=8.2,
                    failure_reasons=[FailureReason.SILENT_ZERO_VECTOR],
                ),
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "report.json")
            export_report_to_json(report, json_path)
            self.assertTrue(os.path.exists(json_path))

            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertFalse(loaded["all_passed"])
            self.assertIn("gpu", loaded["backends"])
            self.assertEqual(loaded["backends"]["gpu"]["failure_reasons"], ["SILENT_ZERO_VECTOR"])


if __name__ == "__main__":
    unittest.main()
