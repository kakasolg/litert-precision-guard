"""Unit tests for LiteRT Precision Guard metrics, models, and anomaly detection."""

import contextlib
import io
import json
import os
import tempfile
import unittest
import numpy as np

from litert_precision_guard.cli import create_parser, main as cli_main
from litert_precision_guard.metrics import compute_tensor_metrics
from litert_precision_guard.models import (
    BackendConfig,
    BackendReport,
    FailureReason,
    TensorComparison,
    VerificationReport,
)
from litert_precision_guard.report import export_report_to_json
from litert_precision_guard.verifier import PrecisionVerifier, compare_to_reference, verify_output_files


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


class TestReferenceComparison(unittest.TestCase):
    """Outputs that keep a normal norm but point the wrong way, compared against an external reference."""

    def setUp(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(size=(8, 64))
        self.ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)

    def test_bad_rows_hidden_in_overall_cosine_are_caught(self):
        out = self.ref.copy()
        out[3] = -out[3]  # one wrong row; the flattened cosine is still 0.75
        comp = compute_tensor_metrics("emb", baseline=self.ref, target=out, min_cosine_similarity=0.5)
        self.assertGreater(comp.cosine_similarity, 0.5)
        self.assertLess(comp.min_row_cosine, 0.0)
        self.assertFalse(comp.passed)
        self.assertIn(FailureReason.PRECISION_DEGRADATION, comp.failure_reasons)

    def test_norm_preserving_wrong_vectors_flagged_as_reference_mismatch(self):
        rng = np.random.default_rng(1)
        wrong = self.ref + rng.normal(scale=0.08, size=self.ref.shape)
        wrong /= np.linalg.norm(wrong, axis=1, keepdims=True)  # unit norm, no NaN, no zero rows
        comps = compare_to_reference({"emb": self.ref}, {"output_0": wrong.astype(np.float32)})
        self.assertEqual(len(comps), 1)
        c = comps[0]
        self.assertFalse(c.passed)
        self.assertFalse(c.has_nan)
        self.assertFalse(c.is_zero_vector)
        self.assertIn(FailureReason.REFERENCE_MISMATCH, c.failure_reasons)

    def test_matching_outputs_pass_and_reshape(self):
        comps = compare_to_reference({"emb": self.ref.reshape(1, 8, 64)}, {"emb": self.ref.astype(np.float32)})
        self.assertTrue(comps[0].passed)

    def test_missing_name_and_size_mismatch_raise(self):
        with self.assertRaises(ValueError):
            compare_to_reference({"a": self.ref, "b": self.ref}, {"a": self.ref, "c": self.ref})
        with self.assertRaises(ValueError):
            compare_to_reference({"a": self.ref}, {"a": self.ref[:4]})

    def test_verify_output_files_and_cli(self):
        bad = self.ref.copy()
        bad[0] = np.roll(bad[0], 1)
        with tempfile.TemporaryDirectory() as d:
            rp, gp, bp = (os.path.join(d, n) for n in ("ref.npz", "good.npz", "bad.npz"))
            np.savez(rp, emb=self.ref)
            np.savez(gp, emb=self.ref.astype(np.float32))
            np.savez(bp, emb=bad)
            self.assertTrue(verify_output_files(rp, gp).all_passed)
            report = verify_output_files(rp, bp)
            self.assertFalse(report.all_passed)
            self.assertIn(FailureReason.REFERENCE_MISMATCH, report.backend_reports["outputs"].failure_reasons)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["--outputs", gp, "--reference", rp]), 0)
                self.assertEqual(cli_main(["--outputs", bp, "--reference", rp]), 1)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli_main(["--outputs", gp]), 1)
                self.assertEqual(cli_main(["model.tflite", "--reference", rp]), 1)

    def test_verify_requires_inputs_with_reference(self):
        v = PrecisionVerifier(model_path="unused.tflite")
        with self.assertRaises(ValueError):
            v.verify(reference_outputs={"emb": self.ref})


if __name__ == "__main__":
    unittest.main()
