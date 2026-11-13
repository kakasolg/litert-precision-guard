"""CLI Entrypoint and argument parser for LiteRT Precision Guard."""

import argparse
import sys
from typing import List, Optional

from litert_precision_guard.models import BackendConfig
from litert_precision_guard.report import export_report_to_json, render_report_to_terminal
from litert_precision_guard.verifier import PrecisionVerifier, load_npz, verify_output_files


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="litert-precision-guard",
        description="Automated differential diagnostic CLI tool to detect silent zero-vectors, NaN/Inf poisoning, and precision degradation across LiteRT hardware delegates.",
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the .tflite model file to verify (omit when using --outputs).",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default=None,
        help="Optional .npz of real model inputs (name -> array) used instead of synthetic inputs.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Optional .npz of expected outputs for --inputs (e.g. from the original PyTorch model). Every backend, "
             "the CPU included, is also compared against it.",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default=None,
        help="Check an .npz of outputs produced elsewhere (phone, Core AI, ONNX Runtime, browser) against "
             "--reference without running a model.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="cpu,gpu,gpu-fp32",
        help="Comma-separated list of backends to compare against CPU baseline. Options: cpu, gpu, gpu-fp32 (Default: cpu,gpu,gpu-fp32).",
    )
    parser.add_argument(
        "--min-cosine-similarity",
        type=float,
        default=0.99,
        help="Minimum acceptable cosine similarity against CPU baseline (Default: 0.99).",
    )
    parser.add_argument(
        "--min-norm",
        type=float,
        default=1e-4,
        help="Minimum L2 norm threshold below which an output vector is flagged as vanishing/zero (Default: 1e-4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic synthetic input generation (Default: 42).",
    )
    parser.add_argument(
        "--signature",
        type=str,
        default=None,
        help="Optional model signature key (defaults to first signature if omitted).",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to export structured JSON verification results (for CI/CD gates).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed per-tensor verbosity.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.outputs:
        if not args.reference:
            print("--outputs needs --reference", file=sys.stderr)
            return 1
        try:
            report = verify_output_files(args.reference, args.outputs, args.min_cosine_similarity, args.min_norm)
        except Exception as exc:
            print(f"Comparison failed: {exc}", file=sys.stderr)
            return 1
        render_report_to_terminal(report, verbose=args.verbose)
        if args.json_output:
            export_report_to_json(report, args.json_output)
            print(f"JSON report saved to: {args.json_output}")
        return 0 if report.all_passed else 1

    if not args.model_path:
        print("model_path is required unless --outputs is given", file=sys.stderr)
        return 1
    if args.reference and not args.inputs:
        print("--reference needs --inputs (the inputs the reference outputs were produced from)", file=sys.stderr)
        return 1

    try:
        backend_configs = PrecisionVerifier.parse_backend_names(args.backends)
    except ValueError as e:
        print(f"Error parsing --backends: {e}", file=sys.stderr)
        return 1

    verifier = PrecisionVerifier(
        model_path=args.model_path,
        backend_configs=backend_configs,
        min_cosine_similarity=args.min_cosine_similarity,
        min_norm=args.min_norm,
        seed=args.seed,
        signature_key=args.signature,
    )

    try:
        inputs = load_npz(args.inputs) if args.inputs else None
        reference = load_npz(args.reference) if args.reference else None
        report = verifier.verify(custom_inputs=inputs, reference_outputs=reference)
    except Exception as exc:
        print(f"Verification failed with runtime error: {exc}", file=sys.stderr)
        return 1

    # Render terminal table
    render_report_to_terminal(report, verbose=args.verbose)

    # Export JSON if requested
    if args.json_output:
        export_report_to_json(report, args.json_output)
        print(f"JSON report saved to: {args.json_output}")

    # Return exit code: 0 if all passed, 1 if any failed
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
