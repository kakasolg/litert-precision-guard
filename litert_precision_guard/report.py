"""CLI formatting and report rendering for LiteRT Precision Guard."""

import json
import os
import sys
from typing import Optional, TextIO

from litert_precision_guard.models import FailureReason, VerificationReport

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GRAY = "\033[90m"


def render_report_to_terminal(report: VerificationReport, verbose: bool = False, file: Optional[TextIO] = None) -> None:
    """Renders a comprehensive diagnostic report to standard output."""
    file = file or sys.stdout
    file.write(f"\n{BOLD}{CYAN}======================================================================={RESET}\n")
    file.write(f"{BOLD}{CYAN}        LiteRT Delegate Precision Guard (Differential Verifier)        {RESET}\n")
    file.write(f"{BOLD}{CYAN}======================================================================={RESET}\n\n")

    file.write(f"{BOLD}Model Path:{RESET} {report.model_path}\n")
    if report.signature_key:
        file.write(f"{BOLD}Signature:{RESET}  {report.signature_key}\n")
    file.write(f"{BOLD}Inputs:{RESET}     {report.input_shapes}\n")
    file.write(f"{BOLD}Thresholds:{RESET} Min Cosine Sim = {report.min_cosine_similarity:.4f}, Min Norm = {report.min_norm}\n\n")

    has_multi_tensor = any(len(b.tensor_comparisons) > 1 for b in report.backend_reports.values())

    if has_multi_tensor:
        header = f"{'Backend':<20} | {'Tensor':<14} | {'Status':<6} | {'Cosine Sim':<10} | {'Norm (Ref/Tgt)':<16} | {'Max Diff':<9} | {'Diagnostics'}"
        file.write(f"{BOLD}{header}{RESET}\n")
        file.write(f"{'='*20}-+-{'='*14}-+-{'='*6}-+-{'='*10}-+-{'='*16}-+-{'='*9}-+-{'='*20}\n")
    else:
        header = f"{'Backend':<24} | {'Status':<6} | {'Cosine Sim':<11} | {'Norm (Ref/Tgt)':<18} | {'Max Diff':<10} | {'Diagnostics'}"
        file.write(f"{BOLD}{header}{RESET}\n")
        file.write(f"{'='*24}-+-{'='*6}-+-{'='*11}-+-{'='*18}-+-{'='*10}-+-{'='*20}\n")

    for backend_name, b_report in report.backend_reports.items():
        display_name = b_report.config.display_name
        if not has_multi_tensor and len(display_name) > 23:
            display_name = display_name[:20] + "..."
        elif has_multi_tensor and len(display_name) > 19:
            display_name = display_name[:16] + "..."

        if b_report.error_message:
            status_badge = f"{RED}{BOLD}ERROR{RESET}"
            if has_multi_tensor:
                file.write(f"{display_name:<20} | {'all':<14} | {status_badge:<15} | {'N/A':<10} | {'N/A':<16} | {'N/A':<9} | {RED}{b_report.error_message}{RESET}\n")
            else:
                file.write(f"{display_name:<24} | {status_badge:<15} | {'N/A':<11} | {'N/A':<18} | {'N/A':<10} | {RED}{b_report.error_message}{RESET}\n")
            continue

        for comp in b_report.tensor_comparisons:
            if comp.passed:
                status_badge = f"{GREEN}{BOLD}PASS{RESET} "
            else:
                status_badge = f"{RED}{BOLD}FAIL{RESET} "

            cos_str = f"{comp.cosine_similarity:.6f}"
            if comp.min_row_cosine is not None and comp.min_row_cosine < comp.cosine_similarity:
                cos_str = f"{comp.min_row_cosine:.6f}*"  # worst row of a batch
            norms_str = f"{comp.baseline_norm:.2f} / {comp.target_norm:.2f}"
            max_diff_str = f"{comp.max_abs_diff:.6f}" if comp.max_abs_diff != float("inf") else "inf"

            if comp.passed:
                diag = f"{GREEN}OK ({b_report.latency_ms:.1f} ms){RESET}"
            else:
                reasons = [r.value for r in comp.failure_reasons]
                diag = f"{RED}{', '.join(reasons)}{RESET}"

            t_name = comp.name
            if len(t_name) > 13:
                t_name = t_name[:10] + "..."

            if has_multi_tensor:
                file.write(
                    f"{display_name:<20} | {t_name:<14} | {status_badge:<15} | {cos_str:<10} | {norms_str:<16} | {max_diff_str:<9} | {diag}\n"
                )
            else:
                file.write(
                    f"{display_name:<24} | {status_badge:<15} | {cos_str:<11} | {norms_str:<18} | {max_diff_str:<10} | {diag}\n"
                )

    if any(c.min_row_cosine is not None and c.min_row_cosine < c.cosine_similarity
           for b in report.backend_reports.values() for c in b.tensor_comparisons):
        file.write("\n* cosine of the worst row in a batch output\n")

    file.write(f"\n{BOLD}{CYAN}-----------------------------------------------------------------------{RESET}\n")

    gpu_default = report.backend_reports.get("gpu") or report.backend_reports.get("gpu-default")
    gpu_fp32 = report.backend_reports.get("gpu-fp32")

    has_silent_zero = False
    if gpu_default and not gpu_default.passed:
        for comp in gpu_default.tensor_comparisons:
            if comp.is_zero_vector or FailureReason.SILENT_ZERO_VECTOR in comp.failure_reasons:
                has_silent_zero = True

    if has_silent_zero:
        file.write(f"\n{YELLOW}{BOLD}[DIAGNOSTIC FINDING]{RESET}\n")
        file.write(f"  * Detected {RED}SILENT ZERO-VECTOR / VANISHING OUTPUT{RESET} on default GPU delegate.\n")
        file.write(f"  * Root cause pattern: Shader precision underflow during quantized tensor computation.\n")
        file.write(f"  * Known public precedent: {BOLD}google-ai-edge/LiteRT#7693{RESET} & {BOLD}huggingface/transformers.js#1728{RESET}\n")
        if gpu_fp32 and gpu_fp32.passed:
            file.write(f"  * {GREEN}Remediation verified:{RESET} Forcing FP32 on GPU delegate restored healthy cosine similarity (PASS).\n")
        else:
            file.write(f"  * Recommended action: Test with `--backends cpu,gpu-fp32` (or set `enforce_f32=True` in runtime delegate options).\n")

    file.write(f"\n")
    if report.all_passed:
        what = "the reference" if "outputs" in report.backend_reports else "the CPU baseline"
        file.write(f"{GREEN}{BOLD}>>> VERIFICATION PASSED: All outputs match {what} within thresholds. <<<{RESET}\n\n")
    else:
        file.write(f"{RED}{BOLD}>>> VERIFICATION FAILED: Silent errors or precision regressions detected. <<<{RESET}\n\n")


def export_report_to_json(report: VerificationReport, output_path: str) -> None:
    """Exports verification result into a structured JSON file for CI/CD pipelines."""
    data = report.to_dict()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
