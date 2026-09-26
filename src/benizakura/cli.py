from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

from benizakura.comparator import SimpleComparator
from benizakura.models import EvaluationCase
from benizakura.runner import EvaluationRunner, MockEvaluator


def load_cases(path: Path) -> List[EvaluationCase]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_cases = data.get("cases", []) if isinstance(data, dict) else data
    cases = []
    for item in raw_cases:
        cases.append(
            EvaluationCase(
                id=item["id"],
                topic=item["topic"],
                question=item["question"],
                candidate_answer=item["candidate_answer"],
                mode=item.get("mode", "STANDARD"),
                rubric=item.get("rubric"),
                metadata=item.get("metadata", {}),
            )
        )
    return cases


def _run_benchmark_cli(sub_argv: list[str]) -> int:
    from benizakura.benchmark import (
        compute_benchmark_hash,
        generate_blinded_annotation_tasks,
        validate_benchmark,
    )

    bench_parser = argparse.ArgumentParser(
        prog="benizakura benchmark",
        description="Benchmark validation, integrity hashing, and blinding utilities.",
    )
    subparsers = bench_parser.add_subparsers(dest="benchmark_command", required=True)

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate benchmark dataset structure and leakage.")
    validate_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("evals/conquer/benchmark_v1.json"),
        help="Path to benchmark JSON (default: evals/conquer/benchmark_v1.json).",
    )

    # hash
    hash_parser = subparsers.add_parser("hash", help="Compute deterministic SHA-256 hash of benchmark.")
    hash_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("evals/conquer/benchmark_v1.json"),
        help="Path to benchmark JSON (default: evals/conquer/benchmark_v1.json).",
    )
    hash_parser.add_argument(
        "--write",
        action="store_true",
        help="Write computed hash to companion .sha256 file.",
    )

    # blind
    blind_parser = subparsers.add_parser("blind", help="Generate double-blind annotation tasks and private key.")
    blind_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("evals/conquer/benchmark_v1.json"),
        help="Path to benchmark JSON (default: evals/conquer/benchmark_v1.json).",
    )
    blind_parser.add_argument(
        "--out-tasks",
        type=Path,
        default=Path("evals/conquer/annotations/blinded_tasks.json"),
        help="Output path for blinded tasks JSON.",
    )
    blind_parser.add_argument(
        "--out-key",
        type=Path,
        default=Path("evals/conquer/annotations/blinding_key_v1.json"),
        help="Output path for private unblinding key JSON.",
    )
    blind_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic presentation swapping.",
    )

    args = bench_parser.parse_args(sub_argv)

    if args.benchmark_command == "validate":
        report = validate_benchmark(args.path)
        print(report.summary())
        return 0 if report.is_valid else 1

    elif args.benchmark_command == "hash":
        sha = compute_benchmark_hash(args.path)
        print(f"SHA256 ({args.path}):\n{sha}")
        if args.write:
            companion_path = args.path.with_suffix(".sha256")
            with open(companion_path, "w", encoding="utf-8") as f:
                f.write(f"{sha}  {args.path}\n")
            print(f"Wrote checksum to {companion_path}")
        return 0

    elif args.benchmark_command == "blind":
        tasks, key_map = generate_blinded_annotation_tasks(args.path, seed=args.seed)
        args.out_tasks.parent.mkdir(parents=True, exist_ok=True)
        args.out_key.parent.mkdir(parents=True, exist_ok=True)

        with open(args.out_tasks, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
        with open(args.out_key, "w", encoding="utf-8") as f:
            json.dump(key_map, f, indent=2)

        print(f"Successfully generated {len(tasks)} double-blind annotation tasks:")
        print(f"  - Blinded Tasks: {args.out_tasks}")
        print(f"  - Private Key:   {args.out_key}")
        return 0

    return 1


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "benchmark":
        return _run_benchmark_cli(argv[1:])

    parser = argparse.ArgumentParser(
        prog="benizakura",
        description="Benizakura: Regression testing change gate for AI behavior.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evals/conquer/cases.json"),
        help="Path to evaluation cases JSON file.",
    )
    parser.add_argument(
        "--demo",
        choices=["pass", "fail", "inconclusive", "gate", "gate-pass", "gate-fail", "gate-inconclusive"],
        default="pass",
        help="Simulate demonstration scenario (pass, fail, inconclusive, gate, gate-pass, gate-fail, gate-inconclusive). Default: pass.",
    )
    parser.add_argument(
        "--baseline-version",
        default="v1",
        help="Baseline version identifier.",
    )
    parser.add_argument(
        "--candidate-version",
        default="v2",
        help="Candidate version identifier.",
    )

    args = parser.parse_args(argv)

    if args.demo in ("gate", "gate-pass", "gate-fail", "gate-inconclusive"):
        return _run_gate_demo(args.demo)

    cases_path = args.cases
    if not cases_path.is_file():
        # Try relative to repo root if run from a different subfolder
        candidate_path = Path(__file__).resolve().parent.parent.parent / args.cases
        if candidate_path.is_file():
            cases_path = candidate_path
        else:
            print(f"Error: Evaluation cases file not found at {args.cases}", file=sys.stderr)
            return 1

    cases = load_cases(cases_path)

    # Setup mock evaluators based on requested demonstration mode
    if args.demo == "fail":
        baseline_evaluator = MockEvaluator(default_score=8.0)
        # Induce a critical drop on the first case
        critical_case_id = cases[0].id if cases else "case-1"
        candidate_evaluator = MockEvaluator(
            default_score=8.0,
            case_overrides={critical_case_id: {"score": 3.5, "feedback": "Critical regression."}},
        )
    elif args.demo == "inconclusive":
        baseline_evaluator = MockEvaluator(default_score=7.0)
        candidate_evaluator = MockEvaluator(default_score=7.05)
    else:  # "pass"
        baseline_evaluator = MockEvaluator(default_score=6.8)
        candidate_evaluator = MockEvaluator(default_score=8.2)

    baseline_runner = EvaluationRunner(baseline_evaluator)
    candidate_runner = EvaluationRunner(candidate_evaluator)

    baseline_run = baseline_runner.run(cases, version=args.baseline_version)
    candidate_run = candidate_runner.run(cases, version=args.candidate_version)

    comparator = SimpleComparator()
    comparison = comparator.compare(baseline_run, candidate_run)

    print("Benizakura")
    print("==========")
    print()
    print(f"Evaluation cases: {len(cases)}")
    print()
    print(f"Baseline:  {comparison.baseline_version} (avg: {comparison.baseline_avg_score:.2f})")
    print(f"Candidate: {comparison.candidate_version} (avg: {comparison.candidate_avg_score:.2f})")
    print()
    print(f"Verdict: {comparison.verdict.value}")
    print()
    print("Summary:")
    print(comparison.summary)

    if comparison.regressions:
        print()
        print("Detected Regressions:")
        for reg in comparison.regressions:
            print(f"  - [{reg['case_id']}] baseline={reg['baseline_score']} -> candidate={reg['candidate_score']} ({reg['reason']})")

    return 0 if comparison.verdict.value == "PASS" else 1


def _build_demo_statistical_analysis(scenario: str):
    from benizakura.models import ConsistencyOutcome
    from benizakura.statistical import EffectSize, PairedCaseOutcome, StatisticalAnalysis

    if scenario == "gate-pass":
        paired_cases = []
        for i in range(1, 16):
            paired_cases.append(PairedCaseOutcome(case_id=f"case-{i:02d}", outcome=ConsistencyOutcome.CONSISTENT_CANDIDATE_WIN, paired_difference=1.0))
        paired_cases.append(PairedCaseOutcome(case_id="case-16", outcome=ConsistencyOutcome.CONSISTENT_BASELINE_WIN, paired_difference=-1.0))
        paired_cases.append(PairedCaseOutcome(case_id="case-17", outcome=ConsistencyOutcome.CONSISTENT_BASELINE_WIN, paired_difference=-1.0))
        for i in range(18, 21):
            paired_cases.append(PairedCaseOutcome(case_id=f"case-{i:02d}", outcome=ConsistencyOutcome.CONSISTENT_TIE, paired_difference=0.0))
        return StatisticalAnalysis(
            sample_size=20,
            candidate_wins=15,
            baseline_wins=2,
            ties=3,
            inconsistent_cases=0,
            candidate_win_rate=0.75,
            baseline_win_rate=0.10,
            tie_rate=0.15,
            consistency_rate=1.0,
            observed_difference=0.65,
            confidence_level=0.95,
            confidence_interval=(0.40, 0.85),
            num_bootstrap_samples=1000,
            random_seed=42,
            effect_size=EffectSize("net_win_rate_difference", 0.65, cohens_g=0.38, interpretation="Large candidate advantage"),
            paired_cases=paired_cases,
        )
    elif scenario == "gate-fail":
        paired_cases = []
        for i in range(1, 4):
            paired_cases.append(PairedCaseOutcome(case_id=f"case-{i:02d}", outcome=ConsistencyOutcome.CONSISTENT_CANDIDATE_WIN, paired_difference=1.0))
        for i in range(4, 16):
            paired_cases.append(PairedCaseOutcome(case_id=f"case-{i:02d}", outcome=ConsistencyOutcome.CONSISTENT_BASELINE_WIN, paired_difference=-1.0))
        for i in range(16, 21):
            paired_cases.append(PairedCaseOutcome(case_id=f"case-{i:02d}", outcome=ConsistencyOutcome.CONSISTENT_TIE, paired_difference=0.0))
        return StatisticalAnalysis(
            sample_size=20,
            candidate_wins=3,
            baseline_wins=12,
            ties=5,
            inconsistent_cases=0,
            candidate_win_rate=0.15,
            baseline_win_rate=0.60,
            tie_rate=0.25,
            consistency_rate=1.0,
            observed_difference=-0.45,
            confidence_level=0.95,
            confidence_interval=(-0.70, -0.20),
            num_bootstrap_samples=1000,
            random_seed=42,
            effect_size=EffectSize("net_win_rate_difference", -0.45, cohens_g=-0.30, interpretation="Significant regression"),
            paired_cases=paired_cases,
        )
    elif scenario == "gate-inconclusive":
        paired_cases = []
        for i in range(1, 9):
            paired_cases.append(PairedCaseOutcome(case_id=f"case-{i:02d}", outcome=ConsistencyOutcome.CONSISTENT_CANDIDATE_WIN, paired_difference=1.0))
        for i in range(9, 12):
            paired_cases.append(PairedCaseOutcome(case_id=f"case-{i:02d}", outcome=ConsistencyOutcome.CONSISTENT_BASELINE_WIN, paired_difference=-1.0))
        for i in range(12, 15):
            paired_cases.append(PairedCaseOutcome(case_id=f"case-{i:02d}", outcome=ConsistencyOutcome.CONSISTENT_TIE, paired_difference=0.0))
        for i in range(15, 21):
            paired_cases.append(PairedCaseOutcome(case_id=f"case-{i:02d}", outcome=ConsistencyOutcome.POSITION_UNSTABLE, paired_difference=0.0))
        return StatisticalAnalysis(
            sample_size=20,
            candidate_wins=8,
            baseline_wins=3,
            ties=3,
            inconsistent_cases=6,
            candidate_win_rate=0.40,
            baseline_win_rate=0.15,
            tie_rate=0.15,
            consistency_rate=0.70,
            observed_difference=0.25,
            confidence_level=0.95,
            confidence_interval=(-0.05, 0.55),
            num_bootstrap_samples=1000,
            random_seed=42,
            effect_size=EffectSize("net_win_rate_difference", 0.25, cohens_g=0.22, interpretation="Unreliable due to judge ordering bias"),
            paired_cases=paired_cases,
        )
    raise ValueError(f"Unknown gate demo scenario: {scenario}")


def _print_gate_demo(result, title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"Benizakura Release Gate: {title}")
    print(f"{'=' * 72}")
    print(f"Verdict:             {result.verdict.value}")
    print(f"Sample Size:         {result.sample_size} cases")
    print(f"Observed Difference: {result.observed_difference:+.2f} ({result.observed_difference:+.1%})")
    print(f"Confidence Interval: [{result.confidence_interval[0]:+.2f}, {result.confidence_interval[1]:+.2f}]")
    print(f"Regression Rate:     {result.regression_rate:.1%} ({result.regression_count} of {result.sample_size} cases regressed)")
    if result.regressed_cases:
        print(f"Regressed Cases:     {', '.join(result.regressed_cases)}")
    print(f"Instability Rate:    {result.unstable_rate:.1%} ({result.unstable_count} of {result.sample_size} cases position-unstable)")
    if result.unstable_cases:
        print(f"Unstable Cases:      {', '.join(result.unstable_cases)}")
    print("\nPolicy Configuration (Benizakura Defaults):")
    print(f"  - min_cases:            {result.config.min_cases}")
    print(f"  - max_unstable_rate:    {result.config.max_unstable_rate:.1%}")
    print(f"  - regression_tolerance: {result.config.regression_tolerance:.1%}")
    print(f"  - max_regression_rate:  {result.config.max_regression_rate:.1%}")
    print(f"  - min_effect_size:      {result.config.min_effect_size:.1%}")
    print("\nDecision Reason:")
    print(f"  {result.reason}\n")


def _run_gate_demo(mode: str) -> int:
    from benizakura.gate import GateConfig, ReleaseGate
    from benizakura.models import Verdict

    gate = ReleaseGate(GateConfig())
    scenarios = (
        [("PASS Scenario", "gate-pass"), ("FAIL Scenario", "gate-fail"), ("INCONCLUSIVE Scenario", "gate-inconclusive")]
        if mode == "gate"
        else [(f"{mode.upper()} Scenario", mode)]
    )

    last_verdict = None
    for title, scenario_key in scenarios:
        analysis = _build_demo_statistical_analysis(scenario_key)
        result = gate.evaluate(analysis)
        _print_gate_demo(result, title)
        last_verdict = result.verdict

    if mode == "gate":
        return 0
    return 0 if last_verdict == Verdict.PASS else 1


if __name__ == "__main__":
    sys.exit(main())
