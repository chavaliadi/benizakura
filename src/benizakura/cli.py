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

    # annotation-validate
    val_ann_parser = subparsers.add_parser("annotation-validate", help="Validate annotator submissions against schema and benchmark.")
    val_ann_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("evals/conquer/annotations/submissions"),
        help="Path to submissions JSON file or directory (default: evals/conquer/annotations/submissions).",
    )
    val_ann_parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("evals/conquer/benchmark_v1.json"),
        help="Path to benchmark JSON (default: evals/conquer/benchmark_v1.json).",
    )
    val_ann_parser.add_argument(
        "--allowed-annotators",
        type=str,
        default=None,
        help="Comma-separated list of authorized annotator IDs.",
    )

    # annotation-status
    stat_ann_parser = subparsers.add_parser("annotation-status", help="Check progress and coverage of human annotations.")
    stat_ann_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("evals/conquer/annotations/submissions"),
        help="Path to submissions JSON file or directory (default: evals/conquer/annotations/submissions).",
    )
    stat_ann_parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("evals/conquer/benchmark_v1.json"),
        help="Path to benchmark JSON (default: evals/conquer/benchmark_v1.json).",
    )

    # agreement
    agr_parser = subparsers.add_parser("agreement", help="Compute human-vs-human Cohen's kappa and calibration gate decision.")
    agr_parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=Path("evals/conquer/annotations/submissions"),
        help="Directory containing annotator submissions.",
    )
    agr_parser.add_argument(
        "--ann1",
        type=Path,
        default=None,
        help="Path to Annotator 1 submissions file or directory.",
    )
    agr_parser.add_argument(
        "--ann2",
        type=Path,
        default=None,
        help="Path to Annotator 2 submissions file or directory.",
    )
    agr_parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("evals/conquer/benchmark_v1.json"),
        help="Path to benchmark JSON (default: evals/conquer/benchmark_v1.json).",
    )
    agr_parser.add_argument(
        "--key",
        type=Path,
        default=Path("evals/conquer/annotations/blinding_key_v1.json"),
        help="Path to private blinding key JSON (default: evals/conquer/annotations/blinding_key_v1.json).",
    )
    agr_parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Calibration gate Cohen's kappa threshold (default: 0.60).",
    )
    agr_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as machine-readable JSON.",
    )

    # consensus
    cons_parser = subparsers.add_parser("consensus", help="Resolve disagreements and generate consensus ground truth.")
    cons_parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=Path("evals/conquer/annotations/submissions"),
        help="Directory containing annotator submissions.",
    )
    cons_parser.add_argument(
        "--ann1",
        type=Path,
        default=None,
        help="Path to Annotator 1 submissions.",
    )
    cons_parser.add_argument(
        "--ann2",
        type=Path,
        default=None,
        help="Path to Annotator 2 submissions.",
    )
    cons_parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("evals/conquer/benchmark_v1.json"),
        help="Path to benchmark JSON.",
    )
    cons_parser.add_argument(
        "--key",
        type=Path,
        default=Path("evals/conquer/annotations/blinding_key_v1.json"),
        help="Path to blinding key JSON.",
    )
    cons_parser.add_argument(
        "--adjudications",
        type=Path,
        default=None,
        help="Path to adjudications JSON file.",
    )
    cons_parser.add_argument(
        "--out",
        type=Path,
        default=Path("evals/conquer/consensus_v1.json"),
        help="Output path for consensus cases JSON.",
    )
    cons_parser.add_argument(
        "--export-adjudication-template",
        type=Path,
        default=None,
        help="Write template file containing all disagreement cases awaiting adjudication.",
    )

    # ground-truth
    gt_parser = subparsers.add_parser("ground-truth", help="Ground-truth artifact inspection, compilation, and freezing.")
    gt_parser.add_argument(
        "action",
        choices=["status", "verify", "freeze", "compile"],
        nargs="?",
        default="status",
        help="Action: status, verify, freeze, or compile (default: status).",
    )
    gt_parser.add_argument(
        "--path",
        type=Path,
        default=Path("evals/conquer/ground_truth_v1.json"),
        help="Path to ground truth JSON (default: evals/conquer/ground_truth_v1.json).",
    )
    gt_parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("evals/conquer/benchmark_v1.json"),
        help="Path to benchmark JSON.",
    )
    gt_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evals/conquer/ground_truth_v1.manifest.json"),
        help="Path to ground truth manifest.",
    )
    gt_parser.add_argument(
        "--sha256",
        type=Path,
        default=Path("evals/conquer/ground_truth_v1.sha256"),
        help="Path to ground truth .sha256 file.",
    )
    gt_parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=Path("evals/conquer/annotations/submissions"),
        help="Path to submissions directory (for compile).",
    )
    gt_parser.add_argument(
        "--key",
        type=Path,
        default=Path("evals/conquer/annotations/blinding_key_v1.json"),
        help="Path to blinding key JSON (for compile).",
    )
    gt_parser.add_argument(
        "--adjudications",
        type=Path,
        default=None,
        help="Path to adjudications JSON (for compile).",
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

    elif args.benchmark_command == "annotation-validate":
        from benizakura.benchmark import load_benchmark_data
        from benizakura.calibration import load_annotator_submissions, validate_submissions_batch

        bench_data = load_benchmark_data(args.benchmark)
        cases = bench_data.get("cases", [])
        allowed_set = set(args.allowed_annotators.split(",")) if args.allowed_annotators else None

        try:
            submissions = load_annotator_submissions(args.path)
        except Exception as exc:
            print(f"Failed to load submissions: {exc}", file=sys.stderr)
            return 1

        is_valid, errors = validate_submissions_batch(
            [s.to_dict() for s in submissions],
            cases,
            allowed_annotators=allowed_set,
        )

        print(f"Annotation Validation Report")
        print(f"============================")
        print(f"Submissions Checked: {len(submissions)}")
        print(f"Status:              {'VALID' if is_valid else 'INVALID'}")
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for err in errors:
                print(f"  [ERROR] {err}")
            return 1
        print("All submissions passed schema, criteria, and leakage validation.")
        return 0

    elif args.benchmark_command == "annotation-status":
        from benizakura.benchmark import load_benchmark_data
        from benizakura.calibration import load_annotator_submissions

        bench_data = load_benchmark_data(args.benchmark)
        cases = bench_data.get("cases", [])
        all_case_ids = {c["case_id"] for c in cases}

        if not args.path.exists():
            print(f"Annotation Status: No submissions found at {args.path}")
            print(f"Total Benchmark Cases: {len(all_case_ids)}")
            return 0

        submissions = load_annotator_submissions(args.path)
        by_annotator: dict[str, set[str]] = {}
        for s in submissions:
            by_annotator.setdefault(s.annotator_id, set()).add(s.case_id)

        print("Human Annotation Status")
        print("-----------------------")
        print(f"Benchmark Cases: {len(all_case_ids)}")
        print(f"Active Annotators: {len(by_annotator)}")
        print()

        for ann_id, cids in sorted(by_annotator.items()):
            missing = all_case_ids - cids
            pct = (len(cids) / len(all_case_ids)) * 100 if all_case_ids else 0.0
            print(f"  - {ann_id}: {len(cids)}/{len(all_case_ids)} ({pct:.1f}%)")
            if missing and len(missing) <= 5:
                print(f"      Missing: {', '.join(sorted(missing))}")
            elif missing:
                print(f"      Missing: {len(missing)} cases")

        if len(by_annotator) >= 2:
            shared = set.intersection(*by_annotator.values())
            print(f"\nCases evaluated by all active annotators: {len(shared)}/{len(all_case_ids)}")

        return 0

    elif args.benchmark_command == "agreement":
        from benizakura.benchmark import load_benchmark_data
        from benizakura.calibration import (
            calculate_human_agreement,
            load_annotator_submissions,
        )

        bench_data = load_benchmark_data(args.benchmark)
        with open(args.key, "r", encoding="utf-8") as f:
            key_data = json.load(f)

        if args.ann1 and args.ann2:
            subs_1 = load_annotator_submissions(args.ann1)
            subs_2 = load_annotator_submissions(args.ann2)
        else:
            all_subs = load_annotator_submissions(args.submissions_dir)
            by_ann: dict[str, list] = {}
            for s in all_subs:
                by_ann.setdefault(s.annotator_id, []).append(s)
            if len(by_ann) < 2:
                print(
                    f"Error: Found only {len(by_ann)} annotator(s) in {args.submissions_dir}. "
                    "Agreement analysis requires two independent annotators.",
                    file=sys.stderr,
                )
                return 1
            ann_keys = sorted(by_ann.keys())
            subs_1 = by_ann[ann_keys[0]]
            subs_2 = by_ann[ann_keys[1]]

        agreement_report = calculate_human_agreement(
            subs_1,
            subs_2,
            key_data,
            bench_data,
            threshold=args.threshold,
        )

        if args.json:
            print(json.dumps(agreement_report.to_dict(), indent=2))
        else:
            print(agreement_report.summary())

        return 0 if agreement_report.calibration_gate == "PASS" else 1

    elif args.benchmark_command == "consensus":
        from benizakura.benchmark import load_benchmark_data
        from benizakura.calibration import (
            apply_adjudications,
            create_adjudication_item,
            load_annotator_submissions,
        )

        bench_data = load_benchmark_data(args.benchmark)
        with open(args.key, "r", encoding="utf-8") as f:
            key_data = json.load(f)

        if args.ann1 and args.ann2:
            subs_1 = load_annotator_submissions(args.ann1)
            subs_2 = load_annotator_submissions(args.ann2)
        else:
            all_subs = load_annotator_submissions(args.submissions_dir)
            by_ann: dict[str, list] = {}
            for s in all_subs:
                by_ann.setdefault(s.annotator_id, []).append(s)
            if len(by_ann) < 2:
                print(f"Error: Consensus requires two independent annotator sets.", file=sys.stderr)
                return 1
            ann_keys = sorted(by_ann.keys())
            subs_1 = by_ann[ann_keys[0]]
            subs_2 = by_ann[ann_keys[1]]

        consensus_cases, pending_errors = apply_adjudications(
            subs_1,
            subs_2,
            key_data,
            bench_data,
            adjudication_records=args.adjudications,
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"consensus_cases": consensus_cases}, f, indent=2)

        print(f"Generated consensus records ({len(consensus_cases)} cases): {args.out}")
        if pending_errors:
            print(f"\nPending Disagreements / Errors ({len(pending_errors)}):")
            for err in pending_errors:
                print(f"  [PENDING] {err}")

            if args.export_adjudication_template:
                dict_1 = {s.case_id: s for s in subs_1}
                dict_2 = {s.case_id: s for s in subs_2}
                mappings = key_data.get("mappings", {})
                templates = []
                for c in consensus_cases:
                    if c["status"] == "SUBMITTED":
                        cid = c["case_id"]
                        templates.append(create_adjudication_item(cid, dict_1[cid], dict_2[cid], mappings[cid]))
                args.export_adjudication_template.parent.mkdir(parents=True, exist_ok=True)
                with open(args.export_adjudication_template, "w", encoding="utf-8") as f:
                    json.dump({"adjudications": templates}, f, indent=2)
                print(f"Wrote adjudication review template to: {args.export_adjudication_template}")

            return 1
        return 0

    elif args.benchmark_command == "ground-truth":
        from benizakura.benchmark import compute_benchmark_hash, load_benchmark_data
        from benizakura.calibration import (
            apply_adjudications,
            calculate_human_agreement,
            freeze_ground_truth_artifact,
            GroundTruthArtifact,
            GroundTruthStatus,
            load_annotator_submissions,
        )

        if args.action == "status":
            if not args.path.exists():
                print(f"Ground-truth file does not exist at {args.path}")
                return 1
            with open(args.path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            print("Ground-Truth Status")
            print("-------------------")
            print(f"Path:            {args.path}")
            print(f"Benchmark ID:    {gt_data.get('benchmark_id')}")
            print(f"Benchmark SHA:   {gt_data.get('benchmark_sha256')}")
            print(f"Status:          {gt_data.get('status')}")
            print(f"Human Labels:    {gt_data.get('human_labels')}")
            print(f"Cases Total:     {gt_data.get('case_count')}")
            stats = gt_data.get("agreement_statistics")
            if stats:
                print(f"Cohen's kappa:   {stats.get('cohens_kappa')}")
                print(f"Raw agreement:   {stats.get('raw_agreement')}")
            return 0

        elif args.action == "verify":
            if not args.path.exists():
                print(f"Error: Ground truth file not found at {args.path}", file=sys.stderr)
                return 1
            with open(args.path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            expected_benchmark_hash = compute_benchmark_hash(args.benchmark)
            if gt_data.get("benchmark_sha256") != expected_benchmark_hash:
                print(
                    f"Integrity Error: Ground truth references benchmark hash {gt_data.get('benchmark_sha256')}, "
                    f"but benchmark hash is {expected_benchmark_hash}.",
                    file=sys.stderr,
                )
                return 1
            print(f"Ground-truth verified against benchmark {args.benchmark} (SHA256 matches: {expected_benchmark_hash}).")
            return 0

        elif args.action == "compile":
            bench_data = load_benchmark_data(args.benchmark)
            bench_hash = compute_benchmark_hash(args.benchmark)
            with open(args.key, "r", encoding="utf-8") as f:
                key_data = json.load(f)

            all_subs = load_annotator_submissions(args.submissions_dir)
            by_ann: dict[str, list] = {}
            for s in all_subs:
                by_ann.setdefault(s.annotator_id, []).append(s)

            if len(by_ann) < 2:
                print("Error: Compiling ground truth requires at least 2 independent annotator submission sets.", file=sys.stderr)
                return 1

            ann_keys = sorted(by_ann.keys())
            subs_1 = by_ann[ann_keys[0]]
            subs_2 = by_ann[ann_keys[1]]

            agreement_report = calculate_human_agreement(subs_1, subs_2, key_data, bench_data)
            consensus_cases, pending_errors = apply_adjudications(
                subs_1, subs_2, key_data, bench_data, adjudication_records=args.adjudications
            )

            status = (
                GroundTruthStatus.ADJUDICATION_COMPLETE
                if not pending_errors and agreement_report.calibration_gate == "PASS"
                else GroundTruthStatus.HUMAN_AGREEMENT_MEASURED
            )

            gt_artifact = GroundTruthArtifact(
                benchmark_id=bench_data.get("benchmark_id", "conquer-benchmark-v1"),
                benchmark_version=bench_data.get("version", "1.0.0"),
                benchmark_sha256=bench_hash,
                schema_version="1.0.0",
                status=status,
                case_count=len(consensus_cases),
                human_labels="PENDING" if pending_errors else "CONSENSUS_COMPLETE",
                agreement_statistics=agreement_report.to_dict(),
                cases=consensus_cases,
                created_at=datetime.now(timezone.utc).isoformat(),
                frozen_at=None,
            )

            args.path.parent.mkdir(parents=True, exist_ok=True)
            with open(args.path, "w", encoding="utf-8") as f:
                json.dump(gt_artifact.to_dict(), f, indent=2)

            print(f"Compiled ground-truth artifact to {args.path} (status: {status})")
            if pending_errors:
                print(f"Notice: {len(pending_errors)} pending cases/adjudications remain.")
            return 0

        elif args.action == "freeze":
            try:
                artifact, sha = freeze_ground_truth_artifact(
                    artifact_source=args.path,
                    benchmark_source=args.benchmark,
                    out_json=args.path,
                    out_manifest=args.manifest,
                    out_sha256=args.sha256,
                )
                print(f"Successfully froze ground truth artifact:")
                print(f"  Artifact:  {args.path}")
                print(f"  Manifest:  {args.manifest}")
                print(f"  SHA-256:   {sha}")
                return 0
            except Exception as exc:
                print(f"Error freezing ground truth: {exc}", file=sys.stderr)
                return 1

    return 1


def _run_experiment_cli(sub_argv: list[str]) -> int:
    from benizakura.experiment import (
        DEFAULT_JUDGE_A,
        DEFAULT_JUDGE_B,
        ExperimentVariant,
        check_calibration_prerequisites,
        execute_experiment,
        plan_experiment,
        summarize_experiment,
    )

    exp_parser = argparse.ArgumentParser(
        prog="benizakura experiment",
        description="LLM judge calibration experiment planning, prerequisite validation, and execution.",
    )
    subparsers = exp_parser.add_subparsers(dest="experiment_command", required=True)

    # plan
    plan_p = subparsers.add_parser("plan", help="Generate declarative experiment workload plan and request estimates.")
    plan_p.add_argument("--benchmark", type=Path, default=Path("evals/conquer/benchmark_v1.json"), help="Benchmark file.")
    plan_p.add_argument("--ground-truth", type=Path, default=Path("evals/conquer/ground_truth_v1.json"), help="Ground truth file.")
    plan_p.add_argument("--experiment-id", type=str, default="calibration_v1", help="Experiment identifier.")
    plan_p.add_argument("--repeats", type=int, default=1, help="Repeat count.")
    plan_p.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    # validate
    val_p = subparsers.add_parser("validate", help="Check calibration prerequisites (frozen ground truth, kappa >= 0.60, hash).")
    val_p.add_argument("--benchmark", type=Path, default=Path("evals/conquer/benchmark_v1.json"), help="Benchmark file.")
    val_p.add_argument("--ground-truth", type=Path, default=Path("evals/conquer/ground_truth_v1.json"), help="Ground truth file.")

    # run
    run_p = subparsers.add_parser("run", help="Run calibration experiment (requires --execute for live calls; default: dry-run).")
    run_p.add_argument("experiment_id", nargs="?", default="calibration_v1", help="Experiment identifier.")
    run_p.add_argument("--benchmark", type=Path, default=Path("evals/conquer/benchmark_v1.json"), help="Benchmark file.")
    run_p.add_argument("--ground-truth", type=Path, default=Path("evals/conquer/ground_truth_v1.json"), help="Ground truth file.")
    run_p.add_argument("--output-dir", type=Path, default=Path("evals/conquer/experiments"), help="Experiments output directory.")
    run_p.add_argument("--repeats", type=int, default=1, help="Repeat count.")
    run_p.add_argument("--execute", action="store_true", help="Explicit confirmation flag to execute live model API calls.")
    run_p.add_argument("--dry-run", action="store_true", help="Explicit dry-run simulation mode.")
    run_p.add_argument(
        "--allow-unfrozen",
        action="store_true",
        help="DEVELOPMENT/FIXTURE ONLY: Allow offline test harness execution with unfrozen ground truth. Strictly blocked if --execute is passed.",
    )

    # status
    stat_p = subparsers.add_parser("status", help="Inspect status and progress of an experiment run.")
    stat_p.add_argument("experiment_id", nargs="?", default="calibration_v1", help="Experiment identifier.")
    stat_p.add_argument("--experiments-dir", type=Path, default=Path("evals/conquer/experiments"), help="Experiments directory.")

    # summarize
    sum_p = subparsers.add_parser("summarize", help="Compile and display statistical summary and hypothesis metrics.")
    sum_p.add_argument("experiment_id", nargs="?", default="calibration_v1", help="Experiment identifier.")
    sum_p.add_argument("--experiments-dir", type=Path, default=Path("evals/conquer/experiments"), help="Experiments directory.")
    sum_p.add_argument("--benchmark", type=Path, default=Path("evals/conquer/benchmark_v1.json"), help="Benchmark file.")
    sum_p.add_argument("--ground-truth", type=Path, default=Path("evals/conquer/ground_truth_v1.json"), help="Ground truth file.")
    sum_p.add_argument("--json", action="store_true", help="Output as machine-readable JSON.")

    args = exp_parser.parse_args(sub_argv)

    if args.experiment_command == "plan":
        plan = plan_experiment(
            benchmark_source=args.benchmark,
            ground_truth_source=args.ground_truth,
            experiment_id=args.experiment_id,
            repeat_count=args.repeats,
            is_dry_run=True,
        )
        if args.json:
            print(json.dumps(plan.to_dict(), indent=2))
        else:
            print(plan.summary())
        return 0

    elif args.experiment_command == "validate":
        res = check_calibration_prerequisites(
            benchmark_path=args.benchmark,
            ground_truth_path=args.ground_truth,
        )
        print(res.summary())
        return 0 if res.is_ready else 1

    elif args.experiment_command == "run":
        if args.allow_unfrozen and args.execute:
            print(
                "Execution Error: --allow-unfrozen cannot be combined with live execution (--execute). "
                "Official calibration runs require verified, frozen human ground truth.",
                file=sys.stderr,
            )
            return 1

        if args.allow_unfrozen:
            print("[SAFETY WARNING] Running with --allow-unfrozen. This is a DEVELOPMENT/FIXTURE mode only and does not produce official calibration evidence.\n")

        is_dry_run = not args.execute
        if is_dry_run:
            print("DRY RUN — no provider calls made. (Use --execute for live API calls)\n")

        plan = plan_experiment(
            benchmark_source=args.benchmark,
            ground_truth_source=args.ground_truth,
            experiment_id=args.experiment_id,
            repeat_count=args.repeats,
            is_dry_run=is_dry_run,
        )

        try:
            status, info = execute_experiment(
                plan=plan,
                output_dir=args.output_dir,
                benchmark_source=args.benchmark,
                ground_truth_source=args.ground_truth,
                allow_unfrozen=args.allow_unfrozen,
                execute_live=args.execute,
            )
            print(f"Experiment execution finished: {status.value}")
            for k, v in info.items():
                print(f"  {k}: {v}")
            return 0
        except Exception as exc:
            print(f"Execution Error: {exc}", file=sys.stderr)
            return 1

    elif args.experiment_command == "status":
        manifest_file = args.experiments_dir / args.experiment_id / "manifest.json"
        if not manifest_file.is_file():
            print(f"Experiment manifest not found at: {manifest_file}", file=sys.stderr)
            return 1
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Experiment Status: {args.experiment_id}")
        print("================================")
        print(f"Status:          {data.get('status')}")
        print(f"Benchmark:       {data.get('benchmark_id')} ({data.get('case_count')} cases)")
        print(f"Total Requests:  {data.get('total_requests')}")
        print(f"Runs Recorded:   {data.get('runs_recorded', 0)}")
        print(f"Created At:      {data.get('created_at')}")
        print(f"Completed At:    {data.get('completed_at')}")
        return 0

    elif args.experiment_command == "summarize":
        try:
            summary = summarize_experiment(
                experiment_id=args.experiment_id,
                output_dir=args.experiments_dir,
                benchmark_source=args.benchmark,
                ground_truth_source=args.ground_truth,
            )
            if args.json:
                print(json.dumps(summary, indent=2))
            else:
                print(f"Experiment Summary: {args.experiment_id}")
                print("==================================")
                for cfg_key, cfg_data in summary.get("configurations", {}).items():
                    print(f"\nConfiguration: {cfg_key}")
                    print(f"  Decisions:       {cfg_data.get('total_decisions')}")
                    print(f"  Unstable Count:  {cfg_data.get('unstable_count')} ({cfg_data.get('position_instability_rate', 0.0):.1%})")
                    if cfg_data.get("human_evaluated_cases", 0) > 0:
                        print(f"  Human Agreement: {cfg_data.get('raw_human_agreement', 0.0):.1%} (Cohen's kappa: {cfg_data.get('cohens_kappa', 0.0):.2f})")
                        print(f"  False Passes:    {cfg_data.get('false_pass_count')}")
                        print(f"  False Regress.:  {cfg_data.get('false_regression_count')}")
            return 0
        except Exception as exc:
            print(f"Summary Error: {exc}", file=sys.stderr)
            return 1

    return 1


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "benchmark":
        return _run_benchmark_cli(argv[1:])

    if argv and argv[0] == "experiment":
        return _run_experiment_cli(argv[1:])

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
