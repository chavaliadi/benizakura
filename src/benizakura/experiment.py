from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set, Tuple, Union
import urllib.error
import urllib.request

from benizakura.benchmark import (
    REQUIRED_CATEGORY_COUNTS,
    REQUIRED_TOTAL_CASES,
    compute_benchmark_hash,
    load_benchmark_data,
)
from benizakura.calibration import (
    AgreementStatistics,
    ConfusionMatrix,
    compute_cohens_kappa,
)
from benizakura.errors import (
    CalibrationPrerequisiteError,
    ExperimentError,
    JudgeParseError,
    JudgeProviderError,
    JudgeValidationError,
)
from benizakura.models import (
    CriterionAssessment,
    EvaluationCase,
    PairwiseJudgment,
    PairwiseWinner,
    Rubric,
    StandardCriteria,
)
from benizakura.parser import JudgeResponseParser
from benizakura.prompt import (
    JudgePromptBuilder,
    SYSTEM_PROMPT_PAIRWISE_RATIONALE_FIRST_V1,
    SYSTEM_PROMPT_PAIRWISE_RUBRIC_V1,
    SYSTEM_PROMPT_PAIRWISE_V1,
    SYSTEM_PROMPT_POINTWISE_V1,
)
from benizakura.provider import JudgeRequest, JudgeResponse


EXPECTED_BENCHMARK_SHA256 = "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314"
KAPPA_MINIMUM_THRESHOLD = 0.60


class ExperimentVariant(str, Enum):
    """The five controlled research experiment variants."""
    POINTWISE = "A"                     # Simple pointwise scalar scoring
    SINGLE_DIRECTION = "B"              # Single-direction pairwise (Baseline vs Candidate)
    BIDIRECTIONAL = "C"                 # Bidirectional pairwise (forward + reverse order)
    RUBRIC_BIDIRECTIONAL = "D"          # Rubric-based bidirectional pairwise
    FULL_BENIZAKURA = "E"               # Full Benizakura (rubric + evidence + bidirectional + bootstrap CI)


class ExecutionStatus(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class CallStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARSE_ERROR = "PARSE_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    TIMEOUT = "TIMEOUT"


# ============================================================================
# Secret Sanitization Helper
# ============================================================================

_SECRET_KEY_NAMES: Set[str] = {
    "api_key",
    "apikey",
    "authorization",
    "x-api-key",
    "secret",
    "token",
    "password",
    "access_token",
}

_SECRET_PATTERN_ANTHROPIC = re.compile(r"sk-ant-[a-zA-Z0-9_\-]+", re.IGNORECASE)
_SECRET_PATTERN_OPENAI = re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE)
_SECRET_PATTERN_BEARER = re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]+", re.IGNORECASE)


def sanitize_secrets(obj: Any) -> Any:
    """Recursively scrub API keys, tokens, and credentials from dictionaries, lists, and strings."""
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            if str(k).lower() in _SECRET_KEY_NAMES:
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = sanitize_secrets(v)
        return cleaned
    elif isinstance(obj, list):
        return [sanitize_secrets(item) for item in obj]
    elif isinstance(obj, str):
        val = _SECRET_PATTERN_ANTHROPIC.sub("[REDACTED_ANTHROPIC_KEY]", obj)
        val = _SECRET_PATTERN_OPENAI.sub("[REDACTED_OPENAI_KEY]", val)
        val = _SECRET_PATTERN_BEARER.sub("Bearer [REDACTED_TOKEN]", val)
        return val
    return obj


# ============================================================================
# Judge Configuration
# ============================================================================

@dataclass
class JudgeConfig:
    """Configuration record for an LLM evaluation judge."""
    judge_id: str
    provider: str
    model: str
    temperature: float = 0.0
    system_prompt_version: str = "pairwise_v1"
    rubric_version: str = "standard_rubric_v1"
    prompt_schema_version: str = "1.0.0"
    max_tokens: int = 1024
    seed: Optional[int] = None
    model_family: str = "anthropic"
    target_model_family: str = "openai"
    same_family: bool = field(init=False)

    def __post_init__(self) -> None:
        self.same_family = (self.model_family.strip().lower() == self.target_model_family.strip().lower())

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_secrets({
            "judge_id": self.judge_id,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "system_prompt_version": self.system_prompt_version,
            "rubric_version": self.rubric_version,
            "prompt_schema_version": self.prompt_schema_version,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "model_family": self.model_family,
            "target_model_family": self.target_model_family,
            "same_family": self.same_family,
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JudgeConfig:
        return cls(
            judge_id=str(data["judge_id"]),
            provider=str(data["provider"]),
            model=str(data["model"]),
            temperature=float(data.get("temperature", 0.0)),
            system_prompt_version=str(data.get("system_prompt_version", "pairwise_v1")),
            rubric_version=str(data.get("rubric_version", "standard_rubric_v1")),
            prompt_schema_version=str(data.get("prompt_schema_version", "1.0.0")),
            max_tokens=int(data.get("max_tokens", 1024)),
            seed=data.get("seed"),
            model_family=str(data.get("model_family", "anthropic")),
            target_model_family=str(data.get("target_model_family", "openai")),
        )


# Canonical Research Configurations
DEFAULT_JUDGE_A = JudgeConfig(
    judge_id="judge_a_claude",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    temperature=0.0,
    system_prompt_version="pairwise_v1",
    rubric_version="standard_rubric_v1",
    prompt_schema_version="1.0.0",
    max_tokens=1024,
    seed=None,
    model_family="anthropic",
    target_model_family="openai",
)

DEFAULT_JUDGE_B = JudgeConfig(
    judge_id="judge_b_gpt4o",
    provider="openai",
    model="gpt-4o-2024-08-06",
    temperature=0.0,
    system_prompt_version="pairwise_v1",
    rubric_version="standard_rubric_v1",
    prompt_schema_version="1.0.0",
    max_tokens=1024,
    seed=42,
    model_family="openai",
    target_model_family="openai",
)


# ============================================================================
# Provider Abstraction & Adapters
# ============================================================================

class LLMJudgeProvider(Protocol):
    """Protocol for provider execution adapters in calibration experiments."""

    def generate(self, request: JudgeRequest, config: JudgeConfig) -> JudgeResponse:
        ...


class AnthropicJudgeProvider:
    """HTTP client adapter for Anthropic Claude models."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.timeout = timeout

    def generate(self, request: JudgeRequest, config: JudgeConfig) -> JudgeResponse:
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise JudgeProviderError(
                "ANTHROPIC_API_KEY environment variable is not set. Cannot execute live API calls."
            )

        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.user_prompt}],
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )

        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_bytes = resp.read()
                latency_ms = (time.time() - start_time) * 1000.0
                resp_json = json.loads(raw_bytes.decode("utf-8"))

            content_blocks = resp_json.get("content", [])
            text_chunks = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
            content = "\n".join(text_chunks)
            usage = resp_json.get("usage", {})

            return JudgeResponse(
                content=content,
                metadata=sanitize_secrets({
                    "provider": "anthropic",
                    "model": resp_json.get("model", config.model),
                    "latency_ms": latency_ms,
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "stop_reason": resp_json.get("stop_reason"),
                }),
            )
        except urllib.error.HTTPError as err:
            err_body = ""
            try:
                err_body = err.read().decode("utf-8")
            except Exception:
                pass
            raise JudgeProviderError(
                f"Anthropic API HTTP Error {err.code}: {sanitize_secrets(err_body)}"
            ) from err
        except urllib.error.URLError as err:
            raise JudgeProviderError(f"Anthropic network connection failed: {err}") from err
        except Exception as exc:
            raise JudgeProviderError(f"Anthropic generation error: {exc}") from exc


class OpenAIJudgeProvider:
    """HTTP client adapter for OpenAI models."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

    def generate(self, request: JudgeRequest, config: JudgeConfig) -> JudgeResponse:
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise JudgeProviderError(
                "OPENAI_API_KEY environment variable is not set. Cannot execute live API calls."
            )

        url = "https://api.openai.com/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if config.seed is not None:
            payload["seed"] = config.seed

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_bytes = resp.read()
                latency_ms = (time.time() - start_time) * 1000.0
                resp_json = json.loads(raw_bytes.decode("utf-8"))

            choices = resp_json.get("choices", [])
            content = choices[0]["message"]["content"] if choices else ""
            usage = resp_json.get("usage", {})

            return JudgeResponse(
                content=content,
                metadata=sanitize_secrets({
                    "provider": "openai",
                    "model": resp_json.get("model", config.model),
                    "latency_ms": latency_ms,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "system_fingerprint": resp_json.get("system_fingerprint"),
                }),
            )
        except urllib.error.HTTPError as err:
            err_body = ""
            try:
                err_body = err.read().decode("utf-8")
            except Exception:
                pass
            raise JudgeProviderError(
                f"OpenAI API HTTP Error {err.code}: {sanitize_secrets(err_body)}"
            ) from err
        except urllib.error.URLError as err:
            raise JudgeProviderError(f"OpenAI network connection failed: {err}") from err
        except Exception as exc:
            raise JudgeProviderError(f"OpenAI generation error: {exc}") from exc


class MockExperimentJudgeProvider:
    """Deterministic mock provider for unit tests and local dry-run simulations."""

    def __init__(
        self,
        default_response: Optional[Union[str, JudgeResponse]] = None,
        responses_by_case: Optional[Dict[str, Union[str, JudgeResponse]]] = None,
    ):
        self.default_response = default_response or json.dumps({
            "winner": "B",
            "rationale": "Mock default preference for candidate response B.",
            "confidence": 0.85,
            "criterion_assessments": [
                {"criterion_name": "CORRECTNESS", "winner": "B", "rationale": "Accurate."},
                {"criterion_name": "RELEVANCE", "winner": "B", "rationale": "Direct."},
                {"criterion_name": "COMPLETENESS", "winner": "B", "rationale": "Thorough."},
                {"criterion_name": "TECHNICAL_DEPTH", "winner": "B", "rationale": "Detailed."},
                {"criterion_name": "CLARITY", "winner": "B", "rationale": "Clear."},
                {"criterion_name": "GROUNDEDNESS", "winner": "B", "rationale": "Grounded."},
            ],
        })
        self.responses_by_case = responses_by_case or {}
        self.call_history: List[JudgeRequest] = []

    def generate(self, request: JudgeRequest, config: JudgeConfig) -> JudgeResponse:
        self.call_history.append(request)
        case_id = request.metadata.get("case_id", "")
        resp = self.responses_by_case.get(case_id, self.default_response)
        if isinstance(resp, JudgeResponse):
            return resp
        return JudgeResponse(content=str(resp), metadata={"provider": "mock", "latency_ms": 1.0})


# ============================================================================
# Prerequisite Gate
# ============================================================================

@dataclass
class PrerequisiteCheckResult:
    """Status record verifying whether human ground truth prerequisites are satisfied."""
    is_ready: bool
    benchmark_hash_valid: bool
    ground_truth_exists: bool
    ground_truth_frozen: bool
    human_kappa: Optional[float]
    kappa_sufficient: bool
    blocking_reasons: List[str]

    def summary(self) -> str:
        lines = [
            "Human Calibration Prerequisite Check",
            "====================================",
            f"Ready for Live Calibration: {'YES' if self.is_ready else 'BLOCKED'}",
            f"Benchmark Hash Valid:       {self.benchmark_hash_valid}",
            f"Ground Truth File Exists:   {self.ground_truth_exists}",
            f"Ground Truth Frozen:        {self.ground_truth_frozen}",
            f"Human-Human Cohen's kappa:  {self.human_kappa if self.human_kappa is not None else 'N/A'} (Required: >= {KAPPA_MINIMUM_THRESHOLD:.2f})",
        ]
        if self.blocking_reasons:
            lines.append("")
            lines.append(f"Blocking Reasons ({len(self.blocking_reasons)}):")
            for r in self.blocking_reasons:
                lines.append(f"  [BLOCKED] {r}")
        return "\n".join(lines)


def check_calibration_prerequisites(
    benchmark_path: Union[str, Path] = "evals/conquer/benchmark_v1.json",
    ground_truth_path: Union[str, Path] = "evals/conquer/ground_truth_v1.json",
    expected_benchmark_hash: str = EXPECTED_BENCHMARK_SHA256,
    allow_unfrozen: bool = False,
) -> PrerequisiteCheckResult:
    """Verify all prerequisites before allowing automated judge calibration experiments to run."""
    blocking_reasons: List[str] = []
    b_path = Path(benchmark_path)
    gt_path = Path(ground_truth_path)

    # 1. Benchmark file and SHA-256 hash check
    benchmark_hash_valid = False
    if not b_path.is_file():
        blocking_reasons.append(f"Benchmark file not found at: {b_path}")
    else:
        computed_hash = compute_benchmark_hash(b_path)
        if computed_hash != expected_benchmark_hash:
            blocking_reasons.append(
                f"Benchmark SHA-256 mismatch! Expected '{expected_benchmark_hash}', computed '{computed_hash}'."
            )
        else:
            benchmark_hash_valid = True

    # 2. Ground truth artifact check
    ground_truth_exists = gt_path.is_file()
    ground_truth_frozen = False
    human_kappa: Optional[float] = None
    kappa_sufficient = False

    if not ground_truth_exists:
        blocking_reasons.append(f"Ground-truth file not found at: {gt_path}")
    else:
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
        except Exception as exc:
            blocking_reasons.append(f"Failed to read/parse ground-truth JSON: {exc}")
            gt_data = {}

        gt_status = gt_data.get("status")
        if gt_status == "GROUND_TRUTH_FROZEN":
            ground_truth_frozen = True
        else:
            blocking_reasons.append(
                f"Ground truth status is '{gt_status}' (expected 'GROUND_TRUTH_FROZEN'). "
                "Actual human annotations and adjudication must be completed and frozen before judge calibration."
            )

        # Check human agreement kappa
        stats = gt_data.get("agreement_statistics", {})
        if isinstance(stats, dict):
            human_kappa = stats.get("cohens_kappa")
            if human_kappa is not None:
                if human_kappa >= KAPPA_MINIMUM_THRESHOLD:
                    kappa_sufficient = True
                else:
                    blocking_reasons.append(
                        f"Human-human Cohen's kappa is {human_kappa:.4f} < {KAPPA_MINIMUM_THRESHOLD:.2f}. "
                        "Human calibration kill-gate failed; automated judge calibration is blocked."
                    )
            else:
                blocking_reasons.append("Human-human Cohen's kappa is not recorded in ground-truth artifact.")
        else:
            blocking_reasons.append("Ground-truth artifact missing 'agreement_statistics' container.")

    is_ready = len(blocking_reasons) == 0
    if allow_unfrozen:
        # Software test mode only
        is_ready = True

    return PrerequisiteCheckResult(
        is_ready=is_ready,
        benchmark_hash_valid=benchmark_hash_valid,
        ground_truth_exists=ground_truth_exists,
        ground_truth_frozen=ground_truth_frozen,
        human_kappa=human_kappa,
        kappa_sufficient=kappa_sufficient,
        blocking_reasons=blocking_reasons,
    )


# ============================================================================
# Experiment Data Models & Run Logging
# ============================================================================

@dataclass
class ExperimentCaseRun:
    """Individual execution record of an LLM judge on a benchmark case."""
    run_id: str
    case_id: str
    judge_id: str
    variant: str
    orientation: str
    repeat_index: int
    call_status: CallStatus
    raw_winner: Optional[str]
    normalized_winner: Optional[str]
    confidence: Optional[float]
    rationale: Optional[str]
    criterion_scores: Dict[str, str] = field(default_factory=dict)
    latency_ms: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    error_message: Optional[str] = None
    raw_content_sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_secrets({
            "run_id": self.run_id,
            "case_id": self.case_id,
            "judge_id": self.judge_id,
            "variant": self.variant,
            "orientation": self.orientation,
            "repeat_index": self.repeat_index,
            "call_status": self.call_status.value if isinstance(self.call_status, CallStatus) else str(self.call_status),
            "raw_winner": self.raw_winner,
            "normalized_winner": self.normalized_winner,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "criterion_scores": self.criterion_scores,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "error_message": self.error_message,
            "raw_content_sha256": self.raw_content_sha256,
            "metadata": self.metadata,
        })


@dataclass
class ExperimentPlan:
    """Declarative workload plan describing an evaluation experiment run."""
    experiment_id: str
    benchmark_id: str
    benchmark_sha256: str
    ground_truth_id: str
    ground_truth_sha256: str
    judges: List[JudgeConfig]
    variants: List[ExperimentVariant]
    case_count: int
    repeat_count: int
    total_requests: int
    estimated_tokens: int
    is_dry_run: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_secrets({
            "experiment_id": self.experiment_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_sha256": self.benchmark_sha256,
            "ground_truth_id": self.ground_truth_id,
            "ground_truth_sha256": self.ground_truth_sha256,
            "judges": [j.to_dict() for j in self.judges],
            "variants": [v.value for v in self.variants],
            "case_count": self.case_count,
            "repeat_count": self.repeat_count,
            "total_requests": self.total_requests,
            "estimated_tokens": self.estimated_tokens,
            "is_dry_run": self.is_dry_run,
            "created_at": self.created_at,
        })

    def summary(self) -> str:
        lines = [
            f"Experiment Plan: {self.experiment_id}",
            "================================",
            f"Benchmark:        {self.benchmark_id} ({self.case_count} cases)",
            f"Benchmark SHA:    {self.benchmark_sha256}",
            f"Ground Truth ID:  {self.ground_truth_id}",
            f"Execution Mode:   {'DRY RUN (Zero live API calls)' if self.is_dry_run else 'LIVE API EXECUTION'}",
            "",
            f"Judges ({len(self.judges)}):",
        ]
        for j in self.judges:
            fam_str = "SAME FAMILY (GPT)" if j.same_family else "INDEPENDENT FAMILY"
            lines.append(f"  - {j.judge_id:<18} ({j.provider} / {j.model}) [{fam_str}]")

        lines.extend([
            "",
            f"Variants ({len(self.variants)}):",
        ])
        for v in self.variants:
            lines.append(f"  - Variant {v.value}: {v.name}")

        lines.extend([
            "",
            f"Repeat count:     {self.repeat_count}",
            f"Total requests:   {self.total_requests}",
            f"Estimated tokens: ~{self.estimated_tokens:,} tokens",
        ])
        return "\n".join(lines)


# ============================================================================
# Experiment Planning Engine
# ============================================================================

def plan_experiment(
    benchmark_source: Union[str, Path, Dict[str, Any]] = "evals/conquer/benchmark_v1.json",
    ground_truth_source: Union[str, Path, Dict[str, Any]] = "evals/conquer/ground_truth_v1.json",
    experiment_id: str = "calibration_v1",
    judges: Optional[List[JudgeConfig]] = None,
    variants: Optional[List[ExperimentVariant]] = None,
    repeat_count: int = 1,
    is_dry_run: bool = True,
) -> ExperimentPlan:
    """Calculate the precise workload and request count for an evaluation experiment."""
    bench_data = load_benchmark_data(benchmark_source)
    cases = bench_data.get("cases", [])
    b_hash = compute_benchmark_hash(benchmark_source)

    gt_data: Dict[str, Any] = {}
    gt_sha = "unverified"
    if isinstance(ground_truth_source, (str, Path)) and Path(ground_truth_source).is_file():
        with open(ground_truth_source, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gt_sha = hashlib.sha256(json.dumps(gt_data, sort_keys=True).encode("utf-8")).hexdigest()
    elif isinstance(ground_truth_source, dict):
        gt_data = ground_truth_source
        gt_sha = hashlib.sha256(json.dumps(gt_data, sort_keys=True).encode("utf-8")).hexdigest()

    gt_id = gt_data.get("ground_truth_id", "conquer-ground-truth-v1")

    judges_list = judges or [DEFAULT_JUDGE_A, DEFAULT_JUDGE_B]
    variants_list = variants or [
        ExperimentVariant.POINTWISE,
        ExperimentVariant.SINGLE_DIRECTION,
        ExperimentVariant.BIDIRECTIONAL,
        ExperimentVariant.RUBRIC_BIDIRECTIONAL,
        ExperimentVariant.FULL_BENIZAKURA,
    ]

    total_requests = 0
    num_cases = len(cases)

    # Variant request calculation:
    # A (Pointwise): 2 calls per case (eval baseline, eval candidate)
    # B (Single Direction): 1 call per case
    # C (Bidirectional): 2 calls per case (forward, reverse)
    # D (Rubric Bidirectional): 2 calls per case
    # E (Full Benizakura): 2 calls per case
    for v in variants_list:
        calls_per_case = 2 if v in (
            ExperimentVariant.POINTWISE,
            ExperimentVariant.BIDIRECTIONAL,
            ExperimentVariant.RUBRIC_BIDIRECTIONAL,
            ExperimentVariant.FULL_BENIZAKURA,
        ) else 1
        total_requests += calls_per_case * num_cases * len(judges_list) * repeat_count

    # Estimated average tokens per request: ~1,500 input + 400 output = 1,900
    estimated_tokens = total_requests * 1900

    return ExperimentPlan(
        experiment_id=experiment_id,
        benchmark_id=bench_data.get("benchmark_id", "conquer-benchmark-v1"),
        benchmark_sha256=b_hash,
        ground_truth_id=gt_id,
        ground_truth_sha256=gt_sha,
        judges=judges_list,
        variants=variants_list,
        case_count=num_cases,
        repeat_count=repeat_count,
        total_requests=total_requests,
        estimated_tokens=estimated_tokens,
        is_dry_run=is_dry_run,
    )


# ============================================================================
# Experiment Execution Engine
# ============================================================================

def execute_experiment(
    plan: ExperimentPlan,
    output_dir: Union[str, Path] = "evals/conquer/experiments",
    benchmark_source: Union[str, Path] = "evals/conquer/benchmark_v1.json",
    ground_truth_source: Union[str, Path] = "evals/conquer/ground_truth_v1.json",
    provider_factory: Optional[Callable[[JudgeConfig], LLMJudgeProvider]] = None,
    allow_unfrozen: bool = False,
    execute_live: bool = False,
    max_retries: int = 3,
) -> Tuple[ExecutionStatus, Dict[str, Any]]:
    """Execute evaluation experiment according to declarative plan.

    Enforces:
    - If execute_live is False: performs zero external network calls (Dry-run mode).
    - If allow_unfrozen is False: strictly verifies human-ground-truth prerequisites.
    - Records immutable run traces in `runs/`.
    """
    out_base = Path(output_dir) / plan.experiment_id
    runs_dir = out_base / "runs"
    norm_dir = out_base / "normalized"
    runs_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prerequisite verification
    prereq = check_calibration_prerequisites(
        benchmark_path=benchmark_source,
        ground_truth_path=ground_truth_source,
        expected_benchmark_hash=plan.benchmark_sha256,
        allow_unfrozen=allow_unfrozen,
    )
    if not prereq.is_ready:
        manifest_data = plan.to_dict()
        manifest_data["status"] = ExecutionStatus.FAILED.value
        manifest_data["blocking_reasons"] = prereq.blocking_reasons
        with open(out_base / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        raise CalibrationPrerequisiteError(
            f"Calibration blocked by prerequisites:\n{prereq.summary()}"
        )

    # 2. Check if live execution was permitted
    if not execute_live and not plan.is_dry_run:
        # User requested execution without explicit flag
        plan.is_dry_run = True

    bench_data = load_benchmark_data(benchmark_source)
    cases = bench_data.get("cases", [])
    prompt_builder = JudgePromptBuilder()
    parser = JudgeResponseParser()

    rubric = Rubric(
        name="StandardTechnicalRubric",
        instructions="Evaluate technical interview responses based on asymptotic complexity, systems design, and edge cases.",
        criteria=[
            StandardCriteria.CORRECTNESS,
            StandardCriteria.RELEVANCE,
            StandardCriteria.COMPLETENESS,
            StandardCriteria.TECHNICAL_DEPTH,
            StandardCriteria.CLARITY,
            StandardCriteria.GROUNDEDNESS,
        ],
    )

    runs_recorded: List[ExperimentCaseRun] = []
    normalized_decisions: List[Dict[str, Any]] = []

    # If in dry-run mode, serialize planned manifest and return early without making calls
    if plan.is_dry_run:
        manifest_data = plan.to_dict()
        manifest_data["status"] = ExecutionStatus.PLANNED.value
        manifest_data["completed_at"] = None
        with open(out_base / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        return ExecutionStatus.PLANNED, {
            "experiment_id": plan.experiment_id,
            "status": ExecutionStatus.PLANNED.value,
            "message": "Dry-run plan created successfully. Zero provider API calls were executed.",
            "total_requests": plan.total_requests,
            "runs_directory": str(runs_dir),
        }

    # 3. Live or mock execution loop
    for judge in plan.judges:
        # Instantiate provider
        if provider_factory:
            provider = provider_factory(judge)
        elif judge.provider == "anthropic":
            provider = AnthropicJudgeProvider()
        elif judge.provider == "openai":
            provider = OpenAIJudgeProvider()
        else:
            provider = MockExperimentJudgeProvider()

        for variant in plan.variants:
            prompt_ver = (
                "pairwise_rubric_v1"
                if variant in (ExperimentVariant.RUBRIC_BIDIRECTIONAL, ExperimentVariant.FULL_BENIZAKURA)
                else "pairwise_v1"
            )

            for case_dict in cases:
                cid = case_dict["case_id"]
                base_ans = case_dict["baseline_answer"]
                cand_ans = case_dict["candidate_answer"]
                eval_case = EvaluationCase(
                    id=cid,
                    topic=case_dict["category"],
                    question=case_dict["question"],
                    candidate_answer=cand_ans,
                )

                for rep in range(plan.repeat_count):
                    run_prefix = f"{plan.experiment_id}_{judge.judge_id}_{variant.value}_{cid}_rep{rep}"

                    # Handle Variant A (Pointwise)
                    if variant == ExperimentVariant.POINTWISE:
                        req_base = prompt_builder.build_pointwise(eval_case, base_ans, rubric=rubric)
                        req_cand = prompt_builder.build_pointwise(eval_case, cand_ans, rubric=rubric)

                        res_base = _call_with_retry(provider, req_base, judge, max_retries)
                        res_cand = _call_with_retry(provider, req_cand, judge, max_retries)

                        # Parse scores
                        score_b = _extract_pointwise_score(res_base.content)
                        score_c = _extract_pointwise_score(res_cand.content)

                        diff = score_c - score_b
                        if diff > 0.5:
                            norm_winner = "CANDIDATE"
                        elif diff < -0.5:
                            norm_winner = "BASELINE"
                        else:
                            norm_winner = "TIE"

                        case_run = ExperimentCaseRun(
                            run_id=f"{run_prefix}_pointwise",
                            case_id=cid,
                            judge_id=judge.judge_id,
                            variant=variant.value,
                            orientation="pointwise",
                            repeat_index=rep,
                            call_status=CallStatus.SUCCESS if score_b is not None and score_c is not None else CallStatus.PARSE_ERROR,
                            raw_winner=norm_winner,
                            normalized_winner=norm_winner,
                            confidence=1.0,
                            rationale=f"Candidate score: {score_c:.1f}, Baseline score: {score_b:.1f} (delta={diff:+.1f})",
                            raw_content_sha256=hashlib.sha256(f"{res_base.content}\n{res_cand.content}".encode("utf-8")).hexdigest(),
                        )
                        runs_recorded.append(case_run)
                        normalized_decisions.append({
                            "case_id": cid,
                            "judge_id": judge.judge_id,
                            "variant": variant.value,
                            "repeat_index": rep,
                            "normalized_winner": norm_winner,
                            "is_unstable": False,
                            "outcome": f"CONSISTENT_{norm_winner}_WIN" if norm_winner != "TIE" else "CONSISTENT_TIE",
                        })

                    # Handle Variant B (Single Direction)
                    elif variant == ExperimentVariant.SINGLE_DIRECTION:
                        req_fwd = prompt_builder.build(eval_case, base_ans, cand_ans, rubric=None, prompt_version="pairwise_v1")
                        res_fwd = _call_with_retry(provider, req_fwd, judge, max_retries)

                        try:
                            judgment = parser.parse(res_fwd.content)
                            raw_win = judgment.winner.value
                            norm_win = "BASELINE" if raw_win == "A" else ("CANDIDATE" if raw_win == "B" else "TIE")
                            status = CallStatus.SUCCESS
                            err_msg = None
                        except Exception as exc:
                            raw_win = None
                            norm_win = None
                            status = CallStatus.PARSE_ERROR
                            err_msg = str(exc)

                        case_run = ExperimentCaseRun(
                            run_id=f"{run_prefix}_fwd",
                            case_id=cid,
                            judge_id=judge.judge_id,
                            variant=variant.value,
                            orientation="forward",
                            repeat_index=rep,
                            call_status=status,
                            raw_winner=raw_win,
                            normalized_winner=norm_win,
                            confidence=judgment.confidence if status == CallStatus.SUCCESS else None,
                            rationale=judgment.rationale if status == CallStatus.SUCCESS else None,
                            error_message=err_msg,
                            raw_content_sha256=hashlib.sha256(res_fwd.content.encode("utf-8")).hexdigest(),
                        )
                        runs_recorded.append(case_run)
                        normalized_decisions.append({
                            "case_id": cid,
                            "judge_id": judge.judge_id,
                            "variant": variant.value,
                            "repeat_index": rep,
                            "normalized_winner": norm_win,
                            "is_unstable": False,
                            "outcome": f"CONSISTENT_{norm_win}_WIN" if norm_win in ("CANDIDATE", "BASELINE") else "CONSISTENT_TIE",
                        })

                    # Handle Variants C, D, E (Bidirectional)
                    else:
                        use_rubric = rubric if variant in (ExperimentVariant.RUBRIC_BIDIRECTIONAL, ExperimentVariant.FULL_BENIZAKURA) else None

                        # Pass 1: Forward (A=Baseline, B=Candidate)
                        req_fwd = prompt_builder.build(eval_case, base_ans, cand_ans, rubric=use_rubric, prompt_version=prompt_ver)
                        res_fwd = _call_with_retry(provider, req_fwd, judge, max_retries)

                        # Pass 2: Reverse (A=Candidate, B=Baseline)
                        req_rev = prompt_builder.build(eval_case, cand_ans, base_ans, rubric=use_rubric, prompt_version=prompt_ver)
                        res_rev = _call_with_retry(provider, req_rev, judge, max_retries)

                        try:
                            jdg_fwd = parser.parse(res_fwd.content, rubric=use_rubric)
                            fwd_raw = jdg_fwd.winner.value
                            fwd_norm = "BASELINE" if fwd_raw == "A" else ("CANDIDATE" if fwd_raw == "B" else "TIE")
                        except Exception as exc:
                            fwd_raw = None
                            fwd_norm = None

                        try:
                            jdg_rev = parser.parse(res_rev.content, rubric=use_rubric)
                            rev_raw = jdg_rev.winner.value
                            # In reverse, A is Candidate and B is Baseline
                            rev_norm = "CANDIDATE" if rev_raw == "A" else ("BASELINE" if rev_raw == "B" else "TIE")
                        except Exception as exc:
                            rev_raw = None
                            rev_norm = None

                        is_unstable = (fwd_norm != rev_norm) or (fwd_norm is None)
                        if is_unstable:
                            final_winner = "TIE"
                            outcome = "POSITION_UNSTABLE"
                        else:
                            final_winner = fwd_norm
                            outcome = f"CONSISTENT_{final_winner}_WIN" if final_winner != "TIE" else "CONSISTENT_TIE"

                        run_fwd = ExperimentCaseRun(
                            run_id=f"{run_prefix}_fwd",
                            case_id=cid,
                            judge_id=judge.judge_id,
                            variant=variant.value,
                            orientation="forward",
                            repeat_index=rep,
                            call_status=CallStatus.SUCCESS if fwd_raw else CallStatus.PARSE_ERROR,
                            raw_winner=fwd_raw,
                            normalized_winner=fwd_norm,
                            confidence=jdg_fwd.confidence if fwd_raw else None,
                            rationale=jdg_fwd.rationale if fwd_raw else None,
                            raw_content_sha256=hashlib.sha256(res_fwd.content.encode("utf-8")).hexdigest(),
                        )
                        run_rev = ExperimentCaseRun(
                            run_id=f"{run_prefix}_rev",
                            case_id=cid,
                            judge_id=judge.judge_id,
                            variant=variant.value,
                            orientation="reverse",
                            repeat_index=rep,
                            call_status=CallStatus.SUCCESS if rev_raw else CallStatus.PARSE_ERROR,
                            raw_winner=rev_raw,
                            normalized_winner=rev_norm,
                            confidence=jdg_rev.confidence if rev_raw else None,
                            rationale=jdg_rev.rationale if rev_raw else None,
                            raw_content_sha256=hashlib.sha256(res_rev.content.encode("utf-8")).hexdigest(),
                        )
                        runs_recorded.extend([run_fwd, run_rev])

                        normalized_decisions.append({
                            "case_id": cid,
                            "judge_id": judge.judge_id,
                            "variant": variant.value,
                            "repeat_index": rep,
                            "forward_winner": fwd_norm,
                            "reverse_winner": rev_norm,
                            "is_unstable": is_unstable,
                            "normalized_winner": final_winner,
                            "outcome": outcome,
                        })

    # Save all raw runs to disk
    for r in runs_recorded:
        fname = f"{r.run_id}.json"
        with open(runs_dir / fname, "w", encoding="utf-8") as f:
            json.dump(r.to_dict(), f, indent=2)

    # Save normalized decisions to disk
    with open(norm_dir / "normalized_decisions.json", "w", encoding="utf-8") as f:
        json.dump(normalized_decisions, f, indent=2)

    # Update manifest
    manifest_data = plan.to_dict()
    manifest_data["status"] = ExecutionStatus.COMPLETED.value
    manifest_data["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_data["runs_recorded"] = len(runs_recorded)
    with open(out_base / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return ExecutionStatus.COMPLETED, {
        "experiment_id": plan.experiment_id,
        "status": ExecutionStatus.COMPLETED.value,
        "runs_recorded": len(runs_recorded),
        "normalized_decisions": len(normalized_decisions),
        "output_directory": str(out_base),
    }


def _call_with_retry(
    provider: LLMJudgeProvider,
    request: JudgeRequest,
    config: JudgeConfig,
    max_retries: int = 3,
) -> JudgeResponse:
    """Execute provider call with bounded exponential backoff on transient errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return provider.generate(request, config)
        except JudgeProviderError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(0.5 * (2 ** attempt))
    raise last_exc or JudgeProviderError("Provider call failed after retries.")


def _extract_pointwise_score(content: str) -> float:
    """Extract scalar 0-10 score from pointwise response."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "score" in data:
            return float(data["score"])
    except Exception:
        pass
    # Fallback regex search for score
    match = re.search(r'"score"\s*:\s*([0-9\.]+)', content)
    if match:
        return float(match.group(1))
    return 5.0


# ============================================================================
# Statistical Analysis & Hypotheses Verification
# ============================================================================

def summarize_experiment(
    experiment_id: str,
    output_dir: Union[str, Path] = "evals/conquer/experiments",
    benchmark_source: Union[str, Path] = "evals/conquer/benchmark_v1.json",
    ground_truth_source: Union[str, Path] = "evals/conquer/ground_truth_v1.json",
) -> Dict[str, Any]:
    """Compile comprehensive analysis comparing experiment results against human ground truth."""
    exp_dir = Path(output_dir) / experiment_id
    norm_file = exp_dir / "normalized" / "normalized_decisions.json"
    if not norm_file.is_file():
        raise FileNotFoundError(f"Normalized decisions not found at: {norm_file}")

    with open(norm_file, "r", encoding="utf-8") as f:
        normalized_decisions = json.load(f)

    bench_data = load_benchmark_data(benchmark_source)
    bench_cases = {c["case_id"]: c for c in bench_data.get("cases", [])}

    gt_data: Dict[str, Any] = {}
    if Path(ground_truth_source).is_file():
        with open(ground_truth_source, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

    gt_cases = {c["case_id"]: c for c in gt_data.get("cases", [])}

    # Group decisions by judge and variant
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for d in normalized_decisions:
        key = (d["judge_id"], d["variant"])
        grouped.setdefault(key, []).append(d)

    summary_results: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configurations": {},
        "hypotheses_evidence": {},
    }

    # Analyze each configuration
    for (judge_id, variant_code), decisions in grouped.items():
        config_key = f"{judge_id}_variant_{variant_code}"

        # Position bias statistics
        total_pairs = len(decisions)
        unstable_count = sum(1 for d in decisions if d.get("is_unstable", False))
        unstable_rate = unstable_count / total_pairs if total_pairs > 0 else 0.0

        # Agreement with human ground truth (if available)
        judge_ratings: List[str] = []
        human_ratings: List[str] = []
        false_pass_count = 0
        false_regression_count = 0

        for d in decisions:
            cid = d["case_id"]
            j_win = d.get("normalized_winner", "TIE")
            gt_case = gt_cases.get(cid, {})
            consensus = gt_case.get("consensus", {})
            h_win = consensus.get("winner") if consensus else None

            if h_win:
                judge_ratings.append(j_win)
                human_ratings.append(h_win)

                # False Pass: Human says BASELINE won (candidate worse), Judge says CANDIDATE won
                if h_win == "BASELINE" and j_win == "CANDIDATE":
                    false_pass_count += 1
                # False Regression: Human says CANDIDATE won, Judge says BASELINE won
                elif h_win == "CANDIDATE" and j_win == "BASELINE":
                    false_regression_count += 1

        kappa, se, ci, conf_mat = compute_cohens_kappa(judge_ratings, human_ratings) if judge_ratings else (0.0, 0.0, (0.0, 0.0), None)
        raw_agree = sum(1 for j, h in zip(judge_ratings, human_ratings) if j == h) / len(judge_ratings) if judge_ratings else 0.0

        summary_results["configurations"][config_key] = {
            "judge_id": judge_id,
            "variant": variant_code,
            "total_decisions": total_pairs,
            "unstable_count": unstable_count,
            "position_instability_rate": unstable_rate,
            "human_evaluated_cases": len(judge_ratings),
            "raw_human_agreement": raw_agree,
            "cohens_kappa": kappa,
            "kappa_se": se,
            "confusion_matrix": conf_mat.to_dict() if conf_mat else None,
            "false_pass_count": false_pass_count,
            "false_regression_count": false_regression_count,
        }

    # Evaluate research hypotheses
    h_evidence: Dict[str, Any] = {}

    # H1: Bidirectional evaluation lowers position-dependent error compared with single-direction
    sd_keys = [k for k in summary_results["configurations"] if "variant_B" in k]
    bd_keys = [k for k in summary_results["configurations"] if "variant_C" in k]
    if sd_keys and bd_keys:
        h_evidence["H1_bidirectional_lowers_position_bias"] = {
            "hypothesis": "Bidirectional evaluation detects and neutralizes position-dependent flips compared with single-direction.",
            "single_direction_variants": sd_keys,
            "bidirectional_variants": bd_keys,
            "observed_instability_detected": [
                summary_results["configurations"][k]["unstable_count"] for k in bd_keys
            ],
            "conclusion": "Hypothesis testable once live evaluation is executed.",
        }

    # H2: Rubric decomposition improves agreement
    h_evidence["H2_rubric_improves_agreement"] = {
        "hypothesis": "Rubric decomposition improves human agreement compared with unconstrained pairwise judging.",
        "status": "Awaiting frozen ground truth and live execution.",
    }

    # H3: Rationale/criterion-first reduces instability
    h_evidence["H3_rationale_first_reduces_instability"] = {
        "hypothesis": "Rationale/criterion-first structured evaluation (R2) reduces instability compared to winner-first (R1).",
        "status": "Schema variant R2 implemented; testable in comparative live sweeps.",
    }

    # H4: Independent judge family behaves differently on hallucinations
    h_evidence["H4_independent_family_hallucination_divergence"] = {
        "hypothesis": "Anthropic (independent family) diverges from OpenAI (same family) on hallucination probes.",
        "status": "Probes identified (dsa-005, sys-005, backend-006, behavioral-006); testable on live run.",
    }

    # H5: Bootstrap confidence intervals reduce false regression alarms
    h_evidence["H5_bootstrap_ci_reduces_false_regressions"] = {
        "hypothesis": "Bootstrap confidence intervals and INCONCLUSIVE gating reduce false regression alarms.",
        "status": "Gate logic implemented; testable against full benchmark runs.",
    }

    summary_results["hypotheses_evidence"] = h_evidence

    # Save summary artifact
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    return summary_results
