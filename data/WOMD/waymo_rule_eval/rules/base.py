"""
Base classes for rule evaluation.

Provides the two-phase rule evaluation pattern:
1. Applicability detection: Does this rule apply to the scenario?
2. Violation evaluation: If applicable, evaluate severity of violation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.context import ScenarioContext


@dataclass
class FrameViolation:
    """Violation information for a single frame."""

    frame_idx: int
    severity: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApplicabilityResult:
    """Result from checking if a rule applies to a scenario."""

    applies: bool
    confidence: float = 1.0
    reasons: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    # Optional metadata for tracking
    rule_id: Optional[str] = None
    rule_level: Optional[int] = None
    name: Optional[str] = None


@dataclass
class ViolationResult:
    """Result from evaluating a rule violation."""

    severity: float = 0.0  # 0 = no violation, higher = worse
    severity_normalized: float = 0.0  # 0-1 normalized severity
    measurements: Dict[str, Any] = field(default_factory=dict)
    explanation: Any = ""  # Can be str or List[str]
    frame_violations: List[Any] = field(
        default_factory=list
    )  # List[int] or List[FrameViolation]
    # Optional metadata for tracking
    rule_id: Optional[str] = None
    rule_level: Optional[int] = None
    name: Optional[str] = None
    # Timeseries data for detailed analysis
    timeseries: Dict[str, Any] = field(default_factory=dict)
    # Confidence in the evaluation
    confidence: float = 1.0

    @property
    def has_violation(self) -> bool:
        return self.severity > 0

    @property
    def violation_count(self) -> int:
        """Number of frames with violations."""
        return len(self.frame_violations)


@dataclass
class RuleResult:
    """Combined result from applicability and violation evaluation."""

    rule_id: str
    level: int
    name: str
    applies: bool
    applicability: Optional[ApplicabilityResult] = None
    violation: Optional[ViolationResult] = None
    error: Optional[str] = None

    @property
    def has_violation(self) -> bool:
        return self.violation is not None and self.violation.severity > 0

    @property
    def severity(self) -> float:
        return self.violation.severity if self.violation else 0.0


class ApplicabilityDetector(ABC):
    """Abstract base class for rule applicability detection."""

    # Subclasses should override these
    rule_id: str = "UNKNOWN"
    level: int = 0
    name: str = "Unknown Rule"

    @abstractmethod
    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Determine if this rule applies to the scenario.

        Args:
            ctx: ScenarioContext with ego, agents, map, signals

        Returns:
            ApplicabilityResult indicating if rule applies and why
        """
        pass


class ViolationEvaluator(ABC):
    """Abstract base class for rule violation evaluation."""

    # Subclasses should override these
    rule_id: str = "UNKNOWN"
    level: int = 0
    name: str = "Unknown Rule"

    @abstractmethod
    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate the severity of rule violation.

        Args:
            ctx: ScenarioContext with ego, agents, map, signals
            applicability: Result from applicability detection

        Returns:
            ViolationResult with severity and details
        """
        pass


class Rule:
    """
    Convenience wrapper bundling detector and evaluator.

    Usage:
        rule = Rule(
            CollisionApplicability(),
            CollisionViolation()
        )
        result = rule.evaluate(context)
    """

    def __init__(self, detector: ApplicabilityDetector, evaluator: ViolationEvaluator):
        self.detector = detector
        self.evaluator = evaluator
        self.rule_id = detector.rule_id
        self.level = detector.level
        self.name = detector.name

    def evaluate(self, ctx: ScenarioContext) -> RuleResult:
        """Evaluate rule on scenario."""
        try:
            applicability = self.detector.detect(ctx)

            result = RuleResult(
                rule_id=self.rule_id,
                level=self.level,
                name=self.name,
                applies=applicability.applies,
                applicability=applicability,
            )

            if applicability.applies:
                violation = self.evaluator.evaluate(ctx, applicability)
                result.violation = violation

            return result

        except Exception as e:
            return RuleResult(
                rule_id=self.rule_id,
                level=self.level,
                name=self.name,
                applies=False,
                error=str(e),
            )
