"""
Unified rule API adapter.

CRITICAL: The codebase has inconsistent rule APIs:
- New API: rule.applies(ctx), rule.violation(ctx, app)
- Old API: rule.detector.detect(ctx), rule.evaluator.evaluate(ctx, app)
- RuleEntry API: rule.applicability.detect(ctx), rule.violation.evaluate(ctx, app)

This adapter handles all variants transparently, providing a single interface
for rule evaluation regardless of the underlying implementation.
"""

from typing import Any, Tuple, Optional
from .base import ApplicabilityResult, ViolationResult
from ..core.context import ScenarioContext


def evaluate_rule(
    rule: Any,
    ctx_scene: ScenarioContext,  # For applicability (scene-only)
    ctx_with_traj: Optional[
        ScenarioContext
    ] = None,  # For violation (with injected trajectory)
) -> Tuple[ApplicabilityResult, Optional[ViolationResult]]:
    """
    Unified rule evaluation adapter.

    Handles all rule API variants (RuleEntry, Rule class, modern .applies()).
    Returns (ApplicabilityResult, ViolationResult or None if inapplicable).
    """
    if ctx_with_traj is None:
        ctx_with_traj = ctx_scene

    # Detect applicability using scene-only context
    applicability = evaluate_applicability_only(rule, ctx_scene)

    if not applicability.applies:
        return applicability, None

    # Evaluate violation using trajectory-conditioned context
    violation = evaluate_violation_only(rule, ctx_with_traj, applicability)

    return applicability, violation


def evaluate_applicability_only(
    rule: Any,
    ctx: ScenarioContext,
) -> ApplicabilityResult:
    """Evaluate only applicability (scene-only, no trajectory needed)."""
    # Try different API patterns

    # Pattern 1: RuleEntry with .detector attribute
    if hasattr(rule, "detector") and hasattr(rule.detector, "detect"):
        return rule.detector.detect(ctx)

    # Pattern 2: Rule class with .applicability attribute
    if hasattr(rule, "applicability") and hasattr(rule.applicability, "detect"):
        return rule.applicability.detect(ctx)

    # Pattern 3: Modern API with .applies() method
    if hasattr(rule, "applies"):
        result = rule.applies(ctx)
        # Handle case where applies() returns bool vs ApplicabilityResult
        if isinstance(result, bool):
            return ApplicabilityResult(applies=result)
        return result

    # Pattern 4: Direct .detect() method
    if hasattr(rule, "detect"):
        return rule.detect(ctx)

    raise TypeError(
        f"Rule {type(rule).__name__} has no compatible applicability API. "
        f"Expected one of: .detector.detect(), .applicability.detect(), .applies(), .detect()"
    )


def evaluate_violation_only(
    rule: Any,
    ctx: ScenarioContext,
    applicability: ApplicabilityResult,
) -> ViolationResult:
    """Evaluate only violation (requires trajectory-conditioned context)."""
    # Pattern 1: RuleEntry with .evaluator attribute
    if hasattr(rule, "evaluator") and hasattr(rule.evaluator, "evaluate"):
        return rule.evaluator.evaluate(ctx, applicability)

    # Pattern 2: Rule class with .violation attribute
    if hasattr(rule, "violation") and hasattr(rule.violation, "evaluate"):
        return rule.violation.evaluate(ctx, applicability)

    # Pattern 3: Modern API with .violation() method (callable)
    if callable(getattr(rule, "violation", None)):
        result = rule.violation(ctx, applicability)
        return result

    # Pattern 4: Direct .evaluate() method
    if hasattr(rule, "evaluate"):
        return rule.evaluate(ctx, applicability)

    raise TypeError(
        f"Rule {type(rule).__name__} has no compatible violation API. "
        f"Expected one of: .evaluator.evaluate(), .violation.evaluate(), .violation(), .evaluate()"
    )


def evaluate_rule_batch(
    rules: list,
    ctx_scene: ScenarioContext,
    ctx_with_traj: Optional[ScenarioContext] = None,
) -> list:
    """
    Evaluate multiple rules using the unified adapter.

    Args:
        rules: List of rule objects
        ctx_scene: ScenarioContext for applicability detection
        ctx_with_traj: ScenarioContext with injected trajectory

    Returns:
        List of (rule_id, ApplicabilityResult, ViolationResult or None) tuples
    """
    results = []
    for rule in rules:
        # Get rule_id
        rule_id = (
            getattr(rule, "rule_id", None)
            or getattr(getattr(rule, "detector", None), "rule_id", None)
            or "UNKNOWN"
        )

        try:
            app, vio = evaluate_rule(rule, ctx_scene, ctx_with_traj)
            results.append((rule_id, app, vio))
        except Exception as e:
            # Return failed result with error info
            app = ApplicabilityResult(
                applies=False, confidence=0.0, reasons=[f"Evaluation error: {e}"]
            )
            results.append((rule_id, app, None))

    return results


def get_rule_id(rule: Any) -> str:
    """
    Extract rule ID from a rule object.

    Args:
        rule: Rule object (any API variant)

    Returns:
        Rule ID string
    """
    # Try different locations for rule_id
    if hasattr(rule, "rule_id"):
        return rule.rule_id
    if hasattr(rule, "detector") and hasattr(rule.detector, "rule_id"):
        return rule.detector.rule_id
    if hasattr(rule, "evaluator") and hasattr(rule.evaluator, "rule_id"):
        return rule.evaluator.rule_id
    return "UNKNOWN"
