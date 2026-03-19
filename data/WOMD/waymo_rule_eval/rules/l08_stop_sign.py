"""
L8.R2: Stop Sign Compliance Rule

Detects when a vehicle fails to come to a complete stop at a stop sign/stop line.

Standards:
- Complete stop: speed < 0.3 m/s
- Must stop before crossing stop line
- Rolling stop is a violation

Severity:
- Based on speed at line crossing and distance past line
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.context import ScenarioContext
from ..utils.constants import (
    L8_NEAR_STOP_M,
    L8_STOP_SPEED_MPS,
    L8_STOPLINE_EPSILON_M,
    L8_WINDOW_FRAMES,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

log = logging.getLogger(__name__)
RULE_NORM = {"L8.R2": 8.0}


def per_frame_stopline_signed_distance(
    ego_x: np.ndarray, ego_y: np.ndarray, stopline_xy: np.ndarray
) -> np.ndarray:
    """
    Calculate signed distance from ego to stop line at each frame.

    Positive distance = before stop line
    Negative distance = past stop line
    """
    n_frames = len(ego_x)
    distances = np.full(n_frames, np.inf)

    if stopline_xy is None or len(stopline_xy) < 2:
        return distances

    # Stop line as a line segment
    p1 = np.array(stopline_xy[0])
    p2 = np.array(stopline_xy[-1])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-6:
        return distances

    line_unit = line_vec / line_len
    # Normal pointing "forward" (perpendicular to line)
    normal = np.array([-line_unit[1], line_unit[0]])

    for i in range(n_frames):
        ego_pos = np.array([ego_x[i], ego_y[i]])
        # Vector from line start to ego
        to_ego = ego_pos - p1
        # Signed distance (positive = same side as normal, negative = opposite)
        distances[i] = np.dot(to_ego, normal)

    return distances


class StopSignApplicability(ApplicabilityDetector):
    """
    Checks if stop sign rule is applicable.

    Applicable when:
    - Stop line geometry is available in map
    - Vehicle approaches near stop line
    """

    rule_id = "L8.R2"
    level = 8
    name = "Stop sign compliance"

    def __init__(self, near_m: float = L8_NEAR_STOP_M):
        """
        Initialize the detector.

        Args:
            near_m: Distance threshold for "near" stop line
        """
        self.near_m = near_m

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Check if stop sign evaluation applies.
        """
        # Check for stop line geometry
        if not hasattr(ctx, "map") or ctx.map is None:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["No map context"],
                features={},
            )

        stopline_xy = getattr(ctx.map, "stopline_xy", None)

        # If no explicit stopline geometry, try to use stop sign positions
        if stopline_xy is None or len(stopline_xy) < 2:
            stop_signs = getattr(ctx.map, "stop_signs", [])
            if stop_signs:
                # Use stop sign positions as synthetic stop line
                sign_positions = []
                for ss in stop_signs:
                    if "x" in ss and "y" in ss:
                        sign_positions.append([ss["x"], ss["y"]])
                if sign_positions:
                    # If only one stop sign, create perpendicular line
                    if len(sign_positions) == 1:
                        # Get ego heading direction from first valid frame
                        valid_idx = ~(np.isnan(ctx.ego.x) | np.isnan(ctx.ego.y))
                        if np.any(valid_idx):
                            idx = np.where(valid_idx)[0][0]
                            yaw = ctx.ego.yaw[idx] if hasattr(ctx.ego, "yaw") else 0
                            # Create perpendicular line at stop sign
                            perp = np.array([-np.sin(yaw), np.cos(yaw)])
                            center = np.array(sign_positions[0])
                            # Line 5m wide across the stop sign
                            stopline_xy = np.array(
                                [center - 2.5 * perp, center + 2.5 * perp]
                            )
                        else:
                            stopline_xy = None
                    else:
                        stopline_xy = np.array(sign_positions)

        if stopline_xy is None or len(stopline_xy) < 2:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["No stop line geometry or stop signs"],
                features={},
            )

        # Calculate signed distance to stop line
        d = per_frame_stopline_signed_distance(
            ctx.ego.x, ctx.ego.y, np.array(stopline_xy)
        )

        # Check if vehicle ever comes near stop line
        near = np.isfinite(d) & (np.abs(d) <= self.near_m)
        came_near = bool(np.any(near))

        features = {"d": d, "v": ctx.ego.speed, "dt": ctx.dt}

        if not came_near:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.5,
                reasons=["Vehicle never near stop line"],
                features=features,
            )

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=True,
            confidence=1.0,
            reasons=["Vehicle approaches stop line"],
            features=features,
        )


class StopSignViolation(ViolationEvaluator):
    """
    Evaluates stop sign violations.

    A violation occurs when:
    - Vehicle crosses stop line without coming to complete stop
    - Speed at line crossing > stop threshold
    """

    rule_id = "L8.R2"
    level = 8
    name = "Stop sign compliance"

    def __init__(
        self,
        v_stop_mps: float = L8_STOP_SPEED_MPS,
        eps_m: float = L8_STOPLINE_EPSILON_M,
        window_frames: int = L8_WINDOW_FRAMES,
    ):
        """
        Initialize the evaluator.

        Args:
            v_stop_mps: Speed threshold for "stopped"
            eps_m: Epsilon for stop line position
            window_frames: Window size for checking stop
        """
        self.v_stop = v_stop_mps
        self.eps = eps_m
        self.win = window_frames

    def evaluate(
        self, ctx: ScenarioContext, app: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate stop sign violations.
        """
        if not app.applies:
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=None,
                measurements={},
                explanation=["Not applicable"],
                confidence=0.0,
            )

        d = app.features["d"]
        v = app.features["v"]
        dt = app.features["dt"]

        # Find crossings (before -> after stop line)
        before_or_on = d >= -self.eps
        after = d < -self.eps

        crossings_idx = np.where((before_or_on[:-1]) & (after[1:]))[0] + 1

        min_abs_d = float(np.nanmin(np.abs(d))) if d.size else float("inf")
        min_v_all = float(np.nanmin(v)) if v.size else 0.0

        if crossings_idx.size == 0:
            meas = {
                "note": "No crossing detected",
                "min_abs_d": min_abs_d,
                "min_v": min_v_all,
            }
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=None,
                measurements=meas,
                explanation=["Stopped / compliant - no crossing"],
                confidence=app.confidence,
            )

        # Check each crossing for proper stop
        any_rolling = False
        max_speed_at_line = 0.0
        max_past_depth = 0.0
        n_events = 0

        for c in crossings_idx:
            s = max(0, c - self.win)
            e = min(len(d), c + self.win)
            local_v = v[s:e]
            local_d = d[s:e]

            min_local_v = float(np.min(local_v)) if local_v.size else 0.0
            crossed = np.any(local_d < -self.eps)

            if crossed:
                n_events += 1
                c0 = max(0, c - 2)
                c1 = min(len(v), c + 3)
                v_at_line = float(np.min(v[c0:c1])) if c1 > c0 else float(v[c])
                max_speed_at_line = max(max_speed_at_line, v_at_line)

                past_line_mask = local_d < -self.eps
                if np.any(past_line_mask):
                    local_depth = float(np.max(-local_d[past_line_mask]))
                    max_past_depth = max(max_past_depth, local_depth)

                full_stop = min_local_v <= self.v_stop
                if not full_stop:
                    any_rolling = True

        if not any_rolling:
            meas = {
                "n_events": n_events,
                "max_speed_at_line": max_speed_at_line,
                "max_past_line": max_past_depth,
                "note": "Stopped fully at all crossings",
            }
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=None,
                measurements=meas,
                explanation=["Stopped / compliant"],
                confidence=app.confidence,
            )

        # Violation: rolling stop
        severity = max_speed_at_line * (1.0 + max_past_depth / 5.0)
        norm_factor = RULE_NORM.get(self.rule_id, 8.0)
        normalized = min(severity / norm_factor, 1.0)

        meas = {
            "n_events": n_events,
            "max_speed_at_line": max_speed_at_line,
            "max_past_line": max_past_depth,
        }

        explanation = [
            "Rolled through stop sign",
            f"Speed at line: {max_speed_at_line:.2f} m/s",
            f"Max distance past line: {max_past_depth:.2f} m",
        ]

        return ViolationResult(
            rule_id=self.rule_id,
            name=self.name,
            severity=severity,
            severity_normalized=normalized,
            timeseries=None,
            measurements=meas,
            explanation=explanation,
            confidence=app.confidence,
        )
