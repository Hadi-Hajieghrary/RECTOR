"""
Window scheduling for rule evaluation.

Provides sliding window functionality to evaluate ego trajectory
at defined time intervals (e.g., every 0.1s or 1s) over a
future horizon (e.g., next 5 seconds).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class WindowSpec:
    """Specification for an evaluation window."""

    start_idx: int  # First frame index (inclusive)
    end_idx: int  # Last frame index (exclusive)
    start_ts: float  # Start timestamp in seconds
    end_ts: float  # End timestamp in seconds

    @property
    def n_frames(self) -> int:
        """Number of frames in window."""
        return self.end_idx - self.start_idx

    @property
    def duration_s(self) -> float:
        """Duration of window in seconds."""
        return self.end_ts - self.start_ts


def make_windows(
    timestamps: np.ndarray,
    window_size_s: float = 5.0,
    stride_s: float = 0.1,
    min_frames: int = 2,
) -> List[WindowSpec]:
    """
    Create sliding windows over a trajectory.

    This is the core function for time-stepping through a scenario.
    At each step, we evaluate the ego's trajectory for the next
    `window_size_s` seconds.

    Args:
        timestamps: Array of timestamps in seconds (T,)
        window_size_s: Size of evaluation window in seconds (default 5.0)
        stride_s: Step size between windows in seconds (default 0.1)
        min_frames: Minimum frames required for valid window (default 2)

    Returns:
        List of WindowSpec objects defining each evaluation window

    Example:
        # Evaluate at 10Hz (every 0.1s), looking ahead 5 seconds
        windows = make_windows(timestamps, window_size_s=5.0, stride_s=0.1)

        for win in windows:
            # Evaluate trajectory from start_idx to end_idx
            ego_segment = ego[win.start_idx:win.end_idx]
            result = evaluate_rules(ego_segment, agents, map)
    """
    if len(timestamps) < min_frames:
        return []

    # Compute dt from timestamps
    if len(timestamps) > 1:
        dt = float(np.median(np.diff(timestamps)))
    else:
        dt = 0.1  # Default to 10Hz

    # Convert to frame counts
    if stride_s <= 0:
        log.warning(f"stride_s={stride_s} is non-positive, clamping to dt={dt}")
    window_frames = max(min_frames, int(round(window_size_s / dt)))
    stride_frames = max(1, int(round(stride_s / dt)))

    T = len(timestamps)
    windows: List[WindowSpec] = []

    start_idx = 0
    while start_idx < T:
        # End index is start + window size, capped at total frames
        end_idx = min(T, start_idx + window_frames)

        # Only create window if it has enough frames
        if end_idx - start_idx >= min_frames:
            windows.append(
                WindowSpec(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_ts=float(timestamps[start_idx]),
                    end_ts=float(timestamps[end_idx - 1]),
                )
            )

        # Stop if we've reached the end
        if end_idx >= T:
            break

        # Move to next window
        start_idx += stride_frames

    return windows


def make_receding_horizon_windows(
    timestamps: np.ndarray,
    horizon_s: float = 8.0,
    step_s: float = 1.0,
    overlap_s: float = 0.0,
) -> List[WindowSpec]:
    """
    Create receding horizon windows for prediction evaluation.

    In receding horizon control/evaluation, at each step we:
    1. Look at current state
    2. Evaluate predicted trajectory for next `horizon_s` seconds
    3. Step forward by `step_s` seconds
    4. Repeat

    Args:
        timestamps: Array of timestamps in seconds (T,)
        horizon_s: Prediction horizon in seconds (default 8.0)
        step_s: Step size between evaluations in seconds (default 1.0)
        overlap_s: Overlap between windows in seconds (default 0.0)

    Returns:
        List of WindowSpec objects

    Example:
        # Evaluate 8-second horizon every 1 second
        windows = make_receding_horizon_windows(timestamps, horizon_s=8.0, step_s=1.0)
    """
    effective_stride = step_s - overlap_s
    return make_windows(
        timestamps, window_size_s=horizon_s, stride_s=effective_stride, min_frames=2
    )


def make_windows_timed(
    T: int,
    dt: float,
    window_size_s: float = 8.0,
    stride_s: float = 2.0,
    min_frames: int = 2,
) -> List[WindowSpec]:
    """
    Create windows from frame count and time step (no timestamps array).

    Convenience function when you don't have explicit timestamps
    but know the number of frames and time step.

    Args:
        T: Total number of frames
        dt: Time step between frames in seconds
        window_size_s: Window size in seconds
        stride_s: Stride between windows in seconds
        min_frames: Minimum frames for valid window

    Returns:
        List of WindowSpec objects
    """
    # Generate synthetic timestamps
    timestamps = np.arange(T) * dt
    return make_windows(
        timestamps=timestamps,
        window_size_s=window_size_s,
        stride_s=stride_s,
        min_frames=min_frames,
    )


def slice_scenario_to_window(scenario_context, window: WindowSpec):
    """
    Extract a windowed portion of a ScenarioContext.

    Creates a new ScenarioContext containing only the frames
    specified by the window.

    Note: Prefer using extract_window() from rule_executor module
    which handles the current context API correctly.

    Args:
        scenario_context: Full ScenarioContext
        window: WindowSpec defining the slice

    Returns:
        New ScenarioContext with sliced data
    """
    from .rule_executor import extract_window

    return extract_window(
        scenario_context,
        window.start_idx,
        window.end_idx,
    )
