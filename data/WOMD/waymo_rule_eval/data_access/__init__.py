"""Data access modules for Waymo Rule Evaluation."""

from .adapter_motion_scenario import MotionScenarioReader

# Backward-compatible alias for legacy code
MotionScenarioAdapter = MotionScenarioReader
