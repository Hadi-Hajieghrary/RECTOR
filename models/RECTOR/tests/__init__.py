"""
RECTOR Test Suite

This package contains all tests for validating RECTOR functionality.

Test Hierarchy (Gate-based):

    Gate #1: test_sensitivity.py
        - CRITICAL: Must pass before any infrastructure work
        - Validates conditional model responds to influencer changes
        - Exit 0 = PASS, Exit 1 = FAIL, Exit 2 = ERROR

    Gate #2: test_latency.py (future)
        - Validates batched forward pass < 50ms for B=48

    Gate #3: test_safety.py (future)
        - Validates no ghost interactions (plans within non-coop envelope)

    Gate #4: test_e2e_latency.py (future)
        - Validates full planning loop < 100ms

Usage:
    # Run sensitivity test only (recommended first)
    python models/RECTOR/tests/test_sensitivity.py

    # Run all tests with pytest
    python -m pytest models/RECTOR/tests/ -v
"""
