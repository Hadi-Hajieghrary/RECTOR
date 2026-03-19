# waymo_rule_eval/utils/wre_logging.py
# -*- coding: utf-8 -*-
"""Simple logging utilities for waymo rule evaluation."""
from __future__ import annotations

import contextvars
import logging
from typing import Any, Dict, List, Tuple

# Context variables for structured logging
_run_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "run_id", default="default"
)
_scenario_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "scenario_id", default=""
)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_ctx(**kwargs) -> List[Tuple[contextvars.ContextVar, contextvars.Token]]:
    """Set context variables for logging. Returns tokens for reset."""
    tokens = []
    if "run_id" in kwargs:
        tokens.append((_run_id, _run_id.set(kwargs["run_id"])))
    if "scenario_id" in kwargs:
        tokens.append((_scenario_id, _scenario_id.set(kwargs["scenario_id"])))
    return tokens


def reset_ctx(tokens: List[Tuple[contextvars.ContextVar, contextvars.Token]]) -> None:
    """Reset context variables using tokens from set_ctx."""
    for var, token in tokens:
        var.reset(token)


def get_run_id() -> str:
    """Get the current run ID."""
    return _run_id.get()


def get_scenario_id() -> str:
    """Get the current scenario ID."""
    return _scenario_id.get()
