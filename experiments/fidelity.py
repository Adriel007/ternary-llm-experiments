from __future__ import annotations

import math
from typing import Optional

class ForwardFaithfulnessError(AssertionError):
    pass

def assert_forward_faithful(
    measured: float,
    expected: float,
    *,
    tol: float,
    label: str = "forward-faithfulness gate",
    relative: bool = False,
) -> float:
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    if not (math.isfinite(measured) and math.isfinite(expected)):
        raise ForwardFaithfulnessError(
            f"{label} FAILED: non-finite value (measured={measured}, expected={expected})"
        )
    abs_dev = abs(measured - expected)
    if relative:
        if expected == 0:
            raise ValueError("relative=True requires a non-zero `expected`")
        dev = abs_dev / abs(expected)
    else:
        dev = abs_dev
    if dev > tol:
        kind = "fractional" if relative else "absolute"
        raise ForwardFaithfulnessError(
            f"{label} FAILED: measured {measured:.6g} vs expected {expected:.6g} "
            f"({kind} dev {dev:.6g} > tol {tol:.6g}) — the forward pass is not faithful; "
            f"aborting (no result downstream of a broken forward is interpretable)."
        )
    return dev

def assert_ppl_faithful(
    measured_ppl: float,
    expected_ppl: float,
    *,
    tol: float = 0.20,
    label: Optional[str] = None,
) -> float:
    return assert_forward_faithful(
        measured_ppl, expected_ppl, tol=tol,
        label=label or f"base-ternary PPL forward-faithfulness gate (tol {tol})",
    )
