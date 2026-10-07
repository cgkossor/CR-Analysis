"""Hierarchy-preserving model reduction and the crossed-model term hierarchy.

Reduction removes terms that the data do not support, but never at the cost of
hierarchy: if an interaction stays, its parent main effects stay too, even when
individually insignificant. A non-hierarchical polynomial is not invariant to
recoding the factors, so its coefficients mean something different depending on
where you put the origin -- which is not a property any formulator should have
to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.design.matrix import (
    CompositionDegree,
    ModelSpec,
    build_model_matrix,
    term_names,
)
from pipeline.surfaces.model import SurfaceFit, fit_surface, sequential_sum_of_squares


@dataclass(frozen=True)
class ReducedModel:
    """The model chosen for a response, and why."""

    spec: ModelSpec
    kept_terms: tuple[str, ...]
    dropped_terms: tuple[str, ...]
    fit: SurfaceFit
    rationale: str


def _parents(term: str) -> set[str]:
    """Terms whose presence is required for ``term`` to be hierarchical.

    A crossed term ``comp:v^k`` has as parents the same composition term at lower
    process powers, and the composition parents at the same power.
    """
    parents: set[str] = set()
    comp, _, proc = term.partition(":")

    if proc:
        power = 2 if proc == "v^2" else 1
        for lower in range(power):
            suffix = "" if lower == 0 else (":v" if lower == 1 else f":v^{lower}")
            parents.add(f"{comp}{suffix}")

    factors = comp.split("*")
    if len(factors) > 1:
        for k in range(1, len(factors)):
            for i in range(len(factors)):
                subset = factors[:i] + factors[i + 1 :]
                if len(subset) == len(factors) - k:
                    name = "*".join(subset)
                    parents.add(f"{name}:{proc}" if proc else name)
    return parents


def enforce_hierarchy(kept: set[str], available: set[str]) -> set[str]:
    """Add back every parent of every kept term."""
    result = set(kept)
    changed = True
    while changed:
        changed = False
        for term in list(result):
            for parent in _parents(term):
                if parent in available and parent not in result:
                    result.add(parent)
                    changed = True
    return result


def choose_composition_degree(
    composition: np.ndarray,
    process: np.ndarray,
    response: np.ndarray,
    process_power: int,
    alpha: float = 0.05,
) -> tuple[CompositionDegree, str]:
    """Pick the composition degree by sequential sums of squares (AC4).

    The richest degree whose incremental F test is significant wins. When nothing
    beyond linear is significant, linear is chosen -- adding terms that the data
    do not support inflates prediction variance without improving prediction, and
    the FDS curve in AC3 shows exactly what that costs.
    """
    degrees: tuple[CompositionDegree, ...] = ("linear", "quadratic", "special_cubic")
    matrices: dict[str, np.ndarray] = {
        str(d): build_model_matrix(
            composition, process, ModelSpec("scheffe", d, process_power)
        )
        for d in degrees
    }
    steps = sequential_sum_of_squares(matrices, response, list(degrees))
    if not steps:
        return "linear", "sequential SS not computable; defaulted to linear"

    chosen: CompositionDegree = "linear"
    for step in steps:
        if step.model != "linear" and np.isfinite(step.p_value) and step.p_value < alpha:
            chosen = step.model  # type: ignore[assignment]

    detail = "; ".join(
        f"{s.model}: +{s.added_terms} terms, F={s.f_value:.2f}, p={s.p_value:.4g}"
        for s in steps
        if s.model != "linear"
    )
    return chosen, f"sequential SS -> {chosen} ({detail})"


def reduce_model(
    composition: np.ndarray,
    process: np.ndarray,
    response: np.ndarray,
    spec: ModelSpec,
    *,
    response_name: str = "response",
    alpha: float = 0.05,
) -> ReducedModel:
    """Drop unsupported terms from ``spec``, preserving hierarchy."""
    full_matrix = build_model_matrix(composition, process, spec)
    labels = term_names(spec)
    full = fit_surface(
        full_matrix, response, labels, response_name=response_name, model_label=spec.label
    )
    if not full.estimable:
        return ReducedModel(spec, tuple(labels), (), full, "full model not estimable")

    # Backward elimination, one term at a time. A single cut at p < alpha in
    # the full fit loses real effects: with three grades the v and v^2 terms
    # overlap, so each can look insignificant alone while together they carry
    # a clear grade effect, and the whole grade dependence was dropped at once
    # (every grade then predicted the same line). Removing only the weakest
    # term that nothing else depends on, and refitting, lets the survivor
    # show its real significance. Linear blending terms are never removed:
    # in a Scheffe model they are the pure-component responses.
    kept = set(labels)
    current = full
    while True:
        protected = {t for t in labels if ":" not in t and "*" not in t}
        for term in kept:
            protected |= _parents(term)
        removable = [
            c for c in current.coefficients
            if c.name not in protected and np.isfinite(c.p_value) and c.p_value >= alpha
        ]
        if not removable:
            break
        weakest = max(removable, key=lambda c: (c.p_value, c.name))
        kept.discard(weakest.name)
        idx = [i for i, name in enumerate(labels) if name in kept]
        current = fit_surface(
            full_matrix[:, idx], response, [labels[i] for i in idx],
            response_name=response_name, model_label=spec.label,
        )
        if not current.estimable:
            kept.add(weakest.name)
            break
    significant = {c.name for c in current.coefficients
                   if np.isfinite(c.p_value) and c.p_value < alpha}
    kept = enforce_hierarchy(kept, set(labels))
    if not kept:
        return ReducedModel(spec, tuple(labels), (), full, "no term reached significance")

    keep_idx = [i for i, name in enumerate(labels) if name in kept]
    reduced_matrix = full_matrix[:, keep_idx]
    reduced_labels = [labels[i] for i in keep_idx]
    reduced = fit_surface(
        reduced_matrix,
        response,
        reduced_labels,
        response_name=response_name,
        model_label=f"{spec.label} (reduced)",
    )

    if not reduced.estimable or reduced.pred_r_squared < full.pred_r_squared - 1e-9:
        return ReducedModel(
            spec,
            tuple(labels),
            (),
            full,
            "reduction did not improve predicted R^2; kept the full model",
        )

    dropped = tuple(name for name in labels if name not in kept)
    added_back = sorted(kept - significant)
    rationale = (
        f"kept {len(reduced_labels)} of {len(labels)} terms at alpha={alpha}; "
        f"predicted R^2 {full.pred_r_squared:.4f} -> {reduced.pred_r_squared:.4f}"
    )
    if added_back:
        rationale += f"; retained for hierarchy: {', '.join(added_back)}"
    return ReducedModel(spec, tuple(reduced_labels), dropped, reduced, rationale)
