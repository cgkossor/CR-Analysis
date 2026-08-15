"""Model-matrix construction for a mixture-by-process-variable design.

The composition components sum to a constant, so they are exactly collinear and
only two of the three are independent. Two equivalent parameterisations are
supported, and the equivalence is not a detail -- it is the reason the A1 result
does not, by itself, invalidate a surface:

``scheffe``
    Canonical Scheffe polynomial, **no intercept**: ``b1 x1 + b2 x2 + b3 x3``
    (+ cross terms, + ``x1 x2 x3``). Coefficients read directly as the response
    at each pure component, which is what a formulator wants.

``slack``
    One component is dropped and the rest modelled with an intercept, in the
    ordinary polynomial way.

For matching degree these two span the **identical function space** on the
simplex: substituting ``x3 = S - x1 - x2`` turns one into the other. They give
the same fitted values, the same predictions and the same cross-validated error.
What is fatal is neither of them -- it is fitting an intercept *together with*
all three components, which is exactly singular.

The process variable is coded ``log10(nominal viscosity)`` scaled to [-1, +1]
across the tested grades, entering as a continuous factor (AC4). With three
grades only powers 0, 1 and 2 are identifiable; anything higher is dropped and
reported rather than silently absorbed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np

COMPONENTS: tuple[str, str, str] = ("api", "hpmc", "lactose")

Parameterisation = Literal["scheffe", "slack", "singular"]
CompositionDegree = Literal["linear", "quadratic", "special_cubic"]

#: Number of process-variable powers that a k-level factor can identify.
def max_process_power(n_levels: int) -> int:
    """Highest identifiable polynomial power for a factor with ``n_levels`` levels."""
    return max(int(n_levels) - 1, 0)


@dataclass(frozen=True)
class ModelSpec:
    """A crossed composition x process-variable model."""

    parameterisation: Parameterisation
    composition_degree: CompositionDegree
    process_power: int
    dropped_component: str | None = None

    @property
    def label(self) -> str:
        base = f"{self.parameterisation}:{self.composition_degree}"
        return f"{base} x v^{self.process_power}"


def _composition_terms(
    degree: CompositionDegree, parameterisation: Parameterisation, dropped: int
) -> list[tuple[str, tuple[int, ...]]]:
    """Return ``(name, component-index-tuple)`` for the composition side.

    An empty index tuple denotes the intercept, which exists only in the slack
    parameterisation. Scheffe form has none: its linear terms already span the
    constant, since ``x1 + x2 + x3`` is itself constant on the simplex.
    """
    if parameterisation == "scheffe":
        keep = [0, 1, 2]
        terms: list[tuple[str, tuple[int, ...]]] = [
            (COMPONENTS[i], (i,)) for i in keep
        ]
        if degree in ("quadratic", "special_cubic"):
            terms += [
                (f"{COMPONENTS[i]}*{COMPONENTS[j]}", (i, j))
                for i, j in itertools.combinations(keep, 2)
            ]
        if degree == "special_cubic":
            terms.append(("api*hpmc*lactose", (0, 1, 2)))
        return terms

    # The "singular" form is deliberately over-parameterised: an intercept
    # alongside ALL THREE components. It exists so the diagnostics can
    # demonstrate the failure AC3 warns about rather than merely assert it.
    # It is never fitted to a response.
    is_singular = parameterisation == "singular"
    keep = [0, 1, 2] if is_singular else [i for i in range(3) if i != dropped]

    terms = [("intercept", ())]
    terms += [(COMPONENTS[i], (i,)) for i in keep]
    if degree in ("quadratic", "special_cubic"):
        terms += [(f"{COMPONENTS[i]}^2", (i, i)) for i in keep]
        terms += [
            (f"{COMPONENTS[i]}*{COMPONENTS[j]}", (i, j))
            for i, j in itertools.combinations(keep, 2)
        ]
    if degree == "special_cubic":
        # Under x3 = S - x1 - x2 the Scheffe term x1*x2*x3 becomes
        # x1*x2*(S - x1 - x2): ONE new dimension, not two. Adding x1^2*x2 and
        # x1*x2^2 separately would span a strictly larger space and the two
        # parameterisations would stop matching.
        terms.append(("api*hpmc*lactose", (0, 1, 2)))
    return terms


def term_names(spec: ModelSpec) -> list[str]:
    """Human-readable names of every term in ``spec``, in matrix-column order."""
    dropped = COMPONENTS.index(spec.dropped_component) if spec.dropped_component else 2
    comp = _composition_terms(spec.composition_degree, spec.parameterisation, dropped)
    names: list[str] = []
    for power in range(spec.process_power + 1):
        for name, _ in comp:
            if power == 0:
                names.append(name)
            elif name == "intercept":
                names.append(f"v^{power}" if power > 1 else "v")
            else:
                suffix = f"v^{power}" if power > 1 else "v"
                names.append(f"{name}:{suffix}")
    return names


def build_model_matrix(
    composition: np.ndarray, process: np.ndarray, spec: ModelSpec
) -> np.ndarray:
    """Assemble the model matrix.

    ``composition`` is ``(n, 3)`` of component *fractions* (rows summing to the
    mixture total). ``process`` is the coded process variable, length ``n``.
    """
    comp = np.asarray(composition, dtype=float)
    proc = np.asarray(process, dtype=float).reshape(-1)
    if comp.ndim != 2 or comp.shape[1] != 3:
        raise ValueError(f"composition must be (n, 3), got {comp.shape}")
    if comp.shape[0] != proc.shape[0]:
        raise ValueError(
            f"composition has {comp.shape[0]} rows but process has {proc.shape[0]}"
        )

    dropped = COMPONENTS.index(spec.dropped_component) if spec.dropped_component else 2
    terms = _composition_terms(spec.composition_degree, spec.parameterisation, dropped)

    columns: list[np.ndarray] = []
    for power in range(spec.process_power + 1):
        pv = proc**power
        for _, idx in terms:
            base = np.ones(comp.shape[0]) if not idx else np.prod(comp[:, list(idx)], axis=1)
            columns.append(base * pv)
    return np.column_stack(columns)


def code_process(values: np.ndarray, levels: np.ndarray | None = None) -> np.ndarray:
    """Scale ``log10(viscosity)`` onto [-1, +1] across the tested grades.

    Coding uses the *tested* range, so +/-1 marks the boundary of the region the
    design actually explored. Anything outside is extrapolation, which G3 forbids
    silently.
    """
    v = np.asarray(values, dtype=float)
    ref = v if levels is None else np.asarray(levels, dtype=float)
    lo, hi = float(np.min(ref)), float(np.max(ref))
    if hi == lo:
        return np.zeros_like(v)
    return 2.0 * (v - lo) / (hi - lo) - 1.0
