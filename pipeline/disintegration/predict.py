"""Which information predicts disintegration: the recipe, the dissolution, or both?

Leave-one-formulation-out prediction of ln DT for nested linear models. All are
evaluated on the same formulations so their Q² values can be compared. For
ordinary least squares the leave-one-out residual is exact from the hat matrix,
e_i / (1 - h_ii), so no refitting loop is needed.

* **recipe**: the DoE model selected for log DT (Scheffé composition x
  log-viscosity, reduced). What you know before running any test.
* **dissolution**: ln Td alone, i.e. one dissolution number and no recipe.
* **dissolution x grade**: ln Td with a line per grade (the ANCOVA).
* **recipe + dissolution**: the DoE terms plus ln Td. A gain over *recipe*
  means dissolution carries information about DT that the recipe does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.design.matrix import build_model_matrix, term_names
from pipeline.doe.analysis import ResponseAnalysis


@dataclass(frozen=True)
class LooModel:
    key: str
    label: str
    n: int
    n_params: int
    q2: float
    rmse_ln: float
    observed: np.ndarray
    predicted: np.ndarray
    grades: tuple[str, ...]

    @property
    def fold_error(self) -> float:
        """Typical multiplicative prediction error, e.g. 1.15 = within ~15%."""
        return float(np.exp(self.rmse_ln))


def loo(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """(LOO predictions, Q², RMSE) for OLS of y on x; None when not estimable."""
    n, p = x.shape
    if n <= p + 1 or np.linalg.matrix_rank(x) < p:
        return None
    pinv = np.linalg.pinv(x)
    hat = x @ pinv
    h = np.clip(np.diag(hat), 0.0, 1.0 - 1e-12)
    resid = y - hat @ y
    press_resid = resid / (1.0 - h)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    press = float(np.sum(press_resid**2))
    q2 = 1.0 - press / ss_tot if ss_tot > 0 else float("nan")
    return y - press_resid, q2, float(np.sqrt(press / n))


def _recipe_matrix(frame: pd.DataFrame, ra: ResponseAnalysis) -> np.ndarray:
    comp = frame[["api_wt", "hpmc_wt", "lactose_wt"]].to_numpy(dtype=float) / 100.0
    proc = frame["v_coded"].to_numpy(dtype=float)
    labels = term_names(ra.spec)
    keep = [i for i, name in enumerate(labels) if name in set(ra.kept_terms)]
    return build_model_matrix(comp, proc, ra.spec)[:, keep]


def compare(
    matched: pd.DataFrame, recipe_fit: ResponseAnalysis | None, order: list[str]
) -> tuple[LooModel, ...]:
    usable = matched[~matched["dt_censored"] & np.isfinite(matched["ln_td"])].reset_index(
        drop=True
    )
    if len(usable) < 6:
        return ()
    y = usable["ln_dt"].to_numpy(dtype=float)
    ln_td = usable["ln_td"].to_numpy(dtype=float)
    ln_td_c = ln_td - ln_td.mean()
    grades = [g for g in order if g in set(usable["grade"])]
    gmat = np.column_stack([(usable["grade"] == g).to_numpy(dtype=float) for g in grades])

    candidates: list[tuple[str, str, np.ndarray]] = []
    if recipe_fit is not None:
        recipe = _recipe_matrix(usable, recipe_fit)
        candidates.append(("recipe", "Recipe (DoE terms)", recipe))
    candidates.append(("dissolution", "ln Td only", np.column_stack([np.ones(len(y)), ln_td])))
    candidates.append(
        ("dissolution_grade", "ln Td × grade", np.column_stack([gmat, gmat * ln_td_c[:, None]]))
    )
    if recipe_fit is not None:
        candidates.append(
            ("recipe_dissolution", "Recipe + ln Td",
             np.column_stack([_recipe_matrix(usable, recipe_fit), ln_td]))
        )

    labels = tuple(str(g) for g in usable["grade"])
    out: list[LooModel] = []
    for key, label, x in candidates:
        fit = loo(x, y)
        if fit is None:
            continue
        pred, q2, rmse = fit
        out.append(LooModel(key, label, len(y), x.shape[1], q2, rmse, y, pred, labels))
    return tuple(out)
