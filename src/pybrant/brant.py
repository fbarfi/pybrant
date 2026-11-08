"""Brant test for the proportional-odds (ordinal logit) model.

This module implements Brant's (1990) test that the slope coefficients are
invariant across the cumulative binary splits in a proportional-odds model.
It works with `statsmodels`' `OrderedModel` fitted via a logit link and
supports both formula-based and array-based specifications.

Overview
--------
Given an ordinal response with J ordered categories, a proportional-odds
model implies that the K-dimensional slope vector is equal across the J-1
cumulative logits. Brant's test fits J-1 separate binary logit models for the
cumulative splits, collects their slope estimates and covariances, and then
forms adjacent-threshold contrasts. The resulting quadratic form follows a
chi-square distribution under the null.

Key features
------------
- Accepts `OrderedResults` or `OrderedResultsWrapper`.
- Works for formula-based and array-based models.
- Computes the omnibus test and, optionally, per-variable tests that jointly
  assess all expanded columns for each original term (e.g., a factor's dummies).
- Uses Moore–Penrose pseudoinverses for numerical stability.

References
----------
Brant, R. (1990). Assessing proportionality in the proportional odds model
for ordinal logistic regression. *Biometrics*, 46(4), 1171–1178.

Example
-------
>>> import pandas as pd, numpy as np
>>> from statsmodels.miscmodels.ordinal_model import OrderedModel
>>> from pybrant import BrantTest
>>> np.random.seed(42)
>>> n = 500
>>> df = pd.DataFrame({"x1": np.random.randn(n), "x2": np.random.randn(n)})
>>> df["y"] = np.random.randint(0, 4, size=n)
>>> res = OrderedModel.from_formula("y ~ x1 + x2", df).fit(method="bfgs", disp=False)
>>> bt = BrantTest(res, use_formula=True, by_var=True)
>>> print(bt.summary())
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.linalg import pinv
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel, OrderedResults
from statsmodels.miscmodels.ordinal_model import OrderedResultsWrapper


ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]


def _formula_spec(model):
    """The formula specification of a statsmodels model, or None.

    statsmodels keeps it on ``model.data`` (``Model._handle_data`` leaves it
    there): as ``model_spec`` from 0.15, as ``design_info`` before. It is a
    patsy ``DesignInfo`` or, with the formulaic engine, a formulaic
    ``ModelSpec``.
    """
    data = getattr(model, "data", None)
    for obj in (model, data):
        for name in ("model_spec", "design_info"):
            spec = getattr(obj, name, None) if obj is not None else None
            if spec is not None:
                return spec
    return None


def _term_slices(spec) -> Dict[str, slice]:
    """{term name: column slice} from a patsy DesignInfo (``term_name_slices``)
    or a formulaic ModelSpec (``term_slices``, keyed by Term objects)."""
    named = getattr(spec, "term_name_slices", None)
    if named is not None:
        return dict(named)
    by_term = getattr(spec, "term_slices", None)
    if by_term is not None:
        return {str(t): s for t, s in by_term.items()}
    raise ValueError("cannot read the term slices of the formula specification "
                     f"({type(spec).__name__}).")


@dataclass
class BrantResult:
    """Container for Brant test results.

    Parameters
    ----------
    omnibus_stat:
        Chi-square statistic for the omnibus proportional-odds test.
    omnibus_df:
        Degrees of freedom for the omnibus test, equal to `(J-2) * K`.
    omnibus_p:
        Upper-tail p-value for the omnibus test.
    by_variable:
        Optional DataFrame with per-term tests containing columns
        `['Test for', 'Chi-square', 'df', 'p-value']`.
    """

    omnibus_stat: float
    omnibus_df: int
    omnibus_p: float
    by_variable: Optional[pd.DataFrame]


class BrantTest:
    """Brant test for the proportional-odds assumption.

    This class computes Brant's omnibus test and optional per-variable tests
    from a fitted `statsmodels` `OrderedModel` result with a logit link.

    Parameters
    ----------
    fitted:
        A fitted `OrderedResults` or `OrderedResultsWrapper` from
        `statsmodels.miscmodels.ordinal_model.OrderedModel`.
    use_formula:
        Indicates whether the original model was specified using a Patsy
        formula. If `True`, term-level grouping follows `design_info`.
    data:
        The DataFrame used to fit the model (needed for better name recovery
        when `use_formula=False` and `exog_names` are not provided).
    exog_names:
        Names for the exogenous variables when `use_formula=False`.
    by_var:
        If `True`, compute per-variable tests in addition to the omnibus test.
    standardize:
        If `True`, z-score standardize predictors internally for all binary
        logits before fitting; names remain unchanged.
    cov_fallback:
        Strategy for slope covariance of the binary logits. One of
        {"auto", "oim", "hc0", "ridge"}. "auto" tries OIM then falls back to
        ridge Fisher information.
    ridge:
        Nonnegative ridge strength used when `cov_fallback='ridge'` or when
        `cov_fallback='auto'` falls back. Default `1e-8`.

    Notes
    -----
    - Requires at least three ordered outcome levels.
    - Assumes a logit link; results are not validated for other links.
    - Intercepts (cutpoints) are handled by the ordinal model; binary logits
      include an explicit intercept which is removed before stacking slopes.
    """

    def __init__(
        self,
        fitted: Union[OrderedResults, OrderedResultsWrapper],
        *,
        use_formula: bool,
        data: Optional[pd.DataFrame] = None,
        exog_names: Optional[List[str]] = None,
        by_var: bool = True,
        standardize: bool = False,
        cov_fallback: str = "auto",
        ridge: float = 1e-8,
    ) -> None:
        if not isinstance(fitted, (OrderedResults, OrderedResultsWrapper)):
            raise TypeError("`fitted` must be statsmodels OrderedResults/Wrapper")
        self.res = fitted
        self.model = fitted.model
        self.use_formula = bool(use_formula)
        self.by_var = bool(by_var)
        self.input_data = data
        self._exog_names = exog_names
        self.standardize = bool(standardize)
        self.cov_fallback = str(cov_fallback).lower()
        if self.cov_fallback not in {"auto", "oim", "hc0", "ridge"}:
            raise ValueError("cov_fallback must be one of {'auto','oim','hc0','ridge'}")
        self.ridge = float(ridge) if ridge is not None else 0.0
        if self.ridge < 0:
            raise ValueError("ridge must be nonnegative")

        self.y = np.asarray(self.model.endog)
        self.X = np.asarray(self.model.exog)
        self.nobs = int(self.model.nobs)
        self.levels = np.sort(np.unique(self.y))
        if self.levels.ndim != 1 or self.levels.size < 3:
            raise ValueError("Brant test needs >= 3 ordered outcome levels")
        self.Jm1 = int(self.levels.size - 1)
        self.K = int(self.X.shape[1])

        if self.use_formula:
            spec = _formula_spec(self.model)
            if spec is None:
                raise ValueError(
                    "use_formula=True but the model carries no formula specification "
                    "(model_spec or design_info); was it fitted with "
                    "OrderedModel.from_formula?")
            self.var_names = list(spec.column_names)
            if len(self.var_names) != self.K:
                raise ValueError(
                    f"the formula specification has {len(self.var_names)} columns and "
                    f"the model exog has {self.K}")
            self.term_to_indices: Dict[str, List[int]] = {}
            for term, slc in _term_slices(spec).items():
                if term.lower() in ("intercept", "1"):
                    continue
                self.term_to_indices[term] = list(range(slc.start, slc.stop))
        else:
            if self._exog_names is not None:
                if len(self._exog_names) != self.K:
                    raise ValueError("len(exog_names) must match number of predictors")
                self.var_names = list(self._exog_names)
            elif self.input_data is not None:
                yname = getattr(self.model, "endog_names", None)
                cols = [c for c in self.input_data.columns if c != yname]
                self.var_names = cols if len(cols) == self.K else [f"x{i}" for i in range(self.K)]
            else:
                self.var_names = [f"x{i}" for i in range(self.K)]
            self.term_to_indices = {name: [i] for i, name in enumerate(self.var_names)}

        self.df_x = pd.DataFrame(self.X, columns=self.var_names)
        self.df_y = pd.Series(self.y, name="y")
        if self.standardize:
            mu = self.df_x.mean(axis=0).to_numpy()
            sd = self.df_x.std(axis=0, ddof=0).to_numpy()
            sd = np.where(sd == 0, 1.0, sd)
            self._mu = mu
            self._sd = sd
            self.X_fit = (self.X - mu) / sd
        else:
            self._mu = None
            self._sd = None
            self.X_fit = self.X

    def _prepare_binary_data(self) -> pd.DataFrame:
        """Construct binary outcomes for each cumulative split.

        Returns
        -------
        DataFrame
            A DataFrame containing columns `z_m` for `m=0..J-2` where
            `z_m = 1{y > level_m}`, along with the original exogenous matrix.
        """
        z = {f"z_{m}": (self.df_y.values > self.levels[m]).astype(int) for m in range(self.Jm1)}
        return pd.concat([pd.DataFrame(z, index=self.df_x.index), self.df_x], axis=1)

    def _fit_binary_logits(self) -> Tuple[List[sm.discrete.discrete_model.BinaryResults], np.ndarray, np.ndarray]:
        """Fit binary logits for each cumulative split and collect slopes and covariances.

        Returns
        -------
        models:
            List of fitted `Logit` results, length `J-1`.
        beta_star:
            Stacked slope vector of length `(J-1)*K`, ordered by split then variable.
        V:
            Block-diagonal covariance matrix for `beta_star` of shape `((J-1)K, (J-1)K)`.
        """
        tmp = self._prepare_binary_data()
        Xc = sm.add_constant(self.X_fit, has_constant="add")
        models = []
        betas = []
        covs = []
        for m in range(self.Jm1):
            y_m = tmp[f"z_{m}"].values
            mod = sm.Logit(y_m, Xc).fit(method="bfgs", disp=False, maxiter=1000)
            models.append(mod)
            p = np.asarray(mod.params)
            if self.cov_fallback == "oim":
                try:
                    V_full = np.asarray(mod.cov_params())
                except Exception:
                    w = (mod.predict() * (1.0 - mod.predict())).astype(float)
                    XtWX = Xc.T @ (w[:, None] * Xc)
                    V_full = pinv(XtWX + self.ridge * np.eye(XtWX.shape[0]))
            elif self.cov_fallback == "hc0":
                p_hat = mod.predict()
                u = (y_m - p_hat).astype(float)
                w = (p_hat * (1.0 - p_hat)).astype(float)
                bread = pinv(Xc.T @ (w[:, None] * Xc))
                meat = Xc.T @ (Xc * (u * u)[:, None])          # X' diag(u^2) X
                V_full = bread @ meat @ bread
            elif self.cov_fallback == "ridge":
                p_hat = mod.predict()
                w = (p_hat * (1.0 - p_hat)).astype(float)
                XtWX = Xc.T @ (w[:, None] * Xc)
                V_full = pinv(XtWX + self.ridge * np.eye(XtWX.shape[0]))
            else:
                try:
                    V_full = np.asarray(mod.cov_params())
                except Exception:
                    p_hat = mod.predict()
                    w = (p_hat * (1.0 - p_hat)).astype(float)
                    XtWX = Xc.T @ (w[:, None] * Xc)
                    V_full = pinv(XtWX + self.ridge * np.eye(XtWX.shape[0]))
            betas.append(p[1:])
            covs.append(V_full[1:, 1:])
        beta_star = np.hstack(betas)
        V = np.zeros((self.Jm1 * self.K, self.Jm1 * self.K))
        for m in range(self.Jm1):
            i0, i1 = m * self.K, (m + 1) * self.K
            V[i0:i1, i0:i1] = covs[m]
        return models, beta_star, V

    def _cross_covariances(self, models, V: np.ndarray) -> np.ndarray:
        """Fill off-diagonal covariance blocks via weight-matrix formula.

        Parameters
        ----------
        models:
            Sequence of fitted `Logit` models for each split.
        V:
            Current covariance matrix with diagonal blocks set.

        Returns
        -------
        ndarray
            Covariance matrix with off-diagonal slope-slope blocks populated.
        """
        X = sm.add_constant(self.X_fit, has_constant="add")
        for m in range(self.Jm1 - 1):
            pm = models[m].predict()
            Wm = np.diag((pm * (1.0 - pm)).astype(float))
            XmWX_inv = pinv(X.T @ Wm @ X)
            for l in range(m + 1, self.Jm1):
                pl = models[l].predict()
                Wl = np.diag((pl * (1.0 - pl)).astype(float))
                Wml = np.diag((pl - pm * pl).astype(float))
                XlWX_inv = pinv(X.T @ Wl @ X)
                Mfull = XmWX_inv @ (X.T @ Wml @ X) @ XlWX_inv
                M = Mfull[1:, 1:]
                i_m = slice(m * self.K, (m + 1) * self.K)
                i_l = slice(l * self.K, (l + 1) * self.K)
                V[i_m, i_l] = M
                V[i_l, i_m] = M.T
        return V

    @staticmethod
    def _contrast_matrix(Jm1: int, K: int) -> np.ndarray:
        """Build the omnibus contrast matrix.

        Parameters
        ----------
        Jm1:
            Number of cumulative splits, `J-1`.
        K:
            Number of predictors.

        Returns
        -------
        ndarray
            Matrix `D = kron(M, I_K)` where `M` encodes adjacent differences
            across splits, shape `((J-2)K, (J-1)K)`.
        """
        if Jm1 < 2:
            raise ValueError("Need at least J-1 >= 2")
        M = np.zeros((Jm1 - 1, Jm1))
        for r in range(Jm1 - 1):
            M[r, r] = 1.0
            M[r, r + 1] = -1.0
        D = np.kron(M, np.eye(K))
        return D

    @staticmethod
    def _selector_for_var(K: int, Jm1: int, j: int) -> np.ndarray:
        """Selector for coefficients of variable `j` across all splits.

        Returns
        -------
        ndarray
            Matrix of shape `(J-1, (J-1)K)` that picks the `j`-th slope from
            each split in the stacked parameter vector.
        """
        S = np.zeros((Jm1, Jm1 * K))
        for r in range(Jm1):
            S[r, r * K + j] = 1.0
        return S

    def _compute_stats(self) -> Tuple[float, int, float, Optional[pd.DataFrame]]:
        """Compute omnibus and per-variable Brant statistics.

        Returns
        -------
        X2:
            Omnibus chi-square statistic.
        df:
            Degrees of freedom for the omnibus test.
        p:
            P-value for the omnibus test.
        by_var_df:
            Optional DataFrame of per-variable tests.
        """
        models, beta_star, V = self._fit_binary_logits()
        V = self._cross_covariances(models, V)
        D = self._contrast_matrix(self.Jm1, self.K)
        dd = D @ beta_star
        VV = D @ V @ D.T
        X2 = float(dd.T @ pinv(VV) @ dd)
        df = D.shape[0]
        p = float(chi2.sf(X2, df))

        by_var_df = None
        if self.by_var:
            rows = []
            for name, idxs in self.term_to_indices.items():
                if not idxs:
                    continue
                S_blocks = [self._selector_for_var(self.K, self.Jm1, j) for j in idxs]
                S = np.vstack(S_blocks)
                M = np.zeros((self.Jm1 - 1, self.Jm1))
                for r in range(self.Jm1 - 1):
                    M[r, r] = 1.0
                    M[r, r + 1] = -1.0
                D_term = np.kron(np.eye(len(idxs)), M)
                ddj = D_term @ (S @ beta_star)
                VVj = D_term @ (S @ V @ S.T) @ D_term.T
                X2j = float(ddj.T @ pinv(VVj) @ ddj)
                dfj = (self.Jm1 - 1) * len(idxs)
                rows.append({"Test for": name, "Chi-square": X2j, "df": dfj, "p-value": float(chi2.sf(X2j, dfj))})
            by_var_df = pd.DataFrame(rows) if rows else None
        return X2, df, p, by_var_df

    def _format_byvar_table(self, df: pd.DataFrame, width: int = 80) -> str:
        """Return an aligned ASCII table for by-variable results."""
        cols = ["Test for", "Chi-square", "df", "p-value"]
        df = df.copy()[cols]
        def fmt_p(x: float) -> str:
            return "<0.001" if float(x) < 1e-3 else f"{float(x):.3f}"
        df["Chi-square"] = df["Chi-square"].astype(float).map(lambda x: f"{x:.3f}")
        df["df"] = df["df"].astype(int).map(str)
        df["p-value"] = df["p-value"].astype(float).map(fmt_p)
        minw = {"Test for": 8, "Chi-square": 10, "df": 2, "p-value": 7}
        widths = {c: max(minw[c], len(c), df[c].map(len).max()) for c in cols}
        total = sum(widths.values()) + 3 * (len(cols) - 1)
        if total > width:
            overflow = total - width
            cut = min(overflow, widths["Test for"] - minw["Test for"])
            widths["Test for"] -= cut
            df["Test for"] = df["Test for"].map(lambda s: (s[: widths["Test for"] - 1] + "…") if len(s) > widths["Test for"] else s)
        header = f"{cols[0]:<{widths[cols[0]]}}  {cols[1]:>{widths[cols[1]]}}  {cols[2]:>{widths[cols[2]]}}  {cols[3]:>{widths[cols[3]]}}"
        sep = f"{'-' * widths[cols[0]]}  {'-' * widths[cols[1]]}  {'-' * widths[cols[2]]}  {'-' * widths[cols[3]]}"
        rows = [
            f"{str(df.iloc[i][cols[0]]):<{widths[cols[0]]}}  {str(df.iloc[i][cols[1]]):>{widths[cols[1]]}}  {str(df.iloc[i][cols[2]]):>{widths[cols[2]]}}  {str(df.iloc[i][cols[3]]):>{widths[cols[3]]}}"
            for i in range(len(df))
        ]
        return "\n".join([header, sep] + rows)  # aligned ASCII table

    def test_results(self) -> BrantResult:
        """Return structured Brant test results.

        Returns
        -------
        BrantResult
            Object containing omnibus statistic, degrees of freedom, p-value,
            and optional per-variable tests.
        """
        X2, df, p, by_var = self._compute_stats()
        return BrantResult(omnibus_stat=X2, omnibus_df=df, omnibus_p=p, by_variable=by_var)

    @property
    def is_formula_model(self) -> bool:
        """Compatibility alias used by the summary template."""
        return bool(self.use_formula)

    def summary(self) -> str:
        """Generate a clean, report-ready summary string.

        Returns
        -------
        str
            Formatted string containing test results and interpretation.
        """
        r = self.test_results()
        model_type = "Formula-based" if self.use_formula else "Array-based"

        header = [
            "Brant Test of Parallel Regression Assumption",
            "=" * 50,
            f"Model type: {model_type}",
            f"Number of observations: {self.nobs}",
            f"Number of predictors: {len(self.var_names)}",
            f"Number of outcome levels: {self.levels.size}",
            "-" * 50,
            "Test Results:",
        ]

        rows = [
            {"Test for": "Omnibus", "Chi-square": r.omnibus_stat, "df": r.omnibus_df, "p-value": r.omnibus_p}
        ]
        if self.by_var and r.by_variable is not None and not r.by_variable.empty:
            rows_df = pd.DataFrame(rows)
            results_df = pd.concat(
                [rows_df, r.by_variable.loc[:, ["Test for", "Chi-square", "df", "p-value"]]],
                ignore_index=True,
            )
        else:
            results_df = pd.DataFrame(rows)

        results_str = results_df.round({"Chi-square": 3, "p-value": 3}).to_string(index=False)

        interpretation = [
            "-" * 50,
            "Interpretation:",
            "* H0: Parallel Regression Assumption holds",
            "* p-value < 0.05 suggests violation of the parallel regression assumption",
            "* Test examines if effects are consistent across outcome levels",
        ]

        return "\n".join(header + [results_str] + interpretation)
