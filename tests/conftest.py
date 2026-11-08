"""Data and fitted OrderedModel results shared by the tests."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.miscmodels.ordinal_model import OrderedModel

CUTS = np.array([-0.8, 0.6, 1.8])          # four outcome categories


def _ordinal(latent):
    return np.searchsorted(CUTS, latent).astype(int)


@pytest.fixture(scope="session")
def po_data():
    """Proportional odds holds: logistic latent variable with a common slope."""
    rng = np.random.default_rng(42)
    n = 1500
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                       "g": rng.choice(["a", "b", "c"], n)})
    eff = {"a": 0.0, "b": 0.4, "c": -0.3}
    lat = 0.8 * df.x1 - 0.5 * df.x2 + df.g.map(eff) + rng.logistic(size=n)
    df["y"] = _ordinal(lat)
    return df


@pytest.fixture(scope="session")
def npo_data():
    """Proportional odds violated in x2: the latent scale grows with x2."""
    rng = np.random.default_rng(7)
    n = 3000
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    lat = 0.8 * df.x1 + np.exp(0.7 * df.x2) * rng.logistic(size=n)
    df["y"] = _ordinal(lat)
    return df


def fit_array(df, cols):
    return OrderedModel(df["y"].to_numpy(), df[cols].to_numpy(float),
                        distr="logit").fit(method="bfgs", disp=False, maxiter=2000)


def fit_formula(df, formula):
    d = df.copy()
    d["y"] = pd.Categorical(d["y"], ordered=True)
    return OrderedModel.from_formula(formula, d, distr="logit").fit(
        method="bfgs", disp=False, maxiter=2000)


@pytest.fixture(scope="session")
def array_res(po_data):
    return fit_array(po_data, ["x1", "x2"])


@pytest.fixture(scope="session")
def formula_res(po_data):
    return fit_formula(po_data, "y ~ x1 + x2")


@pytest.fixture(scope="session")
def factor_res(po_data):
    return fit_formula(po_data, "y ~ x1 + x2 + C(g)")
