"""Brant test after an ordered logit fitted with statsmodels.

    python examples/brant_example.py
"""

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from pybrant import BrantTest

rng = np.random.default_rng(42)
n = 1500
df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                   "region": rng.choice(["Africa", "Asia", "Europe"], n)})
# x2 changes the spread of the latent variable: proportional odds fails for x2
latent = 0.8 * df.x1 + np.exp(0.6 * df.x2) * rng.logistic(size=n)
df["y"] = pd.Categorical(np.searchsorted([-0.8, 0.6, 1.8], latent), ordered=True)

# formula-based model: a factor is tested jointly over its dummies
res = OrderedModel.from_formula("y ~ x1 + x2 + C(region)", df, distr="logit").fit(
    method="bfgs", disp=False)
bt = BrantTest(res, use_formula=True, by_var=True)
print(bt.summary())

# array-based model
X = df[["x1", "x2"]].to_numpy()
res2 = OrderedModel(df["y"], X, distr="logit").fit(method="bfgs", disp=False)
r = BrantTest(res2, use_formula=False, exog_names=["x1", "x2"]).test_results()
print(r.omnibus_stat, r.omnibus_df, r.omnibus_p)
print(r.by_variable)
