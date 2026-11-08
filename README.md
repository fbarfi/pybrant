# pybrant

Brant's (1990) test of the proportional-odds (parallel regression) assumption
for ordered logit models fitted with `statsmodels`' `OrderedModel`.

The proportional-odds model assumes that each slope is the same at every
cumulative split of the outcome, $\Pr(y>j)$, $j=1,\dots,J-1$. Brant's test fits
a binary logit at each split, collects the slope estimates and their joint
covariance, and tests whether the slopes are equal across splits with a Wald
statistic that is $\chi^2$ under the null. It gives an omnibus test over all
predictors and a test for each predictor (a factor is tested jointly over its
dummies).

## Install

```
pip install git+https://github.com/fbarfi/pybrant.git
```

Requires Python 3.9+, numpy, pandas, scipy and statsmodels (0.13 or later).

## Use

```python
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pybrant import BrantTest

res = OrderedModel.from_formula("y ~ x1 + x2 + C(region)", df, distr="logit").fit(
    method="bfgs", disp=False)

bt = BrantTest(res, use_formula=True, by_var=True)
print(bt.summary())

r = bt.test_results()     # BrantResult
r.omnibus_stat, r.omnibus_df, r.omnibus_p
r.by_variable             # DataFrame: Test for, Chi-square, df, p-value
```

For a model fitted on arrays, pass `use_formula=False` and the predictor names:

```python
res = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)
bt = BrantTest(res, use_formula=False, exog_names=["x1", "x2"])
```

(`examples/brant_example.py` runs both.)

## Options

| argument | meaning |
|---|---|
| `use_formula` | the model was fitted with a formula; per-variable tests then follow its terms, so a factor is tested jointly over its dummies |
| `exog_names` / `data` | predictor names for an array-based model (otherwise taken from `data`, or `x0, x1, ...`) |
| `by_var` | also test each predictor (default `True`) |
| `standardize` | standardize the predictors before fitting the binary logits (the statistic is unchanged; can help numerically) |
| `cov_fallback` | covariance of the binary-logit slopes: `"auto"` (observed information, falling back to a ridge-stabilized inverse), `"oim"`, `"hc0"` (heteroskedasticity-robust within each split; the blocks between splits stay model-based) or `"ridge"` |
| `ridge` | ridge strength for the fallback (default `1e-8`) |

## What is computed

With $J$ ordered categories and $K$ predictors, the binary logits for
$z_j = 1\{y>j\}$ give slope vectors $\hat\beta_1,\dots,\hat\beta_{J-1}$. Their
covariance has diagonal blocks from each logit and off-diagonal blocks
$(X'W_mX)^{-1}X'W_{ml}X(X'W_lX)^{-1}$ with $W_{ml}=\mathrm{diag}(\pi_l-\pi_m\pi_l)$
for $m<l$ (Brant 1990). The omnibus statistic is the Wald statistic of the
adjacent differences $\hat\beta_j-\hat\beta_{j+1}$, with $(J-2)K$ degrees of
freedom; the test for a predictor uses its own differences, with $(J-2)$ degrees
of freedom per column. The test requires at least three outcome categories and
a logit link.

## Tests

```
pip install ".[test]"
pytest
```

The tests check the statistics against an independent numpy implementation of
Brant's test, the formula and array interfaces against each other, the joint
test of a factor, rejection under a simulated violation and not under the null,
the invariance to standardization, the covariance options, and input errors.

## Citation

Please cite Brant, R. (1990). Assessing proportionality in the proportional odds
model for ordinal logistic regression. *Biometrics*, 46(4), 1171–1178. See
`CITATION.cff` for this package.

## License

MIT.
