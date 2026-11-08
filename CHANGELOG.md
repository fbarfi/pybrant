# Changelog

## 0.1.0 (2026-11-08)

First release.

* `BrantTest` for fitted `statsmodels` `OrderedModel` results (logit link),
  formula- or array-based: the omnibus test and per-term tests (a factor is
  tested jointly over its dummies).
* Slope covariances of the binary logits from the observed information, with
  HC0 and ridge alternatives (`cov_fallback`), and optional internal
  standardization of the predictors.
* `BrantResult` with the statistics, and a report-ready `summary()`.

Fixes relative to the pre-release script (`brant_test.py`):

* formula models: the formula specification is read where statsmodels keeps it,
  on `model.data`, as `model_spec` (statsmodels 0.15) or `design_info` (0.14),
  with either the patsy or the formulaic engine;
* `cov_fallback="hc0"`: the robust "meat" is `X' diag(u^2) X` (the previous
  expression raised a shape error);
* the aligned by-variable table now has line breaks.
