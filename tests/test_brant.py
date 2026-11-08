"""Tests of pybrant.BrantTest against an independent implementation and the
properties the test must have."""

import numpy as np
import pytest
from scipy.stats import chi2

from pybrant import BrantResult, BrantTest

from _reference import brant_reference
from conftest import fit_array


def test_matches_reference_array(po_data, array_res):
    r = BrantTest(array_res, use_formula=False, exog_names=["x1", "x2"]).test_results()
    stat, df, p, by = brant_reference(po_data["y"], po_data[["x1", "x2"]],
                                      {"x1": [0], "x2": [1]})
    assert isinstance(r, BrantResult)
    assert r.omnibus_df == df == (4 - 2) * 2
    assert r.omnibus_stat == pytest.approx(stat, rel=1e-3)
    assert r.omnibus_p == pytest.approx(chi2.sf(r.omnibus_stat, r.omnibus_df))
    tab = r.by_variable.set_index("Test for")
    for name in ("x1", "x2"):
        assert tab.loc[name, "Chi-square"] == pytest.approx(by[name][0], rel=1e-3)
        assert tab.loc[name, "df"] == by[name][1] == 2


def test_formula_equals_array(array_res, formula_res):
    a = BrantTest(array_res, use_formula=False, exog_names=["x1", "x2"]).test_results()
    f = BrantTest(formula_res, use_formula=True).test_results()
    assert f.omnibus_stat == pytest.approx(a.omnibus_stat, rel=1e-3)
    assert list(f.by_variable["Test for"]) == ["x1", "x2"]


def test_factor_is_tested_jointly(po_data, factor_res):
    bt = BrantTest(factor_res, use_formula=True)
    r = bt.test_results()
    tab = r.by_variable.set_index("Test for")
    assert "C(g)" in tab.index
    assert tab.loc["C(g)", "df"] == (4 - 2) * 2               # two dummies
    assert r.omnibus_df == (4 - 2) * bt.K == (4 - 2) * 4
    # the same statistic from the reference, with the dummies grouped
    X = bt.df_x.to_numpy()
    groups = {t: idx for t, idx in bt.term_to_indices.items()}
    stat, _, _, by = brant_reference(bt.y, X, groups)
    assert r.omnibus_stat == pytest.approx(stat, rel=1e-3)
    assert tab.loc["C(g)", "Chi-square"] == pytest.approx(by["C(g)"][0], rel=1e-3)


def test_null_and_violation(po_data, array_res, npo_data):
    null = BrantTest(array_res, use_formula=False).test_results()
    assert null.omnibus_p > 0.001
    viol = BrantTest(fit_array(npo_data, ["x1", "x2"]), use_formula=False,
                     exog_names=["x1", "x2"]).test_results()
    assert viol.omnibus_p < 0.001
    tab = viol.by_variable.set_index("Test for")
    assert tab.loc["x2", "p-value"] < 0.001
    assert tab.loc["x2", "Chi-square"] > tab.loc["x1", "Chi-square"]


def test_standardize_leaves_statistic_unchanged(array_res):
    a = BrantTest(array_res, use_formula=False).test_results()
    b = BrantTest(array_res, use_formula=False, standardize=True).test_results()
    assert b.omnibus_stat == pytest.approx(a.omnibus_stat, rel=1e-3)


@pytest.mark.parametrize("cov", ["auto", "oim", "hc0", "ridge"])
def test_covariance_options_run(array_res, cov):
    r = BrantTest(array_res, use_formula=False, cov_fallback=cov).test_results()
    assert np.isfinite(r.omnibus_stat) and 0.0 <= r.omnibus_p <= 1.0


def test_oim_equals_auto(array_res):
    a = BrantTest(array_res, use_formula=False, cov_fallback="auto").test_results()
    o = BrantTest(array_res, use_formula=False, cov_fallback="oim").test_results()
    assert o.omnibus_stat == pytest.approx(a.omnibus_stat)


def test_names(po_data, array_res):
    assert BrantTest(array_res, use_formula=False).var_names == ["x0", "x1"]
    named = BrantTest(array_res, use_formula=False, data=po_data[["x1", "x2", "y"]])
    assert named.var_names == ["x1", "x2"]


def test_summary_and_table(array_res):
    bt = BrantTest(array_res, use_formula=False, exog_names=["x1", "x2"])
    s = bt.summary()
    assert "Omnibus" in s and "x1" in s and "x2" in s
    table = bt._format_byvar_table(bt.test_results().by_variable)
    assert table.count("\n") == 3                              # header, rule, 2 rows


def test_by_var_off(array_res):
    r = BrantTest(array_res, use_formula=False, by_var=False).test_results()
    assert r.by_variable is None


def test_input_errors(po_data, array_res):
    with pytest.raises(TypeError):
        BrantTest(object(), use_formula=False)
    with pytest.raises(ValueError):
        BrantTest(array_res, use_formula=False, exog_names=["only_one"])
    with pytest.raises(ValueError):
        BrantTest(array_res, use_formula=False, cov_fallback="sandwich")
    with pytest.raises(ValueError):
        BrantTest(array_res, use_formula=False, ridge=-1.0)
    with pytest.raises(ValueError):
        BrantTest(array_res, use_formula=True)                  # no design_info
    two = po_data.assign(y=(po_data["y"] > 1).astype(int))
    with pytest.raises(ValueError):
        BrantTest(fit_array(two, ["x1", "x2"]), use_formula=False)
