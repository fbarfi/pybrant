"""pybrant: Brant's (1990) test of the proportional-odds assumption for
statsmodels' OrderedModel (logit link).

    from pybrant import BrantTest
    bt = BrantTest(results, use_formula=True, by_var=True)
    print(bt.summary())
    r = bt.test_results()        # BrantResult: omnibus_stat, omnibus_df, omnibus_p, by_variable
"""

from .brant import BrantResult, BrantTest

__all__ = ["BrantTest", "BrantResult"]
__version__ = "0.1.0"
