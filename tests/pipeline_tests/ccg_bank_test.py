import unittest
from lambeq import CCGBankParser
from parameterized import parameterized

from helpers.UnitTestBaseClass import UnitTestBaseClass
from helpers.ccg_to_diag_test_pipeline import ccg_to_diag_test

config = {
    "draw_result": False,
    "draw_steps": False,
    "type_check_ccg": True,
}

ccgbankparser = CCGBankParser("../data/ccgbank")
trees = ccgbankparser.section2trees(0) # there is a total of 25 sections


class CCGToDiagTests(UnitTestBaseClass):
    """
    Runs the ccg_to_diag_test for all trees in the first section of the ccgbank.
    """
    @parameterized.expand(trees)
    def test_sequence(self, name):
        print(trees[name])
        print(trees[name].deriv())
        self.test_logger = name
        ccg_to_diag_test(self, config, trees[name])
