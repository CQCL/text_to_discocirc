import unittest

from lambeq import BobcatParser, CCGBankParser
from parameterized import parameterized

from pipeline.ccg_to_diag_test_pipeline import ccg_to_diag_test

config = {
    "draw_result": False,
    "draw_steps": False,
    "type_check_ccg": True,
    "compare_type_expansions": False,
}

ccgbankparser = CCGBankParser("../ccgbank_1_1")
trees = ccgbankparser.section2trees(1)


class CCGToDiagTests(unittest.TestCase):
    @parameterized.expand(trees)
    def test_sequence(self, name):
        print(trees[name])
        ccg_to_diag_test(self, config, trees[name])

    result_overview = {'ok': 0}
    def tearDown(self):
        if hasattr(self._outcome, 'errors'):
            # Python 3.4 - 3.10  (These two methods have no side effects)
            result = self.defaultTestResult()
            self._feedErrorsToResult(result, self._outcome.errors)
        else:
            # Python 3.11+
            result = self._outcome.result
        ok = all(
            test != self for test, text in result.errors + result.failures)

        # Demo output:  (print short info immediately - not important)
        if ok:
            self.result_overview['ok'] += 1
        for typ, errors in (
        ('ERROR', result.errors), ('FAIL', result.failures)):
            for test, text in errors:
                if test is self:
                    print("Test: ")
                    print(test)
                    #  the full traceback is in the variable `text`
                    msg = [x for x in text.split('\n')[1:]
                           if not x.startswith(' ')][0].split(":")[0]
                    if msg in self.result_overview.keys():
                        self.result_overview[msg] += 1
                    else:
                        self.result_overview[msg] = 1

        print(self.result_overview)