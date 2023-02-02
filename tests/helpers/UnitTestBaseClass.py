import unittest

class UnitTestBaseClass(unittest.TestCase):
    result_overview = {'ok': 0}
    logger_overview = {}
    test_logger = None

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
                    #  the full traceback is in the variable `text`
                    msg = [x for x in text.split('\n')[1:]
                           if not x.startswith(' ')][0].split(":")[0]
                    if msg in self.result_overview.keys():
                        self.result_overview[msg] += 1
                        if self.test_logger is not None:
                            self.logger_overview[msg].append(self.test_logger)
                    else:
                        self.result_overview[msg] = 1
                        if self.test_logger is not None:
                            self.logger_overview[msg] = [self.test_logger]

        self.test_logger = None
        print(self.result_overview)
        print(self.logger_overview)
