import unittest

from lambeq import BobcatParser
from parameterized import parameterized

from pipeline.ccg_to_diag_test_pipeline import ccg_to_diag_test

sentences = [
    'Looking around he found the letter',
    'Alice quickly and slowly runs',
    'Alice and Bob',
    'Alice eats',
    'Alice quickly gives flowers',
    'Alice who Bob likes walks',
    'Alice who likes Bob walks',
    'Alice gives flowers with Bob',
    'Alice gives flowers to Bob',
    'Alice fully loves Bob',
    'Alice quickly eats',
    'Alice quickly eats fish',
    'Alice quickly eats red fish',
    'Alice quickly loves very red Bob',
    'Alice quickly rapidly loudly loves very very red Bob',
    'Alice quickly and rapidly loves Bob and very red Claire',
    'Alice quickly and rapidly loves Bob',
    'I know of Alice loving and hating Bob',
    'I surely know certainly of Alice quickly loving Bob',
    'I know certainly of Alice quickly loving Bob',
    "Alice knows of Bob liking Claire and Dave hating Eve",
    'Alice knows that Bob loves Claire',
    'Alice runs',
    'Alice runs to the kitchen',
    'Alice knows that Bob loves Claire, Claire hates Bob',
    'Alice loves Bob and Claire loves Dave',
    'Alice loves Bob and Bob loves Claire',
    'The lawyers went to work',
    'Before he went to the treasury',
    'I know of Alice loving Bob',
    'I know of Alice quickly loving Bob',
    'I know certainly of Alice quickly loving Bob',
    'I know that Alice hates Bob and Claire loves Dave',
    'Alice loves Bob and Claire hates Dave',
    'Alice quickly loves and hates Bob',
    'I dreamt that Alice went to the shop and bought ice cream',
    'Bob likes flowers that Claire picks',
    'Fred is no longer in the office',
    'How many objects is Mary carrying',
    'Alice quickly and hastily eats'
]
parser = BobcatParser()
config = {
    "draw_result": False,
    "draw_steps": False,
    "type_check_ccg": True,
    "compare_type_expansions": False,
}

class CCGToDiagTests(unittest.TestCase):
    @parameterized.expand(sentences)
    def test_sequence(self, sentence):
        print(sentence)

        ccg_parse = parser.sentence2tree(sentence)

        ccg_to_diag_test(self, config, ccg_parse)

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