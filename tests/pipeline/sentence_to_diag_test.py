import traceback
import unittest

from lambeq import BobcatParser
from parameterized import parameterized

from discocirc.diag import expand_s_types
from discocirc.diag import Frame
from discocirc.expr import Expr
from discocirc.expr import expr_to_diag
from discocirc.expr import pull_out
from discocirc.expr import type_expand
from discocirc.helpers.closed import biclosed_to_closed, Ty

sentences = [
    'Looking around he found the letter',
    'Alice quickly and slowly runs',
    'Alice and Bob',
    'Alice eats',
    'Alice quickly gives flowers',
    'Alice quickly eats',
    'Alice who Bob likes walks',  # (composition)
    'Alice who likes Bob walks',
    'Alice gives flowers with Bob',
    'Alice gives flowers to Bob',
    'Alice quickly loves very red Bob',
    'Alice fully loves Bob',
    'Alice quickly eats',
    'Alice quickly eats fish',
    'Alice quickly eats red fish',
    'Alice quickly loves very red Bob',
    'Alice quickly rapidly loudly loves very very red Bob',
    'Alice quickly and rapidly loves Bob and very red Claire',
    'Alice quickly and rapidly loves Bob',
    'I know of Alice loving and hating Bob',
    'I surely know certainly of Alice quickly loving Bob',  # (composition)
    'I know certainly of Alice quickly loving Bob',  # (composition, UNARY)
    "Alice knows of Bob liking Claire and Dave hating Eve",  # (UNARY)
    'Alice knows that Bob loves Claire',
    'Alice runs',
    'Alice runs to the kitchen',
    'Alice knows that Bob loves Claire, Claire hates Bob',  # (UNARY)
    'Alice loves Bob and Claire loves Dave',
    # (TODO: runs into recursion error)
    'Alice loves Bob and Bob loves Claire',
    'Alice knows that Bob loves Claire',
    'The lawyers went to work',
    'Before he went to the treasury',  # TODO: pulling out does not work
    'I know of Alice loving Bob',
    'I know of Alice quickly loving Bob',
    'I know certainly of Alice quickly loving Bob',  # (composition, UNARY)
    'I know that Alice hates Bob and Claire loves Dave',
    'Alice loves Bob and Claire hates Dave',
    'Alice knows that Bob loves Claire',
    'Alice quickly loves and hates Bob',
    'I dreamt that Alice went to the shop and bought ice cream',
    'Bob likes flowers that Claire picks',  # (composition)
    'Fred is no longer in the office',
    'How many objects is Mary carrying',
    'Alice quickly and hastily eats'
]
parser = BobcatParser()
config = {
    "draw_result": False,
    "draw_steps": False,
    "check_ccg_types": True,
    "compare_type_expansions": False,
    "compare_ccg_creations": True,
}


def biclosed_to_expr(diagram):
    terms = []
    for box, offset in zip(diagram.boxes, diagram.offsets):
        if not box.dom:  # is word
            typ = biclosed_to_closed(box.cod)
            terms.append(Expr.literal(box.name, typ))
        else:
            if len(box.dom) == 2:
                if box.name.startswith("FA"):
                    term = terms[offset](terms[offset + 1])
                elif box.name.startswith("BA"):
                    term = terms[offset + 1](terms[offset])
                elif box.name.startswith("FC"):
                    x = Expr.literal("temp",
                                     terms[offset + 1].typ.input)
                    term = Expr.lmbda(x, terms[offset](terms[offset + 1](x)))
                elif box.name.startswith("BC") or box.name.startswith(
                        "BX"):
                    x = Expr.literal("temp",
                                     terms[offset].typ.input)
                    term = Expr.lmbda(x,
                                      terms[offset + 1](terms[offset](x)))
                else:
                    raise NotImplementedError
                terms[offset:offset + 2] = [term]
            elif box.name == "Curry(BA(n >> s))":
                x = Expr.literal("temp", Ty('n') >> Ty('s'))
                terms[offset] = Expr.lmbda(x, x(terms[offset]))
            else:
                raise NotImplementedError(box)
    return terms[0]

class CCGToDiagTests(unittest.TestCase):
    def compare_ccg_creation(self, ccg_parse, expr):
        try:
            expr2 = biclosed_to_expr(ccg_parse.to_biclosed_diagram())
            self.assertEqual(expr, expr2)
        except NotImplementedError:
            traceback.print_exc()

    def compare_type_expansions(self, expr):
        test_diag = expr_to_diag(expr)
        expand_s_types(test_diag)
        expr = type_expand(expr)
        test_diag2 = expr_to_diag(expr)
        self.assertEqual(test_diag, test_diag2)

    @parameterized.expand(sentences)
    def test_sequence(self, sentence):
        print(sentence)

        # ------- Step 1: Sentence to ccg -----------
        ccg_parse = parser.sentence2tree(sentence)

        # ------- Step 2: CCG to Expr -----------
        expr = Expr.ccg_to_expr(ccg_parse)
        if config["check_ccg_types"]:
            self.assertTrue(expr.type_check(), msg="Typechecking ccg to expr")

        if config["compare_ccg_creations"]:
            self.compare_ccg_creation(ccg_parse, expr)


        if config["draw_steps"]:
            diag = expr_to_diag(expr)
            diag = (Frame.get_decompose_functor())(diag)
            diag.draw()

        # ------- Step 3: Pulling out -----------
        expr = pull_out(expr)
        if config["check_ccg_types"]:
            self.assertTrue(expr.type_check(),
                            msg="Typechecking pulled out expr")

        # TODO: write test to check that all types have been pulled out
        #  correctly (Issue #14)

        if config["draw_steps"]:
            diag = expr_to_diag(expr)
            diag = (Frame.get_decompose_functor())(diag)
            diag.draw()

        # ------- Step 4: Type expansion -----------
        if config["compare_type_expansions"]:
            self.compare_type_expansions(expr)

        expr = type_expand(expr)
        if config["check_ccg_types"]:
            self.assertTrue(expr.type_check(),
                            msg="Typechecking expanded expr")

        # TODO: write test to check that all types have been expanded
        #  correctly (Issue #14)

        if config["draw_steps"]:
            diag = expr_to_diag(expr)
            diag = (Frame.get_decompose_functor())(diag)
            diag.draw()

        # ------- Step 5: Expr to Diag -----------
        diag = expr_to_diag(expr)
        diag = (Frame.get_decompose_functor())(diag)

        expr_uncurried = Expr.uncurry(expr)
        diag_uncurried = expr_to_diag(expr_uncurried)
        diag_uncurried = (Frame.get_decompose_functor())(diag_uncurried)

        self.assertEqual(diag, diag_uncurried)

        if config["draw_result"] or config["draw_steps"]:
            diag.draw()

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
                    #  the full traceback is in the variable `text`
                    msg = [x for x in text.split('\n')[1:]
                           if not x.startswith(' ')][0]
                    if msg in self.result_overview.keys():
                        self.result_overview[msg] += 1
                    else:
                        self.result_overview[msg] = 1

        print(self.result_overview)