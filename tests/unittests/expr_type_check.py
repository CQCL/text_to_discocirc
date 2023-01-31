import unittest

from discocirc.expr.ccg_type_check import expr_type_check
from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Ty

# Unit tests to test the basic functionality of the Expr.type_check() function.
# These are by no means comprehensive.

class CCGToDiagTests(unittest.TestCase):
    def test_type_check(self):
        state1 = Expr.literal("test", Ty('n1'))
        state2 = Expr.literal("test2", Ty('n2'))
        state3 = Expr.literal("test3", Ty('n3') @ Ty('n4'))
        self.assertTrue(expr_type_check(state1))
        self.assertTrue(expr_type_check(state2))
        self.assertTrue(expr_type_check(state3))

        process1 = Expr.literal("test", Ty('i1') >> Ty('p1'))
        process2 = Expr.literal("test2", (Ty('i2') @ Ty('i3')) >> Ty('p2'))
        process3 = Expr.literal("test3", Ty('i4') >> (Ty('p3') @ Ty('p4')))
        self.assertTrue(expr_type_check(process1))
        self.assertTrue(expr_type_check(process2))
        self.assertTrue(expr_type_check(process3))

        self.assertTrue(expr_type_check(Expr.lst([state1, state2, state3], interchange=False)))
        self.assertTrue(expr_type_check(Expr.lst([process1, process2, process3], interchange=False)))

        self.assertTrue(expr_type_check(Expr.lmbda(state1, state2)))

