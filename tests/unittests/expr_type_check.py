import unittest


from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Ty

# Unit tests to test the basic functionality of the Expr.type_check() function.
# These are by no means comprehensive.

class CCGToDiagTests(unittest.TestCase):
    def test_type_check(self):
        state1 = Expr.literal("test", Ty('n1'))
        state2 = Expr.literal("test2", Ty('n2'))
        state3 = Expr.literal("test3", Ty('n3') @ Ty('n4'))
        self.assertTrue(state1.type_check())
        self.assertTrue(state2.type_check())
        self.assertTrue(state3.type_check())

        process1 = Expr.literal("test", Ty('i1') >> Ty('p1'))
        process2 = Expr.literal("test2", (Ty('i2') @ Ty('i3')) >> Ty('p2'))
        process3 = Expr.literal("test3", Ty('i4') >> (Ty('p3') @ Ty('p4')))
        self.assertTrue(process1.type_check())
        self.assertTrue(process2.type_check())
        self.assertTrue(process3.type_check())

        self.assertTrue(Expr.lst([state1, state2, state3], interchange=False).type_check())
        self.assertTrue(Expr.lst([process1, process2, process3], interchange=False).type_check())

        self.assertTrue(Expr.lmbda(state1, state2).type_check())

