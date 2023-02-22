import unittest

from discocirc.diag import Frame
from discocirc.expr import Expr, expr_to_diag
from discocirc.helpers.closed import Ty


class CCGToDiagTests(unittest.TestCase):
    def draw_expr(self, expr):
        diag = expr_to_diag(expr)
        diag = (Frame.get_decompose_functor())(diag)
        diag.draw()

    def test_type_check(self):
        x = Expr.literal("x", Ty('x'))
        x2 = Expr.literal("x2", Ty('x2'))
        y = Expr.literal("y", Ty('n') >> Ty('s'))
        z = Expr.literal("z", ((Ty('n') >> Ty('s')) @ Ty('x')) >> Ty('z'))
        self.draw_expr(Expr.lmbda(x, Expr.apply(z, Expr.lst([y, x], interchange=False))))

        self.draw_expr(Expr.lmbda(x, Expr.lst([y, x, y])))
        self.draw_expr(Expr.lmbda(x, Expr.lmbda(x2, Expr.lst([x, x2]))))

        x = Expr.literal("x", Ty('n'))
        y = Expr.literal('y', Ty('n') >> Ty('n'))
        # self.draw_expr(Expr.lst([y, y], interchange=True)(x))

        self.assertTrue(True)