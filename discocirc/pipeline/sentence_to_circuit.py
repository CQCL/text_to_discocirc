from discocirc.expr.type_expand import type_expand, expand_coordination
from discocirc.expr.ccg_to_expr import ccg_to_expr
from discocirc.expr.expr_to_diag import expr_to_diag
from discocirc.diag.frame import Frame
from discocirc.expr.pull_out import pull_out
from discopy import rigid

from discocirc.helpers import closed
from discocirc.expr.expr import Expr

def sentence2circ(parser, sentence):
    ccg = parser.sentence2tree(sentence)
    expr = ccg_to_expr(ccg)
    expr = pull_out(expr)
    expr = expand_coordination(expr)
    expr = type_expand(expr)
    diag = expr_to_diag(expr)
    diag = (Frame.get_decompose_functor())(diag)

    return diag
