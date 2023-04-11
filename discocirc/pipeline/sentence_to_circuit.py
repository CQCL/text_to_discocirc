from discocirc.expr.inverse_beta import inverse_beta
from discocirc.expr.type_expand import type_expand
from discocirc.expr.ccg_to_expr import ccg_to_expr
from discocirc.expr.expr_to_diag import expr_to_diag
from discocirc.diag.frame import Frame
from discocirc.expr.pull_out import pull_out
from discocirc.expr.type_expand import expand_coordination
from discocirc.expr.type_expand import n_expand


def sentence2circ(parser, sentence):
    ccg = parser.sentence2tree(sentence)
    expr = ccg_to_expr(ccg)
    # first round of pull out
    expr = inverse_beta(expr)
    expr = pull_out(expr)
    # expand noun-noun coordination
    expr = expand_coordination(expr)
    # second round of pull out
    expr = inverse_beta(expr)
    expr = pull_out(expr)
    # n expand first
    expr = n_expand(expr)
    # then s expand
    expr = type_expand(expr)
    # convert expr to diagram
    diag = expr_to_diag(expr)
    diag = (Frame.get_decompose_functor())(diag)

    return diag
