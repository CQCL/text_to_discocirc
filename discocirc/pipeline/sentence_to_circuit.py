from discocirc.expr.inverse_beta import inverse_beta
from discocirc.expr.n_type_expand import n_type_expand
from discocirc.expr.s_type_expand import s_type_expand
from discocirc.expr.coordination_expand import coordination_expand
from discocirc.expr.ccg_to_expr import ccg_to_expr
from discocirc.expr.expr_to_diag import expr_to_diag
from discocirc.diag.frame import Frame
from discocirc.expr.pull_out import pull_out
from discocirc.semantics.rewrite import rewrite


def sentence2circ(parser, sentence):
    ccg = parser.sentence2tree(sentence)
    expr = ccg_to_expr(ccg)
    # first round of pull out
    expr = inverse_beta(expr)
    expr = pull_out(expr)
    # expand noun-noun coordination
    expr = coordination_expand(expr)
    # second round of pull out
    expr = inverse_beta(expr)
    expr = pull_out(expr)
    # s expand
    expr = s_type_expand(expr)
    # then n expand
    expr = n_type_expand(expr)
    # convert expr to diagram
    diag = expr_to_diag(expr)
    diag = (Frame.get_decompose_functor())(diag)\
    # apply semantic rewrites
    diag = rewrite(diag, rules='all')

    return diag
