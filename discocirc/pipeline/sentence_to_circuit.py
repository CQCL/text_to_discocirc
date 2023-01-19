from discocirc.diag.expand_s_types import expand_s_types
from discocirc.expr.expr_to_diag import expr_to_diag
from discocirc.diag.frame import Frame
from discocirc.expr.pull_out import pull_out
from discopy import rigid

from discocirc.helpers import closed
from discocirc.expr.expr import Expr


def make_word(name, simple_type, *diags):
    above = rigid.Id()
    insides = []
    i = 0

    while isinstance(simple_type, closed.Func):
        if isinstance(simple_type.input, closed.Func):
            box = diags[i] if i < len(diags) \
                else make_word('?', simple_type.input)
            insides = [box] + insides
        else:
            t = rigid.Ty(simple_type.input[0].name)
            box = diags[i] if i < len(diags) else rigid.Id(t)
            above = box @ above

        simple_type = simple_type.output
        i += 1

    dom = above.cod
    cod = rigid.Ty().tensor(*[rigid.Ty(t.name) for t in simple_type])
    if len(insides) == 0:  # not a frame
        return above >> rigid.Box(name, dom, cod)

    return above >> Frame(name, dom, cod, insides)


def make_diagram(term):
    # term = decomp(term)
    diags = map(make_diagram, term.args)
    return make_word(term.name, term.simple_type, *diags)


def convert_sentence(ccg):
    expr = Expr.ccg_to_expr(ccg)
    expr = pull_out(expr)
    diag = expr_to_diag(expr)
    diag = expand_s_types(diag)
    # diag = (Frame.get_decompose_functor())(diag)

    return diag


def sentence2circ(parser, sentence):
    ccg = parser.sentence2tree(sentence)
    return convert_sentence(ccg)
