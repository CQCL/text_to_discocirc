from discopy import rigid

from discocirc import closed
from discocirc.expand_s_types import expand_s_types
from discocirc.frame import Frame
from discocirc.pulling_out import recurse_pull
from discocirc.term import Term, Compose, decomp, make_term


def make_word(name, simple_type, *diags):
    above = rigid.Id()
    insides = []
    i = 0

    while isinstance(simple_type, closed.Func):
        if isinstance(simple_type.input, closed.Func):
            box = diags[i] if i < len(diags) else make_word('?', simple_type.input)
            insides = [box] + insides
        else:
            t = rigid.Ty(simple_type.input[0].name)
            box = diags[i] if i < len(diags) else rigid.Id(t)
            above = box @ above

        simple_type = simple_type.output
        i += 1

    dom = above.cod
    cod = rigid.Ty(simple_type[0].name)
    if len(insides) == 0:  # not a frame
        return above >> rigid.Box(name, dom, cod)

    return above >> Frame(name, dom, cod, insides)


def make_diagram(term):
    term = decomp(term)
    diags = map(make_diagram, term.args)
    return make_word(term.name, term.simple_type, *diags)


def convert_sentence(diagram):
    term = make_term(diagram)
    recurse_pull(term)

    step = make_diagram(term)
    step = expand_s_types(step)
    step = (Frame.get_decompose_functor())(step)

    return step


def sentence2circ(parser, sentence):
    biclosed_diag = parser.sentence2tree(sentence).to_biclosed_diagram()
    return convert_sentence(biclosed_diag)


