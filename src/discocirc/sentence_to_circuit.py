from __future__ import annotations
from dataclasses import dataclass

from discopy import biclosed, rigid
from discopy.biclosed import Over, Under

from discocirc.discocirc_utils import get_ccg_output, get_ccg_input
from discocirc.expand_s_types import expand_s_types
from discocirc.frame import Frame
from discocirc.pulling_out import recurse_pull


@dataclass
class Term:
    name: str
    ccg: biclosed.Ty
    output_ccg: biclosed.Ty
    args: list[Term]

    def __call__(self, x: Term) -> Term:
        return Term(self.name, self.ccg, self.output_ccg, [*self.args, x])

    def __repr__(self) -> str:
        args = self.args
        if args:
            return f"{self.name}({args=})"
        return self.name


@dataclass
class Compose:
    func1: Term
    func2: Term

    def __call__(self, x: Term) -> Term:
        return self.func1(self.func2(x))

    def __repr__(self) -> str:
        return f"({self.func1} o {self.func2})"


@dataclass
class TR:
    arg: Term

    def __call__(self, x: Term) -> Term:
        return x(self.arg)


def make_term(diagram):
    terms = []
    for box, offset in zip(diagram.boxes, diagram.offsets):
        if not box.dom:  # is word
            terms.append(Term(box.name, box.cod, box.cod, []))
        else:
            if len(box.dom) == 2:
                if box.name.startswith("FA"):
                    term = terms[offset](terms[offset + 1])
                elif box.name.startswith("BA"):
                    term = terms[offset + 1](terms[offset])
                elif box.name.startswith("FC"):
                    term = Compose(terms[offset], terms[offset + 1])
                elif box.name.startswith("BC") or box.name.startswith("BX"):
                    term = Compose(terms[offset + 1], terms[offset])
                else:
                    raise NotImplementedError
                term.output_ccg = box.cod
                terms[offset:offset + 2] = [term]
            elif box.name == "Curry(BA(n >> s))":
                terms[offset] = TR(terms[offset])
            else:
                raise NotImplementedError
    return terms[0]


def make_word(name, ccg, *diags):
    above = rigid.Id()
    insides = []
    i = 0

    while isinstance(ccg, (Over, Under)):
        ccg_input = get_ccg_input(ccg)
        if isinstance(ccg_input, (Over, Under)):
            box = diags[i] if i < len(diags) else make_word('?', ccg_input)
            insides = [box] + insides if isinstance(ccg, Under) \
                else insides + [box]
        else:
            t = rigid.Ty(ccg_input[0].name)
            box = diags[i] if i < len(diags) else rigid.Id(t)
            above = above @ box if isinstance(ccg, Over) \
                else box @ above

        ccg = get_ccg_output(ccg)
        i += 1

    dom = above.cod
    cod = rigid.Ty(ccg[0].name)
    if len(insides) == 0:  # not a frame
        return above >> rigid.Box(name, dom, cod)

    return above >> Frame(name, dom, cod, insides)


def decomp(term):
    if term is None:
        return None

    if isinstance(term, Compose):
        dummy = Term('?', biclosed.Ty('n'), biclosed.Ty('n'), [])
        return term.func1(term.func2(dummy))

    args = [decomp(arg) for arg in term.args]
    return Term(term.name, term.ccg, term.output_ccg, args)


def make_diagram(term):
    term = decomp(term)
    diags = map(make_diagram, term.args)
    return make_word(term.name, term.ccg, *diags)


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


