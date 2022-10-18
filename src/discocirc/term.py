from __future__ import annotations
from dataclasses import dataclass

from discocirc import closed


@dataclass
class Term:
    name: str
    simple_type: closed.Ty
    final_type: closed.Ty
    args: list[Term]

    def __call__(self, x: Term) -> Term:
        return Term(self.name, self.simple_type, self.final_type, [*self.args, x])

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
            simple_type = closed.biclosed_to_closed(box.cod)
            terms.append(Term(box.name, simple_type, simple_type, []))
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
                term.final_type = closed.biclosed_to_closed(box.cod)
                terms[offset:offset + 2] = [term]
            elif box.name == "Curry(BA(n >> s))":
                terms[offset] = TR(terms[offset])
            else:
                raise NotImplementedError
    return terms[0]


def decomp(term):
    if term is None:
        return None

    if isinstance(term, Compose):
        dummy = Term('?', closed.Ty('n'), closed.Ty('n'), [])
        return term.func1(term.func2(dummy))

    args = [decomp(arg) for arg in term.args]
    return Term(term.name, term.simple_type, term.final_type, args)
