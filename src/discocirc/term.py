from __future__ import annotations
from dataclasses import dataclass

from lambeq import CCGRule
from lambeq.bobcat.tree import IndexedWord
from discocirc import closed

@dataclass
class Term:
    name: str
    simple_type: closed.Ty
    final_type: closed.Ty
    args: list[Term]
    head: list[IndexedWord] = None

    def __call__(self, x: Term) -> Term:
        if self.final_type.input != x.final_type:
            raise TypeError(f"Type of {x.name}({x.final_type}) does not match the input type of {self.name}({self.final_type.input})")
        return Term(self.name, self.simple_type, self.final_type.output, [*self.args, x], None)

    def __repr__(self) -> str:
        args = self.args
        if args:
            return f"{self.name}({self.simple_type}, {args=})"
        return f"{self.name}({self.simple_type})"


@dataclass
class Compose:
    func1: Term
    func2: Term
    head: list[IndexedWord] = None

    def __call__(self, x: Term) -> Term:
        return self.func1(self.func2(x))

    def __repr__(self) -> str:
        return f"({self.func1} o {self.func2})"


@dataclass
class TR:
    arg: Term
    head: list[IndexedWord] = None

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
                # term.final_type = closed.biclosed_to_closed(box.cod)
                terms[offset:offset + 2] = [term]
            elif box.name == "Curry(BA(n >> s))":
                terms[offset] = TR(terms[offset])
            else:
                raise NotImplementedError
    return terms[0]

def ccg_to_term(ccg_parse):
    children = [make_term(child) for child in ccg_parse.children]

    result = None
    # Rules with 0 children
    if ccg_parse.rule == CCGRule.LEXICAL:
        closed_type = closed.biclosed_to_closed(ccg_parse.biclosed_type)
        result = Term(ccg_parse.text, closed_type, closed_type, [])

    # Rules with 1 child
    elif ccg_parse.rule == CCGRule.FORWARD_TYPE_RAISING:
        result = TR(children[0])
    elif ccg_parse.rule == CCGRule.UNARY:
        result = children[0]

    # Rules with 2 children
    elif ccg_parse.rule == CCGRule.FORWARD_APPLICATION:
        result = children[0](children[1])
    elif ccg_parse.rule == CCGRule.BACKWARD_APPLICATION:
        result = children[1](children[0])
    elif ccg_parse.rule == CCGRule.FORWARD_COMPOSITION:
        result = Compose(children[0], children[1])
    elif ccg_parse.rule == CCGRule.BACKWARD_COMPOSITION:
        result = Compose(children[1], children[0])

    if result is None:
        raise NotImplementedError(ccg_parse.rule)

    result.head = ccg_parse.original.variable.fillers
    return result

def decomp(term):
    if term is None:
        return None

    if isinstance(term, Compose):
        dummy = Term('?', closed.Ty('n'), closed.Ty('n'), [], None)
        return term.func1(term.func2(dummy))

    args = [decomp(arg) for arg in term.args]
    return Term(term.name, term.simple_type, term.final_type, args, term.head)
