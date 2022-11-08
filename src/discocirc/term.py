from __future__ import annotations
from dataclasses import dataclass

from discocirc import closed
from discocirc.closed import Ty


def b_combinator(f, A):
    """
    This function models the operation of the B-combinator on f,
    which is dependent on the type A.
    We have B = λ f: (B -> C). λ g: (A -> B). λ h: A. f(g(h))

    :param f: The function through which we pull out
    :param A: The type of the term that we pull out
    :return: B f - The new function through which we have pulled out
    """
    typs = []
    simple_type = f.simple_type
    for _ in f.args:
        typs.insert(0, simple_type.input)
        simple_type = simple_type.output
    # assert(f.final_type == simple_type)
    final_type = (A >> f.final_type.input) >> (A >> f.final_type.output)
    new_type = final_type
    for typ in typs:
        new_type = typ >> new_type

    return Term(f.name, new_type, final_type, f.args)

@dataclass
class Term:
    name: str
    simple_type: closed.Ty
    final_type: closed.Ty
    args: list[Term]

    def __call__(self, x: Term) -> Term:
        # if self.final_type.input != x.final_type:
        #     raise TypeError(f"Type of {x.name}({x.final_type}) does not match the input type of {self.name}({self.final_type.input})")
        return Term(self.name, self.simple_type, self.final_type.output,
                    [*self.args, x])

    def __repr__(self) -> str:
        args = self.args
        if args:
            return f"{self.name}({self.simple_type}, {args=})"
        return f"{self.name}({self.simple_type})"

@dataclass
class TR(Term):
    pass


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
                    if isinstance(terms[offset], TR):
                        term = terms[offset + 1]
                        term.final_type = term.final_type.input \
                                          >> term.final_type.output.output
                        term.__call__ = lambda g: terms[offset + 1](g)(terms[offset])
                    else:
                        term = b_combinator(terms[offset],
                                            terms[offset + 1].final_type.input)
                        term = term(terms[offset + 1])
                elif box.name.startswith("BC") or box.name.startswith("BX"):
                    term = b_combinator(terms[offset + 1],
                                        terms[offset].final_type.input)
                    term = term(terms[offset])
                else:
                    raise NotImplementedError(box.name)
                # term.final_type = closed.biclosed_to_closed(box.cod)
                terms[offset:offset + 2] = [term]
            elif box.name == "Curry(BA(n >> s))":
                terms[offset] = TR(terms[offset].name, terms[offset].simple_type, terms[offset].final_type, terms[offset].args)
            else:
                raise NotImplementedError
    return terms[0]


# def decomp(term):
#     if term is None:
#         return None
#
#     if isinstance(term, Compose):
#         dummy = Term('?', closed.Ty('n'), closed.Ty('n'), [])
#         return term.func1(term.func2(dummy))
#
#     args = [decomp(arg) for arg in term.args]
#     return Term(term.name, term.simple_type, term.final_type, args)
