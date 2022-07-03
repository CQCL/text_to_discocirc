# %%
from __future__ import annotations
from dataclasses import dataclass

from discopy import biclosed, rigid
from discopy.biclosed import Over, Under


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
    inside = rigid.Id()
    i = 0
    wire = rigid.Id(rigid.Ty('*'))
    while isinstance(ccg, (Over, Under)):
        if isinstance(ccg, Over):
            if isinstance(ccg.right, (Over, Under)):
                box = diags[i] if i < len(diags) else make_word('?', ccg.right)
                if not inside:
                    inside = wire @ box @ wire
                else:
                    inside = inside @ box @ wire
            else:
                t = rigid.Ty(ccg.right[0].name)
                box = diags[i] if i < len(diags) else rigid.Id(t)
                above = above @ box
            ccg = ccg.left
        else:
            if isinstance(ccg.left, (Over, Under)):
                box = diags[i] if i < len(diags) else make_word('?', ccg.left)
                if not inside:
                    inside = wire @ box @ wire
                else:
                    inside = wire @ box @ inside
            else:
                t = rigid.Ty(ccg.left[0].name)
                box = diags[i] if i < len(diags) else rigid.Id(t)
                above = box @ above
            ccg = ccg.right
        i += 1

    dom = above.cod
    cod = rigid.Ty(ccg[0].name)
    if not inside:  # not a frame
        return above >> rigid.Box(name, dom, cod)
    top = rigid.Box(f'[{name}]', dom, inside.dom)
    bot = rigid.Box(f'[\\{name}]', inside.cod, cod)
    return above >> top >> inside >> bot


def decomp(term):
    if term is None:
        return None
    if isinstance(term, Compose):
        dummy = Term('?', biclosed.Ty('n'), biclosed.Ty('n'), [])
        return term.func1(term.func2(dummy))
    args = [decomp(arg) for arg in term.args]
    return Term(term.name, term.ccg, term.ccg, args)


def make_diagram(term):
    term = decomp(term)
    diags = map(make_diagram, term.args)
    return make_word(term.name, term.ccg, *diags)


# %%
from lambeq import BobcatParser

parser = BobcatParser()


def type_check_term(term):
    """
    Given a term, check if all the arguments match the required ccg.

    :param term: Term - The term which should be type checked.
    :return: None - If term does not type check.
        ccg - The output type of the term, if it type checks.
    """
    ccg = term.ccg
    for arg in term.args:
        if not get_ccg_input(ccg) == type_check_term(arg):
            return None
        ccg = get_ccg_output(ccg)

    return ccg


# %%
def get_hyperholes(term):
    """
    Finds the position of all the holes a given term has,
    i.e. all the inputs that are processes

    :param term: Term - for which the hyperholes should be found.
    :return: List - position in argument list of all hyperholes.
    """
    hyperholes = []
    ccg = term.ccg
    for i, arg in enumerate(term.args):
        if isinstance(get_ccg_input(ccg), (Over, Under)):
            hyperholes.append(i)
        ccg = get_ccg_output(ccg)

    return hyperholes


def get_ccg_input(ccg):
    if isinstance(ccg, Under):
        return ccg.left
    elif isinstance(ccg, Over):
        return ccg.right


def get_ccg_output(ccg):
    if isinstance(ccg, Under):
        return ccg.right
    elif isinstance(ccg, Over):
        return ccg.left


def set_ccg_output(ccg, output):
    if isinstance(ccg, Under):
        ccg.right = output
    elif isinstance(ccg, Over):
        ccg.left = output


def pull_single_hyperhole(term, hole_position):
    """
    Given a hyperbox pull out the arguments of the specified hole.
    For hyperboxes with multiple holes, this has to be called multiple times.

    :param term: Term - hyperbox who's arguments will be pulled out.
                 We assume that all internal hyperboxes are fully pulled out.
    :param hole_position: int - the argument position of the hole which should
                be pulled out.
    """
    inner_term = term.args[hole_position]

    ccg = inner_term.ccg
    pulled_out_args = []

    inner_term_hyperholes = get_hyperholes(inner_term)

    for i, ar in enumerate(inner_term.args.copy()):
        # If current argument should go into a hyper hole: skip
        # (by recursive property of pulling out, we assume all internal
        # hyperboxes to already be pulled out correctly).
        # Thus, they should take exactly one input, which we don't pull out.
        if i in inner_term_hyperholes:
            ccg = get_ccg_output(ccg)
            continue

        # Pull out the argument
        term.args.insert(hole_position + len(pulled_out_args) + 1, ar)
        pulled_out_args.append((type(ccg), ar.output_ccg))
        inner_term.args.remove(ar)

        ccg = get_ccg_output(ccg)

    # Update the ccg_type in reverse order such that the first argument pulled
    # out is the last added to the ccg and thus the next input
    term_ccg = term.ccg
    for i in range(hole_position):
        term_ccg = get_ccg_output(term_ccg)

    for ccg_type, ccg in reversed(pulled_out_args):
        if ccg_type == Over:
            term_ccg.left = Over(term_ccg.left, ccg)
            term_ccg.right = Over(term_ccg.right, ccg)
        elif ccg_type == Under:
            term_ccg.left = Under(ccg, term_ccg.left)
            term_ccg.right = Under(ccg, term_ccg.right)


# %%
def recurse_pull(term):
    """
    Given a term, recursively pull out all the hyperboxes to get a fully
    pulled out term.

    :param term: Term - The term which should be pulled out.
    """
    for i in range(len(term.args)):
        recurse_pull(term.args[i])

    hyper_holes = get_hyperholes(term)
    for hole in hyper_holes:
        pull_single_hyperhole(term, hole)


# %%
# ----- TODO:  This is just for testing purposes. Remove after development ----
def run_sentence(sentence):
    ccg_parse = parser.sentence2tree(sentence).to_biclosed_diagram()

    term = make_term(ccg_parse)
    diag = make_diagram(term)
    # diag.draw()
    assert (type_check_term(term) is not None)

    recurse_pull(term)
    new_diag = make_diagram(term)
    new_diag.draw()
    assert (type_check_term(term) is not None)


sentences = [
    'Alice quickly rapidly gives flowers to Bob',
    'Alice quickly gives flowers to Bob',
    'Alice quickly loves very red Bob',
    'Alice fully loves Bob',
    'Alice quickly eats',
    'Alice quickly eats fish',
    'Alice quickly eats red fish',
    'Alice quickly loves very red Bob',
    'Alice quickly rapidly loudly loves very very red Bob',
    'Alice quickly and rapidly loves Bob and very red Claire',
    'I know of Alice loving Bob',
    'I surely know quickly of Alice quickly loving Bob',
]

for sentence in sentences:
    run_sentence(sentence)

# %%
