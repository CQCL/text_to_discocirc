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
    ccg = term.ccg
    for arg in term.args:
        if not get_ccg_input(ccg) == type_check_term(arg):
            return None
        ccg = get_ccg_output(ccg)

    return ccg


# %%
def is_hyperbox(ccg):
    if isinstance(ccg, Over):
        return isinstance(ccg.right, (Over, Under))
    elif isinstance(ccg, Under):
        return isinstance(ccg.left, (Over, Under))

    return False


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


def normalpull(term):
    """
    Given a hyperbox pull out the arguments.

    :param term: Term - hyperbox who's arguments will be pulled out.
                 We assume that all internal hyperboxes are fully pulled out.
    :return: The same term with arguments pulled out.
    """
    assert (is_hyperbox(term.ccg))

    ccg = term.args[0].ccg
    pulled_out_args = []
    no_pulled_out = 0
    for ar in term.args[0].args.copy():
        # If current ccg is hyperbox, the input should go inside
        # (by recursive property of pulling out, we assume all internal
        # hyperboxes to already be pulled out correctly)
        if is_hyperbox(ccg):
            ccg = get_ccg_output(ccg)
            continue

        pulled_out_args.append((type(ccg), ar.output_ccg))
        term.args.insert(1 + no_pulled_out, ar)
        no_pulled_out += 1
        term.args[0].args.remove(ar)

        ccg = get_ccg_output(ccg)

    for ccg_type, ccg in reversed(pulled_out_args):
        if ccg_type == Over:
            term.ccg.left = Over(term.ccg.left, ccg)
            term.ccg.right = Over(term.ccg.right, ccg)
        elif ccg_type == Under:
            term.ccg.left = Under(ccg, term.ccg.left)
            term.ccg.right = Under(ccg, term.ccg.right)

    return term


def hyperpull(term):
    assert (is_hyperbox(term.ccg))

    if len(term.args) == 0:
        # If no args, then nothing to pull out
        return term

    # If the hyperbox term has multiple holes
    # (i.e after the first hole it's still a hyperbox), continue pulling
    if is_hyperbox(get_ccg_output(term.ccg)):
        new_term = hyperpull(
            Term('', get_ccg_output(term.ccg),
                 get_ccg_output(term.ccg), term.args[1:]))
        set_ccg_output(term.ccg, new_term.ccg)
        term.args = [term.args[0]] + new_term.args

    if get_ccg_input(term.ccg) != term.args[0].ccg:
        term = normalpull(term)

    return term


# %%
def recurse_pull(term):
    for i in range(len(term.args)):
        term.args[i] = recurse_pull(term.args[i])

    if is_hyperbox(term.ccg):
        term = hyperpull(term)

    return term


# %%

def run_sentence(sentence):
    ccg_parse = parser.sentence2tree(sentence).to_biclosed_diagram()
    term = make_term(ccg_parse)
    diag = make_diagram(term)
    # diag.draw()
    assert(type_check_term(term) is not None)

    new_term = recurse_pull(term)
    new_diag = make_diagram(new_term)

    new_diag.draw()
    assert(type_check_term(term) is not None)


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
    'Alice loves Bob',
]

for sentence in sentences:
    run_sentence(sentence)

# %%
