# %%
from __future__ import annotations
from dataclasses import dataclass

from discopy import biclosed, rigid
from discopy.biclosed import Over, Under


@dataclass
class Term:
    name: str
    ccg: biclosed.Ty
    composite_ccg: biclosed.Ty
    args: list[Term]

    def __call__(self, x: Term) -> Term:
        return Term(self.name, self.ccg, self.composite_ccg, [*self.args, x])
        # return Term(self.name, self.ccg, [*self.args, x])

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
                term.composite_ccg = box.cod
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


# %%
def is_hyperbox(ccg):
    if isinstance(ccg, Over):
        if isinstance(ccg.right, (Over, Under)):
            return True
    elif isinstance(ccg, Under):
        if isinstance(ccg.left, (Over, Under)):
            return True
    return False


# %%
def normalpull(term):
    ccg = term.args[0].ccg
    new_left_ccg = term.ccg.left
    new_right_ccg = term.ccg.right

    internal_args = []
    pulled_out_args = []
    for ar in term.args[0].args:
        if isinstance(ccg, Over):
            # If the input is a hyperbox, don't pull it out
            if is_hyperbox(ccg):
                ccg = ccg.left
                internal_args.append(ar)
                continue

            new_left_ccg = Over(new_left_ccg, ar.composite_ccg)
            new_right_ccg = Over(new_right_ccg, ar.composite_ccg)
            ccg = ccg.left
            pulled_out_args.append(ar)

        elif isinstance(ccg, Under):
            # If the input should go into the hyperbox
            if is_hyperbox(ccg):
                ccg = ccg.right
                internal_args.append(ar)
                continue

            new_left_ccg = Under(new_left_ccg, ar.composite_ccg)
            new_right_ccg = Under(new_right_ccg, ar.composite_ccg)
            ccg = ccg.right
            pulled_out_args.append(ar)

    print('original:', term.ccg)
    print(term)
    term.args = [term.args[0]] + pulled_out_args + term.args[1:]
    term.args[0].args = internal_args
    term.ccg.left = new_left_ccg
    term.ccg.right = new_right_ccg
    print('new', term.ccg)
    print(term)
    return term


def hyperpull(term):
    if isinstance(term.ccg, Over):
        # If the hyperbox term has multiple holes (i.e after the first hole it's still a hyperbox), continue pulling
        if is_hyperbox(term.ccg.left):
            new_term = hyperpull(Term('', term.ccg.left, term.ccg.left, term.args[1:]))
            term.ccg.left = new_term.ccg
            term.args = [term.args[0]] + new_term.args

        if len(term.args) > 0:
            if term.ccg.right != term.args[0].ccg:
                print('pull out')
                term = normalpull(term)
            else:
                print('combining types')
                # term.composite_ccg = term.ccg.left
    elif isinstance(term.ccg, Under):
        # print('left is input')
        if is_hyperbox(term.ccg.right):
            new_term = hyperpull(Term('', term.ccg.right, term.ccg.right, term.args[1:]))
            term.ccg.right = new_term.ccg
            term.args = [term.args[0]] + new_term.args

        if len(term.args) > 0:
            if term.ccg.left != term.args[0].ccg:
                print('pull out')
                term = normalpull(term)
            else:
                print('combining types')
                # term.composite_ccg = term.ccg.right
    return term


# %%
def recurse_pull(term):
    for i in range(len(term.args)):
        term.args[i] = recurse_pull(term.args[i])
    print(term.name, term.ccg)
    if is_hyperbox(term.ccg):
        print('This is a hyperbox. Might need to pull out')
        term = hyperpull(term)
    return term


# %%

def run_sentence(sentence):
    ccg_parse = parser.sentence2tree(sentence).to_biclosed_diagram()
    term = make_term(ccg_parse)
    diag = make_diagram(term)
    diag.draw()

    new_term = recurse_pull(term)
    new_diag = make_diagram(new_term)
    new_diag.draw()


sentences = [
    # 'Alice quickly rapdily gives flowers to Bob',
    # 'Alice quickly loves very red Bob',
    # 'Alice fully loves Bob',
    # 'Alice quickly eats',
    # 'Alice quickly eats fish',
    # 'Alice quickly eats red fish',
    'Alice quickly rapidly loudly loves very very red Bob',
    'Alice quickly and rapidly loves Bob and very red Claire'
]
for sentence in sentences:
    run_sentence(sentence)



# %%
