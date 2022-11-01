#%%
from lambeq import BobcatParser
from discocirc.closed import Func
from discocirc.sentence_to_circuit import make_term, make_diagram, sentence2circ
from discocirc.pulling_out import is_higher_order
from discocirc.term import Term
from discocirc.frame import Frame

#%%
parser = BobcatParser()
draw_term = lambda term: (Frame.get_decompose_functor())(make_diagram(term)).draw()

# %%
sentence = "Alice quickly passionately gives Bob flowers"
# sentence = "Alice quickly gives Claire red smelly fish"
diagram = parser.sentence2tree(sentence).to_biclosed_diagram()
term = make_term(diagram)

# %%
def just(term):
    return Term(term.name, term.simple_type, term.simple_type, [])


def pop(term):
    args, h = term.args[:-1], term.args[-1]
    final_type = h.final_type >> term.final_type
    g = Term(term.name, term.simple_type, final_type, args)
    assert term == g(h)
    return g, h


def thrush(f, g, h):
    typs = []
    t = f.simple_type
    for _ in f.args:
        typs.append(0, t.input)
        t = t.output
    t = (h.final_type >> t.input) >> (h.final_type >> t.output)
    final_type = t
    for typ in typs:
        t = typ >> t
    return Term(f.name, t, final_type, f.args), g, h


def exch_t(term, i, n):
    typs = []
    t = term.simple_type
    for _ in range(n):
        typs.insert(0, t.input)
        t = t.output
    t = typs[i] >> t
    for j, typ in enumerate(typs):
        if j != i:
            t = typ >> t
    return Term(term.name, t, t, [])


def exch(term, i):
    g = exch_t(term, i, len(term.args))
    args = term.args
    for j, arg in enumerate(term.args):
        if j != len(args) - i - 1:
            g = g(arg)
    g = g(args[len(args) - i - 1])
    return g


# %%
def check(f, g, h):
    # don't pull out higher order diagrams
    if isinstance(h.final_type, Func):
        return False
    # only pull out from a higher order box:
    if not is_higher_order(f.final_type):
        return False
    return True


def pull_out(term):
    f = just(term)
    for arg in term.args:
        new_arg = pull_out(arg)
        put_back = True
        for i in range(len(new_arg.args)):
            try_arg = exch(new_arg, i)
            g, h = pop(try_arg)
            if check(f, g, h):
                f, g, h = thrush(f, g, h)
                f = pull_out(f(g))(h)
                put_back = False
                break

        if put_back:
            f = f(new_arg)
    return f


new_term = pull_out(term)

draw_term(new_term)
# %%
