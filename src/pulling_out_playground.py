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
    name = term.name
    simple_type = term.simple_type
    return Term(name, simple_type, simple_type, [])


def pop(term):
    name = term.name
    simple_type = term.simple_type
    final_type = term.final_type
    last_arg = term.args[-1]

    new_final_type = last_arg.final_type >> final_type
    return Term(name, simple_type, new_final_type, term.args[:-1]), term.args[-1]


def thrush(f, g, h):
    typs = []
    simple_type = f.simple_type
    for arg in f.args:
        typs.append(simple_type.input)
        simple_type = simple_type.output
    h_type = h.final_type
    simple_type = (h_type >> simple_type.input) >> (h_type >> simple_type.output)
    final_type = simple_type
    for typ in typs[::-1]:
        simple_type = typ >> simple_type
    return Term(f.name, simple_type, final_type, f.args), g, h


# %%
def check(f, g, h):
    # don't pull out higher order diagrams
    if isinstance(h.final_type, Func):
        return False
    # if nothing in g
    if f.simple_type.input == g.simple_type:
        return False
    # only pull out from a higher order box:
    if not is_higher_order(f.final_type):
        return False
    return True


def pull_out(term):
    if len(term.args) == 0:
        return term
    f = just(term)
    for arg in term.args:
        new_arg = pull_out(arg)
        if len(new_arg.args) > 0:
            g, h = pop(new_arg)
            assert new_arg == g(h)
            if check(f, g, h):
                f, g, h = thrush(f, g, h)
                assert pull_out(g) == g
                assert pull_out(h) == h
                f = pull_out(f(g))(h)
            else:
                f = f(new_arg)
        else:
            f = f(new_arg)

    return f


new_term = pull_out(term)

draw_term(new_term)
# %%
