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


#%%
def pull_out(term):
    if len(term.args) == 0:
        return term
    if len(term.args[0].args) == 0:
        return term
    f = term
    g = term.args[0]
    if is_higher_order(g.args[-1].final_type):
        return term
    if not is_higher_order(f.simple_type):
        return term
    if f.simple_type.input == g.simple_type:
        return term
    if is_higher_order(g.simple_type):
        if f.simple_type.input == g.simple_type.output:
            return term
    g = pull_out(g)
    f.args[0] = g
    h = g.args[-1]
    # if isinstance(h, Func):
    #     return term
    h_type = h.final_type
    f_prime_type = (h_type >> f.simple_type.input) >> (h_type >> f.simple_type.output)

    g_prime = Term(g.name, g.simple_type, h_type >> g.final_type, g.args[:-1])
    f_prime = Term(f.name, f_prime_type, f_prime_type, [])

    new_term = f_prime(g_prime)
    new_term = new_term(h)
    for args in f.args[1:]:
        new_term = new_term(args)
    new_term = pull_out(new_term)
    # draw_term(new_term)
    return new_term

new_term = pull_out(term)

draw_term(new_term)
# %%
