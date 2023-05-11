from lambeq import SpacyTokeniser

from discocirc.expr.expr_normal_form import expr_normal_form
from discocirc.expr.expr_possessive_pronouns import expand_coref
from discocirc.expr.inverse_beta import inverse_beta
from discocirc.expr.n_type_expand import n_type_expand
from discocirc.expr.s_type_expand import s_type_expand
from discocirc.expr.coordination_expand import coordination_expand
from discocirc.expr.ccg_to_expr import ccg_to_expr
from discocirc.expr.expr_to_diag import expr_to_diag
from discocirc.diag.frame import Frame
from discocirc.expr.pull_out import pull_out
from discocirc.semantics.rewrite import rewrite

tokenizer = SpacyTokeniser()

def sentence2circ(parser, sentence, semantic_rewrites=True, spacy_model=None):
    tokenized_sentence = tokenizer.tokenise_sentence(sentence)
    ccg = parser.sentence2tree(tokenized_sentence, tokenised=True)
    expr = ccg_to_expr(ccg)
    # first round of pull out
    expr = pull_out(expr)
    # expand noun-noun coordination
    expr = coordination_expand(expr)
    # second round of pull out
    expr = pull_out(expr)
    # then n expand
    expr = n_type_expand(expr)
    # convert expr to diagram
    # s expand
    expr = s_type_expand(expr)
    if spacy_model:
        doc = spacy_model(sentence)
        expr = expand_coref(expr, doc)
    diag = expr_to_diag(expr)
    # apply semantic rewrites
    if semantic_rewrites:
        diag = rewrite(diag, rules='all')
    # decompose diagram
    diag = (Frame.get_decompose_functor())(diag)

    return diag
