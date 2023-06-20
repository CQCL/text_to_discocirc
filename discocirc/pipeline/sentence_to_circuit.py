from lambeq import SpacyTokeniser

from discocirc.expr.expr_possessive_pronouns import expand_coref
from discocirc.expr.n_type_expand import n_type_expand
from discocirc.expr.s_type_expand import p_type_expand, s_type_expand
from discocirc.expr.coordination_expand import coordination_expand
from discocirc.expr.ccg_to_expr import ccg_to_expr
from discocirc.expr.expr_to_diag import expr_to_diag
from discocirc.diag.frame import Frame
from discocirc.expr.pull_out import pull_out
from discocirc.helpers.discocirc_utils import expr_add_indices_to_types
from discocirc.semantics.rewrite import rewrite

tokenizer = SpacyTokeniser()

def sentence2circ(parser, sentence, semantic_rewrites=True, spacy_model=None, add_indices_to_types=True):
    tokenized_sentence = tokenizer.tokenise_sentence(sentence)
    ccg = parser.sentence2tree(tokenized_sentence, tokenised=True)
    expr = ccg_to_expr(ccg)
    expr = pull_out(expr)
    expr = coordination_expand(expr)
    expr = pull_out(expr)
    expr = n_type_expand(expr)
    expr = p_type_expand(expr)
    expr = s_type_expand(expr)
    # if spacy_model:
    #     doc = spacy_model(sentence)
    #     expr = expand_coref(expr, doc)
    if add_indices_to_types:
        expr = expr_add_indices_to_types(expr)
    diag = expr_to_diag(expr)
    if semantic_rewrites:
        diag = rewrite(diag, rules='all')
    diag = (Frame.get_decompose_functor())(diag)

    return diag
