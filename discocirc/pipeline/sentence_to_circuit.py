from lambeq import SpacyTokeniser

import warnings

from discocirc.expr.n_type_expand import n_type_expand
from discocirc.expr.s_type_expand import p_type_expand, s_type_expand
from discocirc.expr.coordination_expand import coordination_expand
from discocirc.expr.ccg_to_expr import ccg_to_expr
from discocirc.expr.to_discopy_diagram import expr_to_diag
from discocirc.diag.frame import Frame
from discocirc.expr.pull_out import pull_out
from discocirc.helpers.discocirc_utils import expr_add_indices_to_types
from discocirc.semantics.rewrite import rewrite
from discocirc.expr.resolve_pronouns import expand_coref

tokenizer = SpacyTokeniser()

def sentence2circ(parser, sentence, semantic_rewrites=True, spacy_model=None, if_expand_coref=True, add_indices_to_types=True, frame_expansion=True):
    """
    Converts a natural language sentence to a DisCoCirc circuit.
    """
    tokenized_sentence = tokenizer.tokenise_sentence(sentence)
    ccg = parser.sentence2tree(tokenized_sentence, tokenised=True)
    expr = ccg_to_expr(ccg)
    expr = pull_out(expr)
    expr = coordination_expand(expr)
    expr = pull_out(expr)
    expr = n_type_expand(expr)
    expr = p_type_expand(expr)
    expr = s_type_expand(expr)
    if spacy_model == None and if_expand_coref:
        warnings.warn('Spacy model not provided. Coreference resolution will not be performed.')
    if spacy_model and if_expand_coref:
        doc = spacy_model(sentence)
        expr = expand_coref(expr, doc)
    if add_indices_to_types:
        expr = expr_add_indices_to_types(expr)
    else: 
        warnings.warn('If you do not add indices to the types, the composition of sentences might be incorrect.')
    diag = expr_to_diag(expr)
    if semantic_rewrites:
        diag = rewrite(diag, rules='all')
    if frame_expansion:
        diag = (Frame.get_decompose_functor())(diag)

    return diag
