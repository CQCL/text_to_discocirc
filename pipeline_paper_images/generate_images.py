from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Ty
from discocirc.pipeline.text_to_circuit import compose_circuits_using_corefs, get_corefs, noun_normal_form

import numpy as np
import re
import spacy
from discopy import hypergraph
from discopy.monoidal import Box, Id, Diagram, Functor
from lambeq import BobcatParser

from discocirc.helpers.discocirc_utils import get_last_initial_noun
from discocirc.diag.drag_up import drag_all
from discocirc.pipeline.sentence_to_circuit import sentence2circ
from discocirc.semantics.rules import passive_to_active_voice, remove_relative_pronouns, remove_to_be, remove_articles
from lambeq import SpacyTokeniser

import warnings

from discocirc.expr.n_type_expand import n_type_expand
from discocirc.expr.normal_form import normal_form
from discocirc.expr.s_type_expand import p_type_expand, s_type_expand
from discocirc.expr.coordination_expand import coordination_expand
from discocirc.expr.ccg_to_expr import ccg_to_expr
from discocirc.expr.to_discopy_diagram import expr_to_diag
from discocirc.expr.pull_out import pull_out
from discocirc.helpers.discocirc_utils import expr_add_indices_to_types
from discocirc.semantics.rewrite import rewrite
from discocirc.expr.resolve_pronouns import expand_coref

parser = BobcatParser(verbose='suppress')
# Load a SpaCy English model
spacy_model = spacy.load('en_core_web_trf')
spacy_model.add_pipe('coreferee')
tokenizer = SpacyTokeniser()

def sentence2circ(parser, sentence, semantic_rewrites=True, spacy_model=None, if_expand_coref=False, add_indices_to_types=True, rules='all', early_break=None):
    """
    Converts a natural language sentence to a DisCoCirc circuit.
    """
    tokenized_sentence = tokenizer.tokenise_sentence(sentence)
    ccg = parser.sentence2tree(tokenized_sentence, tokenised=True)
    expr = ccg_to_expr(ccg)
    if early_break == 'pre_pull_out':
        if add_indices_to_types:
            expr = expr_add_indices_to_types(expr)
        diag = expr_to_diag(expr)
        if semantic_rewrites:
            diag = rewrite(diag, rules=rules)

        return diag
    expr = pull_out(expr)
    # unsure if it suffies for normal_form to be called once, 
    # or if it needs to be called recursively inside coord_expand
    expr = normal_form(expr)
    expr = coordination_expand(expr)
    expr = pull_out(expr)
    if early_break == 'pre_type_expand':
        if add_indices_to_types:
            expr = expr_add_indices_to_types(expr)
        
        diag = expr_to_diag(expr)
        if semantic_rewrites:
            diag = rewrite(diag, rules=rules)

        return diag
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
        diag = rewrite(diag, rules=rules)

    return diag

def sentence_list_to_circuit(context, simplify_swaps=False, wire_order='intro_order', spacy_model=spacy_model, add_indices_to_types=True, frame_expansion=True, doc=None, **kwargs):
    """
    Parameters:
    -----------
    context : list
        List of context sentences.
    simplify_swaps : bool
        Whether to simplify the circuit by removing unnecessary swaps.
    wire_order : str
        The order for the noun wires in the output circuit.
        'update_order' : the most recently updated wires occur on the right
        'intro_order' : the most recently introduced wires occur on the right

    Returns:
    --------
    context_circ : discopy.rigid.Diagram
    """
    sentence_circuits = []
    for sentence in context:
        sentence_diag = sentence2circ(parser,
                                      sentence,
                                      spacy_model = spacy_model,
                                      add_indices_to_types = add_indices_to_types,
                                      **kwargs)
        sentence_circuits.append(sentence_diag)
    corefs, corefs_sent_ids = get_corefs(doc)
    context_circ = compose_circuits_using_corefs(sentence_circuits, corefs, corefs_sent_ids)

    # attempt to remove some redundant swaps
    if simplify_swaps:
        context_circ = noun_normal_form(context_circ)
        num_nouns = get_last_initial_noun(context_circ) + 1
        nouns = context_circ[:num_nouns]
        back_n_forth = lambda f: hypergraph.Diagram.upgrade(f).downgrade()
        context_circ = back_n_forth(context_circ[num_nouns:])
        context_circ = nouns >> context_circ

    return context_circ

def text_to_circuit(text, **kwargs):
    """
    input a given text as a string

    return the corresponding (discopy) circuit for that text
    """
    doc = spacy_model(text)
    sentences = []
    for sent in doc.sents:
        s = sent.text
        sentences.append(s)
    return sentence_list_to_circuit(sentences, spacy_model=spacy_model, doc=doc, **kwargs)

aspect_ratio = 5/8

# Section 1 - Introduction
text_to_circuit("Alice really likes Bob", semantic_rewrites=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section1/example1.png")

text_to_circuit("Claire dislikes Alice", semantic_rewrites=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section1/example2.png")

text_to_circuit("Alice really likes Bob. Claire dislikes Alice.", semantic_rewrites=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section1/example3.png")

# Section 2 - Background

# Section 3 - Lambeq to Diagram
n_type = Ty('n')
s_type_fake = Ty('s ')
expr_to_diag(Expr.literal('Alice', n_type)).draw(figsize=[2, 2*aspect_ratio], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section3/constant_terms0.png")
walkes = Expr.literal('walkes', n_type >> s_type_fake)
expr_to_diag(walkes).draw(figsize=[2, 2*aspect_ratio], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section3/constant_terms1.png")
quickly = Expr.literal('quickly', (n_type >> s_type_fake)>> (n_type >> s_type_fake))
expr_to_diag(quickly(walkes)).draw(figsize=[6, 6*aspect_ratio], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section3/constant_terms2.png")

text_to_circuit("I Alice dislikes Bob who likes Claire", semantic_rewrites=False, early_break="pre_pull_out", add_indices_to_types=False).draw(figsize=[6, 6*aspect_ratio], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section3/final_example.png")


I = Expr.literal('I', n_type)
dreamt = Expr.literal('dreamt', s_type_fake >> (n_type >> s_type_fake))
Bob = Expr.literal('Bob', n_type)
flew = Expr.literal('flew', n_type >> s_type_fake)
expr_to_diag((dreamt(flew(Bob))(I))).draw(figsize=[6, 6*aspect_ratio], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section3/s_type1.png")

# TODO: pipeline drawing is weird here
text_to_circuit("I dreamt Bob flew", semantic_rewrites=False, early_break="pre_pull_out", add_indices_to_types=False).draw(figsize=[6, 6*aspect_ratio], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section3/s_type2.png")

# Section 4 - Lambda to circuit

# Section 4.3 - Noun coordination expand
text_to_circuit("Alice and Bob walk", semantic_rewrites=False, early_break="pre_pull_out").draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/coordination_expand1.png")

text_to_circuit("Alice and Bob walk", semantic_rewrites=False, early_break="pre_type_expand").draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/coordination_expand2.png")

text_to_circuit("Alice , Bob , Claire and Dave walk", semantic_rewrites=False, early_break="pre_pull_out").draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/coordination_expand3.png")

text_to_circuit("Alice , Bob , Claire and Dave walk", semantic_rewrites=False, early_break="pre_type_expand").draw(figsize=[12, 12*aspect_ratio], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/coordination_expand4.png")

# Section 4.4 - Sentence composition
text_to_circuit("Alice likes Bob", semantic_rewrites=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/sentence_composition1.png")
text_to_circuit("He is funny", semantic_rewrites=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/sentence_composition2.png")
text_to_circuit("Alice likes Bob. He is funny", semantic_rewrites=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/sentence_composition3.png")

text_to_circuit("Bob thinks he is smart", semantic_rewrites=False, if_expand_coref=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/noun_contraction1.png")
text_to_circuit("Bob thinks he is smart", semantic_rewrites=False, if_expand_coref=True).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/noun_contraction2.png")

# Section 4.5 - Semantic rewrites
text_to_circuit("Alice pets the cat", semantic_rewrites=False, if_expand_coref=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/determiners1.png")
text_to_circuit("Alice pets the cat", semantic_rewrites=True, if_expand_coref=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/determiners2.png")

text_to_circuit("Alice is red", semantic_rewrites=False, if_expand_coref=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/isdeletion1.png")
text_to_circuit("Alice is red", semantic_rewrites=True, if_expand_coref=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/isdeletion2.png")

text_to_circuit("Bob who loves Alice runs", semantic_rewrites=True, if_expand_coref=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/relativepronoundeletion.png")

text_to_circuit("Alice is bored by the class", semantic_rewrites=False, if_expand_coref=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/passivevoice1.png")
text_to_circuit("Alice is bored by the class", semantic_rewrites=True, if_expand_coref=False, rules={"remove_to_be": remove_to_be, "remove_articles": remove_articles}).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/passivevoice2.png")
text_to_circuit("Alice is bored by the class", semantic_rewrites=True, if_expand_coref=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/passivevoice3.png")

text_to_circuit("Bob loves his dog", semantic_rewrites=False, if_expand_coref=False).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/possessive_pronoun1.png")
text_to_circuit("Bob loves his dog", semantic_rewrites=False, if_expand_coref=True).draw(figsize=[8, 5], margins=[0.1, 0.1], aspect='auto', path="pipeline_paper_images/paper_figures/pipeline_output/Section4/possessive_pronoun2.png")

# Section 5 - Discussion



