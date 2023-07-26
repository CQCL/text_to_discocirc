import random
from lambeq import CCGType, CCGRule

from discocirc.expr import Expr
from discocirc.expr.expr import create_index_mapping_dict, map_expr_indices
from discocirc.helpers.closed import ccg_cat_to_closed, Ty
from discocirc.helpers.discocirc_utils import change_expr_typ, create_random_variable


def ccg_to_expr(ccg_parse):
    """
    Takes in a ccg_parse, and returns the corresponding expr.
    Recurses to the leaves of the ccg tree first, and builds the expr up from leaf to root

    This process is a canonical one, as described by the lambda-calculus semantics
    of CCG.
    There is some stuff at the end about appending the grammatical heads
    """
    children = [ccg_to_expr(child) for child in ccg_parse.children]

    result = None
    # Rules with 0 children
    if ccg_parse.rule == CCGRule.LEXICAL:
        word_index = ccg_parse.text + '_' + str(ccg_parse.metadata['original'].variable.fillers[0].index)
        closed_type = ccg_cat_to_closed(ccg_parse.metadata['original'].cat, str(word_index))
        result = Expr.literal(ccg_parse.text, closed_type)

    # Rules with 1 child
    elif ccg_parse.rule == CCGRule.FORWARD_TYPE_RAISING \
            or ccg_parse.rule == CCGRule.BACKWARD_TYPE_RAISING:
        word_index = ccg_parse.children[0].original.variable.fillers[0].index
        tr_type = ccg_cat_to_closed(ccg_parse.metadata['original'].cat, str(word_index))
        x = create_random_variable(tr_type.input)
        result = Expr.lmbda(x, x(children[0]), tr_type.index)
    elif ccg_parse.rule == CCGRule.UNARY:
        word_index = ccg_parse.metadata['original'].variable.fillers[0].index
        closed_type = ccg_cat_to_closed(ccg_parse.metadata['original'].cat, str(word_index))
        if children[0].typ != closed_type:
            result = change_expr_typ(children[0], closed_type)
        else:
            result = children[0]

    # Rules with 2 children
    elif ccg_parse.rule == CCGRule.FORWARD_APPLICATION:
        result = children[0](children[1])
    elif ccg_parse.rule == CCGRule.BACKWARD_APPLICATION:
        result = children[1](children[0])
    elif ccg_parse.rule == CCGRule.FORWARD_COMPOSITION \
            or ccg_parse.rule == CCGRule.FORWARD_CROSSED_COMPOSITION:
        result = composition(ccg_parse, children[0], children[1])
    elif ccg_parse.rule == CCGRule.BACKWARD_COMPOSITION \
            or ccg_parse.rule == CCGRule.BACKWARD_CROSSED_COMPOSITION:
        result = composition(ccg_parse, children[1], children[0])
    elif ccg_parse.rule == CCGRule.CONJUNCTION:
        first_word = ccg_parse.children[0].biclosed_type
        if CCGType.conjoinable(first_word):
            w0, w1 = 0, 1
        else:
            w0, w1 = 1, 0
        # w0, w1 = 0, 1 if CCGAtomicType.conjoinable(first_word) else 1, 0
        conjunction = children[w0]
        conjunct = children[w1]
        word_index = ccg_parse.children[w0].original.variable.fillers[0].index
        if conjunct.typ == Ty('n'):
            second_conjunct_typ = Ty('n', index=set([str(word_index) + '_x']))
            final_idx = set.union(conjunct.typ.index, second_conjunct_typ.index)
            final_conjunction_type = Ty('n', index=final_idx)
            conjunction_type = conjunct.typ >> (second_conjunct_typ >> final_conjunction_type)
        else:
            conjunction_type = conjunct.typ >> (conjunct.typ >> conjunct.typ)
        conjunction_type.index = conjunction.typ.index
        conjunction = change_expr_typ(conjunction, conjunction_type)
        result = conjunction(conjunct)
    # up to here
    elif ccg_parse.rule == CCGRule.REMOVE_PUNCTUATION_RIGHT:
        # if children[0].typ != biclosed_to_closed(ccg_parse.biclosed_type):
        #     punctuation = Expr.literal(children[1].name, children[0].typ >> biclosed_to_closed(ccg_parse.biclosed_type))
        #     result = punctuation(children[0])
        # else:
        result = children[0]
    elif ccg_parse.rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
        # if children[1].typ != biclosed_to_closed(ccg_parse.biclosed_type):
        #     punctuation = Expr.literal({children[0].name}, children[1].typ >> biclosed_to_closed(ccg_parse.biclosed_type))
        #     result = punctuation(children[1])
        # else:
        result = children[1]

    if result is None:
        raise NotImplementedError(ccg_parse.rule)

    if 'original' in ccg_parse.metadata:
        if ccg_parse.metadata['original'].cat.var in ccg_parse.metadata['original'].var_map.keys():
            result.head = ccg_parse.metadata['original'].variable.fillers
        else:
            result.head = None

    return result


def composition(ccg_parse, f, g):
    """
    returns the composition of f, g as an expression: 
        λx.f(g(x))
    """
    x = create_random_variable(g.typ.input)
    result = Expr.lmbda(x, f(g(x)))
    original_typ = ccg_cat_to_closed(ccg_parse.metadata['original'].cat, str(random.randint(100, 999)))
    index_mapping = create_index_mapping_dict(original_typ, result.typ)
    result = map_expr_indices(result, index_mapping)
    return result
