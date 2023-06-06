import random
import time

from lambeq import CCGAtomicType, CCGRule

from discocirc.expr import Expr
from discocirc.expr.expr import create_index_mapping_dict, map_expr_indices
from discocirc.helpers.closed import biclosed_to_closed, ccg_cat_to_closed
from discocirc.helpers.discocirc_utils import change_expr_typ, create_random_variable


def ccg_to_expr(ccg_parse):
    children = [ccg_to_expr(child) for child in ccg_parse.children]

    result = None
    # Rules with 0 children
    if ccg_parse.rule == CCGRule.LEXICAL:
        word_index = ccg_parse.text + '_' + str(ccg_parse.original.variable.fillers[0].index)
        closed_type = ccg_cat_to_closed(ccg_parse.original.cat, word_index)
        result = Expr.literal(ccg_parse.text, closed_type)

    # Rules with 1 child
    elif ccg_parse.rule == CCGRule.FORWARD_TYPE_RAISING \
            or ccg_parse.rule == CCGRule.BACKWARD_TYPE_RAISING:
        word_index = ccg_parse.children[0].original.variable.fillers[0].index
        tr_type = ccg_cat_to_closed(ccg_parse.original.cat, word_index)
        x = create_random_variable(tr_type.input)
        result = Expr.lmbda(x, x(children[0]), tr_type.index)
    elif ccg_parse.rule == CCGRule.UNARY:
        word_index = ccg_parse.original.variable.fillers[0].index
        closed_type = ccg_cat_to_closed(ccg_parse.original.cat, word_index)
        result = change_expr_typ(children[0], closed_type)

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
    # up to here
    elif ccg_parse.rule == CCGRule.CONJUNCTION:
        left, right = children[0].typ, children[1].typ
        if CCGAtomicType.conjoinable(left):
            type = right >> biclosed_to_closed(ccg_parse.biclosed_type)
            children[0].typ = type
            result = children[0](children[1])
        elif CCGAtomicType.conjoinable(right):
            type = left >> biclosed_to_closed(ccg_parse.biclosed_type)
            children[1].typ = type
            result = children[1](children[0])
    elif ccg_parse.rule == CCGRule.REMOVE_PUNCTUATION_RIGHT:
        if children[0].typ != biclosed_to_closed(ccg_parse.biclosed_type):
            punctuation = Expr.literal(children[1].name, children[0].typ >> biclosed_to_closed(ccg_parse.biclosed_type))
            result = punctuation(children[0])
        else:
            result = children[0]
    elif ccg_parse.rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
        if children[1].typ != biclosed_to_closed(ccg_parse.biclosed_type):
            punctuation = Expr.literal({children[0].name}, children[1].typ >> biclosed_to_closed(ccg_parse.biclosed_type))
            result = punctuation(children[1])
        else:
            result = children[1]

    if result is None:
        raise NotImplementedError(ccg_parse.rule)

    if hasattr(ccg_parse, "original"):
        if ccg_parse.original.cat.var in ccg_parse.original.var_map.keys():
            result.head = ccg_parse.original.variable.fillers
        else:
            result.head = None

    return result

def composition(ccg_parse, f, g):
    x = create_random_variable(g.typ.input)
    result = Expr.lmbda(x, f(g(x)))
    original_typ = ccg_cat_to_closed(ccg_parse.original.cat, random.randint(100, 999))
    index_mapping = create_index_mapping_dict(original_typ, result.typ)
    result = map_expr_indices(result, index_mapping)
    return result
