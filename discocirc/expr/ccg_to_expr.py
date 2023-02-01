import time

from lambeq import CCGAtomicType, CCGRule

from discocirc.expr import Expr
from discocirc.helpers.closed import biclosed_to_closed
from discocirc.helpers.discocirc_utils import change_expr_typ


def ccg_to_expr(ccg_parse):
    children = [ccg_to_expr(child) for child in ccg_parse.children]

    result = None
    # Rules with 0 children
    if ccg_parse.rule == CCGRule.LEXICAL:
        closed_type = biclosed_to_closed(ccg_parse.biclosed_type)
        result = Expr.literal(ccg_parse.text, closed_type)

    # Rules with 1 child
    elif ccg_parse.rule == CCGRule.FORWARD_TYPE_RAISING \
            or ccg_parse.rule == CCGRule.BACKWARD_TYPE_RAISING:
        x = Expr.literal(f"x{time.time()}",
                         biclosed_to_closed(ccg_parse.biclosed_type).input)
        result = Expr.lmbda(x, x(children[0]))
    elif ccg_parse.rule == CCGRule.UNARY:
        result = change_expr_typ(children[0],
                                 biclosed_to_closed(ccg_parse.biclosed_type))

    # Rules with 2 children
    elif ccg_parse.rule == CCGRule.FORWARD_APPLICATION:
        result = children[0](children[1])
    elif ccg_parse.rule == CCGRule.BACKWARD_APPLICATION:
        result = children[1](children[0])
    elif ccg_parse.rule == CCGRule.FORWARD_COMPOSITION \
            or ccg_parse.rule == CCGRule.FORWARD_CROSSED_COMPOSITION:
        x = Expr.literal(f"temp", biclosed_to_closed(
            ccg_parse.children[1].biclosed_type).input)
        result = Expr.lmbda(x, children[0](children[1](x)))
    elif ccg_parse.rule == CCGRule.BACKWARD_COMPOSITION \
            or ccg_parse.rule == CCGRule.BACKWARD_CROSSED_COMPOSITION:
        x = Expr.literal(f"temp", biclosed_to_closed(
            ccg_parse.children[0].biclosed_type).input)
        result = Expr.lmbda(x, children[1](children[0](x)))
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
        result = children[0]
    elif ccg_parse.rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
        result = children[1]

    if result is None:
        raise NotImplementedError(ccg_parse.rule)

    if ccg_parse.original.cat.var in ccg_parse.original.var_map.keys():
        result.head = ccg_parse.original.variable.fillers
    else:
        result.head = None

    return result