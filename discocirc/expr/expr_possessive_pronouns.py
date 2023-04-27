from random import randint

from diag import Frame
from discocirc.helpers.closed import Ty, Func
from expr import Expr, expr_to_diag
from expr.ccg_type_check import expr_type_check
from expr.expr_uncurry import expr_uncurry


def find_word_in_expr(expr, word, pos):
    if expr.expr_type == "literal":
        if expr.name == str(word) and \
                len(expr.head) == 1 and \
                expr.head[0].index == pos + 1:
            return expr
        else:
            return None
    elif expr.expr_type == "application":
        return find_word_in_expr(expr.fun, word, pos) or find_word_in_expr(expr.arg, word, pos)
    elif expr.expr_type == "lambda":
        return find_word_in_expr(expr.body, word, pos)
    elif expr.expr_type == "list":
        for e in expr.expr_list:
            result = find_word_in_expr(e, word, pos)
            if result:
                return result
        return None


def find_parent_in_expr(expr, child):
    if expr.expr_type == "literal":
        return None
    elif expr.expr_type == "application":
        if expr.arg == child or expr.fun == child:
            return expr
        return find_parent_in_expr(expr.fun, child) or find_parent_in_expr(
            expr.arg, child)
    elif expr.expr_type == "lambda":
        return find_parent_in_expr(expr.body, child)
    elif expr.expr_type == "list":
        for e in expr.expr_list:
            result = find_parent_in_expr(e, child)
            if result:
                return result
        return None

def expand_possessive_pronouns(expr, doc, chain, mention_expr):
    parent = find_parent_in_expr(
        expr,
        find_parent_in_expr(
            expr,
            find_parent_in_expr(expr, mention_expr)
        )
    )

    other_args = []
    while not find_word_in_expr(parent.arg,
                doc[chain[chain.most_specific_mention_index].root_index],
                chain[chain.most_specific_mention_index].root_index):
        other_args.append(expr_uncurry(parent.arg))
        parent = find_parent_in_expr(expr, parent)

    new_pp = Expr.literal(
        mention_expr.name,
        Ty('n') >> (Ty('n') >> (Ty('n') @ Ty('n'))),
        head=mention_expr.head
    )

    new_arg = parent.fun
    for _ in other_args:
        new_arg = new_arg.fun

    new_outside = expr_uncurry(new_arg.fun)
    pp = new_pp(new_arg.arg.arg)(parent.arg)

    if len(other_args) == 0:
        return new_outside(pp)

    swap_args = []
    for arg in list(reversed(other_args)) + [parent.arg, new_arg.arg.arg]:
        print(arg.name)
        swap_args.append(Expr.literal(f"x_{randint(1000,9999)}", expr_uncurry(arg).typ))

    swaps = Expr.lmbda(swap_args[-1],
                       Expr.lst([swap_args[-1]] + swap_args[:-2] + [swap_args[-2]], interchange=False))
    for arg in reversed(swap_args[:-1]):
        swaps = Expr.lmbda(arg, swaps)

    applied_swap = swaps
    for arg in reversed(other_args):
        applied_swap = applied_swap(arg)

    return new_outside(Expr.apply(expr_uncurry(applied_swap), pp, reduce=False))


def expand_personal_pronoun(expr, doc, chain, word):
    return expr


def expand_coref(expr, doc):
    for chain in doc._.coref_chains:
        for mention in chain:
            for token_index in mention.token_indexes:
                word = find_word_in_expr(expr, doc[token_index], token_index)
                if word.typ == Func(Ty('n'), Ty('n')):
                    print("Found a possessive pronoun!")
                    assert(len(mention.token_indexes) == 1)
                    expr = expand_possessive_pronouns(expr, doc, chain, word)

                elif word.typ == Ty('n'):
                    print("Found a personal pronoun!")
                    assert(len(mention.token_indexes) == 1)
                    expr = expand_personal_pronoun(expr, doc, chain, word)

    return expr