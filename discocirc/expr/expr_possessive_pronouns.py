from random import randint

from diag import Frame
from discocirc.helpers.closed import Ty, Func
from expr import Expr, expr_to_diag
from expr.ccg_type_check import expr_type_check
from expr.expr_uncurry import expr_uncurry

def create_lambda_swap(new_order):
    temp_vars = []
    for num in range(max(new_order) + 1):
        temp_vars.append(Expr.literal(f"x_{randint(1000, 9999)}", Ty('n')))

    lst = []
    for i in new_order:
        lst.append(temp_vars[i])

    output = Expr.lst(lst)
    for temp_var in temp_vars:
        output = Expr.lmbda(temp_var, output)

    return output

def literal_equal_to(literal, word, pos):
    return literal.expr_type == "literal" and \
        literal.name == str(word) and \
        len(literal.head) == 1 and \
        literal.head[0].index == pos + 1

def find_word_in_expr(expr, word, pos):
    if expr.expr_type == "literal":
        if literal_equal_to(expr, word, pos):
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


def create_pp_block(pp_mentions, expr, most_specific):
    new_type = Ty.tensor(*[Ty('n') for _ in range(len(most_specific) + 1)])
    for _ in range(len(most_specific) + 1):
        new_type = Ty('n') >> new_type

    new_pps = []
    for mention_expr in pp_mentions:
        new_pps.append(Expr.literal(
            mention_expr[1].name,
            new_type,
            head=mention_expr[1].head
        ))

    pp_block = new_pps[0](find_parent_in_expr(expr, pp_mentions[0][1]).arg)

    for mention in reversed(most_specific):
        pp_block = pp_block(mention)

    ids = []
    for i in range(((len(pp_mentions) - 1))):
        lst = []
        for j in range(i + 1):
            temp = Expr.literal(f"x_{randint(1000, 9999)}", Ty('n'))
            lst.append(Expr.lmbda(temp, temp))
        ids.append(Expr.lst(lst))

    for i in range(1, len(new_pps)):
        swap = expr_uncurry(
            create_lambda_swap(
                list(range(0, i - 1)) +
                [i + len(most_specific) - 1] +
                list(range(i - 1, i + len(most_specific) - 1)) +
                [i + len(most_specific)]))
        new_args = Expr.apply(swap, Expr.lst(
            [pp_block, find_parent_in_expr(expr, pp_mentions[i][1]).arg]),
                              reduce=False)
        next_layer = Expr.lst([ids[i - 1], new_pps[i]])
        pp_block = next_layer(new_args)

    # The second to last wire is the most specific mention, which has to be
    # moved back to the beginning.
    no_wires = len(most_specific) + len(pp_mentions)
    unswap = expr_uncurry(create_lambda_swap(
        list(range(no_wires - len(most_specific) - 1, no_wires - 1)) +
        list(range(len(pp_mentions) - 1)) +
        [no_wires - 1]
    ))

    return unswap(pp_block)

def unroll_expr_to_arg(expr, word, id):
    other_args = []
    while not find_word_in_expr(expr.arg,
                                word,
                                id):
        other_args.append(expr.arg)
        expr = expr.fun

    return expr.fun, expr.arg, other_args

def expand_possessive_pronouns(expr, doc, chain, pp_mentions):
    # I am currently assuming some properties of the mentions, which I am
    # asserting below. So far, I have not found counter examples. I think
    # these assertions are true for corefs within a single sentence, however,
    # might not hold for multiple sentences (which is not relevant here).

    # Assertion 1: the most specific index is the first mention in the chain.
    assert(chain.most_specific_mention_index == 0)

    # Assertion 2: the most specific mentions are sorted in ascending order.
    for i in range(len(chain[chain.most_specific_mention_index]) - 1):
        assert(chain[chain.most_specific_mention_index][i] <
               chain[chain.most_specific_mention_index][i + 1])

    # Assertion 3: the most specific mentions are before the others
    assert(chain[chain.most_specific_mention_index][-1] < pp_mentions[0][0])

    # Assertion 4: the pp_mentions are sorted in ascending order.
    for i in range(len(pp_mentions) - 1):
        assert(pp_mentions[i][0] < pp_mentions[i + 1][0])


    # Unroll expr and extract all parts that are affected.
    other_args = [[]]
    remaining_expr = expr
    most_specific = []
    for id in chain[chain.most_specific_mention_index]:
        remaining_expr, arg, prior_args = unroll_expr_to_arg(remaining_expr, doc[id], id)
        other_args[0] += prior_args
        most_specific.append(arg)

    for (id, _) in pp_mentions:
        remaining_expr, _, prior_args = unroll_expr_to_arg(remaining_expr, doc[id], id)
        other_args.append(prior_args)

    # Create the personal pronoun block
    pp_block = create_pp_block(pp_mentions, expr, most_specific)

    new_outside = expr_uncurry(remaining_expr)

    # Reassemble the expr
    if sum([len(arg) for arg in other_args]) == 0:
        return new_outside(pp_block)

    lst = []
    other_arg_counter = len(pp_mentions) + len(most_specific)
    for i in range(0, len(other_args)):
        lst += list(range(other_arg_counter,
                          other_arg_counter + len(other_args[i])))
        if i == 0:
            lst += list(range(0, len(most_specific)))
        else:
            lst += [i + len(most_specific) - 1]
        other_arg_counter += len(other_args[i])

    swaps = create_lambda_swap(lst)

    for args in reversed(other_args):
        for arg in reversed(args):
            swaps = swaps(arg)

    return new_outside(Expr.apply(expr_uncurry(swaps),
                                  pp_block, reduce=False))


def expand_coref(expr, doc):
    for chain in doc._.coref_chains:
        pronoun_mentions = []
        for mention in chain:
            for token_index in mention.token_indexes:
                word_expr = find_word_in_expr(expr, doc[token_index], token_index)
                if word_expr.typ == Func(Ty('n'), Ty('n')):
                    print("Found a possessive pronoun!")
                    assert(len(mention.token_indexes) == 1)
                    pronoun_mentions.append((mention[0], word_expr))

        if len(pronoun_mentions) > 0:
            expr = expand_possessive_pronouns(
                expr, doc, chain, pronoun_mentions)

    return expr