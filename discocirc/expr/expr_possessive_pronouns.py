from random import randint

from diag import Frame
from discocirc.helpers.closed import Ty, Func
from expr import Expr, expr_to_diag
from expr.ccg_type_check import expr_type_check
from expr.expr_uncurry import expr_uncurry


def create_lambda_swap(new_order):
    """
    Given a list of integers, create a lambda expression that swaps the wires
    as specified by the list where the ith wire is swapped to all wires j where
    new_order[j] = i.

    :param new_order: The list of integers specifying the new order of the wires.
    :return: The lambda expression that swaps the wires.
    """
    temp_vars = []
    for num in range(max(new_order) + 1):
        temp_vars.append(Expr.literal(f"x_{randint(10000, 99999)}", Ty('n')))

    lst = []
    for i in new_order:
        lst.append(temp_vars[i])

    output = Expr.lst(lst)
    for temp_var in temp_vars:
        output = Expr.lmbda(temp_var, output)

    return output


def find_word_in_expr(expr, word, pos):
    """
    Given an expr, return the expr of the literal representing
    the word specified by the tuple (word, pos). Return None if this literal
    is not present.

    :param expr: The expr in which the literal is searched for.
    :param word: The string representation of the word to be searched for.
    :param pos: The position in the sentence of the word to be searched for.
    :return: The literal representing the word, or None if it is not present.
    """
    if expr.expr_type == "literal":
        if expr.name == str(word) and \
                len(expr.head) == 1 and \
                expr.head[0].index == pos + 1:
            return expr
        else:
            return None
    elif expr.expr_type == "application":
        return find_word_in_expr(expr.fun, word, pos) or \
               find_word_in_expr(expr.arg, word, pos)
    elif expr.expr_type == "lambda":
        return find_word_in_expr(expr.body, word, pos)
    elif expr.expr_type == "list":
        for e in expr.expr_list:
            result = find_word_in_expr(e, word, pos)
            if result:
                return result
        return None


def create_pp_block(most_specific, pps):
    """
    Given a list of the most specific mentions and a list of exprs corresponding
    to states with possessive pronouns for a single coreference, create an expr
    representing a state that combines the exprs of the pps with the most specific
    mentions.

    :param most_specific: A list of the arguments corresponding to the most
            specific mentions.
    :param pps: A list of arguments corresponding to the possessive pronouns.
    :return: A new expr representing the state that combines the pps with the
            most specific mentions.
    """
    # Create the new possessive pronouns (which have a different type).
    new_type = Ty.tensor(*[Ty('n') for _ in range(len(most_specific) + 1)])
    for _ in range(len(most_specific) + 1):
        new_type = Ty('n') >> new_type

    new_pps = []
    for pp in pps:
        new_pps.append(Expr.literal(
            pp.fun.name, # requires the pp.fun to be the possessive pronoun.
                         # Checked in expand_possessive_pronoun_chain().
            new_type,
            head=pp.fun.head
        ))

    # Put pronouns and arguments back together.
    # First possessive pronouns.
    pp_block = new_pps[0](pps[0].arg)

    for mention in reversed(most_specific):
        pp_block = pp_block(mention)

    # All other possessive pronouns.
    ids = []
    for i in range(((len(pps) - 1))):
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
            [pp_block, pps[i].arg]),
                              reduce=False)
        next_layer = Expr.lst([ids[i - 1], new_pps[i]])
        pp_block = next_layer(new_args)

    # Bring the wires back in order.
    # The second to last wires are the most specific mentions, which have to be
    # moved back to the beginning.
    no_wires = len(most_specific) + len(pps)
    unswap = expr_uncurry(create_lambda_swap(
        list(range(no_wires - len(most_specific) - 1, no_wires - 1)) +
        list(range(len(pps) - 1)) +
        [no_wires - 1]
    ))

    return unswap(pp_block)


def find_arg_in_arg_list(args, word, word_pos):
    """
    Given a list of args, find the argument that contains the word specified
    by the tuple word and word_pos. This argument may contain other words.
    Return the args before the found argument, the found argument and the
    remaining arguments.

    :param args: A list of arguments in which the specified word is to be found.
    :param word: The string representation of the word to be found.
    :param word_pos: The position in the sentence of the word to be found.
    :return: A tuple of the form
        (args_before_found_arg, found_arg, remaining_args) where
        - args_before_found_arg is a list of arguments before the found argument,
        - found_arg is the argument which countains the specified word,
        - remaining_args is a list of arguments after the found argument.
    """
    arg_pos = 0
    while not find_word_in_expr(args[arg_pos], word, word_pos):
        arg_pos += 1

    return args[:arg_pos], args[arg_pos], args[arg_pos + 1:]


def expand_possessive_pronoun_chain(args, most_specific_indices, pp_mentions):
    """
    Given a list of arguments, a list of the most specific mentions and a list
    of the possessive pronouns in one coreference, recombine the arguments to
    express the possessive pronouns.

    :param args: A list of arguments which contains the most specific mentions
            and the possessive pronouns.
    :param most_specific_indices: A list of indices of the most specific
            mentions, each of the form (word, position of word in sentence).
    :param pp_mentions: A list of indices of the possessive pronouns, each of
            the form (word, position of word in sentence).
    :return: An adapted list of the arguments and a list representing the
            swaps required to put the arguments back into their original wire
            order.
    """
    # ================ ASSERTION ================
    # I am currently assuming some properties of the mentions, which I am
    # asserting below. So far, I have not found counter examples. I think
    # these assertions are true for corefs within a single sentence, however,
    # might not hold for multiple sentences (which is not relevant here).

    # Assertion 1: the most specific mentions are sorted in ascending order.
    for i in range(len(most_specific_indices) - 1):
        assert(most_specific_indices[i] <
               most_specific_indices[i + 1])

    # Assertion 2: the most specific mentions are before the others
    assert(most_specific_indices[-1] < pp_mentions[0])

    # Assertion 3: the pp_mentions are sorted in ascending order.
    for i in range(len(pp_mentions) - 1):
        assert(pp_mentions[i] < pp_mentions[i + 1])

    # ================ ASSERTION ================

    # Identify all args that are relevant for this coref.
    other_args = [[]]
    most_specific_args = []
    remaining_args = args
    for word, id in most_specific_indices:
        prior_args, arg, remaining_args = find_arg_in_arg_list(remaining_args, word, id)
        other_args[0] += prior_args
        most_specific_args.append(arg)

    pps = []
    for word, id in pp_mentions:
        prior_args, arg, remaining_args = find_arg_in_arg_list(remaining_args, word, id)
        other_args.append(prior_args)
        pps.append(arg)

        # Assertion currently made create_pp_block()
        assert(find_word_in_expr(arg.fun, word, id) is not None)

    other_args.append(remaining_args)

    # Create the personal pronoun block
    pp_block = create_pp_block(most_specific_args, pps)

    # Create the list of indices for the lambda swap
    lst = []
    other_arg_counter = len(pp_mentions) + len(most_specific_args)
    for i in range(0, len(other_args)):
        other_args_len = sum([len(arg.typ) for arg in other_args[i]])
        lst += list(range(other_arg_counter,
                          other_arg_counter + other_args_len))
        if i == 0:
            lst += list(range(0, len(most_specific_args)))
        elif i != len(other_args) - 1:
            lst += [i + len(most_specific_args) - 1]
        other_arg_counter += other_args_len

    return [pp_block] + \
           [arg for other_arg in other_args for arg in other_arg], lst


def expand_possessive_pronouns(expr, all_pp_chains):
    """
    Given an expr and a list of all possessive pronouns chains, create an expr
    where the possessive pronouns are expanded.

    :param expr: The expr to be expanded.
    :param all_pp_chains: A list of all possessive pronoun chains where
         each chain is a tuple of the form (most_specific, pronoun_mentions)
    :return: The expanded expr.
    """
    # Extract all arguments that are passed to the outermost expr.
    # We assume all relevant arguments for the coref are in there.
    remainder = expr
    # The current implementation extracts all arguments given to the outermost
    # application and combines them for all the corefs before putting them back
    # into one expr. Through this, the difference of exprs with arguments and
    # exprs with lists does not have to be taken into account (where lists are
    # created for the coref expansion).
    args = []
    while remainder.expr_type == "application" and \
            remainder.arg.typ == Ty('n'):
        args.append(remainder.arg)
        remainder = remainder.fun

    new_outside = expr_uncurry(remainder)

    # Expand possessive pronouns
    combined_swaps = None
    for most_specific, pps in reversed(all_pp_chains):
        args, swap = expand_possessive_pronoun_chain(
            args, most_specific, pps)
        if combined_swaps is None:
            combined_swaps = swap
        else:
            combined_swaps = [swap[i] for i in combined_swaps]

    # Build swaps
    swapped_args = create_lambda_swap(combined_swaps)
    arg_counter = 1
    while args[-arg_counter].typ == Ty('n'):
        swapped_args = swapped_args(args[-arg_counter])
        arg_counter += 1

    arg_list = args
    if arg_counter > 1:
        arg_list = args[:-arg_counter + 1]
    swapped_args = Expr.apply(expr_uncurry(swapped_args), Expr.lst(arg_list),
                              reduce=False)

    # Combine
    return new_outside(swapped_args)


def expand_coref(expr, doc):
    """
    Given an expr and a doc containing corefs, create a new expr which expands
    the possessive pronouns.

    :param expr: The expr to be expanded.
    :param doc: The doc containing the corefs.
    :return: The new expr.
    """
    all_pps = []
    for chain in doc._.coref_chains:
        pps_in_chain = []
        for mention in chain:
            for token_index in mention.token_indexes:
                word_expr = find_word_in_expr(expr, doc[token_index], token_index)
                if word_expr.typ == Func(Ty('n'), Ty('n')):
                    print("Found a possessive pronoun!")
                    assert(len(mention.token_indexes) == 1)
                    pps_in_chain.append((doc[mention[0]], mention[0]))

        if len(pps_in_chain) > 0:
            most_specific = []
            for mention in chain[chain.most_specific_mention_index]:
                most_specific.append((doc[mention], mention))
            all_pps.append((most_specific, pps_in_chain))

    if len(all_pps) == 0:
        return expr

    return expand_possessive_pronouns(expr, all_pps)
