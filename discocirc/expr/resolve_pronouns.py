from copy import deepcopy
from discocirc.expr.normal_form import normal_form

from discocirc.helpers.closed import Ty, Func, deep_copy_ty
from discocirc.expr import Expr
from discocirc.expr.uncurry import uncurry
from discocirc.helpers.discocirc_utils import create_random_variable, \
    create_lambda_swap
from discocirc.expr import pull_out
from discocirc.expr.expr import create_index_mapping_dict, map_expr_indices


def literal_equivalent(expr, word, pos):
    """
    Check if an expr is equivalent to a word in a sentence, specified by the
    word and its position.
    """
    return expr.expr_type == "literal" and \
        expr.name == str(word) and \
        len(expr.head) == 1 and \
        expr.head[0].index == pos + 1


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
        if literal_equivalent(expr, word, pos):
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


def replace_literal_in_expr(expr, word, pos, replacement):
    """
    Given an expr, replace the literal representing the word specified by the
    tuple (word, pos) with the expr replacement. Returns the new expr and the
    expr which was replaced.

    :param expr: The expr in which the literal is replaced.
    :param word: The string representation of the word to be replaced.
    :param pos: The position in the sentence of the word to be replaced.
    :param replacement: The expr to replace the literal with.
    :return: A tuple of the new expr with the literal replaced and
            the expr which was replaced.
    """
    if expr.expr_type == "literal":
        if literal_equivalent(expr, word, pos):
            new_expr = replacement
            replaced = [Expr.literal(expr.name, expr.typ)]
        else:
            new_expr = expr
            replaced = []
    elif expr.expr_type == "application":
        new_fun, fun_exprs = replace_literal_in_expr(expr.fun, word, pos, replacement)
        new_arg, arg_exprs = replace_literal_in_expr(expr.arg, word, pos, replacement)
        new_expr = new_fun(new_arg)
        replaced = fun_exprs + arg_exprs
    elif expr.expr_type == "lambda":
        new_body, body_exprs = replace_literal_in_expr(expr.body, word, pos, replacement)
        new_var, var_exprs = replace_literal_in_expr(expr.var, word, pos, replacement)
        new_expr = Expr.lmbda(new_var, new_body)
        replaced = body_exprs + var_exprs
    elif expr.expr_type == "list":
        new_list = [replace_literal_in_expr(e, word, pos, replacement)
         for e in expr.expr_list]
        new_expr = Expr.lst([element[0] for element in new_list])
        replaced = [e for element in new_list for e in element[1]]

    new_expr.head = expr.head
    return new_expr, replaced


def create_pp_block(most_specific, pps):
    """
    Given a list of the most specific mentions and a list of exprs corresponding
    to states with possessive pronouns, create an expr representing a state
    that combines the exprs of the pps with the most specific
    mentions.

    This code is probably unnecessarily complicated.

    :param most_specific: A list of the arguments corresponding to the most
            specific mentions.
    :param pps: A list of arguments corresponding to the possessive pronouns.
    :return: A new expr representing the state that combines the pps with the
            most specific mentions.
    """
    # Create the new possessive pronouns (which have a different type).

    new_pps = []
    for pp in pps:
        assert(pp.fun.typ.input == pp.fun.typ.output) # pp.fun is the pronoun
        assert(pp.fun.typ.input == pp.typ)
        input_type = Ty.tensor(*[ms_expr.typ for ms_expr in most_specific]) @ pp.typ
        new_type = input_type >> input_type

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
        for pp in pps[:i + 1]:
            temp = create_random_variable(pp.typ)
            lst.append(Expr.lmbda(temp, temp))
        ids.append(Expr.lst(lst))

    for i in range(1, len(new_pps)):
        swap = uncurry(
            create_lambda_swap(
                list(range(0, i - 1)) +
                [i + len(most_specific) - 1] +
                list(range(i - 1, i + len(most_specific) - 1)) +
                [i + len(most_specific)], Ty.tensor(*[arg.typ for arg in [pp_block, pps[i].arg]])))
        new_args = Expr.apply(swap, Expr.lst(
            [pp_block, pps[i].arg]),
                              reduce=False)
        next_layer = Expr.lst([ids[i - 1], new_pps[i]])
        pp_block = next_layer(new_args)

    # Bring the wires back in order.
    # The second to last wires are the most specific mentions, which have to be
    # moved back to the beginning.
    no_wires = len(most_specific) + len(pps)
    unswap = uncurry(create_lambda_swap(
        list(range(no_wires - len(most_specific) - 1, no_wires - 1)) +
        list(range(len(pps) - 1)) +
        [no_wires - 1]
    , pp_block.typ))

    return unswap(pp_block)


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
        arg_pos = 0
        while not find_word_in_expr(remaining_args[arg_pos], word, id):
            arg_pos += 1

        other_args[0] += remaining_args[:arg_pos]
        most_specific_args.append(remaining_args[arg_pos])
        remaining_args = remaining_args[arg_pos + 1:]

    pps = []
    for word, id in pp_mentions:
        arg_pos = 0
        while not find_word_in_expr(remaining_args[arg_pos], word, id):
            arg_pos += 1

        other_args.append(remaining_args[:arg_pos])
        pps.append(remaining_args[arg_pos])

        # Assertion currently made create_pp_block()
        assert(find_word_in_expr(remaining_args[arg_pos].fun, word, id) is not None)

        remaining_args = remaining_args[arg_pos + 1:]

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

    new_outside = uncurry(remainder)

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
    swapped_args = create_lambda_swap(combined_swaps,
                                      Ty.tensor(*[arg.typ for arg in args]))
    arg_counter = 1
    while args[-arg_counter].typ == Ty('n'):
        swapped_args = swapped_args(args[-arg_counter])
        arg_counter += 1

    arg_list = args
    if arg_counter > 1:
        arg_list = args[:-arg_counter + 1]
    swapped_args = Expr.apply(uncurry(swapped_args), Expr.lst(arg_list),
                              reduce=False)

    # Combine
    return new_outside(swapped_args)


def expand_personal_pronouns(expr, all_personal):
    """
    Given an expr and a list of all personal pronouns chains, create an expr
    where the personal pronouns are expanded.

    :param expr: The expr to be expanded.
    :param all_personal: A list of all personal pronoun chains where
         each chain is a tuple of the form (most_specific, pronoun_mentions)
    :return: The expanded expr.
    """
    for typ in expr.typ:
        assert(typ == Ty('n'))

    new_expr = deepcopy(expr)

    for most_specific, personal in all_personal:
        pre_coref_type = deep_copy_ty(new_expr.typ)
        most_specific_exprs = []
        for occurance in most_specific:
            typ = find_word_in_expr(expr, occurance[0], occurance[1]).typ
            temp = create_random_variable(typ)
            body = replace_literal_in_expr(new_expr, occurance[0], occurance[1], temp)
            new_expr = Expr.lmbda(temp, body[0])
            most_specific_exprs += body[1]

        most_specific_typ = Ty('n', index=set.union(*[t.typ.index for t in most_specific_exprs]))
        typs_to_remove = []
        for occurance in personal:
            typ = find_word_in_expr(expr, occurance[0], occurance[1]).typ
            temp = create_random_variable(typ)
            body = replace_literal_in_expr(new_expr, occurance[0], occurance[1], temp)
            new_expr = Expr.lmbda(temp, body[0])
            typs_to_remove.append(typ.index)
            index_mapping = create_index_mapping_dict(typ, most_specific_typ)
            new_expr = map_expr_indices(new_expr, index_mapping, reduce=False)

        new_type = Ty().tensor(*[t for t in pre_coref_type if t.index not in typs_to_remove])
        for ms_expr in most_specific_exprs:
            new_type = ms_expr.typ >> new_type
        new_frame = Expr.literal("coref",
                                 new_expr.typ >> new_type)

        composed = new_frame(new_expr)
        for most_specific_expr in reversed(most_specific_exprs):
            composed = composed(most_specific_expr)

        new_expr = pull_out(composed)
    return new_expr


def expand_coref(expr, doc):
    """
    Given an expr and a doc containing corefs, create a new expr which expands
    the possessive pronouns.

    :param expr: The expr to be expanded.
    :param doc: The doc containing the corefs.
    :return: The new expr.
    """
    all_possessive = []
    all_personal = []
    for chain in doc._.coref_chains:
        possessive_in_chain = []
        personal_in_chain = []
        for mention in chain:
            if mention == chain[chain.most_specific_mention_index]:
                continue

            for token_index in mention.token_indexes:
                word_expr = find_word_in_expr(expr, doc[token_index], token_index)
                if word_expr is None:
                    assert(False)
                if word_expr.typ == Func(Ty('n'), Ty('n')):
                    assert(len(mention.token_indexes) == 1)
                    possessive_in_chain.append((doc[mention[0]], mention[0]))

                if word_expr.typ == Ty('n'):
                    assert(mention == chain[chain.most_specific_mention_index] or len(mention.token_indexes) == 1)
                    personal_in_chain.append((doc[mention[0]], mention[0]))

        most_specific = [(doc[m], m) for m in chain[chain.most_specific_mention_index]]

        if len(possessive_in_chain) > 0:
            all_possessive.append((most_specific, possessive_in_chain))

        if len(personal_in_chain) > 0:
            all_personal.append((most_specific, personal_in_chain))

    if len(all_possessive) > 0:
        expr = normal_form(expr)
        expr = expand_possessive_pronouns(expr, all_possessive)

    if len(all_personal) > 0:
        expr = expand_personal_pronouns(expr, all_personal)

    return expr
