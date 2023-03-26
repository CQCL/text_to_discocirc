def find_word_in_expr(expr, word, pos):
    if expr.expr_type == "literal":
        if expr.name == str(word):
            return expr
        else:
            return None
    elif expr.expr_type == "application":
        return find_word_in_expr(expr.fun, word, pos) or find_word_in_expr(expr.arg, word, pos)
    elif expr.expr_type == "lambda":
        return find_word_in_expr(expr.expr, word, pos)
    elif expr.expr_type == "list":
        for e in expr.expr_list:
            result = find_word_in_expr(e, word, pos)
            if result:
                return result
        return None


def expand_possessive_pronouns(expr, doc):
    for coref in doc._.coref_chains:
        for mention in coref:
            for token_index in mention.token_indexes:
                word = find_word_in_expr(expr, doc[token_index], token_index)

    return expr