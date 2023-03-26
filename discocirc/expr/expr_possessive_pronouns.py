def find_word_in_expr(expr, word):
    if expr.expr_type == "literal":
        if expr.name == word:
            return expr
        else:
            return None


def expand_possessive_pronouns(expr, corefs):
    for coref in corefs:
        for mention in coref:
            word = find_word_in_expr(expr, mention)