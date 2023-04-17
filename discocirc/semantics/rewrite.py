from discocirc.semantics.rules import remove_relative_pronouns, remove_to_be, remove_articles


rules_dict = {
    "remove_articles": remove_articles,
    "remove_to_be": remove_to_be,
    "remove_relative_pronouns": remove_relative_pronouns,
}


def rewrite(diagram, rules='all'):
    """
    Applies a list of rules to a diagram
    """
    if rules == 'all':
        rules = rules_dict.keys()
    for rule in rules:
        diagram = rules_dict[rule](diagram)
    return diagram
