from discocirc.semantics.rules import passive_to_active_voice, remove_relative_pronouns, remove_to_be, remove_articles


rules_dict = {
    "remove_articles": remove_articles,
    "remove_to_be": remove_to_be,
    "remove_relative_pronouns": remove_relative_pronouns,
    "passive_to_active_voice": passive_to_active_voice,
}


def rewrite(diagram, rules='all'):
    """
    Applies a list of rewrite rules to a diagram
    """
    if rules == 'all':
        rules = rules_dict.keys()
    for rule in rules:
        diagram = rules_dict[rule](diagram)
    return diagram
