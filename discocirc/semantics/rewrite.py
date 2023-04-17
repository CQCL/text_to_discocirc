from discocirc.semantics.rules import remove_is, remove_the


rules_dict = {
    "remove_the": remove_the,
    "remove_is": remove_is,
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
