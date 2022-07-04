from discopy.biclosed import Over, Under

from discocirc.discocirc_utils import get_ccg_input, get_ccg_output


def get_holes(term):
    """
    Finds the position of all the holes a given term has,
    i.e. all the inputs that are processes

    :param term: Term - for which the holes should be found.
    :return: List - position in argument list of all holes.
    """
    holes = []
    ccg = term.ccg
    for i in range(len(term.args)):
        if isinstance(get_ccg_input(ccg), (Over, Under)):
            holes.append(i)
        ccg = get_ccg_output(ccg)

    return holes

def pull_single_hole(term, hole_position):
    """
    Given a hyperbox pull out the arguments of the specified hole.
    For hyperboxes with multiple holes, this has to be called multiple times.

    :param term: Term - hyperbox who's arguments will be pulled out.
                 We assume that all internal hyperboxes are fully pulled out.
    :param hole_position: int - the argument position of the hole which should
                be pulled out.
    """
    inner_term = term.args[hole_position]

    ccg = inner_term.ccg
    pulled_out_args = []

    inner_term_holes = get_holes(inner_term)

    for i, ar in enumerate(inner_term.args.copy()):
        # If current argument should go into a hyper hole: skip
        # (by recursive property of pulling out, we assume all internal
        # hyperboxes to already be pulled out correctly).
        # Thus, they should take exactly one input, which we don't pull out.
        if i in inner_term_holes:
            ccg = get_ccg_output(ccg)
            continue

        # Pull out the argument
        term.args.insert(hole_position + len(pulled_out_args) + 1, ar)
        pulled_out_args.append((type(ccg), ar.output_ccg))
        inner_term.args.remove(ar)

        ccg = get_ccg_output(ccg)

    # Update the ccg_type in reverse order such that the first argument pulled
    # out is the last added to the ccg and thus the next input
    term_ccg = term.ccg
    for _ in range(hole_position):
        term_ccg = get_ccg_output(term_ccg)

    for ccg_type, ccg in reversed(pulled_out_args):
        if ccg_type == Over:
            term_ccg.left = Over(term_ccg.left, ccg)
            term_ccg.right = Over(term_ccg.right, ccg)
        elif ccg_type == Under:
            term_ccg.left = Under(ccg, term_ccg.left)
            term_ccg.right = Under(ccg, term_ccg.right)


# %%
def recurse_pull(term):
    """
    Given a term, recursively pull out all the hyperboxes to get a fully
    pulled out term.

    :param term: Term - The term which should be pulled out.
    """
    for i in range(len(term.args)):
        recurse_pull(term.args[i])

    hyper_holes = get_holes(term)
    for hole in hyper_holes:
        pull_single_hole(term, hole)
