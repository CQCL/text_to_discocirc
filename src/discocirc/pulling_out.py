from discopy import rigid
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
                # or get_ccg_input(ccg) == rigid.Ty('p')\
                # or get_ccg_input(ccg) == rigid.Ty('s'):
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
    hole_filling_args = []

    inner_term_holes = get_holes(inner_term)

    for i, arg in enumerate(inner_term.args.copy()):
        # If current argument should go into a hyper hole: skip
        # (by recursive property of pulling out, we assume all internal
        # hyperboxes to already be pulled out correctly).
        # Thus, they should take exactly one input, which we don't pull out.
        if i in inner_term_holes:
            arg_ccg = arg.ccg
            for _ in arg.args:
                arg_ccg = get_ccg_output(arg_ccg)

            hole_filling_args.append((type(ccg), arg_ccg))
            ccg = get_ccg_output(ccg)
            continue

        # Pull out the argument
        term.args.insert(hole_position + len(pulled_out_args) + 1, arg)
        pulled_out_args.append((type(ccg), arg.output_ccg))
        inner_term.args.remove(arg)

        ccg = get_ccg_output(ccg)

    # Update the ccg_type in reverse order such that the first argument pulled
    # out is the last added to the ccg and thus the next input
    new_inner_ccg = inner_term.ccg
    for _ in range(len(pulled_out_args) + len(hole_filling_args)):
        new_inner_ccg = get_ccg_output(new_inner_ccg)

    term_ccg = term.ccg
    for _ in range(hole_position):
        term_ccg = get_ccg_output(term_ccg)

    for ccg_type, ccg in reversed(pulled_out_args):
        if ccg_type == Over:
            term_ccg.left = Over(term_ccg.left, ccg)
            term_ccg.right = Over(term_ccg.right, ccg)
            new_inner_ccg = Over(new_inner_ccg, ccg)
        elif ccg_type == Under:
            term_ccg.left = Under(ccg, term_ccg.left)
            term_ccg.right = Under(ccg, term_ccg.right)
            new_inner_ccg = Under(ccg, new_inner_ccg)

    for ccg_type, ccg in reversed(hole_filling_args):
        if ccg_type == Over:
            new_inner_ccg = Over(new_inner_ccg, ccg)
        elif ccg_type == Under:
            new_inner_ccg = Under(ccg, new_inner_ccg)

    inner_term.ccg = new_inner_ccg


def recurse_pull(term):
    """
    Given a term, recursively pull out all the hyperboxes to get a fully
    pulled out term.

    :param term: Term - The term which should be pulled out.
    """
    for i in range(len(term.args)):
        recurse_pull(term.args[i])

    hyper_holes = get_holes(term)
    num_holes = len(hyper_holes)
    for i in range(len(hyper_holes)):
        pull_single_hole(term, hyper_holes[i])

        # As we pull out arguments, the position of the holes changes.
        # The number of holes should not. Hence the assertion.
        hyper_holes = get_holes(term)
        assert(len(hyper_holes) == num_holes)
