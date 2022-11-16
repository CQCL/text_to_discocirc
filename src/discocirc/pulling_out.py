from discopy import rigid
from discopy.biclosed import Over, Under

from discocirc.closed import Func, Ty


def get_holes(term):
    """
    Finds the position of all the holes a given term has,
    i.e. all the inputs that are processes

    :param term: Term - for which the holes should be found.
    :return: List - position in argument list of all holes.
    """
    holes = []
    simple_type = term.simple_type
    for i in range(len(term.args)):
        if is_higher_order(simple_type):
            holes.append(i)
        simple_type = simple_type.output

    return holes

def is_higher_order(simple_type):
    if not isinstance(simple_type, Func):
        return False
    return isinstance(simple_type.input, Func) \
                or simple_type.input == Ty('p')\
                or simple_type.input == Ty('s')


def pull_single_hole(term, hole_position):
    """
    Given a hyperbox, pull out the arguments of the specified hole.
    For hyperboxes with multiple holes, this has to be called multiple times.

    :param term: Term - hyperbox whose arguments will be pulled out.
                 We assume that all internal hyperboxes are fully pulled out.
    :param hole_position: int - the argument position of the hole which should
                be pulled out.
    """
    inner_term = term.args[hole_position]

    inner_term_type = inner_term.simple_type
    pulled_out_args = []
    hole_filling_args = []

    inner_term_holes = get_holes(inner_term)

    for i, arg in enumerate(inner_term.args.copy()):
        # If current argument should go into a hole: skip
        # (by recursive property of pulling out, we assume all internal
        # hyperboxes to already be pulled out correctly).
        # Thus, they should take exactly one input, which we don't pull out.
        if i in inner_term_holes:
            arg_type = arg.simple_type
            for _ in arg.args:
                arg_type = arg_type.output

            hole_filling_args.append(arg_type)
            inner_term_type = inner_term_type.output
            continue

        # Pull out the argument
        term.args.insert(hole_position + len(pulled_out_args) + 1, arg)
        pulled_out_args.append(arg.final_type)
        inner_term.args.remove(arg)

        inner_term_type = inner_term_type.output

    # Update the type in reverse order such that the first argument pulled
    # out is the last added to the type and thus the next input
    new_inner_type = inner_term.simple_type
    for _ in range(len(pulled_out_args) + len(hole_filling_args)):
        new_inner_type = new_inner_type.output

    term_type = term.simple_type
    for _ in range(hole_position):
        term_type = term_type.output

    for pulled_out_arg in reversed(pulled_out_args):
        term_type.input = Func(pulled_out_arg, term_type.input)
        term_type.output = Func(pulled_out_arg, term_type.output)
        new_inner_type = Func(pulled_out_arg, new_inner_type)

    for hole_filling_arg in reversed(hole_filling_args):
        new_inner_type = Func(hole_filling_arg, new_inner_type)

    inner_term.simple_type = new_inner_type


def recurse_pull(term):
    """
    Given a term, recursively pull out all the hyperboxes to get a fully
    pulled out term.

    :param term: Term - The term which should be pulled out.
    """
    for i in range(len(term.args)):
        recurse_pull(term.args[i])

    holes = get_holes(term)
    num_holes = len(holes)
    for i in range(len(holes)):
        pull_single_hole(term, holes[i])

        # As we pull out arguments, the position of the holes changes.
        # The number of holes should not. Hence, the assertion.
        holes = get_holes(term)
        assert(len(holes) == num_holes)
