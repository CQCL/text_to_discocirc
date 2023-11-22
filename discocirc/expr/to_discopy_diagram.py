
from discopy import monoidal
from discocirc.diag.drag_up import drag_all

from discocirc.expr import Expr
from discocirc.diag.frame import Frame
from discocirc.helpers.closed import downgrade_to_monoidal
from discocirc.helpers import closed


def _literal_to_diag(expr, context, expand_lambda_frames):
    """
    Creating a diagram for an expr literal.

    :param expr: The expr for which the diag should be created
        (assumed to be of type literal).
    :param context: A set of bound variables through lambda expressions.
    :param expand_lambda_frames: Boolean whether lambda expressions with states
        as variables should be drawn as wires (not used here).
    :return: A diagram corresponding to expr.
    """
    name = expr.name
    if expr in context:
        name = f"context: {expr.name}: {expr.typ}"

        if expand_lambda_frames:
            # If we expand the lambda frames, we use wires to draw how the
            # variables of the lambda expression will be manipulated.
            # Therefore, in that case, each var must be drawn as a state.
            return monoidal.Box(name, monoidal.Ty(), downgrade_to_monoidal(expr.typ))

    output = expr.typ
    if isinstance(output, closed.Func):
        input = monoidal.Ty()
        while isinstance(output, closed.Func):
            input = output.input @ input
            output = output.output
        return monoidal.Box(name, downgrade_to_monoidal(input), downgrade_to_monoidal(output))
    else:
        return monoidal.Box(name, monoidal.Ty(), downgrade_to_monoidal(output))

def _lambda_to_diag_frame(expr, context, expand_lambda_frames):
    """
    Creating a diagram for a lambda expr, drawing it as a frame.

    :param expr: The expr for which the diag should be created
        (assumed to be of type lambda).
    :param context: A set of bound variables through lambda expressions.
    :param expand_lambda_frames: Boolean whether lambda expressions with states
        as variables should be drawn as wires.
    :return: A diagram corresponding to expr.
    """
    context.add(expr.var)
    fun = expr_to_diag(expr.body, context, expand_lambda_frames)
    context.remove(expr.var)
    if expr.var.expr_type == "list":
        var_name = str(list(reversed([str(v).splitlines()[0] for v in expr.var.expr_list])))
    else:
        var_name = expr.var.name
    return Frame(f"λ: {var_name}: {expr.var.typ}",
                 fun.dom @ expr.var.typ,
                 fun.cod,
                 [fun]
                 )


def get_instances_of_var(diag, var):
    """
    Return a list of all the layer numbers where var appears in diag.

    :param diag: The diagram in which all instances should be found.
    :param var: The var which should be found in diag.
    :return: A list of all the layer numbers where var appears in diag.
    """

    instances = []
    for i, (left, box, right) in enumerate(diag.layers):
        if box.name == f"context: {var.name}: {var.typ}":
            instances.append(i)

    return instances


def remove_state_at_layer(diag, layer_no):
    """
    Given a diag, remove the state on layer specified by layer_no.
    Instead of the removed state a new wire is introduced and pulled all the
    way to the top.

    :param diag: The diagram from which the state should be removed.
    :param layer_no: The layer on which the state to be removed is.
    :return: Tuple: (The new diag, the number of the new wire).
    """
    removed_left, removed_box, removed_right = diag.layers[layer_no]
    assert(removed_box.dom == monoidal.Ty())

    new_diag = monoidal.Id(diag.cod)
    for left, box, right in reversed(diag.layers[layer_no + 1:]):
        new_diag = monoidal.Id(left) @ box @ monoidal.Id(right) >> new_diag

    typ = downgrade_to_monoidal(removed_box.cod)
    # position of the new wire that has to be introduced.
    wire_no = len(removed_left)

    for left, box, right in reversed(diag.layers[:layer_no]):
        if wire_no <= len(left):
            new_left = left[:wire_no] @ typ @ left[wire_no:]
            new_right = right
        else:
            no = wire_no - len(left) - len(box.cod)
            new_right = right[:no] @ typ @ right[no:]
            new_left = left
            wire_no += len(box.dom) - len(box.cod)

        new_diag = monoidal.Id(new_left) @ box @ monoidal.Id(new_right) >> new_diag

    return new_diag, wire_no


def swap_wire_to_left(dom, original_pos, expected_pos):
    """
    Given the wires specified by dom, add swaps such that the wire at original_pos
    will end up at expected_pos.

    :param dom: The wires for which the swaps should be created.
    :param original_pos: The position of the wire which should be swapped left.
    :param expected_pos: The position where the specified wire should end up.
    :return: The diag with swaps such that the wire at end_pos is swaped
    """
    assert(original_pos >= expected_pos)
    swaps = monoidal.Id(dom)
    for i in range(expected_pos, original_pos):
        swaps = monoidal.Id(swaps.dom[:i]) @ \
                monoidal.Swap(monoidal.Ty(swaps.dom[i + 1]), monoidal.Ty(swaps.dom[i])) @ \
                monoidal.Id(swaps.dom[i + 2:]) >> swaps
    return swaps


def _lambda_to_diag_open_wire(expr, context, expand_lambda_frames):
    """
    Creating a diagram for a lambda expr drawing the input as an open wire.

    :param expr: The expr for which the diag should be created
        (assumed to be of type lambda).
    :param context: A set of bound variables through lambda expressions
    :param expand_lambda_frames: Boolean whether lambda expressions with states
        as variables should be drawn as wires.
    :return: A diagram corresponding to expr.
    """
    if expr.var.expr_type == 'list':
        # Curry list to be able to use normal draw function
        output = Expr.lmbda(expr.var.expr_list[0], expr.body)
        for var in (expr.var.expr_list[1:]):
            output = Expr.lmbda(var, output)
        return expr_to_diag(output, context, expand_lambda_frames)

    context.add(expr.var)
    body = expr_to_diag(expr.body, context, expand_lambda_frames)
    context.remove(expr.var)

    body = drag_all(body)
    var_instances_layer = get_instances_of_var(body, expr.var)
    assert(len(var_instances_layer) == 1)

    # remove all instances of var
    # keep track of the position of the wires corresponding to the removed boxes
    wire_no_of_removed_boxes = []
    for var_layer in var_instances_layer:
        body, wire_no = remove_state_at_layer(body, var_layer)

        i = 0
        while i < len(wire_no_of_removed_boxes) and \
                wire_no_of_removed_boxes[i] > wire_no:
            i += 1

        for j in range(len(expr.var.typ)):
            wire_no_of_removed_boxes.insert(i, wire_no + j)

        for j in range(i + len(expr.var.typ), len(wire_no_of_removed_boxes)):
            wire_no_of_removed_boxes[j] += len(expr.var.typ)

    # move all instances of var to right
    swaps = monoidal.Id(body.dom)
    for i, wire_no in enumerate(wire_no_of_removed_boxes):
        swaps = swap_wire_to_left(swaps.dom, len(swaps.dom) - i - 1, wire_no) >> swaps


    # make copy box
    var_type = downgrade_to_monoidal(expr.var.typ)
    if len(var_instances_layer) == 0:
        copy_box = monoidal.Box("lambda", var_type, monoidal.Ty())
    elif len(var_instances_layer) == 1:
        copy_box = monoidal.Id(var_type)
    else:
        copy_box = monoidal.Box("lambda", var_type, monoidal.Ty().tensor(
            *[var_type for _ in var_instances_layer]))

    return (monoidal.Id(swaps.dom[:-len(copy_box.cod)]) @ copy_box) >> swaps >> body


def _application_to_diag(expr, context, expand_lambda_frames):
    """
    Creating a diagram for an expr of type application.

    :param expr: The expr for which the diag should be created
        (assumed to be of type application).
    :param context: A set of bound variables through lambda expressions
    :param expand_lambda_frames: Boolean whether lambda expressions with states
        as variables should be drawn as wires.
    :return: A diagram corresponding to expr.
    """
    if expr.arg.expr_type == "list":
        fun = expr_to_diag(expr.fun, context, expand_lambda_frames)
        for arg_expr in reversed(expr.arg.expr_list):
            arg = expr_to_diag(arg_expr, context, expand_lambda_frames)
            fun = _compose_diags(arg, fun)
        return fun

    arg = expr_to_diag(expr.arg, context, expand_lambda_frames)
    fun = expr_to_diag(expr.fun, context, expand_lambda_frames)
    return _compose_diags(arg, fun)


def _compose_diags(arg, fun):
    """
    Composing two diags, ensuring that functions are placed inside
    higher order boxes.

    :param arg: The argument that is supposed to be composed with the function.
    :param fun: The function which is applied to the argument.
    :return: The composed diag of fun(arg).
    """
    if arg.dom == monoidal.Ty() and arg.cod != monoidal.Ty('s') and arg.cod != monoidal.Ty('p'):
        new_args = monoidal.Id(fun.dom[:-len(arg.cod)]) @ arg
        return new_args >> fun

    else:
        # Arg is of type Func and should therefore be placed inside fun.
        # TODO: this currently assumes that the type of arg is equal to fun.dom[-1]
        # If, for example, arg.typ == fun.dom[-2:] more of the dom would have
        # to be removed. This does not seem to happen in our tests.
        # Not sure, if this will be a problem in the future.
        new_dom = fun.dom[:-1]

        inputs = monoidal.Id(new_dom)
        for left, box, right in fun.layers[:-1]:
            assert(len(right) == 0)
            if box.dom == monoidal.Ty():
                inputs = inputs @ box
            else:
                inputs = inputs >> (monoidal.Id(inputs.cod[:-len(box.dom)]) @ box)

        left, box, right = fun.layers[-1]
        frame_dom = inputs.cod[len(left):]
        if len(right) > 0:
            frame_dom = frame_dom[:-len(right)]

        if isinstance(box, Frame):
            frame = Frame(box, frame_dom, box.cod,
                          [arg] + box.insides)
        else:
            frame = Frame(box, frame_dom, box.cod, [arg])

        return inputs >> (monoidal.Id(left) @ frame @ monoidal.Id(right))


def _list_to_diag(expr, context, expand_lambda_frames):
    """
    Creating a diagram for an expr of type list.

    :param expr: The expr for which the diag should be created
        (assumed to be of type list).
    :param context: A set of bound variables through lambda expressions
    :param expand_lambda_frames: Boolean whether lambda expressions with states
        as variables should be drawn as wires.
    :return: A diagram corresponding to expr.
    """
    output = monoidal.Id(monoidal.Ty())
    for val in expr.expr_list:
        diag = expr_to_diag(val, context, expand_lambda_frames)
        output = output @ diag

    return output


def expr_to_diag(expr, context=None, expand_lambda_frames=True):
    """
    Creating a diagram for an expr.

    :param expr: The expr for which the diag should be created.
    :param context: A set of bound variables through lambda expressions.
    :param expand_lambda_frames: Boolean whether lambda expressions with states
        as variables should be drawn as wires.
    :return: A diagram corresponding to expr.
    """
    if context is None:
        context = set()

    if expr.expr_type == "literal":
        return _literal_to_diag(expr, context, expand_lambda_frames)
    elif expr.expr_type == "lambda":
        if expand_lambda_frames:
            return _lambda_to_diag_open_wire(expr, context, expand_lambda_frames)
        else:
            return _lambda_to_diag_frame(expr, context, expand_lambda_frames)
    elif expr.expr_type == "application":
        return _application_to_diag(expr, context, expand_lambda_frames)
    elif expr.expr_type == "list":
        return _list_to_diag(expr, context, expand_lambda_frames)
    else:
        raise NotImplementedError(expr.expr_type)


def expr_to_frame_diag(expr):
    return Frame.get_decompose_functor()(expr_to_diag(expr))
