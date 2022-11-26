from discopy import rigid, Diagram

from discocirc.closed import Func, Ty, uncurry_types
from discocirc.frame import Frame

class LambdaType(rigid.Ty):
    def __init__(self, arg):
        super().__init__(*arg)

def _literal_to_diag(expr, context, expand_state_lambda):
    name = expr.name
    if expr in context:
        if not isinstance(expr.final_type, Func) and expand_state_lambda:
            return rigid.Id(LambdaType(expr.final_type))
        name = f"context: {expr.name}: {expr.final_type}"

    uncurried = uncurry_types(expr.final_type)
    if isinstance(uncurried, Func):
        return rigid.Box(name, uncurried.input, uncurried.output)
    else:
        return rigid.Box(name, Ty(), uncurried)


def _add_swaps(domaine, no_swaps, final_pos):
    swaps = rigid.Id(domaine)
    for i in reversed(range(1, no_swaps + 1)):
        swap_layer = rigid.Id(domaine[:-i - 1 - final_pos]) @ \
                     Diagram.swap(rigid.Ty(domaine[-i - final_pos]),
                                  rigid.Ty(domaine[-i - 1 - final_pos]))
        if i + final_pos != 1:
            swap_layer = swap_layer @ rigid.Id(domaine[-i + 1 - final_pos:])
        swaps = swap_layer >> swaps
        domaine = swaps.dom
    return swaps

def expand_lambda(var, body):
    occurance_counter = 0

    # move all occurrences of var to the right
    for i, inp in enumerate(reversed(body.cod[0:])):
        if isinstance(inp, LambdaType) and inp == var:
            swap_layer = _add_swaps(body.dom, i - occurance_counter,
                                    occurance_counter)
            body = swap_layer >> body
            occurance_counter += 1


    if occurance_counter == 0:
        copy_box = rigid.Box(f"del: {var}", var.final_type,
                             Ty())
    elif occurance_counter == 1:
        copy_box = rigid.Id(var.final_type)
    else:
        copy_box = rigid.Box(f"copy: {var}", var.final_type,
                             rigid.Ty.tensor(
                                 *[var.final_type for _ in
                                   range(occurance_counter)]))

    body = rigid.Id(body.dom[:-occurance_counter]) @ copy_box >> body

    return body

def _lambda_to_diag(expr, context, expand_state_lambda):
    context.add(expr.var)
    body = expr_to_diag(expr.expr, context, expand_state_lambda)
    context.remove(expr.var)

    if isinstance(expr.var.final_type, Func) or not expand_state_lambda:
        return Frame(f"lambda: {expr.var.name}: {expr.var.final_type}",
                     body.dom @ expr.var.final_type,
                     body.cod,
                     [body]
                )
    else:
        return expand_lambda(expr.var, body)


def get_next_input(inputs):
    for i in range(1, len(inputs) + 1):
        if isinstance(inputs[-i], rigid.Ty) or \
                isinstance(inputs[-i], Func):
            index = len(inputs) - i
            break

    return index

def _application_to_diag(expr, context, expand_state_lambda):
    arg = expr_to_diag(expr.arg, context, expand_state_lambda)
    body = expr_to_diag(expr.expr, context, expand_state_lambda)

    input_index = get_next_input(body.cod)

    if not isinstance(expr.arg.final_type, Func):
        new_args = rigid.Id(body.dom[:input_index - len(arg.cod) + 1]) @ \
                    arg @ \
                    rigid.Id(body.dom[input_index + len(arg.cod):])
        return new_args >> body

    else:
        new_dom = body.dom[:input_index] @ \
                    body.dom[input_index + 1:]

        # TODO: this assumes that the thing we apply to is on the last layer
        inputs = rigid.Id(new_dom)
        wire_delete_idx = input_index
        for left, box, right in body.layers[:-1]:
            if len(left) > wire_delete_idx:
                left = left[:wire_delete_idx] @ left[wire_delete_idx + 1:]
            elif len(left) == wire_delete_idx:
                raise RuntimeError('Something wrong')
            else:
                idx = wire_delete_idx - len(left) - 1
                right = right[:idx] @ right[idx + 1:]
            inputs = inputs >> (rigid.Id(left) @ box @ rigid.Id(right))

        if isinstance(body[-1], Frame):
            body[-1].insides.append(arg)
            frame = body[-1]
        else:
            frame = Frame(body[-1], inputs.cod, body.cod, [arg])

        return inputs >> frame



def _list_to_diag(expr, context, expand_state_lambda):
    output = rigid.Id(Ty())
    for val in expr.expr_list:
        diag = expr_to_diag(val, context, expand_state_lambda)
        output = output @ diag

    return output


def expr_to_diag(expr, context=None, expand_state_lambda=False):
    if context is None:
        context = set()

    if expr.expr_type == "literal":
        return _literal_to_diag(expr, context, expand_state_lambda)
    elif expr.expr_type == "lambda":
        return _lambda_to_diag(expr, context, expand_state_lambda)
    elif expr.expr_type == "application":
        return _application_to_diag(expr, context, expand_state_lambda)
    elif expr.expr_type == "list":
        return _list_to_diag(expr, context, expand_state_lambda)
    else:
        raise NotImplementedError(expr.expr_type)