from discopy import rigid, Ob

from discocirc.frame import Frame


def expand_wires(wires, last_n_n):
    new_n_n = []
    new_wires = rigid.Ty()

    for type in wires:
        if type == Ob('s'):
            new_wires = new_wires @ last_n_n[0]
            new_n_n.append(last_n_n[0])
            last_n_n = last_n_n[1:]
        else:
            new_wires = new_wires @ rigid.Ty(type)

    return new_wires, last_n_n, new_n_n


def expand_box(box, last_n_n):
    new_insides = []
    if isinstance(box, Frame):
        for inside in box.insides:
            new_insides.append(expand_s_types(inside))

    n, s = map(rigid.Ty, 'ns')

    # Expand dom
    new_dom = rigid.Ty()
    for type in box.dom:
        if type == Ob('s'):
            new_dom = new_dom @ last_n_n[0]
            last_n_n = last_n_n[1:]
        else:
            new_dom = new_dom @ rigid.Ty(type)

    # Expand cod
    new_cod = box.cod
    new_type = []
    if s.objects[0] in box.cod:
        assert box.cod.count(s) == 1
        n_n = n ** (new_dom.count(n) - box.cod.count(n))
        pos = box.cod.objects.index(s.objects[0])
        left, right = box.cod[:pos], box.cod[pos + 1:]
        new_cod = left @ n_n @ right

        new_type = [n_n]

    if isinstance(box, Frame):
        expanded = Frame(box.name, new_dom, new_cod, new_insides)
    else:
        expanded = rigid.Box(box.name, new_dom, new_cod)

    return expanded, last_n_n, new_type


def expand_layer(layer, last_n_n):
    left, box, right = layer
    left, last_n_n, new_n_n = expand_wires(left, last_n_n)

    box, last_n_n, new = expand_box(box, last_n_n)
    new_n_n += new

    right, last_n_n, new = expand_wires(right, last_n_n)
    new_n_n += new

    assert(len(last_n_n) == 0)

    return left, box, right, new_n_n


def expand_s_types(diagram):
    new_diag = rigid.Id(diagram.dom)
    last_n_n = []

    for layer in diagram.layers:
        left, box, right, last_n_n = expand_layer(layer, last_n_n)

        left, right = map(rigid.Id, (left, right))
        new_diag = new_diag >> left @ box @ right

    return new_diag
