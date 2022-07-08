from discopy import rigid

from discocirc.frame import Frame


def expand_box(box, last_n_n):
    n, s = map(rigid.Ty, 'ns')
    if isinstance(box, Frame):
        name, dom, cod = box.name, box.dom, box.cod
        insides = [expand_s_types(b) for b in box.insides]
        box = Frame(name, dom, cod, insides)

    if s.objects[0] in box.dom.objects:
        pos = box.dom.objects.index(s.objects[0])
        left, right = box.dom[:pos], box.dom[pos + 1:]
        box = rigid.Box(box.name, left @ last_n_n @ right, box.cod)

    if s.objects[0] not in box.cod:
        return box, None

    assert box.cod.count(s) == 1
    n_n = n ** box.dom.count(n)
    pos = box.cod.objects.index(s.objects[0])
    left, right = box.cod[:pos], box.cod[pos + 1:]

    if isinstance(box, Frame):
        expanded = Frame(box.name, box.dom, left @ n_n @ right, box.insides)
    else:
        expanded = rigid.Box(box.name, box.dom, left @ n_n @ right)
    return expanded, n_n


def expand_s_types(diagram):
    new_diag = rigid.Id(diagram.dom)
    last_n_n = None
    for left, box, right in diagram.layers:
        box, n_n = expand_box(box, last_n_n)
        if n_n:
            last_n_n = n_n
        left, right = map(rigid.Id, (left, right))
        try:
            new_diag = new_diag >> left @ box @ right
        except Exception:
            n, s = map(rigid.Ty, 'ns')
            n_n = n ** (len(new_diag.cod) - len((left @ box @ right).dom) + 1)
            pos = box.dom.objects.index(s.objects[0])
            _left, _right = rigid.Id(box.dom[:pos]), rigid.Id(box.dom[pos+1:])
            expander = rigid.Box('x', n_n, s)
            new_diag = (
                new_diag >> left @ (_left @ expander @ _right >> box) @ right)
    return new_diag