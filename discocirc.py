"""
WARNING: this code is NOT ready for public consumption.
Please don't copy around into notebooks, or else I will be forced
to maintain old copies of this code forever.
"""

from discopy.biclosed import Over, Under
from discopy import biclosed, rigid
from discopy.rewriting import InterchangerError


def logical_form(diagram):
    if isinstance(diagram, Frame):
        insides = [logical_form(d) for d in diagram._insides]
        slots = ['?'] * len(diagram._slots)
        return f'{diagram.name}({", ".join(insides + slots)})'
    if isinstance(diagram, rigid.Box):
        return diagram.name
    if isinstance(diagram, rigid.Diagram) and len(diagram) == 1:
        return logical_form(diagram.boxes[0])
    return logical_form(diagram.boxes[-1]) + '(' + logical_form(diagram[:-1]) + ')'


class Frame(rigid.Box):
    def __init__(self, name, dom, cod, insides, slots):
        self._insides = insides
        self._slots = slots
        super().__init__(name, dom, cod)
        self.drawing_name = logical_form(self)

    def insert(self, inside):
        if not self._slots:
            raise Exception('No slot!')
        slot = self._slots[0]
        if inside.dom != slot.dom or inside.cod != slot.cod:
            raise ValueError("inside doesn't match box")
        name, dom, cod = self.name, self.dom, self.cod
        insides, slots = self._insides, self._slots
        return type(self)(name, dom, cod, insides + [inside], slots[1:])

    def _decompose(self):
        s = rigid.Ty('*')
        inside_dom = rigid.Ty().tensor(
            *[s @ b.dom for b in self._insides + self._slots]) @ s
        inside_cod = rigid.Ty().tensor(
            *[s @ b.cod for b in self._insides + self._slots]) @ s
        w = rigid.Id(s)
        inside = [w @ decomp(b)
                  for b in self._insides + self._slots]
        top = rigid.Box(f'[{self.name}]', self.dom, inside_dom)
        bot = rigid.Box(f'[\\{self.name}]', inside_cod, self.cod)
        mid = rigid.Id().tensor(*inside) @ w
        # equation(top, mid, bot)
        return top >> mid >> bot


def swap_right(diagram, i):
    left, box, right = diagram.layers[i]
    if box.dom:
        raise ValueError(f"{box} is not a word.")

    new_left, new_right = left @ right[0:1], right[1:]
    new_layer = rigid.Id(new_left) @ box @ rigid.Id(new_right)
    return (
        diagram[:i]
        >> new_layer.permute(len(new_left), len(new_left) - 1)
        >> diagram[i+1:])


def drag_out(diagram, i):
    box = diagram.boxes[i]
    if box.dom:
        raise ValueError(f"{box} is not a word.")
    while i > 0:
        try:
            diagram = diagram.interchange(i-1, i)
            i -= 1
        except InterchangerError:
            diagram = swap_right(diagram, i)
    return diagram


def drag_all(diagram):
    i = len(diagram) - 1
    stop = 0
    while i >= stop:
        box = diagram.boxes[i]
        if not box.dom:  # is word
            diagram = drag_out(diagram, i)
            i = len(diagram) - 1
            stop += 1
        i -= 1
    return diagram


def expand_box(box):
    n, s = map(rigid.Ty, 'ns')
    if isinstance(box, Frame):
        name, dom, cod = box.name, box.dom, box.cod
        insides = [expand_diagram(b) for b in box._insides]
        box = Frame(name, dom, cod, insides, box._slots)
    if s.objects[0] not in box.cod:
        return box
    assert box.cod.count(s) == 1
    n_n = n ** box.dom.count(n)
    pos = box.cod.objects.index(s.objects[0])
    left, right = rigid.Id(box.cod[:pos]), rigid.Id(box.cod[pos+1:])
    expander = rigid.Box('x', s, n_n)
    return box >> left @ expander @ right


def expand_diagram(diagram):
    new_diag = rigid.Id(diagram.dom)
    for left, box, right in diagram.layers:
        box = expand_box(box)
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


def merge_x(diagram):
    if isinstance(diagram, Frame):
        name, dom, cod = diagram.name, diagram.dom, diagram.cod
        insides, slots = diagram._insides, diagram._slots
        insides = [merge_x(inside) for inside in insides]
        return type(diagram)(name, dom, cod, insides, slots)

    new_diag = rigid.Id(diagram.dom)
    layers = list(diagram.layers)
    i = 0
    while i < len(layers):
        left, box, right = layers[i]
        if isinstance(box, Frame):
            box = merge_x(box)
        if i + 1 < len(layers):
            _, next_box, _ = layers[i + 1]
            if next_box.name == 'x':
                # time to merge
                name, dom, cod = box.name, box.dom, next_box.cod
                if not isinstance(box, Frame):
                    box = type(box)(name, dom, cod)
                else:
                    insides, slots = box._insides, box._slots
                    box = type(box)(name, dom, cod, insides, slots)
                layers.pop(i + 1)
        new_diag >>= rigid.Id(left) @ box @ rigid.Id(right)
        i += 1
    return new_diag


def order(t):
    if isinstance(t, Over):
        return max(order(t.right) + 1, order(t.left))
    if isinstance(t, Under):
        return max(order(t.left) + 1, order(t.right))
    return 0

# over: out << inp
# under: inp >> out


def convert_type(ccg_t):
    outside_dom, outside_cod = rigid.Ty(), rigid.Ty()
    insides = []

    div = 0
    while isinstance(ccg_t, (Over, Under)):
        if isinstance(ccg_t, Over):
            inp, out = ccg_t.right, ccg_t.left
        else:  # is under
            inp, out = ccg_t.left, ccg_t.right
        if not isinstance(inp, (Over, Under)):  # is basic type
            new_t = biclosed.biclosed2rigid(inp)
            if isinstance(ccg_t, Over):
                outside_dom = outside_dom @ new_t
            else:  # is under
                outside_dom = new_t @ outside_dom
                div += 1
        else:  # time to bubble
            # TODO whats up with _inside?
            _dom, _cod, _div, _inside = convert_type(inp)
            insides.append((_dom, _cod, _div))
        ccg_t = out
    outside_cod = biclosed.biclosed2rigid(ccg_t)
    return outside_dom, outside_cod, div, insides


def convert_word(word):
    t = word.cod
    dom, cod, div, insides = convert_type(t)
    if insides:
        # TODO treat _div properly
        slots = [rigid.Box('[]', dom, cod)
                 for (dom, cod, _div) in insides]
        return Frame(word.name, dom, cod, [], slots), div
    return rigid.Box(word.name, dom, cod), div


def insert_frame(diagram, new_box):
    for i, (left, box, right) in enumerate(diagram.layers):
        if isinstance(box, Frame):
            layer = rigid.Id(left) @ box.insert(new_box) @ rigid.Id(right)
            return diagram[:i] >> layer >> diagram[i+1:]
    raise Exception('No frame!')


def convert_sentence(diagram):
    diags = []

    for box, offset in zip(diagram.boxes, diagram.offsets):
        i = 0
        off = offset
        # find the first box to contract
        while i < len(diags) and off >= len(diags[i][0].cod):
            off -= len(diags[i][0].cod)
            i += 1
        if off == 0 and not box.dom:
            diags.insert(i, convert_word(box))
        else:
            # always a binary box  TODO unary rules
            if len(box.dom) == 1:
                raise NotImplementedError
            (left, left_div), (right, right_div) = diags[i], diags[i+1]
            if "FA" in box.name:
                ord = order(box.dom[0:1].right)
                if ord == 0:  # compose
                    new_diag = (
                        rigid.Id(left.dom[:left_div]) @ right @
                        rigid.Id(left.dom[left_div + 1:])
                        >> left)
                    div = left_div
                elif ord >= 1:  # put inside
                    new_diag = insert_frame(left, right)
                    div = left_div
                else:
                    raise NotImplementedError
            elif "BA" in box.name:
                ord = order(box.dom[1:2].left)
                if ord == 0:  # compose
                    new_diag = (
                        rigid.Id(right.dom[:right_div - 1]) @ left @
                        rigid.Id(right.dom[right_div:])
                        >> right)
                    div = right_div - 1
                elif ord >= 1:  # put inside
                    new_diag = insert_frame(right, left)
                    div = right_div
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            diags[i:i+2] = [(new_diag, div)]
        step = rigid.Id().tensor(*[d[0] for d in diags])
    step = merge_x(decomp(expand_diagram(drag_all(step))))
    # step = merge_x(expand_diagram(step))
    # res = merge_x(decomp(step))
    # res = drag_all(res)
    # return res
    return step


decomp = rigid.Functor(
    ob=lambda x: x,
    ar=lambda b: b._decompose() if hasattr(b, '_decompose') else b)
