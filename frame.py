from discopy import rigid


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

decomp = rigid.Functor(
    ob=lambda x: x,
    ar=lambda b: b._decompose() if hasattr(b, '_decompose') else b)
