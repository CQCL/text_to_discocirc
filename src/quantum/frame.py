from discopy import rigid


def logical_form(diagram):
    if isinstance(diagram, Frame):
        insides = [logical_form(d) for d in diagram._insides]
        return f'{diagram.name}({", ".join(insides)})'
    if isinstance(diagram, rigid.Box):
        return diagram.name
    if isinstance(diagram, rigid.Diagram) and len(diagram) == 1:
        return logical_form(diagram.boxes[0])
    return logical_form(diagram.boxes[-1]) + '(' + logical_form(
        diagram[:-1]) + ')'


class Frame(rigid.Box):
    def __init__(self, name, dom, cod, insides):
        self._insides = insides
        super().__init__(name, dom, cod)
        self.drawing_name = logical_form(self)

    def insert(self, position, inside):
        if position < 0 or position >= len(self._insides):
            raise Exception('Specified slot out of bound!')
        slot = self._insides[position]

        if inside.dom != slot.dom or inside.cod != slot.cod:
            raise ValueError("Inside doesn't match box")

        self._insides[position] = inside

    def _decompose(self):
        s = rigid.Ty('n')
        if len(self.dom) == 1:
            inside_dom = rigid.Ty().tensor(
                *[b.dom @ s for b in self._insides])
            inside_cod = rigid.Ty().tensor(
                *[b.cod @ s for b in self._insides])
            w = rigid.Id(s)
            inside = [(Frame.get_decompose_functor())(b) @ w
                    for b in self._insides]
            top = rigid.Box(f'[{self.name}]', self.dom, inside_dom)
            bot = rigid.Box(f'[\\{self.name}]', inside_cod, self.cod)
            mid = rigid.Id().tensor(*inside)

        elif len(self.dom) == 2:
            inside_dom = rigid.Ty().tensor(
                *[b.dom @ s for b in self._insides]) @ s
            inside_cod = rigid.Ty().tensor(
                *[b.cod @ s for b in self._insides]) @ s
            w = rigid.Id(s)
            inside = [(Frame.get_decompose_functor())(b) @ w
                    for b in self._insides]
            top = rigid.Box(f'[{self.name}]', self.dom, inside_dom)
            bot = rigid.Box(f'[\\{self.name}]', inside_cod, self.cod)
            mid = rigid.Id().tensor(*inside) @ w
            # equation(top, mid, bot)
        
        return top >> mid >> bot

    @staticmethod
    def get_decompose_functor():
        decomp = rigid.Functor(
            ob=lambda x: x,
            ar=lambda b: b._decompose() if hasattr(b, '_decompose') else b)
        return decomp

    @property
    def insides(self):
        return self._insides
