from discopy import monoidal


def logical_form(diagram):
    if isinstance(diagram, Frame):
        insides = [logical_form(d) for d in diagram._insides]
        return f'{diagram.name}({", ".join(insides)})'
    if isinstance(diagram, monoidal.Box):
        return diagram.name
    if isinstance(diagram, monoidal.Diagram) and len(diagram) == 1:
        return logical_form(diagram.boxes[0])
    return logical_form(diagram.boxes[-1]) + '(' + logical_form(
        diagram[:-1]) + ')'


class Frame(monoidal.Box):
    def __init__(self, name, dom, cod, insides):
        self._insides = insides
        super().__init__(str(name), dom, cod)
        self.drawing_name = logical_form(self)

    def insert(self, position, inside):
        if position < 0 or position >= len(self._insides):
            raise Exception('Specified slot out of bound!')
        slot = self._insides[position]

        if inside.dom != slot.dom or inside.cod != slot.cod:
            raise ValueError("Inside doesn't match box")

        self._insides[position] = inside

    def _decompose(self):
        s = monoidal.Ty('*')
        inside_dom = monoidal.Ty().tensor(
            *[s @ b.dom for b in self._insides]) @ s
        inside_cod = monoidal.Ty().tensor(
            *[s @ b.cod for b in self._insides]) @ s
        w = monoidal.Id(s)
        inside = [w @ (Frame.get_decompose_functor())(b)
                  for b in self._insides]
        top = monoidal.Box(f'[{self.name}]', self.dom, inside_dom)
        bot = monoidal.Box(f'[\\{self.name}]', inside_cod, self.cod)
        mid = monoidal.Id().tensor(*inside) @ w
        # equation(top, mid, bot)
        return top >> mid >> bot

    @staticmethod
    def get_decompose_functor():
        decomp = monoidal.Functor(
            ob=lambda x: x,
            ar=lambda b: b._decompose() if hasattr(b, '_decompose') else b)
        return decomp

    @property
    def insides(self):
        return self._insides


class Diagram(monoidal.Diagram):
    pass

class Functor(monoidal.Functor):
    def __init__(self, ob, ar, frame, ob_factory=monoidal.Ty, ar_factory=Diagram):
        super().__init__(ob, ar, ob_factory, ar_factory)
        self._frame = frame

    def __call__(self, diagram):
        if isinstance(diagram, Frame):
            return self._frame(Frame(diagram.name, self(diagram.dom), self(diagram.cod),
                               [self(i) for i in diagram.insides]))
        return super().__call__(diagram)
