from discopy import monoidal

class Frame(monoidal.Box):
    """
    A subclass of monoidal.Box that supports higher order boxes, i.e. box that has another diagram inside it. 
    """
    def __init__(self, name, dom, cod, insides):
        """
        Initializes a Frame object with a name, domain, codomain, and a list of insides.
        """
        self._insides = insides
        super().__init__(str(name), dom, cod)
        self.drawing_name = logical_form(self)

    def _decompose(self):
        """
        Decomposes a Frame object into a diagram of boxes, where the insides are sandwiched by top 
        and bottom boxes with *-wires acting as walls between multiple holes.
        This is a temporary solution, as discopy does not support drawing for higher order boxes.
        """
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
        return top >> mid >> bot

    @staticmethod
    def get_decompose_functor():
        """
        Returns a functor that decomposes a Frame object into a diagram of boxes.
        """
        decomp = monoidal.Functor(
            ob=lambda x: x,
            ar=lambda b: b._decompose() if hasattr(b, '_decompose') else b)
        return decomp

    @property
    def insides(self):
        """
        Returns the list of insides of a Frame object.
        """
        return self._insides


class Diagram(monoidal.Diagram):
    """
    A subclass of monoidal.Diagram
    """
    pass

class Functor(monoidal.Functor):
    """
    A subclass of monoidal.Functor that supports higher order boxes.
    """
    def __init__(self, ob, ar, frame, ob_factory=monoidal.Ty, ar_factory=Diagram):
        super().__init__(ob, ar, ob_factory, ar_factory)
        self._frame = frame

    def __call__(self, diagram):
        if isinstance(diagram, Frame):
            return self._frame(Frame(diagram.name, self(diagram.dom), self(diagram.cod),
                               [self(i) for i in diagram.insides]))
        return super().__call__(diagram)


def logical_form(diagram):
    """
    Returns the string representing frame and its insides. 
    """
    if isinstance(diagram, Frame):
        insides = [logical_form(d) for d in diagram._insides]
        return f'{diagram.name}({", ".join(insides)})'
    if isinstance(diagram, monoidal.Box):
        return diagram.name
    if isinstance(diagram, monoidal.Diagram) and len(diagram) == 1:
        return logical_form(diagram.boxes[0])
    return logical_form(diagram.boxes[-1]) + '(' + logical_form(
        diagram[:-1]) + ')'
