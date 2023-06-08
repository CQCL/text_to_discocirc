# -*- coding: utf-8 -*-

"""
Implements the free closed monoidal category.
"""

from discopy import monoidal, biclosed, messages
from lambeq import BobcatParser


class Ty(monoidal.Ty):
    """
    Objects in a free closed monoidal category.
    Generated by the following grammar:

        ty ::= Ty(name) | ty @ ty | ty >> ty

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> print(x >> y >> x)
    ((x → y) → x)
    >>> print((y >> x >> y) @ x)
    ((y → x) → y) @ x
    """

    def __init__(self, *objects, input=None, output=None, index=None):
        super().__init__()
        self.input, self.output, self.index = input, output, index
        if len(objects) > 1:
            self._objects = tuple(x if isinstance(x, Ty) else Ty(x) for x in objects)
        elif len(objects) == 1:
            if isinstance(objects[0], Ty):
                if self.index == None:
                    self.index = objects[0].index
                self._objects = objects[0].downgrade()
            elif isinstance(objects[0], monoidal.Ty):
                self._objects = objects[0]
            else:
                self._objects = monoidal.Ty(objects[0])

    def __rshift__(self, other):
        return Func(self, other)

    def __str__(self):
        if len(self._objects) > 1:
            return f'({super().__str__()}){self.index}'
        return super().__str__() + f'{self.index}'

    def tensor(self, *others):
        for other in others:
            if not isinstance(other, monoidal.Ty):
                raise TypeError(messages.type_err(monoidal.Ty, other))
        objects = []
        for t in (self,) + others:
            if len(t.objects) > 1:
                objects += t.objects
            elif len(t.objects) == 1:
                objects.append(t)
        return Ty(*objects)

    @staticmethod
    def upgrade(old):
        if len(old) == 1 and isinstance(old[0], Func):
            return old[0]
        return Ty(*old.objects)

    def downgrade(self):
        if isinstance(self, Func):
            return self
        return super().downgrade()


class Func(Ty):
    """ Function types. """
    def __init__(self, input=None, output=None, index=None):
        name = f'({repr(input)} → {repr(output)})'
        super().__init__(name, input=input, output=output, index=index)

    def __repr__(self):
        return "({} → {})".format(repr(self.input), repr(self.output))

    def __str__(self):
        return "({} → {}){}".format(self.input, self.output, self.index)

    def __eq__(self, other):
        if not isinstance(other, Func):
            return False
        return self.input == other.input and self.output == other.output

    def __hash__(self):
        return hash(repr(self))


def biclosed_to_closed(x):
    """Converts the biclosed types to closed types."""
    if isinstance(x, biclosed.Under):
        return Func(biclosed_to_closed(x.left), biclosed_to_closed(x.right))
    elif isinstance(x, biclosed.Over):
        return Func(biclosed_to_closed(x.right), biclosed_to_closed(x.left))
    elif isinstance(x, biclosed.Ty):
        return Ty(*[biclosed_to_closed(y) for y in x.objects])
    else:
        return x

def ccg_cat_to_closed(cat, word_str=None):
    if cat.atomic:
        typ = biclosed_to_closed(BobcatParser._to_biclosed(cat))
    else:
        result_typ = ccg_cat_to_closed(cat.result, word_str)
        argument_typ = ccg_cat_to_closed(cat.argument, word_str)
        typ = argument_typ >> result_typ
    idx = str(word_str) + '_' + str(cat.var) if word_str else str(cat.var)
    typ.index = set([idx])
    return typ

def uncurry_types(typ, uncurry_everything=False):
    if isinstance(typ, Func) and isinstance(typ.output, Func):
        if uncurry_everything:
            inp = uncurry_types(typ.input, uncurry_everything=True)
            out_inp = uncurry_types(typ.output.input, uncurry_everything=True)
            out_out = uncurry_types(typ.output.output, uncurry_everything=True)
        else:
            inp = typ.input
            out_inp = typ.output.input
            out_out = typ.output.output
        return uncurry_types((out_inp @ inp) >> out_out)

    else:
        return typ
