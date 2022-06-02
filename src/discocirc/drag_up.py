from discopy import rigid
from discopy.rewriting import InterchangerError

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


def drag_out(diagram, i, stop):
    box = diagram.boxes[i]
    while i > stop:
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
            diagram = drag_out(diagram, i, stop)
            i = len(diagram)
            stop += 1
        i -= 1
    return diagram
