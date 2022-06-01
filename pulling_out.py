from discopy import Diagram, Ty, Box

from discocirc import Frame, drag_all
from generate_context_circuit import init_nouns


def pull_out_of_frame(diagram, opening_box_index):
    # ---------- 1. find closing box -----------
    opening_box = diagram.boxes[opening_box_index]
    closing_box_index = -1
    closing_name = "[\\" + diagram.boxes[opening_box_index].name[1:]
    # TODO: current assumption that names are unique
    for i, box in enumerate(diagram.boxes[opening_box_index + 1:], opening_box_index + 1):
        if box.name == closing_name:
            closing_box_index = i
            closing_box = box
            break

    # closing_box_index = [index for (index, box) in enumerate(diagram.boxes[opening_box_index + 1:]) if box.name == closing_name]
    # --------- 2. create sub_diagram and recurse ---------
    sub_diagram = Diagram(diagram.layers[opening_box_index].cod,
                          diagram.layers[closing_box_index].dom,
                          diagram.boxes[opening_box_index + 1:closing_box_index],
                          diagram.offsets[opening_box_index + 1:closing_box_index])

    sub_diagram = pulling_out_diagram(sub_diagram)
    # TODO: check is this assignment necessary? (just out of interest)
    # sub_diagram.draw()

    # -------- 3. merge diagrams ---------
    merged_boxes = diagram.boxes[:opening_box_index + 1] \
                   + sub_diagram.boxes \
                   + diagram.boxes[closing_box_index:]

    merged_offsets = diagram.offsets[:opening_box_index + 1] \
                        + sub_diagram.offsets \
                        + diagram.offsets[closing_box_index:]

    diagram = Diagram(diagram.dom, diagram.cod, merged_boxes, merged_offsets)



    # -------- 4. pull out for current frame ---------

    # 4.1 find nouns in frame
    for i, box in enumerate(diagram.boxes[opening_box_index + 1:closing_box_index], opening_box_index + 1):
        if box.dom == Ty():
            # swap till at the top of frame
            for swap_index in range(i, opening_box_index + 1, -1):
                diagram = diagram.interchange(swap_index, swap_index - 1)
                # diagram.draw()

            noun_index = opening_box_index + 1
            # diagram.draw()

            # pull outside of top of frame
            # TODO: we are currently hardcoding a -1 here as we assume that the box has more wires inside than outside
            dom = opening_box.dom[:diagram.offsets[noun_index] - diagram.offsets[opening_box_index] - 1] @ box.cod @ opening_box.dom[diagram.offsets[noun_index] - diagram.offsets[opening_box_index] - 1:]
            cod = opening_box.cod[:diagram.offsets[noun_index] - diagram.offsets[opening_box_index]] @ box.cod @ opening_box.cod[diagram.offsets[noun_index] - diagram.offsets[opening_box_index]:]

            new_top = Box(opening_box.name, dom, cod)

            new_boxes = diagram.boxes[:opening_box_index] + [box, new_top] + diagram.boxes[opening_box_index + 2:]
            new_offsets = diagram.offsets[:opening_box_index] + [diagram.offsets[noun_index] - 1, diagram.offsets[opening_box_index]] + diagram.offsets[opening_box_index + 2:]

            diagram = Diagram(diagram.dom, diagram.cod, new_boxes, new_offsets)

            opening_box_index += 1
            opening_box = new_top

            # diagram.draw()
        # 4.2 for each state: pull it out

    # diagram.draw()

    return diagram


def pulling_out_diagram(diagram):
    # find all frames
    # TODO: check: do we have to restart once we find a frame?
    for i, box in enumerate(diagram.boxes):
        # identify start of a frame
        if "[" in box.name and not '\\' in box.name:
            diagram = pull_out_of_frame(diagram, i)

    return diagram

def pulling_out_diagram_new(diagram):
    # find all frames
    # TODO: check: do we have to restart once we find a frame?
    for i, box in enumerate(diagram.boxes):
        # identify start of a frame
        if not isinstance(box, Frame):
            return   
        
        insides_with_nouns_removed = []
        inside_nouns = []
        for inside in box._insides:
            x, y = pull_out_of_frame_new(inside)
            insides_with_nouns_removed.append(x)
            inside_nouns.append(y)

    return diagram

def pull_out_of_frame_new(diagram):
    diagram = drag_all(diagram)

    no_nouns = init_nouns(diagram)+1
    insides_with_nouns_removed = diagram[no_nouns:]
    inside_nouns = diagram[:no_nouns]
    return insides_with_nouns_removed, inside_nouns