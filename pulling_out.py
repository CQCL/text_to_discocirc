from discopy import Diagram, Ty, Box, rigid

from frame import Frame
from drag_up import drag_all, drag_out
from discocirc_utils import init_nouns


def pull_out_of_frame_old(diagram, opening_box_index):
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

    sub_diagram = pulling_out_diagram_old(sub_diagram)
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


def pulling_out_diagram_old(diagram):
    # find all frames
    # TODO: check: do we have to restart once we find a frame?
    for i, box in enumerate(diagram.boxes):
        # identify start of a frame
        if "[" in box.name and not '\\' in box.name:
            diagram = pull_out_of_frame_old(diagram, i)

    return diagram

def pulling_out_diagram(diagram):
    new_diag = rigid.Id(diagram.dom)
    for left, box, right in diagram.layers:
        if not isinstance(box, Frame):
            new_diag = new_diag >> rigid.Id(left) @ box @ rigid.Id(right)
            continue
        insides_with_nouns_removed = []
        inside_nouns = []
        num_pulled_out = 0
        for inside in box._insides:
            x, y, num_nouns = pull_out_of_frame(inside)
            num_pulled_out += num_nouns
            insides_with_nouns_removed.append(x)
            inside_nouns.append(y)
        if num_pulled_out > 0:
            pulled_out_nouns = rigid.Id()
            for nouns in inside_nouns:
                pulled_out_nouns = pulled_out_nouns @ nouns
            new_diag = new_diag >> rigid.Id(left) @ pulled_out_nouns @ rigid.Id(right)
        frame_domain = box.dom @ Ty('n') ** num_pulled_out
        new_frame = Frame(box.name, frame_domain, box.cod, insides_with_nouns_removed, box._slots)
        new_diag = new_diag >> rigid.Id(left) @ new_frame @ rigid.Id(right)
    return new_diag

def pull_out_of_frame(diagram):
    diagram = pulling_out_diagram(diagram)
    diagram = drag_all(diagram)
    num_nouns = init_nouns(diagram)+1
    if num_nouns == 0:
        return diagram, diagram[:0], 0
    inside_nouns = diagram[:num_nouns]
    nouns_offset_begin = inside_nouns.offsets[0]
    nouns_offset_end = inside_nouns.offsets[-1]
    pull_out_index = num_nouns
    i = 0
    for box, offset in zip(diagram[num_nouns:].boxes, diagram[num_nouns:].offsets):
        if len(box.dom) <= num_nouns \
            and offset >= nouns_offset_begin \
            and offset + len(box.dom) - 1 <= nouns_offset_end:
            diagram = drag_out(diagram, num_nouns+i, pull_out_index)
            pull_out_index += 1
        i += 1

    new_insides = diagram[pull_out_index:]
    pulled_out_stuff = diagram[:pull_out_index]
    return new_insides, pulled_out_stuff, num_nouns
