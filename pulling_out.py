from discopy import Diagram, Ty, Box


def pull_out_of_frame(diagram, opening_box_index):
    # ---------- 1. find closing box -----------
    opening_box = diagram.boxes[opening_box_index]
    closing_box_index = -1
    closing_name = "[\\" + diagram.boxes[opening_box_index].name[1:]
    for i, box in enumerate(diagram.boxes[opening_box_index + 1:], opening_box_index + 1):
        if box.name == closing_name:
            closing_box_index = i
            closing_box = box
            break

    # closing_box_index = [index for (index, box) in enumerate(diagram.boxes[opening_box_index + 1:]) if box.name == closing_name]

    # --------- 2. create sub_diagram and recurse ---------
    # TODO: change dom and cod of sub_diagram to incorporate wires to the left and right of boxes
    sub_diagram = Diagram(opening_box.cod,
                          closing_box.dom,
                          diagram.boxes[opening_box_index + 1:closing_box_index],
                          diagram.offsets[opening_box_index + 1:closing_box_index])

    sub_diagram = pulling_out_diagram(sub_diagram)
    # TODO: check is this assignment necessary? (just out of interest)

    # -------- 3. merge diagrams ---------
    merged_boxes = diagram.boxes[:opening_box_index + 1] \
                   + sub_diagram.boxes \
                   + diagram.boxes[closing_box_index:]

    merged_offsets = diagram.offsets[:opening_box_index + 1] \
                        + sub_diagram.offsets \
                        + diagram.offsets[closing_box_index:]

    merged = Diagram(diagram.dom, diagram.cod, merged_boxes, merged_offsets)

    # -------- 4. pull out for current frame ---------

    # 4.1 find nouns in frame
    for i, box in enumerate(diagram.boxes[opening_box_index + 1:closing_box_index], opening_box_index + 1):
        if box.dom == Ty():
            # swap till at the top of frame
            for swap_index in range(i, opening_box_index, -1):
                # TODO: implement swapping of boxes
                print(i)

            # pull outside of top of frame

            # TODO: we are currently hardcoding a -1 here as we assume that the box has more wires inside than outside
            dom = opening_box.dom[:diagram.offsets[i] - 1] @ Ty('n') @ opening_box.dom[diagram.offsets[i] - 1:]
            cod = opening_box.cod[:diagram.offsets[i]] @ Ty('n') @ opening_box.cod[diagram.offsets[i]:]

            print(dom)

            print("i", i, " opening_index", opening_box_index)
            print(diagram.offsets[i])
            new_top = Box(opening_box.name, dom, cod)

            print(diagram.boxes[opening_box_index])
            new_boxes = diagram.boxes[:opening_box_index] + [box, new_top] + diagram.boxes[opening_box_index + 2:]
            new_offsets = diagram.offsets[:opening_box_index] + [1, 0] + diagram.offsets[opening_box_index + 2:]

            diagram = Diagram(diagram.dom, diagram.cod, new_boxes, new_offsets)
        # 4.2 for each state: pull it out

    diagram.draw()

    return merged


def pulling_out_diagram(diagram):
    # find all frames
    # TODO: check: do we have to restart once we find a frame?
    for i, box in enumerate(diagram.boxes):
        # identify start of a frame
        if "[" in box.name and not '\\' in box.name:
            diagram = pull_out_of_frame(diagram, i)

    return diagram