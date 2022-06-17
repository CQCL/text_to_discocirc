from discopy import Ty, rigid

from discocirc.frame import Frame
from discocirc.drag_up import drag_all, drag_out
from discocirc.discocirc_utils import init_nouns


def pulling_out_diagram(diagram):
    """
    Return diagram where all states inside of frames have been pulled out.

    :param diagram: Diagram -- Original diagram.
    :return: Diagram -- New diagram with states pulled out of frames.
    """
    new_diag = rigid.Id(diagram.dom)

    # Iterate over all layers and rebuild the diagram by pulling all insides out of frame
    for left, box, right in diagram.layers:
        if not isinstance(box, Frame):
            # Current layer not a frame
            new_diag = new_diag >> rigid.Id(left) @ box @ rigid.Id(right)
            continue

        list_insides_without_nouns = []
        list_former_inside_states = []
        num_pulled_out = 0

        # Pull out each individual inside
        for inside in box._insides:
            new_insides, former_inside_states = pull_inside_out_of_frame(inside)
            num_pulled_out += len(former_inside_states)
            list_insides_without_nouns.append(new_insides)
            list_former_inside_states.append(former_inside_states)

        # Rebuild pulled out layer
        if num_pulled_out > 0:
            new_pulled_out_nouns = rigid.Id()
            for states_diag in list_former_inside_states:
                new_pulled_out_nouns = new_pulled_out_nouns @ states_diag
            new_diag = new_diag >> rigid.Id(left) @ new_pulled_out_nouns @ rigid.Id(right)

        frame_domain = box.dom @ Ty('n') ** num_pulled_out
        new_frame = Frame(box.name, frame_domain, box.cod, list_insides_without_nouns, box._slots)
        new_diag = new_diag >> rigid.Id(left) @ new_frame @ rigid.Id(right)

    return new_diag


def pull_inside_out_of_frame(diagram):
    """
    Return the updated insides of a frame after pulling out the states and boxes only acting on pulled out states.

    :param diagram: (Single) insides of a frame to pull states out of.
    :return: Diagram -- new insides,
            Diag -- States and boxes acting on states which were pulled out.
    """
    # Iteratively pull out diagram
    diagram = pulling_out_diagram(diagram)
    diagram = drag_all(diagram)

    num_nouns = init_nouns(diagram) + 1
    if num_nouns == 0:
        # Nothing to pull out
        return diagram, diagram[:0]

    inside_nouns = diagram[:num_nouns]
    nouns_offset_begin = inside_nouns.offsets[0]
    nouns_offset_end = inside_nouns.offsets[-1]

    # Find all processes that only act on states that are being pulled out. These will be pulled out too
    pull_out_index = num_nouns
    addition_pull_outs = 0
    for box, offset in zip(diagram[num_nouns:].boxes, diagram[num_nouns:].offsets):
        if len(box.dom) <= num_nouns \
                and offset >= nouns_offset_begin \
                and offset + len(box.dom) - 1 <= nouns_offset_end:
            diagram = drag_out(diagram, num_nouns + addition_pull_outs, pull_out_index)
            pull_out_index += 1
        addition_pull_outs += 1

    # New insides are all remaining non-state boxes
    new_insides = diagram[pull_out_index:]

    # Build diagram of pulled out boxes without any identities surrounding the states
    pulled_out_nouns = rigid.Id()
    current_cod = Ty()
    for state in diagram.boxes[:pull_out_index]:
        pulled_out_nouns = pulled_out_nouns >> (rigid.Id(current_cod) @ state)
        current_cod = current_cod @ state.cod

    return new_insides, pulled_out_nouns
