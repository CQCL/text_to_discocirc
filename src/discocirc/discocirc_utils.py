from discopy.rigid import Ty

def init_nouns(circ):
    """
    takes in a circuit with some number of nouns as the initial boxes
    returns the index of the last of these initial nouns
    """

    index = -1
    for i in range(len(circ.boxes)-1):
        if circ.boxes[i].dom ==Ty() and circ.boxes[i+1].dom != Ty():
            index = i # index of the last n oun
            break

    return index
