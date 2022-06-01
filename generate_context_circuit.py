############################################################
# generate context circuits from context sentences
############################################################
#%%

from discocirc import convert_sentence
from lambeq import BobcatParser
from utils import get_star_removal_functor

from discopy.rigid import Ty, Diagram, Box
from discocirc import drag_all # drags nouns to top of circuit diagram
import numpy as np # for inverse permutation


# parser = BobcatParser(verbose='suppress')
parser = BobcatParser(model_name_or_path='C:/Users/jonat/bert/')


#%%

test_context = ['Mary moved to the bathroom',
 'Sandra journeyed to the bedroom',
 'Mary got the football there',
 'John went to the kitchen',
 'Mary went back to the kitchen',
 'Mary went back to the garden',
 'Sandra went back to the office',
 'John moved to the office',
 'Sandra journeyed to the hallway',
 'Daniel went back to the kitchen',
 'Mary dropped the football',
 'John got the milk there']



# %%

def init_nouns(circ):
    """
    takes in a circuit with some number of nouns as the initial boxes
    returns the index of the last of these initial nouns
    """
    # check there actually are initial nouns
    assert circ.boxes[0].dom == Ty()

    index = -1
    for i in range(len(circ.boxes)-1):
        if circ.boxes[i].dom ==Ty() and circ.boxes[i+1].dom != Ty():
            index = i # index of the last n oun
            break

    return index

# %%

# NOTE: this function may become redundant
def noun_sort(circ):
    """
    takes in a circuit with some number of nouns as the initial boxes
    rearranges the order of these so that their offsets are 0, 1, 2, ...
    """

    # first box must be a noun
    assert circ.boxes[0].dom == Ty() 

    # find how many initial nouns there are
    index = init_nouns(circ)
    

    # perform bubblesort on nouns
    swapped = True
    while swapped == True: # keep going as long as swaps are happening
        swapped = False
        for i in range(index,0,-1):
            if circ.offsets[i] <= circ.offsets[i-1]:
                # perform the swap, and remember something swapped
                circ = circ.interchange(i, i-1)
                swapped = True
    
    return circ



#%%

def compose_circuits(circ1, circ2):
    """
    Return the sequential composite roughly corresponding 
    to circ2 << circ 1, where common noun 
    wires are composed     
    """

    # pull the nouns to the top
    circ1 = drag_all(circ1)
    circ2 = drag_all(circ2)
    # ensure the top nouns are ordered with offsets 0, 1, 2, ...
    circ1 = noun_sort(circ1)
    circ2 = noun_sort(circ2)

    # identify number of initial nouns in circ1, circ2
    no_nouns1 = init_nouns(circ1)+1
    no_nouns2 = init_nouns(circ2)+1


    # TODO: assume for now that the no. of output wires
    # = the number of initial nouns, and that no swapping occurs
    if (circ1.cod != Ty(*['n']*no_nouns1) or 
        circ2.cod != Ty(*['n']*no_nouns2)):
        print(repr(circ1))
        print(repr(circ2))
        raise Exception("The types do not lign up.")

    # record pulled up nouns
    nouns_circ1 = circ1.boxes[:no_nouns1]
    nouns_circ2 = circ2.boxes[:no_nouns2]

    # nouns in circ2 not in circ1
    new_nouns = [x for x in nouns_circ2 if x not in nouns_circ1]
    # nouns in circ1 not in circ2
    unused_nouns = [x for x in nouns_circ1 if x not in nouns_circ2]


    # TESTING
    # print("circ1 nouns are ", nouns_circ1)
    # print("circ2 nouns are ", nouns_circ2)
    # print("new nouns are ", new_nouns)
    # print("unused nouns are ", unused_nouns)


    # construct new circ1, circ2 by tensoring required nouns
    for noun in new_nouns:
        circ1 = circ1 @ noun
        # circ1.draw()
    # pull up and order again
    # print(repr(circ1))
    # TODO: drag_all is a little bit broken
    circ1 = noun_sort((drag_all(circ1)))
    print(repr(circ1))
    circ1.draw()

    for noun in unused_nouns:
        circ2 = circ2 @ noun
        # circ2.draw()
    # print(repr(circ2))
    # TODO: drag_all
    circ2 = noun_sort((drag_all(circ2)))
    print(repr(circ2))
    circ2.draw()


    # record new pulled up nouns
    nouns_circ1 = circ1.boxes[:init_nouns(circ1)+1]
    nouns_circ2 = circ2.boxes[:init_nouns(circ2)+1]

    assert len(nouns_circ1) == len(nouns_circ2)

    print("nouns circ 1: ", nouns_circ1)
    print("nouns circ 2: ", nouns_circ2)

    # generate the required permutation (as a list)
    perm = [nouns_circ2.index(x) for x in nouns_circ1]
    # generate the inverse permutation (as a list)
    inv_perm = list(np.argsort(perm))

    print("perm is ",perm)
    print("inv_perm is ",inv_perm)

    # TODO: consider case where nouns perfect overlap, and
    # perm and inv_perm are empty ...

    

    # circ1.permute(*perm).draw()
    # circ2[len(nouns_circ2):].draw()
    # circ2[len(nouns_circ2):].permute(*inv_perm).draw()

    # construct the composite circuit
    circ = circ2[len(nouns_circ2):].permute(*inv_perm) << circ1.permute(*perm)
    return circ


#%%

# generate the circuits for the individual sentences
sentence_circuits = []

for line in test_context:
    line_diag = parser.sentence2tree(line).to_biclosed_diagram()
    line_diag = convert_sentence(line_diag)
    sentence_circuits.append(line_diag)

context_circ = sentence_circuits[0]

for circ in sentence_circuits[1:]:
    context_circ = compose_circuits(context_circ, circ)

context_circ.draw()

# %%
