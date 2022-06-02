###############################################################################
# Useful functions for constructing task data
###############################################################################

# import pickle   # for saving data into file
from lambeq import BobcatParser

from discopy.rigid import Ty, Diagram, Box, Swap, Id
from drag_up import drag_all # drags nouns to top of circuit diagram
import numpy as np # for inverse permutation
from discocirc_utils import init_nouns

parser = BobcatParser(verbose='suppress')
# parser = BobcatParser(model_name_or_path='C:/Users/jonat/bert/')


#%%

# read the .txt file
def task_file_reader(path):
    """
    reads the .txt file at path
    returns 3 lists of equal length
    - context sentences, questions, and answers
    """
    with open(path) as f:
        lines = f.readlines()


    # split the lines into stories
    # record the first line location of new stories
    story_splits = [i for i, line in enumerate(lines) if line[0:2] == '1 ']
    # have no more need for line indices - delete these
    lines = [' '.join(line.split(' ')[1:]) for line in lines]
    # also delete . and \n
    lines = [line.replace('.', '').replace('\n','') for line in lines]
    stories = [lines[i:j] for i, j in zip(story_splits, story_splits[1:]+[None])]

    # create context and QnA pairs
    contexts = []
    qnas = []
    for story in stories:
        # record the lines in the story corresponding to questions
        question_splits = [i for i, line in enumerate(story) if '?' in line]
        for index in question_splits:
            # record the context corresponding to each question
            contexts.append([line for line in story[:index] if '?' not in line])
            # record the question
            qnas.append(story[index])


    # split qna into questions and answers
    questions = [qna.split(' \t')[0] for qna in qnas]
    answers = [qna.split('\t')[1] for qna in qnas]
    return contexts, questions, answers


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
    while swapped == True:  # keep going as long as swaps are happening
        swapped = False
        for i in range(index, 0, -1):
            if circ.offsets[i] <= circ.offsets[i - 1]:
                # perform the swap, and remember something swapped
                circ = circ.interchange(i, i - 1)
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
    no_nouns1 = init_nouns(circ1) + 1
    no_nouns2 = init_nouns(circ2) + 1

    # TODO: assume for now that the no. of output wires
    # = the number of initial nouns, and that no swapping occurs
    if (circ1.cod != Ty(*['n'] * no_nouns1) or
            circ2.cod != Ty(*['n'] * no_nouns2)):
        print(repr(circ1))
        print(repr(circ2))
        raise Exception("The types do not align.")

    # record pulled up nouns
    nouns_circ1 = circ1.boxes[:no_nouns1]
    nouns_circ2 = circ2.boxes[:no_nouns2]

    # nouns in circ2 not in circ1
    new_nouns = [x for x in nouns_circ2 if x not in nouns_circ1]
    # nouns in circ1 not in circ2
    unused_nouns = [x for x in nouns_circ1 if x not in nouns_circ2]

    # construct new circ1, circ2 by tensoring required nouns
    for noun in new_nouns:
        circ1 = circ1 @ noun
        # circ1.draw()
    # pull up and order again
    # print(repr(circ1))
    # TODO: drag_all is a little bit broken
    circ1 = noun_sort((drag_all(circ1)))
    # circ1.draw()

    for noun in reversed(unused_nouns):
        circ2 = noun @ circ2

    # TODO: drag_all
    circ2 = noun_sort((drag_all(circ2)))
    # circ2.draw()

    # record new pulled up nouns
    nouns_circ1 = circ1.boxes[:init_nouns(circ1) + 1]
    nouns_circ2 = circ2.boxes[:init_nouns(circ2) + 1]

    assert len(nouns_circ1) == len(nouns_circ2)

    # generate the required permutation (as a list)
    perm = [nouns_circ2.index(x) for x in nouns_circ1]
    # generate the inverse permutation (as a list)
    inv_perm = list(np.argsort(perm))

    # TODO: switch inv_perm and perm once the permute() function has been fixed
    circ = circ1.permute(*inv_perm) >> circ2[len(nouns_circ2):].permute(*perm)
    return circ
