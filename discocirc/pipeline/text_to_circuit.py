import numpy as np
import spacy
from discopy import symmetric
from discopy.monoidal import Functor, Ty
from discopy.symmetric import Hypergraph, Box
from lambeq import BobcatParser

from discocirc.helpers.discocirc_utils import get_last_initial_noun
from discocirc.diag.drag_up import drag_all
from discocirc.pipeline.sentence_to_circuit import sentence2circ

parser = BobcatParser(verbose='suppress')
# Load a SpaCy English model
spacy_model = spacy.load('en_core_web_trf')
spacy_model.add_pipe('coreferee')

# convert to symmetric monoidal category
sym_func = Functor(
    cod=symmetric.Category,
    ob=lambda ob: ob,
    ar=lambda box: symmetric.Box(box.name, box.dom, box.cod))


# NOTE: this function may become redundant
def noun_sort(circ):
    """
    takes in a (discopy) circuit with some number of nouns as the initial boxes

    rearranges the order of these so that their offsets are 0, 1, 2, ...
    """

    # first box must be a noun
    assert circ.boxes[0].dom == Ty()

    # find how many initial nouns there are
    index = get_last_initial_noun(circ)

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


def sentence_list_to_circuit(context, simplify_swaps=True, wire_order='intro_order', spacy_model=spacy_model):
    """
    Parameters:
    -----------
    context : list
        List of context sentences.
    simplify_swaps : bool
        Whether to simplify the circuit by removing unnecessary swaps.
    wire_order : str
        The order for the noun wires in the output circuit.
        'update_order' : the most recently updated wires occur on the right
        'intro_order' : the most recently introduced wires occur on the right

    Returns:
    --------
    context_circ : discopy.rigid.Diagram
    """
    sentence_circuits = []
    for sentence in context:
        # sentence_diag = sentence2circ(parser, sentence)
        sentence_diag = sentence2circ(parser, sentence, spacy_model=spacy_model)
        sentence_circuits.append(sentence_diag)
    context_circ = sentence_circuits[0]
    for circ in sentence_circuits[1:]:
        context_circ = compose_circuits(context_circ, circ, wire_order)

    # attempt to remove some redundant swaps
    if simplify_swaps:
        context_circ = noun_normal_form(context_circ)
        num_nouns = get_last_initial_noun(context_circ) + 1
        nouns = context_circ[:num_nouns]
        back_n_forth = lambda f: Hypergraph.from_diagram(f).to_diagram()
        context_circ = back_n_forth(context_circ[num_nouns:])
        context_circ = nouns >> context_circ

    return context_circ

def text_to_circuit(text, **kwargs):
    """
    input a given text as a string

    return the corresponding (discopy) circuit for that text
    """
    doc = spacy_model(text)
    sentences = []
    for sent in doc.sents:
        s = sent.text
        sentences.append(s)
    return sentence_list_to_circuit(sentences, spacy_model=spacy_model, **kwargs)

def noun_normal_form(circuit):
    """
    Takes in a circuit, and returns it in a normal form, where all the
    nouns are dragged to the top, and ordered such that their offsets
    are 0, 1, 2, ...
    """
    # NOTE: drag_all is a little bit broken
    return noun_sort(drag_all(circuit))

def collect_normal_nouns(circuit):
    """
    Takes in a circuit in noun normal form,
    and returns a list of the pulled up nouns
    """
    return circuit.boxes[:get_last_initial_noun(circuit) + 1]

def compose_circuits(circ1, circ2, wire_order='intro_order'):
    """
    Return the sequential composite roughly corresponding
    to circ2 << circ 1, where common noun
    wires are composed

    Parameters:
    -----------
    circ1 : discopy.rigid.Diagram
        The first circuit to compose.
    circ2 : discopy.rigid.Diagram
        The second circuit to compose.
    wire_order : str
        The order in which the wires occur in the final circuit.
        'update_order' : the most recently updated wires occur on the right
        'intro_order' : the most recently introduced wires occur on the right
    """

    # pull nouns to top and order them with offsets 0, 1, 2, ...

    circ1 = sym_func(circ1)
    circ2 = sym_func(circ2)

    circ1 = noun_normal_form(circ1)
    circ2 = noun_normal_form(circ2)

    # record pulled up nouns
    nouns_circ1 = collect_normal_nouns(circ1)
    nouns_circ2 = collect_normal_nouns(circ2)
    nouns_circ1_name = [x.name for x in nouns_circ1]
    nouns_circ2_name = [x.name for x in nouns_circ2]

    ob_map = {}
    for i in range(len(nouns_circ2)):
        if nouns_circ2_name[i] in nouns_circ1_name:
            ob_map[nouns_circ2[i].cod] = nouns_circ1[nouns_circ1_name.index(nouns_circ2_name[i])].cod

    # the two functions below are used to define a functor
    def ob_map2(obj):
        if obj in ob_map.keys():
            return ob_map[obj]
        return obj

    def ar_map(box):
        return Box(box.name, functor(box.dom), functor(box.cod))

    functor = Functor(ob_map2, ar_map, cod=symmetric.Category)
    circ2 = functor(circ2)
    nouns_circ2 = collect_normal_nouns(circ2)
    # collect nouns in circ2 not in circ1
    nouns_circ2[0] == nouns_circ1[1]
    new_nouns = [x for x in nouns_circ2 if x not in nouns_circ1]
    # collect nouns in circ1 not in circ2
    unused_nouns = [x for x in nouns_circ1 if x not in nouns_circ2]

    # construct new circ1, circ2 by tensoring required nouns
    for noun in new_nouns:
        circ1 = circ1 @ noun
    circ1 = noun_normal_form(circ1)

    # reverse order because we are adding wires to the left
    for noun in reversed(unused_nouns):
        circ2 = noun @ circ2

    circ2 = noun_normal_form(circ2)

    # record new pulled up nouns
    nouns_circ1 = collect_normal_nouns(circ1)
    nouns_circ2 = collect_normal_nouns(circ2)

    assert len(nouns_circ1) == len(nouns_circ2)

    # generate the required permutation (as a list)
    perm = [nouns_circ2.index(x) for x in nouns_circ1]
    # generate the inverse permutation (as a list)
    inv_perm = np.argsort(perm).astype(int).tolist()

    # NOTE: inv_perm and perm behaviour in the permute() function
    if wire_order == 'intro_order':
        # adopt the noun ordering of circ1
        final_circ = circ1.permute(*inv_perm) >> circ2[len(nouns_circ2):].permute(*perm)
    elif wire_order == 'update_order':
        # adopt the noun ordering of circ2
        final_circ = circ2[:len(nouns_circ2)]

        final_circ = final_circ.permute(*inv_perm)\
            >> circ1[len(nouns_circ1):].permute(*perm)\
            >> circ2[len(nouns_circ2):]
    else:
        raise Exception("Invalid wire_order.")

    return final_circ
