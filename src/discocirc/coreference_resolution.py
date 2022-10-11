#%%
import sys
sys.path = sys.path[2:]
sys.path.append('/home/rshaikh/Neural-DisCoCirc/src/')

from discopy import rigid
from lambeq import BobcatParser
import neuralcoref
import spacy
import string

from discocirc.sentence_to_circuit import sentence2circ, make_term, make_diagram
from discocirc.discocirc_utils import init_nouns
from discocirc.drag_up import drag_all
from discocirc.text_to_circuit import compose_circuits, noun_sort, noun_normal_form, collect_normal_nouns, sentence_list_to_circuit
from discocirc.expand_s_types import expand_s_types
from discocirc.frame import Frame
from discocirc.pulling_out import recurse_pull

# Loadone of SpaCy English models
nlp = spacy.load('en_core_web_md')
# Add neural coref to SpaCy's pipe
neuralcoref.add_to_pipe(nlp)

parser = BobcatParser()


def get_sentence_id(span_or_token, sentences):
    return sentences.index(span_or_token.sent)


def text_to_circ(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    circs = [sentence2circ(parser, sentence) for sentence in sentences]
    corefs, corefs_sent_ids = get_corefs(doc)
    return compose_circuits_using_corefs(circs, corefs, corefs_sent_ids)


def get_corefs(doc):
    doc_sents = list(doc.sents)
    corefs = []
    corefs_sent_ids = []
    for cluster in doc._.coref_clusters:
        corefs.append([mention.text for mention in cluster.mentions])
        corefs_sent_ids.append([
            get_sentence_id(mention, doc_sents) 
            for mention in cluster.mentions
        ])
        
    return corefs, corefs_sent_ids


def compose_circuits_using_corefs(circs, corefs, corefs_sent_ids):
    merged_circuit = circs[0] 
    for i in range(1, len(circs)):
        # print(i)
        corefs_to_merge = []
        for coref, sent_ids in zip(corefs, corefs_sent_ids):
            if i in sent_ids and sent_ids.index(i) != 0:
                corefs_to_merge.append((coref[0], coref[sent_ids.index(i)]))
        # print(corefs_to_merge)
        merged_circuit = merge_circuits(merged_circuit, circs[i], corefs_to_merge)
    return merged_circuit


def merge_circuits(circ1, circ2, corefs_to_merge):
    """
    Merge two circuits, and coreferent mentions.
    """
    # pull the nouns to the top and sort them based on offsets
    circ1 = noun_sort(drag_all(circ1))
    circ2 = noun_sort(drag_all(circ2))
    # get noun boxes
    nouns_circ1 = circ1.boxes[:init_nouns(circ1) + 1]
    nouns_circ2 = circ2.boxes[:init_nouns(circ2) + 1]
    # resolve coreferences between nouns
    for noun1, noun2 in corefs_to_merge:
        noun1_box = find_box_using_name(nouns_circ1, noun1)
        noun2_box = find_box_using_name(nouns_circ2, noun2)
        #replace noun2_box with noun1_box in nouns_circ2
        nouns_circ2[nouns_circ2.index(noun2_box)] = noun1_box
    nouns_diagram = rigid.Id()
    for noun in nouns_circ2:
        nouns_diagram = nouns_diagram @ noun
    coref_resolved_circ2 = nouns_diagram >> circ2[len(nouns_circ2):]
    return compose_circuits(circ1, coref_resolved_circ2)

def find_box_using_name(boxes, name):
    for box in boxes:
        # remove punctuation from box name
        box_name = box.name.translate(str.maketrans('', '', string.punctuation))
        if box_name == name:
            return box
    print('Box not found')
    print(name)
    print(boxes)
    return None

#%%

# text = 'The smart king likes pies. He eats strawberries.'
# text = 'Alice went to the kitchen. She cooked pies. She liked them.'
text = 'Alice likes Bob. She is happy'
composed_circ = text_to_circ(text)
composed_circ.draw()

# %%
