from discopy import rigid
from lambeq import CCGBankParser, BobcatParser

from diag import Frame
from explore_coref import ccg_find_word
from helpers.discocirc_utils import get_last_initial_noun

import coreferee, spacy

from pipeline.sentence_to_circuit import sentence2circ

nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')

def find_coref(ccg):
    doc = nlp(ccg.text)
    found = []

    for chain in doc._.coref_chains:
        for element in chain:
            for token_index in element.token_indexes:
                word = str(doc[token_index])
                found.append(ccg_find_word(ccg, word, token_index))
    return found

def all_corefs(sentences):
    coref_dict = {}
    for i, sentence in enumerate(sentences):
        found = find_coref(sentences[sentence])
        for coref in found:
            if coref.biclosed_type not in coref_dict:
                coref_dict[coref.biclosed_type] = [(coref, sentence)]
            else:
                coref_dict[coref.biclosed_type].append((coref, sentence))

        if i % 10 == 0:
            print(i, " out of ", len(sentences), " sentences processed")
    return coref_dict

def add_coref(diag, corefs):
    no_states_in_diag = get_last_initial_noun(diag)
    no_unique_nouns = no_states_in_diag
    for coref in corefs:
        no_unique_nouns -= len(coref) - 1

    # Build new diagram without the first no_unique_nouns nouns
    new_diag = diag[no_unique_nouns:]
    return Frame("coref",
                 rigid.Ty().tensor(*([rigid.Ty('n')] * no_unique_nouns)),
                 diag.cod,
                 new_diag)

if __name__ == "__main__":
    ccgbankparser = CCGBankParser("../../data/ccgbank")
    trees = ccgbankparser.section2trees(0) # there is a total of 25 sections
    # coref_dict = (all_corefs(trees))
    # for key in coref_dict.keys():
    #     print(key, len(coref_dict[key]), coref_dict[key][0])

    single_sentence_id = 'wsj_0085.42'
    sentence = trees[single_sentence_id].text
    sentence = "Bob , a lawyer who has himself received the forms, talked"
    sentence = "Alice and Bob walked"
    print(sentence)
    parser = BobcatParser()
    sentence2circ(parser, sentence).draw()
    corefs = nlp(sentence)._.coref_chains
    print("corefs: ", corefs)
"""
Results: 
n                       1231    (CCGTree('group'), 'wsj_0003.1')
(n << n)                356     (CCGTree('its'), 'wsj_0003.3')
((n << n) << (n << n))  3       (CCGTree('Groton'), 'wsj_0003.17')
(n >> n)                28      (CCGTree('N.H.'), 'wsj_0013.5')
s                       1       (CCGTree('Carolina'), 'wsj_0044.40')
((n >> s) >> (n >> s))  1       (CCGTree('himself'), 'wsj_0049.10')
((s << s) << n)         1       (CCGTree('the'), 'wsj_0085.42')
(s >> s)                1       (CCGTree('themselves'), 'wsj_0089.27')
(s << n)                1       (CCGTree('her'), 'wsj_0095.2')
"""