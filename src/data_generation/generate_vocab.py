# %%

import os
import sys
p = os.path.abspath('..') # this should be the path to \src
print("THE PATH TO src: ", p)
sys.path.insert(1, p)

from discocirc.sentence_to_circuit import convert_sentence  # Richie's CCG to Circ
from lambeq import BobcatParser
from discocirc.discocirc_utils import get_star_removal_functor
import pickle   # for saving vocab file

parser = BobcatParser(verbose='suppress')

#%%
# read the file
path = os.path.abspath('../..') # this should be the path to \Neural-DisCoCirc
print("THE PATH TO Neural-DisCoCirc: ", path)

with open(path+'/data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt') as f:
    lines = f.readlines()


# %%
# filter out the lines involving questions for now
no_question_lines = [line for line in lines if '?' not in line]
# delete initial line numbers
no_question_lines = [' '.join(line.split(' ')[1:]) for line in no_question_lines]
# delete . and \n
no_question_lines = [line.replace('\n','').replace('.',' ') for line in no_question_lines]


# %%
# record all unique vocabulary boxes (word, CCG type)

vocab = []

# get the star removal functor to deal with frames
functor = get_star_removal_functor()

for i, line in enumerate(no_question_lines):

    # obtain circ for the line
    line_diag = parser.sentence2tree(line).to_biclosed_diagram()
    try:  # TODO: sentences invovlving cross-composition are not supported yet
        line_circ = convert_sentence(line_diag)
    except:
        print("problematic line: {}".format(line))

    # apply the star removal functor
    line_circ = functor(line_circ)

    line_boxes = line_circ.boxes

    for box in line_boxes:
        if box not in vocab:
            vocab.append(box)

    if i % 50 == 0:
        print("{} of {}".format(i,len(no_question_lines)))
        print("vocab size = {}".format(len(vocab)))

print(vocab)

#%%

# add additional vocab from questions, using a representative question
line_diag = parser.sentence2tree('Where is the apple').to_biclosed_diagram()
line_circ = convert_sentence(line_diag)
# star removal
line_circ = functor(line_circ)
line_boxes = line_circ.boxes
for box in line_boxes:
    if box not in vocab:
        vocab.append(box)


#%%
# save vocab file
pickle.dump(vocab, open(path+"/data/task_vocab_dicts/en_qa2.p", "wb"))

# %%
