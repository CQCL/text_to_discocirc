# %%

# import parser
from lambeq import BobcatParser

parser = BobcatParser(verbose='suppress')
print('parser imported')

#%%

# import Richie's CCG to Circ

import sys

sys.path.append('C:\\Users\\jonat\\documents\\py-projects\\text2circ')

from discocirc import convert_sentence


#%%
# read the file
path = 'C:/Users/jonat/documents/py-projects/neural-discocirc/'

with open(path+'tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt') as f:
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

for i, line in enumerate(no_question_lines):

    # obtain circ for the line
    line_diag = parser.sentence2tree(line).to_biclosed_diagram()
    try:  # TODO: sentences invovlving cross-composition are not supported yet
        line_circ = convert_sentence(line_diag)
    except:
        print("problematic line is: {}".format(line))

    line_boxes = line_circ.boxes

    for box in line_boxes:
        if box not in vocab:
            vocab.append(box)

    if i % 50 == 0:
        print("{} of {}".format(i,len(no_question_lines)))
        print("vocab size = {}".format(len(vocab)))


print(vocab)

#%%