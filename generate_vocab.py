###############################################################################
# this script generates a list of vocabulary from the data
###############################################################################

#%% 
# import parser
from lambeq import BobcatParser

parser = BobcatParser(verbose='suppress')
print('parser imported')

# %%
# a recursive function that takes in a CCGTree in dict (json) form,
# and returns a list of all (word, CCG type) in the tree 
def return_boxes(dict):
    """
    This function takes a dictionary (json) representation of
    a CCG derivation tree and returns a list of the boxes used.

    Essentially, it traverses the CCGTree and records the leaves
    """
    boxes = []

    if "children" in dict:
        for child in dict["children"]:
            boxes = boxes+return_boxes(child)

    elif "text" in dict:
        boxes.append([dict["text"],dict["type"]])

    else: # error with CCGTree json
        print('error')

    return boxes

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

    # obtain CCGTree for the line using Bobcat
    line_CCGTree = parser.sentence2tree(line)
    # convert to json
    line_CCGTree = line_CCGTree.to_json()

    line_boxes = return_boxes(line_CCGTree)


    for word in line_boxes:
        if word not in vocab:
            vocab.append(word)

    if i % 50 == 0:
        print("{} of {}".format(i,len(no_question_lines)))
        print("vocab size = {}".format(len(vocab)))
# %%
