#%%
######################################

# path nonsense
import os, sys

# we want p = the absolute path to \src
p = os.path.abspath('.\src')
# p = os.path.abspath('..') 

print('PATH TO src ', p)
sys.path.insert(1, p)

######################################

import pickle
from data_generation.prepare_data_utils import task_file_reader
from discocirc.discocirc_utils import get_star_removal_functor
from discocirc.text_to_circuit import sentence_list_to_circuit

#%%

# this should be the path to \Neural-DisCoCirc
p = os.path.abspath('.')
# p = os.path.abspath('../..') 

print('PATH TO Neural-DisCoCirc ', p)

TASK_FILE = p+'/data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt'
SAVE_FILE = p+'/data/pickled_dataset/updateorder_textspace_dataset_task1_test.pkl'
# TASK_FILE = p+'\\data\\tasks_1-20_v1-2\\en\\qa1_single-supporting-fact_test.txt'
# SAVE_FILE = p+'\\data\\pickled_dataset\\textspace_dataset_task1_test.pkl'

contexts, questions, answers = task_file_reader(TASK_FILE)

# %%

# generate context circuits from context sentences
context_circuits = []
for i, context in enumerate(contexts):
    # a list of all the circuits for sentences in this context
    context_circ = sentence_list_to_circuit(context, simplify_swaps=False, wire_order='update_order')
    context_circuits.append(context_circ)
    print('finished context {}'.format(i), end='\r')

# processing - delete the '?' at the end of questions
questions = [question[:-1] for question in questions]

# generate question circuits
question_circuits = []
for i, question in enumerate(questions):
    question_circ = sentence_list_to_circuit([question])
    question_circuits.append(question_circ)
    print('finished question {}'.format(i), end ='\r')

star_removal_functor = get_star_removal_functor()

dataset = []
for context_circ, question_circ, answer in zip(context_circuits, question_circuits, answers):
    context_circ = star_removal_functor(context_circ)
    question_circ = star_removal_functor(question_circ)
    dataset.append((context_circ, (question_circ, answer)))

#%%

with open(SAVE_FILE, "wb") as f:
    pickle.dump(dataset, f)

# %%
