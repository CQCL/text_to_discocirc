#%%
######################################

# path nonsense
import os, sys

# p = os.path.abspath('.\src')
p = os.path.abspath('..') # this should the the path to \src
print('PATH IS ', p)
sys.path.insert(1, p)

######################################

import pickle
from data_generation.prepare_data_utils import generate_context_circuit, task_file_reader
from discocirc.discocirc_utils import get_star_removal_functor

#%%

p = os.path.abspath('../..') # this should be the path to \Neural-DisCoCirc
TASK_FILE = p+'/data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt'
SAVE_FILE = p+'/data/pickled_dataset/textspace_dataset_task1_train.pkl'
# p = os.path.abspath('../..') # this should be the path to \Neural-DisCoCirc
# TASK_FILE = p+'\\data\\tasks_1-20_v1-2\\en\\qa1_single-supporting-fact_train.txt'
# SAVE_FILE = p+'\\data\\pickled_dataset\\textspace_dataset_task1_test.pkl'


contexts, questions, answers = task_file_reader(TASK_FILE)



# %%

# generate context circuits from context sentences
context_circuits = []
for i, context in enumerate(contexts):
    # a list of all the circuits for sentences in this context
    context_circ = generate_context_circuit(context)
    context_circuits.append(context_circ)
    print('finished context {}'.format(i), end='\r')

# processing - delete the '?' at the end of questions
questions = [question[:-1] for question in questions]

# generate question circuits
question_circuits = []
for i, question in enumerate(questions):
    question_circ = generate_context_circuit([question])
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
