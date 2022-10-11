############################################################
# generate data for 'IsIn model' (tasks 1, 2)
############################################################

#%%

# path nonsense
import os, sys

# some instructions specific to task 1
import pickle

from data_generation.generate_answer_pair_number import get_qa_numbers
from data_generation.prepare_data_utils import task_file_reader
from discocirc.discocirc_utils import get_star_removal_functor
from discocirc.text_to_circuit import sentence_list_to_circuit


TASK_FILE = '/../../data/tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt'
SAVE_FILE = '/../../data/pickled_dataset/test_dataset_task2_train.pkl'

#%%

# want p = absolute path to \Neural-DisCoCirc
# p = os.path.abspath('../..')
p = os.path.abspath('.')
print('PATH TO Neural-DisCoCirc: ', p)
contexts, questions, answers = task_file_reader(p+TASK_FILE)

# # smaller dataset for testing
# contexts = contexts[:20]
# questions = questions[:20]
# answers = answers[:20]

#%%

# generate context circuits from context sentences
context_circuits = []
for i, context in enumerate(contexts):
    # generate a circuit using all sentences in this example's context
    context_circ = sentence_list_to_circuit(context, simplify_swaps=False, wire_order = 'intro_order')
    context_circuits.append(context_circ)
    print('finished context {}'.format(i), end='\r')


star_removal_functor = get_star_removal_functor()
q_a_number_pairs = get_qa_numbers(context_circuits, questions, answers)
dataset = []
for circ, q_a_number_pair, question, answer in \
        zip(context_circuits, q_a_number_pairs, questions, answers):
    circ = star_removal_functor(circ)
    dataset.append((circ, (q_a_number_pair[0], answer)))

#%%

with open(p+SAVE_FILE, "wb") as f:
    pickle.dump(dataset, f)


# %%
