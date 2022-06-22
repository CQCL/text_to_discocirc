############################################################
# generate data for task 1
############################################################


# path nonsense
import os, sys
p = os.path.abspath('..') # this should the the path to \src
sys.path.insert(1, p)


# some instructions specific to task 1
import pickle

from data_generation.generate_answer_pair_number import get_qa_numbers
from data_generation.prepare_data_utils import generate_context_circuit, task_file_reader
from discocirc.discocirc_utils import get_star_removal_functor


TASK_FILE = 'data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt'
SAVE_FILE = 'data/pickled_dataset/dataset_task1_test.pkl'


contexts, questions, answers = task_file_reader(TASK_FILE)

# generate context circuits from context sentences
context_circuits = []
for i, context in enumerate(contexts):
    # a list of all the circuits for sentences in this context
    context_circ = generate_context_circuit(context)
    context_circuits.append(context_circ)
    print('finished context {}'.format(i), end='\r')


star_removal_functor = get_star_removal_functor()
q_a_number_pairs = get_qa_numbers(context_circuits, questions, answers)
dataset = []
for circ, test in zip(context_circuits, q_a_number_pairs):
    circ = star_removal_functor(circ)
    dataset.append((circ, test))

with open(SAVE_FILE, "wb") as f:
    pickle.dump(dataset, f)

