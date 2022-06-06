############################################################
# generate data for task 1
############################################################


#%%
# some instructions specific to task 1
import pickle

from lambeq import BobcatParser
from data_generation.generate_answer_pair_number import get_qa_numbers
from data_generation.prepare_data_utils import compose_circuits, task_file_reader
from discocirc.discocirc import convert_sentence
from discocirc.discocirc_utils import get_star_removal_functor


TASK_FILE = 'data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt'
SAVE_FILE = 'data/pickled_dataset/dataset_task1_test.pkl'

parser = BobcatParser(verbose='suppress')

contexts, questions, answers = task_file_reader(TASK_FILE)

# generate context circuits from context sentences
context_circuits = []
for i, context in enumerate(contexts):
    # a list of all the circuits for sentences in this context
    sentence_circuits = []
    for sentence in context:
        sentence_diag = parser.sentence2tree(sentence).to_biclosed_diagram()
        sentence_diag = convert_sentence(sentence_diag)
        sentence_circuits.append(sentence_diag)
    context_circ = sentence_circuits[0]
    for circ in sentence_circuits[1:]:
        context_circ = compose_circuits(context_circ, circ)
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

