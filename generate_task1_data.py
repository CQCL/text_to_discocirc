############################################################
# generate data for task 1
############################################################


#%%
# some instructions specific to task 1
import pickle

from lambeq import BobcatParser
from prepare_data_utils import compose_circuits, task_file_reader
from discocirc import convert_sentence
from utils import get_star_removal_functor

parser = BobcatParser(verbose='suppress')
# parser = BobcatParser(model_name_or_path='C:/Users/jonat/bert/')

#%%

contexts, questions, answers = task_file_reader('tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt')

# convert question answers to statements
q_a_pairs = []

for question, answer_word in zip(questions, answers):
    question_word = question.split()[-1][:-1]
    q_a_pairs.append((question_word, answer_word))


# %%
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

    if i % 10 == 0:
        print('finished context {}'.format(i))

    if i % 50 == 0:
        with open('context_circuits.pkl', 'wb') as fh:
            pickle.dump(context_circuits, fh)

    context_circuits.append(context_circ)

with open('context_circuits.pkl', 'wb') as fh:
    pickle.dump(context_circuits, fh)

#%%

# test_context = [
#     'Mary moved to the bathroom',
#     'Sandra journeyed to the bathroom',
#     'Mary got the football there',
#     'John went to the kitchen',
#     'Mary went back to the kitchen',
#     'Mary went back to the garden',
#     'Sandra went back to the office',
#     'John moved to the office',
#     'Sandra journeyed to the hallway',
#     'Daniel went back to the kitchen',
#     'Mary dropped the football',
#     'John got the milk there'
# ]

# %%

# generate the circuits for the individual sentences
# sentence_circuits = []
#
# for line in test_context:
#     line_diag = parser.sentence2tree(line).to_biclosed_diagram()
#     line_diag = convert_sentence(line_diag)
#     sentence_circuits.append(line_diag)
#
# context_circ = sentence_circuits[0]
#
# for circ in sentence_circuits[1:]:
#     context_circ = compose_circuits(context_circ, circ)
#
# context_circ.draw(figsize=[20, 80], path="circuit.pdf")

# %%
