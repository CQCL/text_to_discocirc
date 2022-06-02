import pickle

from discopy import Ty

from prepare_data_utils import task_file_reader

def find_wire(diagram, name):
    index = 0
    box = diagram.boxes[index]
    while box.dom == Ty():
        if box.name == name:
            return diagram.offsets[index]

        index += 1
        box = diagram.boxes[index]

    raise Exception('wire not found')



def get_qa_numbers(task_file='tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt',
                   context_circuits="context_circuits.pkl"):
    contexts, questions, answers = task_file_reader(task_file)

    q_a_pairs = []
    for question, answer_word in zip(questions, answers):
        question_word = question.split()[-1][:-1]
        q_a_pairs.append((question_word, answer_word))

    pickle_off = open(context_circuits, "rb")
    diags = pickle.load(pickle_off)

    q_a_number_pairs = []
    for i, (question, answer) in enumerate(q_a_pairs):
        q_id = find_wire(diags[i], question)
        a_id = find_wire(diags[i], answer)
        q_a_pairs.append((q_id, a_id))

    return q_a_number_pairs
