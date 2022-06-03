import sys
sys.setrecursionlimit(10000)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from data_generation.generate_answer_pair_number import get_qa_numbers
from discocirc.discocirc_utils import get_star_removal_functor
import pickle
import tensorflow as tf
from tensorflow import keras

from network.model import DisCoCircTrainer

     
with open('data/task_vocab_dicts/en_qa1_train.p', 'rb') as f:
    vocab = pickle.load(f)

print('initializing model...')
discocirc_trainer = DisCoCircTrainer(vocab, 20)

print('loading pickled dataset...')
with open("data/discocirc_diagrams/context_circuits.pkl", "rb") as f:
    context_circuits = pickle.load(f)
q_a_number_pairs = get_qa_numbers()

star_removal_functor = get_star_removal_functor()
dataset = []
for circ, test in zip(context_circuits, q_a_number_pairs):
    circ = star_removal_functor(circ)
    dataset.append((circ, test))


print('compiling dataset')
discocirc_trainer.compile_dataset(dataset)
discocirc_trainer.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
discocirc_trainer(0)
discocirc_trainer.save('./saved_models/compiled_model')


# input_index_dataset = tf.data.Dataset.from_tensor_slices(list(range(len(dataset))))
input_index_dataset = tf.data.Dataset.from_tensor_slices(list(range(2)))
# input_index_dataset = tf.data.Dataset.range(len(dataset))
input_index_dataset = input_index_dataset.shuffle(len(dataset))
input_index_dataset = input_index_dataset.batch(2)

print('training...')
discocirc_trainer.fit(input_index_dataset, epochs=100)

discocirc_trainer.save('saved_models/trained_model')
