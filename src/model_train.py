import sys
sys.setrecursionlimit(10000)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pickle
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from data_generation.generate_answer_pair_number import get_qa_numbers
from discocirc.discocirc_utils import get_star_removal_functor
from network.model import DisCoCircTrainer

with open('data/task_vocab_dicts/en_qa1_train.p', 'rb') as f:
    vocab = pickle.load(f)

print('initializing model...')
discocirc_trainer = DisCoCircTrainer.from_lexicon(vocab, 20)

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
tbCallBack = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(datetime.now().strftime("%B_%d_%H_%M")), 
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True,
                                         update_freq='batch',
                                         )
print('training...')
discocirc_trainer.fit(epochs=200, batch_size=32, callbacks=[tbCallBack])

discocirc_trainer.save_models('./saved_models/trained_model_boxes_' + datetime.utcnow().strftime("%B_%d_%H_%M") +'.pkl')

