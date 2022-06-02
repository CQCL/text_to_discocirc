import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pickle

from lambeq import BobcatParser
import tensorflow as tf
from tensorflow import keras

from discocirc import convert_sentence
from model import DisCoCircTrainer
from utils import get_star_removal_functor

with open('task_vocab_dicts/en_qa1_test.p', 'rb') as f:
    vocab = pickle.load(f)

print('bobcat parser...')
parser = BobcatParser()

print('initializing model...')
discotrainer = DisCoCircTrainer(vocab, 20)


print('parsing sentences...')
diag = parser.sentence2tree('Mary went').to_biclosed_diagram()
diag = convert_sentence(diag)
diag = get_star_removal_functor()(diag)


print('compiling dataset...')
dataset = [(diag, None) for _ in range(5)]

discotrainer.compile_dataset(dataset)
discotrainer.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)

input_index_dataset = tf.data.Dataset.from_tensor_slices(list(range(len(dataset))))
# input_index_dataset = tf.data.Dataset.range(len(dataset))
input_index_dataset = input_index_dataset.shuffle(len(dataset))
input_index_dataset = input_index_dataset.batch(2)

print('training...')
discotrainer.fit(input_index_dataset, epochs=10)
