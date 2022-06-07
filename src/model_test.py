import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pickle
import numpy as np
from tensorflow import keras

from network.model import DisCoCircTrainer
from network.utils import get_test_accuracy


print('initializing model...')
discocirc_trainer = DisCoCircTrainer.load_models('./saved_models/trained_model_boxes.pkl')

print('loading pickled dataset...')
with open("data/pickled_dataset/dataset_task1_test.pkl", "rb") as f:
    dataset = pickle.load(f)

print('compiling dataset')
discocirc_trainer.compile_dataset(dataset)
discocirc_trainer.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
discocirc_trainer(0)



accuracy = get_test_accuracy(discocirc_trainer, dataset)

print("The accuracy on the test set is", accuracy)
