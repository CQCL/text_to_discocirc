import os

from network.utils import get_test_accuracy


import pickle
from datetime import datetime
from tensorflow import keras

from network.model import DisCoCircTrainer


WIRE_DIMENSION = 20

print('loading vocabulary...')
with open('data/task_vocab_dicts/en_qa1_train.p', 'rb') as f:
    vocab = pickle.load(f)

print('initializing model...')
discocirc_trainer = DisCoCircTrainer.from_lexicon(vocab, WIRE_DIMENSION)

print('loading pickled dataset...')
with open("data/pickled_dataset/dataset_task1_train.pkl", "rb") as f:
    dataset = pickle.load(f)

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
discocirc_trainer.fit(epochs=100, batch_size=32, callbacks=[tbCallBack])

accuracy = get_test_accuracy(discocirc_trainer, dataset)
print("The accuracy on the test set is", accuracy)

discocirc_trainer.save_models('./saved_models/trained_model_boxes_' + datetime.utcnow().strftime("%B_%d_%H_%M") +'.pkl')
