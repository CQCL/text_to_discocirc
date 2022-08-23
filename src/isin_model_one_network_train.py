import os
from network.one_big_network import NeuralDisCoCirc

from network.utils import get_accuracy_one_network

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pickle
from datetime import datetime
from tensorflow import keras


WIRE_DIMENSION = 10


###########################################################

print('loading vocabulary...')
with open('data/task_vocab_dicts/en_qa1_train.p', 'rb') as f:
    vocab = pickle.load(f)

print('initializing model...')
neural_discocirc = NeuralDisCoCirc(vocab, WIRE_DIMENSION, hidden_layers=[10])

print('loading pickled dataset...')
with open("data/pickled_dataset/dataset_task1_train.pkl", "rb") as f:
    dataset = pickle.load(f)

# dataset = dataset[:5]

# train_dataset, validation_dataset = train_test_split(dataset, test_size=0.1, random_state=1)

print('compiling model...')
neural_discocirc.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)

tb_callback = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(datetime.now().strftime("%B_%d_%H_%M")), 
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True,
                                         update_freq='batch',
                                         )

print('training...')
neural_discocirc.fit(dataset, epochs=1, batch_size=32, callbacks=[tb_callback])

print('getting accuracy...')
accuracy = get_accuracy_one_network(neural_discocirc, dataset)
print("The accuracy on the train set is", accuracy)

save_location = './saved_models/isin_trained_model_' + datetime.utcnow().strftime("%B_%d_%H_%M")
neural_discocirc.save(save_location, save_traces=False)
