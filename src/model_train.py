#%%

import os
# this should the the path to \Neural-DisCoCirc
p = os.path.abspath('.')
# p = os.path.abspath('..') 

from network.utils import get_accuracy_isin

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pickle
from datetime import datetime
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback

from network.callbacks import ValidationAccuracy
from network.model import DisCoCircTrainerIsIn

from sklearn.model_selection import train_test_split


config = {
  "epochs": 100,
  "batch_size": 32,
  "wire_dimension": 10,
  "hidden_layers": [10]
}
wandb.init(project="discocirc", entity="domlee", config=config)

print('loading vocabulary...')
with open(p+'/data/task_vocab_dicts/en_qa1.p', 'rb') as f:
    vocab = pickle.load(f)

print('initializing model...')
discocirc_trainer = DisCoCircTrainerIsIn.from_lexicon(
    vocab,
    wire_dimension = config['wire_dimension'],
    hidden_layers = config['hidden_layers']
)

print('loading pickled dataset...')
with open(p+"/data/pickled_dataset/isin_dataset_task1_train.pkl", "rb") as f:
    # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
    dataset = pickle.load(f)

train_dataset, validation_dataset = train_test_split(dataset, test_size=0.1, random_state=1)


print('compiling train dataset')
discocirc_trainer.compile_dataset(train_dataset)
print('compiling validation dataset')
discocirc_trainer.compile_dataset(validation_dataset, validation=True)

discocirc_trainer.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
tb_callback = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(datetime.now().strftime("%B_%d_%H_%M")), 
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True,
                                         update_freq='batch',
                                         )
validation_callback = ValidationAccuracy(get_accuracy_isin, interval=1)                                         
print('training...')
discocirc_trainer.fit(
    epochs = config['epochs'],
    batch_size = config['batch_size'],
    # callbacks = [tb_callback, validation_callback]
    callbacks = [tb_callback, validation_callback, WandbCallback()]
)

discocirc_trainer.save_models('./saved_models/trained_model_boxes_' + datetime.utcnow().strftime("%B_%d_%H_%M") +'.pkl')
