#%%
import os
# this should the the path to \Neural-DisCoCirc
p = os.path.abspath('.')
# p = os.path.abspath('..') 

from network.utils import get_accuracy_textspace

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pickle
from datetime import datetime
from tensorflow import keras

from network.callbacks import ValidationAccuracy
from network.lstm_model import DisCoCircTrainerLSTM

from sklearn.model_selection import train_test_split



#%%

print('loading vocabulary...')
with open(p+'/data/task_vocab_dicts/en_qa1.p', 'rb') as f:
    vocab = pickle.load(f)

print('initializing model...')
kwargs = {
    "lstm_dimension": 20,
    "wire_dimension": 10,
    "lexicon": vocab,
    "hidden_layers": [5]
}

discocirc_trainer = DisCoCircTrainerLSTM.from_lexicon(**kwargs)

#%%

print('loading pickled dataset...')
with open(p+"/data/pickled_dataset/textspace_dataset_task1_train.pkl", "rb") as f:
    # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
    dataset = pickle.load(f)

dataset = dataset[:20]

train_dataset, validation_dataset = train_test_split(dataset, test_size=0.1, random_state=1)

#%%

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
validation_callback = ValidationAccuracy(get_accuracy_textspace, interval=1)                                         

#%%

print('training...')
discocirc_trainer.fit(epochs=100, batch_size=32, callbacks=[tb_callback, validation_callback])
# discocirc_trainer.fit(epochs=100, batch_size=32, callbacks=[tb_callback])

accuracy = get_accuracy_textspace(discocirc_trainer, discocirc_trainer.dataset)
print("The accuracy on the train set is", accuracy)

#%%

discocirc_trainer.save_models(p+'/saved_models/lstm_trained_model_' + datetime.utcnow().strftime("%B_%d_%H_%M") +'.pkl')
