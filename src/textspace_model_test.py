#%%
import os

# this should the the path to \Neural-DisCoCirc
# p = os.path.abspath('..')
p = os.path.abspath('.') 


os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pickle
from tensorflow import keras

from network.textspace_model import DisCoCircTrainerTextspace
from network.utils import get_accuracy_textspace

#%%

print('initializing model...')
discocirc_trainer = DisCoCircTrainerTextspace.load_models(p+'/saved_models/textspace_trained_model_boxes_July_01_17_16.pkl')

print('loading pickled dataset...')
with open(p+"/data/pickled_dataset/textspace_dataset_task1_test.pkl", "rb") as f:
    dataset = pickle.load(f)

# dataset = dataset[:20]

#%%
print('compiling dataset')
discocirc_trainer.compile_dataset(dataset)
discocirc_trainer.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)


accuracy = get_accuracy_textspace(discocirc_trainer, discocirc_trainer.dataset)

print("The accuracy on the test set is", accuracy)

# %%
