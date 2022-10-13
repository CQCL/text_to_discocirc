#%%
import os
import pickle

from sklearn.model_selection import train_test_split

from lambeq import IQPAnsatz
from lambeq.core.types import AtomicType
from discopy import Box, Ty, Id, Bra

# this should be the path to \Neural-DisCoCirc
p = os.path.abspath('../..') 
SAVE_FILE = '/data/pickled_dataset/isin_dataset_task1_no_higher_order_quantum_train_no_perm.pkl'
WIRE_DIMENSION = 2
N = AtomicType.NOUN
NB_LAYERS = 3
D = {N: WIRE_DIMENSION}
ANSATZ = IQPAnsatz(D, n_layers=NB_LAYERS)

print('loading pickled dataset...')
with open(p+"/data/pickled_dataset/isin_dataset_task1_no_higher_order_train.pkl", "rb") as f:
    # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
    dataset = pickle.load(f)
#%%
all_circuits = []
i = 0

for data in dataset:
    print(i)
    circ = ANSATZ(data[0])
    all_circuits.append((circ, data[1]))
    i+=1

with open(p+SAVE_FILE, "wb") as f:
    pickle.dump(all_circuits, f)

