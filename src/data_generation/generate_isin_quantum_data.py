#%%
import os
import pickle

from sklearn.model_selection import train_test_split

from lambeq import IQPAnsatz
from lambeq.ansatz.circuit import Sim14Ansatz, StronglyEntanglingAnsatz
from lambeq.core.types import AtomicType
from discopy import Box, Ty, Id, Bra, Word
from discopy.quantum import Discard
from tqdm import tqdm

# this should be the path to \Neural-DisCoCirc
p = os.path.abspath('../..') 

save_file = '/data/pickled_dataset/isin_dataset_task1_no_higher_order_quantum_train_perm_Ans14.pkl'
wire_dim = 2
N = AtomicType.NOUN
n_layers = 2
D = {N: wire_dim}
ansatz = Sim14Ansatz(D, n_layers=n_layers)

print('loading pickled dataset...')
with open(p+"/data/pickled_dataset/isin_dataset_task1_no_higher_order_train.pkl", "rb") as f:
    # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
    dataset = pickle.load(f)
#%%

def create_perms(data):
    diag = data[0]
    q_word = data[1][0]
    a_word = data[1][1]
    perms = []
    unpermuted = list(range(len(diag.cod)))
    import itertools

    if q_word != len(diag.cod)-1:
        for i in unpermuted:
            if i != q_word:
                p = unpermuted.copy()
                j = p[q_word+1]
                p[q_word+1] = i
                p[i] = j
                d = diag.permute(p)
                c = ansatz(d)
                if i == a_word:
                    perms.append((diag.permute(p), 1))
                else:
                    perms.append((diag.permute(p), 0))

                diag.permute(p)
    else:
        for i in unpermuted:
            p = unpermuted.copy()
            j = p[q_word-1]
            p[q_word-1] = q_word
            p[q_word] = j
            if i != q_word-1:
                j = p[q_word]
                p[q_word] = i
                p[i] = j
                d = diag.permute(p)
                c = ansatz(d)

                if i == a_word:
                    perms.append((diag.permute(p), 1))
                else:
                    perms.append((diag.permute(p), 0))

    return perms




def permute(data, is_in):
    diag = data[0]
    q_word = data[1][0]
    a_word = data[1][1]

    q_neighbour = q_word + 1
    q_small_neighbour = q_word - 1

    perms = []
    for i in range(len(diag.cod)):
        if q_word + 1 != len(diag.cod):
            if i != q_word:
                if i < q_neighbour:
                    c = diag.permute(q_neighbour, i)

                elif i > q_neighbour:
                    c = diag.permute(i, q_neighbour)

                elif i == q_neighbour:
                    c = diag

        else:
            c = diag.permute(q_word, q_small_neighbour)
            c = c.permute(q_small_neighbour, i)
            c.draw(figsize=(8,25))
        
        c = ansatz(c)
        if q_word*wire_dim > 0:
            c >>= Discard(c.cod[:(q_word*wire_dim)]) @ is_in.dagger() @ Discard(c.cod[(q_word*wire_dim+2*wire_dim):])

        else:
            c >>= is_in.dagger() @ Discard(c.cod[(q_word*wire_dim+2*wire_dim):])

        if i == a_word:
            perms.append((c, 1))
        else:
            perms.append((c, 0))


        
    return perms

is_in = Word("is_in", N @ N)
is_in = ansatz(is_in)

#%%

all_circuits = []
for data in tqdm(dataset):
    circ = ansatz(data[0])
    q_word = data[1][0] * wire_dim
    a_word = data[1][1] * wire_dim

    # circ.draw()
    perms = create_perms(data, is_in)
    all_circuits.append((circ, data[1]))
# %%

with open(p+save_file, "wb") as f:
    pickle.dump(all_circuits, f)

# %%
