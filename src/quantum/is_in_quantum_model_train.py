#%%
import os
import os

# this should the the path to \Neural-DisCoCirc
# p = os.path.abspath('../../')
import pickle
from is_in_quantum_model import qDisCoCircIsIn

from sklearn.model_selection import train_test_split

from lambeq import IQPAnsatz
from lambeq.core.types import AtomicType
import torch
from tqdm import tqdm

from utils import get_train_valid
import wandb

#%%
# this should be relative path to \Neural-DisCoCirc
p = os.path.abspath("../..")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ansatz config
WIRE_DIMENSION = 2
N = AtomicType.NOUN
NB_LAYERS = 3
D = {N: WIRE_DIMENSION}
ANSATZ = IQPAnsatz(D, n_layers=NB_LAYERS)

# train config
LEARNING_RATE = 0.001
EPOCHS = 50
# BATCH_SIZE = ?

config = {
    "ansatz": "IQP",
    "layers": NB_LAYERS,
    "wire_dimension": WIRE_DIMENSION,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE
}

wandb.init(project="is_in_prob_dist_01", config=config)

#%%
print("Loading pickled dataset...")
with open(
    p
    + "/data/pickled_dataset/isin_dataset_task1_no_higher_order_quantum_train_no_perm.pkl",
    "rb",
) as f:
    # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
    dataset = pickle.load(f)

train_dataset, validation_dataset = train_test_split(
    dataset, test_size=0.1, random_state=1
)
train_circuits, valid_circuits, train_labels, valid_labels = get_train_valid(
    train_dataset, validation_dataset
)

#%%
print("Initialise model...")
model = qDisCoCircIsIn(train_circuits, valid_circuits, ANSATZ, WIRE_DIMENSION)

#%%
print("Training...")
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.is_in_model.parameters(), lr=LEARNING_RATE)

loss_list = []
acc_list = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for text_circuit, quesans in tqdm(zip(train_circuits, train_labels)):
        results, targets = model.forward(text_circuit, quesans)
        optimizer.zero_grad()
        loss = loss_func(results, targets)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    loss_list.append(epoch_loss)
    valid_accuracy = model.validation_accuracy(valid_labels)
    acc_list.append(valid_accuracy)
    wandb.log({"loss": epoch_loss, "acc": valid_accuracy})

    if epoch % 1 == 0:
        model.is_in_model.save(f"{p}/saved_models/is_in_model_epoch_{epoch}")
        model.text_model.save(f"{p}/saved_models/text_model_epoch_{epoch}")
        pickle.dump(loss_list, open(f"{p}/saved_models/loss_list.pickle", "wb"))
        pickle.dump(acc_list, open(f"{p}/saved_models/acc_list.pickle", "wb"))

    # train_accuracy = accuracy(train_circuits, train_labels, text_model, is_in_model, is_in_circ, is_in_circ_swapped)

    print(f"Epoch: {epoch}    epoch_loss: {epoch_loss}")
    print(f"Valid acc: {valid_accuracy}")

print("FIT FINISHED")

#%%
