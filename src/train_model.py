import os
import shutil

from network.big_network_models.is_in_one_big_network import \
    IsInOneNetworkTrainer
from network.big_network_models.one_network_trainer_base import \
    OneNetworkTrainerBase

from network.individual_networks_models.is_in_trainer import \
    IsInIndividualNetworksTrainer
from network.individual_networks_models.individual_networks_trainer_base_class import \
    IndividualNetworksTrainerBase

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
from datetime import datetime
from pathlib import Path

from tensorflow import keras
import wandb
from wandb.integration.keras import WandbCallback

from network.utils.callbacks import ValidationAccuracy, \
    ModelCheckpointWithoutSaveTraces
from sklearn.model_selection import train_test_split

# this should the the path to \Neural-DisCoCirc
base_path = os.path.abspath('..')
# base_path = os.path.abspath('.')
config = {
    "batch_size": 32,
    "dataset": "isin_dataset_task1_train.pkl",
    "epochs": 1,
    "learning_rate": 0.001,
    "log_wandb": True,
    "trainer": IsInIndividualNetworksTrainer,
    "vocab": "en_qa1.p",
}
model_config = {
    "hidden_layers": [10, 10],
    "is_in_hidden_layers": [10],
    "wire_dimension": 10,
    # "softmax_relevancies": False,
    # "softmax_logits": False
    # "relevance_hidden_layers": [3],
}
config.update(model_config)


def train(base_path, save_path, vocab_path,
          data_path):
    trainer_class = config['trainer']

    print('Training: {} with data {}'
          .format(trainer_class.__name__, config["dataset"]))

    print('loading vocabulary...')
    with open(base_path + vocab_path + config["vocab"], 'rb') as file:
        lexicon = pickle.load(file)

    print('initializing model...')

    discocirc_trainer = trainer_class(lexicon=lexicon, **model_config)

    print('loading pickled dataset...')
    with open(base_path + data_path + config['dataset'],
              "rb") as f:
        # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
        dataset = pickle.load(f)[:5]

    train_dataset, validation_dataset = train_test_split(dataset,
                                                         test_size=0.1,
                                                         random_state=1)

    discocirc_trainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        run_eagerly=True
    )

    datetime_string = datetime.now().strftime("%B_%d_%H_%M")

    tb_callback = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(datetime_string),
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        update_freq='batch',
    )

    save_base_path = base_path + "/checkpoints/"
    checkpoint_callback = ModelCheckpointWithoutSaveTraces(
        filepath='{}/{}'.format(save_base_path, datetime_string),
        save_freq=20 * config["batch_size"]
    )

    validation_callback = ValidationAccuracy(discocirc_trainer.get_accuracy,
                                             interval=1,
                                             log_wandb=config["log_wandb"])

    print('training...')

    callbacks = [tb_callback, validation_callback, checkpoint_callback]

    if config["log_wandb"]:
        callbacks.append(WandbCallback())

    discocirc_trainer.fit(
        train_dataset,
        validation_dataset,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks
    )

    accuracy = discocirc_trainer.get_accuracy(discocirc_trainer.dataset)
    print("The accuracy on the train set is", accuracy)

    save_base_path = base_path + save_path + trainer_class.__name__
    Path(save_base_path).mkdir(parents=True, exist_ok=True)
    name = save_base_path + "/" + trainer_class.__name__ + "_" \
           + datetime.utcnow().strftime("%h_%d_%H_%M")


    discocirc_trainer.save(name, save_traces=False)

    shutil.make_archive(name, 'zip', name)

    if config["log_wandb"]:
        wandb.save(name + '.zip')


if config["log_wandb"]:
    wandb.init(project="discocirc", entity="domlee", config=config)

if __name__ == "__main__":
    train(base_path,
          "/saved_models/",
          '/data/task_vocab_dicts/',
          "/data/pickled_dataset/")
