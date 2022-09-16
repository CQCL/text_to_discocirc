import os

import pickle
from datetime import datetime
from pathlib import Path

from tensorflow import keras
import wandb
from wandb.integration.keras import WandbCallback

from network.callbacks import ValidationAccuracy
from sklearn.model_selection import train_test_split

from network.models.add_logits_trainer import DisCoCircTrainerAddLogits
from network.models.add_scaled_logits_trainer import \
    DisCoCircTrainerAddScaledLogits
from network.models.is_in_trainer import DisCoCircTrainerIsIn
from network.models.lstm_trainer import DisCoCircTrainerLSTM
from network.models.textspace_trainer import DisCoCircTrainerTextspace

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(base_path, save_path, vocab_path,
          data_path):
    trainer_class = config['trainer']

    print('Training: {} with data {}'
          .format(trainer_class.__name__, data_path))

    print('loading vocabulary...')
    with open(base_path + vocab_path + config["vocab"], 'rb') as file:
        lexicon = pickle.load(file)

    print('initializing model...')
    discocirc_trainer = trainer_class.from_lexicon(lexicon,
                                                   config['wire_dimension'])

    print('loading pickled dataset...')
    with open(base_path + data_path + config['dataset'],
              "rb") as f:
        # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
        dataset = pickle.load(f)[:5]

    train_dataset, validation_dataset = train_test_split(dataset,
                                                         test_size=0.1,
                                                         random_state=1)

    print('compiling train dataset (size: {})...'.format(len(train_dataset)))
    discocirc_trainer.compile_dataset(train_dataset)
    print('compiling validation dataset (size: {})...'
          .format(len(validation_dataset)))
    discocirc_trainer.compile_dataset(validation_dataset, validation=True)

    discocirc_trainer.compile(optimizer=keras.optimizers.Adam(),
                              run_eagerly=True)

    tb_callback = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(datetime.now().strftime("%B_%d_%H_%M")),
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        update_freq='batch',
    )

    validation_callback = ValidationAccuracy(discocirc_trainer.get_accuracy,
                                             interval=1, log_wandb=config["log_wandb"])

    print('training...')

    callbacks = [tb_callback, validation_callback]
    if config["log_wandb"]:
        callbacks.append(WandbCallback())
    discocirc_trainer.fit(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks
    )

    accuracy = discocirc_trainer.get_accuracy(discocirc_trainer.dataset)

    print("The accuracy on the train set is", accuracy)

    save_base_path = base_path + save_path + trainer_class.__name__
    Path(save_base_path).mkdir(parents=True, exist_ok=True)
    discocirc_trainer.save_models(
        save_base_path + "/"
        + trainer_class.__name__ + "_"
        + datetime.utcnow().strftime("%h_%d_%H_%M") + '.pkl')

# this should the the path to \Neural-DisCoCirc
base_path = os.path.abspath('..')
# base_path = os.path.abspath('.')

config = {
    "epochs": 100,
    "batch_size": 32,
    "wire_dimension": 10,
    "trainer": DisCoCircTrainerAddScaledLogits,
    "dataset": "add_logits_dataset_task1_train.pkl",
    "vocab": "en_qa1.p",
    "log_wandb": False
}

if config["log_wandb"]:
    wandb.init(project="discocirc", entity="domlee", config=config)

if __name__ == "__main__":
    train(base_path,
          "/saved_models/",
          '/data/task_vocab_dicts/',
          "/data/pickled_dataset/")
