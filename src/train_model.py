import os

from network.big_network_models.is_in_one_big_network import TrainerIsIn
from network.big_network_models.one_big_network import NeuralDisCoCirc
from network.trainer_base_class import DisCoCircTrainerBase

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

from network.add_logits_trainer import DisCoCircTrainerAddLogits
from network.add_scaled_logits_trainer import DisCoCircTrainerAddScaledLogits
from network.added_wires_to_logits_trainer import DisCoCircTrainerAddedWiresToLogits
from network.is_in_trainer import DisCoCircTrainerIsIn
from network.lstm_trainer import DisCoCircTrainerLSTM
from network.textspace_trainer import DisCoCircTrainerTextspace


# this should the the path to \Neural-DisCoCirc
base_path = os.path.abspath('..')
# base_path = os.path.abspath('.')
config = {
    "epochs": 100,
    "batch_size": 8,
    "trainer": TrainerIsIn,
    "dataset": "isin_dataset_task1_train.pkl",
    "vocab": "en_qa1.p",
    "log_wandb": False
}
model_config = {
    "wire_dimension": 2,
    "hidden_layers": [5],
    # "is_in_hidden_layers": [10],
    # "relevance_hidden_layers": [3],
}
config.update(model_config)


def train(base_path, save_path, vocab_path,
          data_path):
    trainer_class = config['trainer']

    print('Training: {} with data {}'
          .format(trainer_class.__name__, data_path))

    print('loading vocabulary...')
    with open(base_path + vocab_path + config["vocab"], 'rb') as file:
        lexicon = pickle.load(file)

    print('initializing model...')
    if issubclass(trainer_class, NeuralDisCoCirc):
        discocirc_trainer = trainer_class(lexicon=lexicon, **model_config)
    else:
        discocirc_trainer = trainer_class.from_lexicon(lexicon, **model_config)

    print('loading pickled dataset...')
    with open(base_path + data_path + config['dataset'],
              "rb") as f:
        # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
        dataset = pickle.load(f)[:5]

    train_dataset, validation_dataset = train_test_split(dataset,
                                                         test_size=0.1,
                                                         random_state=1)

    if issubclass(trainer_class, DisCoCircTrainerBase):
        print('compiling train dataset (size: {})...'.format(len(train_dataset)))
        discocirc_trainer.compile_dataset(train_dataset)
        print('compiling validation dataset (size: {})...'
              .format(len(validation_dataset)))
        discocirc_trainer.compile_dataset(validation_dataset, validation=True)

    discocirc_trainer.compile(optimizer=keras.optimizers.Adam(),
                              run_eagerly=True)

    datetime_string = datetime.now().strftime("%B_%d_%H_%M")

    tb_callback = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(datetime_string),
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        update_freq='batch',
    )

    checkpoint_callback = ModelCheckpointWithoutSaveTraces(
        filepath='checkpoints/{}'.format(datetime_string),
    )

    validation_callback = ValidationAccuracy(discocirc_trainer.get_accuracy,
                                             interval=1, log_wandb=config["log_wandb"])

    print('training...')

    callbacks = [tb_callback, validation_callback, checkpoint_callback]
    callbacks = [tb_callback, checkpoint_callback]
    if config["log_wandb"]:
        callbacks.append(WandbCallback())

    if issubclass(trainer_class, NeuralDisCoCirc):
        discocirc_trainer.fit(
            train_dataset,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks
        )
    else:
        discocirc_trainer.fit(
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks
        )

    accuracy = discocirc_trainer.get_accuracy(discocirc_trainer.dataset)

    print("The accuracy on the train set is", accuracy)

    save_base_path = base_path + save_path + trainer_class.__name__
    Path(save_base_path).mkdir(parents=True, exist_ok=True)
    name = save_base_path + "/" + trainer_class.__name__ + "_" \
           + datetime.utcnow().strftime("%h_%d_%H_%M") + '.pkl'
    discocirc_trainer.save(name, save_traces=False)
    # for normal: discocirc_trainer.save_models(name)
    if config["log_wandb"]:
        wandb.save(name)

if config["log_wandb"]:
    wandb.init(project="discocirc", entity="domlee", config=config)

if __name__ == "__main__":
    train(base_path,
          "/saved_models/",
          '/data/task_vocab_dicts/',
          "/data/pickled_dataset/")
