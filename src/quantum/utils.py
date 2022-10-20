def get_train_valid(train_dataset, validation_dataset):
    # TODO: is this necessary? Could dataset be given to model.fit directly? Or could dataset creation be adjusted?
    train_circuits = [d[0] for d in train_dataset]
    # not really labels yet!! only question word and answer word
    train_labels = [d[1] for d in train_dataset]
    valid_circuits = [d[0] for d in validation_dataset]
    # not really labels yet!! only question word and answer word
    valid_labels = [d[1] for d in validation_dataset]
    return train_circuits, valid_circuits, train_labels, valid_labels
