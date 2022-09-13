import tensorflow as tf
import wandb

class ValidationAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, get_accuracy_fn, interval=1, accuracy_fun_data=None):
        super(ValidationAccuracy, self).__init__()
        self.get_accuracy_fn = get_accuracy_fn
        self.interval = interval
        self.accuracy_fun_data = accuracy_fun_data


    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.interval == 0 and self.model.validation_dataset:
            if self.accuracy_fun_data != None:
                score = self.get_accuracy_fn(self.model, self.model.validation_dataset, self.accuracy_fun_data)
            else:
                score = self.get_accuracy_fn(self.model, self.model.validation_dataset)
            tf.print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
            tf.summary.scalar('validation accuracy', data=score, step=epoch)
            wandb.log({'validation accuracy': score})
