import tensorflow as tf
import wandb


class ValidationAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, get_accuracy_fn, interval=1, log_wandb=False):
        super(ValidationAccuracy, self).__init__()
        self.get_accuracy_fn = get_accuracy_fn
        self.interval = interval
        self.log_wandb = log_wandb

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.interval == 0 and self.model.validation_dataset:
            score = self.get_accuracy_fn(self.model.validation_dataset)
            tf.print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
            tf.summary.scalar('validation accuracy', data=score, step=epoch)
            if self.log_wandb:
                wandb.log({'validation accuracy': score})


class ModelCheckpointWithoutSaveTraces(tf.keras.callbacks.ModelCheckpoint):
    ################################################################
    # I HAVE ONLY MODIFIED TWO LINES FROM THE ORIGINAL SOURCE CODE #
    ################################################################
    def _save_model(self, epoch, batch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
              is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        from keras.utils import io_utils
        from keras.utils import tf_utils
        from tensorflow.python.platform import tf_logging as logging

        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, batch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning(
                            'Can save best model only with %s available, '
                            'skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f'\nEpoch {epoch + 1}: {self.monitor} improved '
                                    f'from {self.best:.5f} to {current:.5f}, '
                                    f'saving model to {filepath}')
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(
                                    filepath, overwrite=True,
                                    options=self._options)
                            else:
                                #######################################
                                #  I HAVE CHANGED THE FOLLOWING LINE  #
                                #######################################
                                self.model.save(filepath, overwrite=True,
                                                save_traces=False,
                                                options=self._options)
                        else:
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f'\nEpoch {epoch + 1}: '
                                    f'{self.monitor} did not improve from {self.best:.5f}')
                else:
                    if self.verbose > 0:
                        io_utils.print_msg(
                            f'\nEpoch {epoch + 1}: saving model to {filepath}')
                    if self.save_weights_only:
                        self.model.save_weights(
                            filepath, overwrite=True, options=self._options)
                    else:
                        #######################################
                        #  I HAVE CHANGED THE FOLLOWING LINE  #
                        #######################################
                        self.model.save(filepath, overwrite=True,
                                        save_traces=False,
                                        options=self._options)

                self._maybe_remove_file()
            except IsADirectoryError as e:  # h5py 3.x
                raise IOError('Please specify a non-directory filepath for '
                              'ModelCheckpoint. Filepath used is an existing '
                              f'directory: {filepath}')
            except IOError as e:  # h5py 2.x
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in str(e.args[0]).lower():
                    raise IOError(
                        'Please specify a non-directory filepath for '
                        'ModelCheckpoint. Filepath used is an existing '
                        f'directory: f{filepath}')
                # Re-throw the error for any other causes.
                raise e
