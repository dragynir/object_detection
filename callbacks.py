import numpy as np
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging



class TrainLogger():
    def __init__(self, log_dir, logfile_name):
        self.logfile_name = str(Path(f'{log_dir}/{logfile_name}'))
    
    def log(self, log_metrics):
        with open(self.logfile_name, 'a') as f:
            f.write(str(log_metrics) + '\n')


class CancellationPoint():
    def __init__(self):
        pass
    
    def check(self):
        return False

class ModelCheckpoint():
  """
  """

  def __init__(self,
               checkpoint,
               save_name,
               mode='min',
               baseline=None,
               ):

    self.stopped_epoch = 0
    self.checkpoint = checkpoint
    self.baseline = baseline
    self.save_name = str(Path(save_name)/Path('ckpt'))


    if mode not in ['min', 'max']:
      logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'min'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    
    self.on_train_begin()

  def on_train_begin(self):
    # Allow instances to be re-used
    self.stopped_epoch = 0
    if self.baseline is not None:
      self.best = self.baseline
    else:
      self.best = np.Inf if self.monitor_op == np.less else -np.Inf

  def step(self, epoch, current):
    if current is None:
      return
    n = self.save_name
    if self.monitor_op(current, self.best):
      self.best = current
      self.stopped_epoch = epoch
      # checkpoint.read("/tmp/ckpt").assert_consumed() read example
      self.checkpoint.write(self.save_name)
      return True

    return False     


class EarlyStopping():
  """Stop training when a monitored metric has stopped improving.
  Assuming the goal of a training is to minimize the loss. With this, the
  metric to be monitored would be `'loss'`, and mode would be `'min'`.
    min_delta: Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement.
    patience: Number of epochs with no improvement
        after which training will be stopped.
    verbose: verbosity mode.
    mode: One of `{"min", "max"}`. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `"max"`
        mode it will stop when the quantity
        monitored has stopped increasing;
  Example:
  >>> callback = EarlyStopping(patience=3)
  >>> # This callback will stop the training when there is no improvement in
  >>> # the validation loss for three consecutive epochs.
  >>> callback.step(epoch=your_epoch_number, current=your_current_loss)
  >>> # step return if Early stopping performed
  """

  def __init__(self,
               min_delta=0,
               patience=0,
               mode='min',
               baseline=None):

    self.patience = patience
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0
    self.baseline = baseline


    if mode not in ['min', 'max']:
      logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'min'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1
    
    self.on_train_begin()

  def on_train_begin(self):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    if self.baseline is not None:
      self.best = self.baseline
    else:
      self.best = np.Inf if self.monitor_op == np.less else -np.Inf

  def step(self, epoch, current):
    if current is None:
      return
    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        return True
    return False        