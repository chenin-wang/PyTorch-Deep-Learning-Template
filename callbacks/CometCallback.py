from ..loggers.logging_colors import get_logger
logger = get_logger()


class SampleCallback:
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs):
        self.experiment.log_metrics(logs)