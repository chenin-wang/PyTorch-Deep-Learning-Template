from ..loggers.logging_colors import get_logger

logger = get_logger()


# define a custom metric as a function
def my_metric(y_true, y_pred):
    pass


# or as a class when we need to accumulate
class MyEpochMetric:
    def __call__(self, y_pred, y_true):
        """
        Computes the metric for the current batch.
        Args:
            y_pred: The prediction of the model.
            y_true: Target to evaluate the model.
        """
        pass

    def reset(self):
        """Resets the metric's state at the start of a new epoch."""
        pass

    def update(self, y_pred, y_true):
        """
        Updates the metric's state with the results of the current batch.
        Args:
            y_pred: The prediction of the model.
            y_true: Target to evaluate the model.
        """
        pass

    def compute(self):
        """Computes and returns the metric based on the accumulated state."""
        pass
