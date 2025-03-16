from typing import Tuple, Union

import comet_ml
import evaluate
import numpy as np
from datasets import Dataset
from loguru import logger


class BertModelEvaluator:
    """
    Handles evaluation metrics computation and logging to CometML.
    """

    def __init__(self, validation_dataset: Dataset, id2label: dict) -> None:
        """
        Initializes the evaluator with validation dataset and evaluation metrics.

        Args:
            validation_dataset: The dataset used for validation.
        """
        logger.info('Initializing BertModelEvaluator')
        self.validation_dataset = validation_dataset
        self.id2label = id2label
        self.accuracy = evaluate.load('accuracy')
        self.precision = evaluate.load('precision')
        self.recall = evaluate.load('recall')
        self.f1 = evaluate.load('f1')

    def get_example(self, index: int):
        """
        Retrieves an example text from the validation dataset.

        Args:
            index (int): Index of the sample in the validation dataset.
        """
        return self.validation_dataset[index]['title_prepared']

    def compute_metrics(self, pred: Union[np.ndarray, Tuple[np.ndarray]]) -> dict:
        """
        Computes classification metrics and logs the confusion matrix to CometML.

        Args:
            pred: Predictions from the model.

        Returns:
            dict: Dictionary containing accuracy, precision, recall, and F1 scores.
        """
        logger.info('Computing evaluation metrics')
        experiment = comet_ml.get_running_experiment()
        preds, labels = pred
        preds = np.argmax(preds, axis=1)

        acc = self.accuracy.compute(predictions=preds, references=labels)
        precision = self.precision.compute(
            predictions=preds, references=labels, average='macro'
        )
        recall = self.recall.compute(
            predictions=preds, references=labels, average='macro'
        )
        f1 = self.f1.compute(predictions=preds, references=labels, average='macro')

        if experiment:
            epoch = (
                int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
            )
            experiment.set_epoch(epoch)
            experiment.log_confusion_matrix(
                y_true=labels,
                y_predicted=preds,
                file_name=f'confusion-matrix-epoch-{epoch}.json',
                labels=list(self.id2label.values()),
                index_to_example_function=self.get_example,
            )
            logger.info('Logged confusion matrix to CometML')

        return {**acc, **precision, **recall, **f1}
