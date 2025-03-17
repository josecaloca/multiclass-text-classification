"""
Module for evaluating a fine-tuned BERT model.

This module provides the `BertModelEvaluator` class, which computes evaluation metrics 
such as accuracy, precision, recall, and F1-score for a given validation dataset. 
It also logs a confusion matrix to CometML for better performance tracking.
"""

from typing import Tuple, Union

import comet_ml
import evaluate
import numpy as np
from datasets import Dataset
from loguru import logger


class BertModelEvaluator:
    """
    Handles evaluation metric computation and logging for a fine-tuned BERT model.

    This class:
    - Computes classification metrics: accuracy, precision, recall, and F1-score.
    - Retrieves sample examples from the validation dataset.
    - Logs a confusion matrix to CometML for tracking model performance.

    Attributes:
        validation_dataset (Dataset): The dataset used for validation.
        id2label (dict): Mapping of label indices to their corresponding labels.
        accuracy: Evaluation metric for accuracy.
        precision: Evaluation metric for precision.
        recall: Evaluation metric for recall.
        f1: Evaluation metric for F1-score.
    """

    def __init__(self, validation_dataset: Dataset, id2label: dict) -> None:
        """
        Initializes the evaluator with the validation dataset and evaluation metrics.

        Args:
            validation_dataset (Dataset): The dataset used for model validation.
            id2label (dict): Dictionary mapping label indices to label names.
        """
        logger.info("Initializing BertModelEvaluator")
        self.validation_dataset = validation_dataset
        self.id2label = id2label
        self.accuracy = evaluate.load("accuracy")
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load("recall")
        self.f1 = evaluate.load("f1")

    def get_example(self, index: int) -> str:
        """
        Retrieves an example text from the validation dataset.

        Args:
            index (int): Index of the sample in the validation dataset.

        Returns:
            str: The preprocessed title of the news article at the specified index.
        """
        return self.validation_dataset[index]["title_prepared"]

    def compute_metrics(self, pred: Union[np.ndarray, Tuple[np.ndarray]]) -> dict:
        """
        Computes classification metrics and logs the confusion matrix to CometML.

        This method:
        - Extracts predictions and true labels.
        - Computes accuracy, precision, recall, and F1-score using the `evaluate` library.
        - Logs a confusion matrix to CometML for performance tracking.

        Args:
            pred (Union[np.ndarray, Tuple[np.ndarray]]): Model predictions and true labels.

        Returns:
            dict: A dictionary containing computed metrics:
                  - "accuracy": Accuracy score.
                  - "precision": Precision score (macro average).
                  - "recall": Recall score (macro average).
                  - "f1": F1-score (macro average).
        """
        logger.info("Computing evaluation metrics")
        experiment = comet_ml.get_running_experiment()
        preds, labels = pred
        preds = np.argmax(preds, axis=1)

        acc = self.accuracy.compute(predictions=preds, references=labels)
        precision = self.precision.compute(
            predictions=preds, references=labels, average="macro"
        )
        recall = self.recall.compute(
            predictions=preds, references=labels, average="macro"
        )
        f1 = self.f1.compute(predictions=preds, references=labels, average="macro")

        if experiment:
            epoch = (
                int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
            )
            experiment.set_epoch(epoch)
            experiment.log_confusion_matrix(
                y_true=labels,
                y_predicted=preds,
                file_name=f"confusion-matrix-epoch-{epoch}.json",
                labels=list(self.id2label.values()),
                index_to_example_function=self.get_example,
            )
            logger.info("Logged confusion matrix to CometML")

        return {**acc, **precision, **recall, **f1}
