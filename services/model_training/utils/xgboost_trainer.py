"""
Module for training an XGBoost model using DistilBERT embeddings.

This module defines the `XGBoostTrainingPipeline` class, which:
- Loads a tokenized dataset from the Hugging Face Hub.
- Uses a pre-trained DistilBERT model to extract sentence embeddings.
- Trains an XGBoost classifier on the extracted embeddings.
- Evaluates model performance using precision, recall, and F1-score.
- Logs results and a confusion matrix to CometML.
- Saves the trained XGBoost model for later use.
"""

from typing import Dict, Tuple, Any

import comet_ml
import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from datasets import DatasetDict, load_dataset
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import DistilBertModel

from utils.paths import MODELS_DIR

from .base import TrainingPipeline


class XGBoostTrainingPipeline(TrainingPipeline):
    """
    A pipeline for training an XGBoost model using DistilBERT embeddings.

    This class:
    - Loads a tokenized dataset.
    - Extracts embeddings using a pre-trained DistilBERT model.
    - Converts embeddings into a format suitable for training.
    - Trains an XGBoost classifier.
    - Logs evaluation metrics and saves the trained model.

    Attributes:
        project_name (str): Name of the CometML project for logging.
        hf_dataset_registry (str): Hugging Face dataset identifier.
        pre_trained_bert_model (str): Pre-trained DistilBERT model for embeddings.
        frac_sample_reduction_training (float): Fraction of training data to retain.
        frac_sample_reduction_validation (float): Fraction of validation data to retain.
        frac_sample_reduction_testing (float): Fraction of testing data to retain.
        id2label (dict): Mapping from class indices to labels.
        random_state (int): Seed for reproducibility.
    """

    def __init__(
        self,
        project_name: str,
        hf_dataset_registry: str,
        pre_trained_bert_model: str,
        frac_sample_reduction_training: float,
        frac_sample_reduction_validation: float,
        frac_sample_reduction_testing: float,
        id2label: Dict[int, str],
        random_state: int,
    ) -> None:
        """
        Initializes the training pipeline.

        Args:
            project_name (str): CometML project name for logging experiments.
            hf_dataset_registry (str): Hugging Face dataset identifier.
            pre_trained_bert_model (str): Pre-trained DistilBERT model to use for embedding extraction.
            frac_sample_reduction_training (float): Fraction of training dataset to retain.
            frac_sample_reduction_validation (float): Fraction of validation dataset to retain.
            frac_sample_reduction_testing (float): Fraction of test dataset to retain.
            id2label (Dict[int, str]): Dictionary mapping class indices to label names.
            random_state (int): Seed for reproducibility.
        """
        self.project_name = project_name
        self.hf_dataset_registry = hf_dataset_registry
        self.pre_trained_bert_model = pre_trained_bert_model
        self.frac_sample_reduction_training = frac_sample_reduction_training
        self.frac_sample_reduction_validation = frac_sample_reduction_validation
        self.frac_sample_reduction_testing = frac_sample_reduction_testing
        self.id2label = id2label
        self.random_state = random_state

        logger.info("Initializing XGBoostTrainingPipeline")
        comet_ml.login(project_name=self.project_name)

        # Load dataset
        self.tokenized_dataset_dict = load_dataset(self.hf_dataset_registry)
        self._prepare_datasets()

        # Load pre-trained DistilBERT model
        self.model = DistilBertModel.from_pretrained(self.pre_trained_bert_model)
        self.model.eval()

        # Extract embeddings
        self.train_dataset, self.validation_dataset, self.test_dataset = (
            self._extract_embeddings()
        )

        # Convert extracted embeddings to DataFrames
        self.df_train, self.df_val, self.df_test = self._convert_to_dataframe()

    def _prepare_datasets(self) -> None:
        """
        Reduces dataset sizes based on configured fraction reduction.
        This helps in controlling dataset size for efficient training.
        """
        logger.info("Subsampling datasets")
        sample_fractions = {
            "train": self.frac_sample_reduction_training,
            "validation": self.frac_sample_reduction_validation,
            "test": self.frac_sample_reduction_testing,
        }
        self.tokenized_dataset_dict = DatasetDict(
            {
                split: dataset.shuffle(seed=self.random_state).select(
                    range(int(dataset.num_rows * sample_fractions[split]))
                )
                for split, dataset in self.tokenized_dataset_dict.items()
            }
        )
        logger.info("Datasets subsampled successfully")

    def _compute_embeddings(self, batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Computes embeddings from DistilBERT for a batch of tokenized text.

        Args:
            batch (Dict[str, Any]): A batch of tokenized sequences containing 'input_ids' and 'attention_mask'.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing extracted embeddings.
        """
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in batch["input_ids"]],
            batch_first=True,
            padding_value=0,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in batch["attention_mask"]],
            batch_first=True,
            padding_value=0,
        )

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return {"embeddings": outputs.last_hidden_state.mean(dim=1).numpy()}

    def _extract_embeddings(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts DistilBERT embeddings for each dataset split.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Training, validation, and test embeddings with labels.
        """
        logger.info("Extracting embeddings from DistilBERT")

        self.tokenized_dataset_dict = self.tokenized_dataset_dict.map(
            self._compute_embeddings, batched=True, batch_size=32
        )

        X_train = np.array(self.tokenized_dataset_dict["train"]["embeddings"])
        y_train = np.array(self.tokenized_dataset_dict["train"]["label"])

        X_val = np.array(self.tokenized_dataset_dict["validation"]["embeddings"])
        y_val = np.array(self.tokenized_dataset_dict["validation"]["label"])

        X_test = np.array(self.tokenized_dataset_dict["test"]["embeddings"])
        y_test = np.array(self.tokenized_dataset_dict["test"]["label"])

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _convert_to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Converts NumPy arrays into Pandas DataFrames.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and test DataFrames.
        """
        logger.info("Converting embeddings to DataFrame")

        df_train = pd.DataFrame(self.train_dataset[0])
        df_train["label"] = self.train_dataset[1]

        df_val = pd.DataFrame(self.validation_dataset[0])
        df_val["label"] = self.validation_dataset[1]

        df_test = pd.DataFrame(self.test_dataset[0])
        df_test["label"] = self.test_dataset[1]

        logger.info("DataFrames created successfully")
        return df_train, df_val, df_test

    def train(self) -> None:
        """
        Trains an XGBoost model on the extracted embeddings and logs it to CometML.
        """
        logger.info("Starting XGBoost training")
        experiment = comet_ml.start(project_name=self.project_name)

        X_train, y_train = self.df_train.drop(columns=["label"]), self.df_train["label"]
        X_val, y_val = self.df_val.drop(columns=["label"]), self.df_val["label"]

        model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=4,
            eval_metric="mlogloss",
            eta=0.1,
            max_depth=6,
            use_label_encoder=False,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        y_pred = model.predict(X_val)

        experiment.log_confusion_matrix(
            y_true=y_val, y_predicted=y_pred, labels=self.id2label.values()
        )

        joblib.dump(model, MODELS_DIR / "xgboost_model.pkl")
        experiment.end()
