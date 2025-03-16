from typing import Tuple

import comet_ml
import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import DistilBertModel

from src.config import Config, config
from src.paths import MODELS_DIR


class XGBoostTrainingPipeline:
    """
    Encapsulates the training pipeline for an XGBoost model using DistilBERT embeddings.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the pipeline by loading the dataset, extracting embeddings, and preparing data.

        Args:
            config (Config): Configuration settings for the pipeline.
        """
        logger.info('Initializing XGBoostTrainingPipeline')
        comet_ml.login(project_name=config.project_name)

        # Load dataset
        self.tokenized_dataset_dict = load_dataset(config.hf_dataset_registry)
        self._prepare_datasets(config)

        # Load pretrained DistilBERT model
        self.model = DistilBertModel.from_pretrained(config.pre_trained_bert_model)
        self.model.eval()

        # Extract embeddings
        self.train_dataset, self.validation_dataset, self.test_dataset = (
            self._extract_embeddings()
        )

        # Convert to DataFrames
        self.df_train, self.df_val, self.df_test = self._convert_to_dataframe()

    def _prepare_datasets(self, config: Config) -> None:
        """Reduces dataset sizes based on configured fraction reduction."""
        logger.info('Subsampling datasets')
        sample_fractions = {
            'train': config.frac_sample_reduction_training,
            'validation': config.frac_sample_reduction_validation,
            'test': config.frac_sample_reduction_testing,
        }
        self.tokenized_dataset_dict = DatasetDict(
            {
                split: dataset.shuffle(seed=config.random_state).select(
                    range(int(dataset.num_rows * sample_fractions[split]))
                )
                for split, dataset in self.tokenized_dataset_dict.items()
            }
        )
        logger.info('Datasets subsampled successfully')

    def _compute_embeddings(self, batch):
        """Helper function to compute embeddings from DistilBERT."""
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in batch['input_ids']],
            batch_first=True,
            padding_value=0,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in batch['attention_mask']],
            batch_first=True,
            padding_value=0,
        )

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return {'embeddings': outputs.last_hidden_state.mean(dim=1).numpy()}

    def _extract_embeddings(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts DistilBERT embeddings for each dataset split."""
        logger.info('Extracting embeddings from DistilBERT')

        self.tokenized_dataset_dict = self.tokenized_dataset_dict.map(
            self._compute_embeddings, batched=True, batch_size=32
        )

        X_train = np.array(self.tokenized_dataset_dict['train']['embeddings'])
        y_train = np.array(self.tokenized_dataset_dict['train']['label'])

        X_val = np.array(self.tokenized_dataset_dict['validation']['embeddings'])
        y_val = np.array(self.tokenized_dataset_dict['validation']['label'])

        X_test = np.array(self.tokenized_dataset_dict['test']['embeddings'])
        y_test = np.array(self.tokenized_dataset_dict['test']['label'])

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _convert_to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Converts NumPy arrays into Pandas DataFrames."""
        logger.info('Converting embeddings to DataFrame')

        df_train = pd.DataFrame(self.train_dataset[0])
        df_train['label'] = self.train_dataset[1]

        df_val = pd.DataFrame(self.validation_dataset[0])
        df_val['label'] = self.validation_dataset[1]

        df_test = pd.DataFrame(self.test_dataset[0])
        df_test['label'] = self.test_dataset[1]

        logger.info('DataFrames created successfully')
        return df_train, df_val, df_test

    def train(self) -> None:
        """
        Trains an XGBoost model on the extracted embeddings and logs it to Comet ML.
        """
        logger.info('Starting XGBoost training')
        experiment = comet_ml.start(project_name=config.project_name)

        # Separate features and labels
        X_train, y_train = self.df_train.drop(columns=['label']), self.df_train['label']
        X_val, y_val = self.df_val.drop(columns=['label']), self.df_val['label']

        # Define XGBoost model
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=4,
            eval_metric='mlogloss',
            eta=0.1,
            max_depth=6,
            use_label_encoder=False,
        )

        # Train the model
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        # Evaluate the model
        y_pred = model.predict(X_val)
        test_accuracy = model.score(X_val, y_val)
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')

        # Log metrics
        experiment.log_metric('test_accuracy', test_accuracy)
        experiment.log_metric('precision', precision)
        experiment.log_metric('recall', recall)
        experiment.log_metric('f1_score', f1)
        experiment.log_confusion_matrix(
            y_true=y_val,
            y_predicted=y_pred,
            labels=config.id2label.values(),
        )

        # Save and log the trained model
        model_path = MODELS_DIR / 'xgboost_model.pkl'
        joblib.dump(model, model_path)
        experiment.log_model('xgboost_model', file_or_folder=str(model_path))

        experiment.end()


if __name__ == '__main__':
    load_dotenv('settings.env')
    logger.info('Starting XGBoostTrainingPipeline execution')
    pipeline = XGBoostTrainingPipeline(config=config)
    pipeline.train()
    logger.info('Pipeline execution completed')
