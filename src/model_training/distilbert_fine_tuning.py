from typing import Tuple

import comet_ml
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import Config, config
from src.model_training.model_evaluations import BertModelEvaluator
from src.paths import MODELS_DIR


class BertTrainingPipeline:
    """
    Encapsulates the training and evaluation pipeline for fine-tuning a BERT model.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the training pipeline.

        Args:
            config (Config): Configuration settings for the pipeline.
        """
        logger.info('Initializing BertTrainingPipeline')
        comet_ml.login(project_name=config.project_name)

        self.tokenized_dataset_dict = load_dataset(config.hf_dataset_registry)
        self.tokenizer = AutoTokenizer.from_pretrained(config.pre_trained_bert_model)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.train_dataset, self.validation_dataset, self.test_dataset = (
            self._prepare_datasets(config)
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.pre_trained_bert_model,
            num_labels=4,
            id2label=config.id2label,
            label2id=config.label2id,
        )
        self.evaluator = BertModelEvaluator(self.validation_dataset)

    def _prepare_datasets(self, config: Config) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepares training, validation, and test datasets.

        Reduces the training dataset size if `frac_sample_reduction_training` is less than 1.

        Args:
            config: Configuration settings.

        Returns:
            Train, validation, and test datasets.
        """
        logger.info('Preparing datasets')
        training_sample_size = int(
            self.tokenized_dataset_dict['train'].num_rows
            * config.frac_sample_reduction_training
        )
        if training_sample_size != self.tokenized_dataset_dict['train'].num_rows:
            train_dataset = (
                self.tokenized_dataset_dict['train']
                .shuffle(seed=config.random_state)
                .select(range(training_sample_size))
            )
        else:
            train_dataset = self.tokenized_dataset_dict['train']
        validation_dataset = self.tokenized_dataset_dict['validation']
        test_dataset = self.tokenized_dataset_dict['test']
        logger.info('Datasets prepared successfully')
        return train_dataset, validation_dataset, test_dataset

    def train(self) -> None:
        """
        Trains the BERT model using the specified training arguments.
        """
        logger.info('Starting model training')
        training_args = TrainingArguments(
            seed=config.random_state,
            output_dir=MODELS_DIR / config.project_name,
            overwrite_output_dir=True,
            num_train_epochs=1,
            do_train=True,
            do_eval=True,
            eval_strategy='steps',
            eval_steps=25,
            save_strategy='steps',
            save_total_limit=10,
            save_steps=25,
            per_device_train_batch_size=8,
            push_to_hub=True,
            report_to=['comet_ml'],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            compute_metrics=self.evaluator.compute_metrics,
            data_collator=self.data_collator,
        )

        trainer.train()
        logger.info('Training completed')
        comet_ml.get_running_experiment().end()
        trainer.push_to_hub(config.hf_model_registry)
        logger.info('Model pushed to hub successfully')


if __name__ == '__main__':
    logger.info('Starting BertTrainingPipeline execution')
    load_dotenv('settings.env')
    pipeline = BertTrainingPipeline(config=config)
    pipeline.train()
    logger.info('Pipeline execution completed')
