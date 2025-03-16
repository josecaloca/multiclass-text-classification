from typing import Tuple

import comet_ml
from datasets import Dataset, load_dataset
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .model_evaluations import BertModelEvaluator
from .paths import MODELS_DIR
from .base import TrainingPipeline


class BertFineTuningPipeline(TrainingPipeline):
    """
    Encapsulates the training and evaluation pipeline for fine-tuning a BERT model.
    """

    def __init__(self, 
                project_name: str,
                hf_dataset_registry: str,
                pre_trained_bert_model: str,
                hf_model_registry: str,
                id2label: dict,
                label2id: dict,
                frac_sample_reduction_training: float,
                random_state: int,
                ) -> None:
        """
        Initializes the training pipeline.
        """
        self.project_name = project_name
        self.hf_dataset_registry = hf_dataset_registry
        self.pre_trained_bert_model = pre_trained_bert_model
        self.hf_model_registry = hf_model_registry
        self.id2label = id2label
        self.label2id = label2id
        self.frac_sample_reduction_training = frac_sample_reduction_training
        self.random_state = random_state
        
        
        logger.info('Initializing BertTrainingPipeline')
        comet_ml.login(project_name=self.project_name)

        self.tokenized_dataset_dict = load_dataset(self.hf_dataset_registry)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pre_trained_bert_model)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.train_dataset, self.validation_dataset, self.test_dataset = (
            self._prepare_datasets()
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pre_trained_bert_model,
            num_labels=4,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.evaluator = BertModelEvaluator(self.validation_dataset, self.id2label)

    def _prepare_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepares training, validation, and test datasets. Reduces the training dataset size if `frac_sample_reduction_training` is less than 1.
        """
        logger.info('Preparing datasets')
        training_sample_size = int(
            self.tokenized_dataset_dict['train'].num_rows
            * self.frac_sample_reduction_training
        )
        if training_sample_size != self.tokenized_dataset_dict['train'].num_rows:
            train_dataset = (
                self.tokenized_dataset_dict['train']
                .shuffle(seed=self.random_state)
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
            seed=self.random_state,
            output_dir=MODELS_DIR / self.project_name,
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
        trainer.push_to_hub(self.hf_model_registry)
        self.tokenizer.push_to_hub(self.hf_model_registry)
        logger.info('Model and custom Tokenizer pushed to hub successfully')

