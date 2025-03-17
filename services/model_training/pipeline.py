"""
This script loads environment variables, initializes, and executes two training pipelines:
1. BertFineTuningPipeline - Fine-tunes a DistilBERT model for text classification.
2. XGBoostTrainingPipeline - Trains an XGBoost model using precomputed embeddings.

The configurations are loaded from `config`, and logs are recorded using `loguru.logger`.
"""

from config import config
from dotenv import load_dotenv
from loguru import logger
from utils.distilbert_fine_tuning import BertFineTuningPipeline
from utils.xgboost_trainer import XGBoostTrainingPipeline

if __name__ == "__main__":
    load_dotenv("settings.env")  # Load environment variables

    # Execute DistilBERT fine-tuning pipeline
    logger.info("Starting BertFineTuningPipeline execution")
    bert_pipeline = BertFineTuningPipeline(
        project_name=config.project_name,
        hf_dataset_registry=config.hf_dataset_registry,
        pre_trained_bert_model=config.pre_trained_bert_model,
        hf_model_registry=config.hf_model_registry,
        id2label=config.id2label,
        label2id=config.label2id,
        frac_sample_reduction_training=config.frac_sample_reduction_training,
        random_state=config.random_state,
    )
    bert_pipeline.train()
    logger.info("BertFineTuningPipeline execution completed")

    # Execute XGBoost training pipeline
    logger.info("Starting XGBoostTrainingPipeline execution")
    xgboost_pipeline = XGBoostTrainingPipeline(
        project_name=config.project_name,
        hf_dataset_registry=config.hf_dataset_registry,
        pre_trained_bert_model=config.pre_trained_bert_model,
        frac_sample_reduction_training=config.frac_sample_reduction_training,
        frac_sample_reduction_validation=config.frac_sample_reduction_validation,
        frac_sample_reduction_testing=config.frac_sample_reduction_testing,
        id2label=config.id2label,
        random_state=config.random_state,
    )
    xgboost_pipeline.train()
    logger.info("XGBoostTrainingPipeline execution completed")
