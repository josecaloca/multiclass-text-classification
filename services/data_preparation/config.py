from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='settings.env', env_file_encoding='utf-8'
    )

    hf_token: Optional[str]

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    project_name: str = 'multiclass-text-classification'
    pre_trained_bert_model: str = 'distilbert/distilbert-base-uncased'
    hf_dataset_registry: str = f'josecaloca/{project_name}-dataset'


config = Config()
