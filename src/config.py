from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='settings.env', env_file_encoding='utf-8'
    )

    comet_api_key: Optional[str]
    comet_ml_workspace: Optional[str]
    comet_ml_project_name: Optional[str]
    comet_log_assets: Optional[str]
    comet_mode: Optional[str]
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    pre_trained_bert_model: str = 'distilbert/distilbert-base-uncased'


config = Config()
