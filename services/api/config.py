from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='settings.env', env_file_encoding='utf-8'
    )

    hf_token: Optional[str]

    pre_trained_bert_model: str = 'distilbert/distilbert-base-uncased'
    project_name: str = 'multiclass-text-classification'
    hf_model_registry: str = f'josecaloca/{project_name}'

    id2label: dict[int, str] = {
        0: 'Business',
        1: 'Science & Technology',
        2: 'Entertainment',
        3: 'Health',
    }


config = Config()
