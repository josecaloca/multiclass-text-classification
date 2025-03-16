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
    comet_auto_log_graph: Optional[bool]
    comet_auto_log_metrics: Optional[bool]
    comet_auto_log_parameters: Optional[bool]
    comet_auto_log_cli_arguments: Optional[bool]
    hf_token: Optional[str]

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    pre_trained_bert_model: str = 'distilbert/distilbert-base-uncased'
    project_name: str = 'multiclass-text-classification'
    hf_dataset_registry: str = f'josecaloca/{project_name}-dataset'
    hf_model_registry: str = f'josecaloca/test-{project_name}-model'

    id2label: dict[int, str] = {
        0: 'Business',
        1: 'Science & Technology',
        2: 'Entertainment',
        3: 'Health',
    }
    label2id: dict[str, int] = {v: k for k, v in id2label.items()}

    random_state: int = 123
    frac_sample_reduction_training: float = 0.01
    frac_sample_reduction_validation: float = 0.01
    frac_sample_reduction_testing: float = 0.01


config = Config()
