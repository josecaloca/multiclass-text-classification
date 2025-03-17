import zipfile
from io import BytesIO
from typing import Tuple

import pandas as pd
import requests
from config import Config, config
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from loguru import logger
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from utils.paths import RAW_DATA_DIR
from utils.text_cleaner import clean_text


class DataProcessingPipeline:
    """
    A pipeline for processing the News Aggregator Dataset, including downloading,
    cleaning, tokenizing, and preparing the data for fine-tuning a BERT-type model
    and training an XGBoost model.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the data processing pipeline with necessary configurations.

        Args:
            config (Config): Configuration object containing data splitting ratios
                             and pre-trained model details.
        """
        logger.info('Initializing DataProcessingPipeline')
        self.train_ratio = config.train_ratio
        self.val_ratio = config.val_ratio
        self.test_ratio = config.test_ratio
        self.label_mapping = {
            'b': 0,
            't': 1,
            'e': 2,
            'm': 3,
        }  # Maps category labels to numerical values.
        self.tokenizer = AutoTokenizer.from_pretrained(config.pre_trained_bert_model)

    def download_and_extract_dataset(self) -> None:
        """
        Downloads the News Aggregator Dataset from the UCI repository and extracts it to the raw data directory.
        """
        logger.info('Downloading and extracting dataset')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip'
        response = requests.get(url)
        response.raise_for_status()

        with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)

        logger.info(f'Dataset downloaded and extracted to {RAW_DATA_DIR}')

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the raw dataset.
        """
        logger.info('Loading dataset into a DataFrame')
        news_columns = [
            'id',
            'title',
            'url',
            'publisher',
            'label',
            'story',
            'hostname',
            'timestamp',
        ]
        news_df = pd.read_csv(
            RAW_DATA_DIR / 'newsCorpora.csv', delimiter='\t', names=news_columns
        )
        logger.info(f'Loaded {len(news_df)} records.')
        return news_df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the dataset by sorting, mapping labels to numerical values, and cleaning text data.

        Args:
            df (pd.DataFrame): The raw dataset.

        Returns:
            pd.DataFrame: A cleaned dataset with only the required columns for training.
        """
        logger.info('Preprocessing data')
        df['timestamp'] = pd.to_datetime(
            df['timestamp'], unit='ms'
        )  # Convert timestamp to datetime.
        df = df.sort_values(by='timestamp')  # Sort by timestamp.
        df['label'] = df['label'].map(
            self.label_mapping
        )  # Convert labels to numeric values.
        df['title_prepared'] = df['title'].apply(clean_text)  # Clean text titles.
        df = df[['title_prepared', 'label']]
        logger.info(f'Data preprocessing complete. Shape: {df.shape}')
        return df

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset into training, validation, and test sets using stratified sampling.

        Args:
            df (pd.DataFrame): The preprocessed dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - train_df: Training dataset
                - val_df: Validation dataset
                - test_df: Test dataset
        """
        logger.info('Splitting data into train, validation, and test sets')
        train_df, temp_df = train_test_split(
            df, test_size=(1 - self.train_ratio), stratify=df['label'], random_state=123
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(self.test_ratio / (self.test_ratio + self.val_ratio)),
            stratify=temp_df['label'],
            random_state=123,
        )
        logger.info(
            f'Data split complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}'
        )
        return train_df, val_df, test_df

    def convert_to_dataset_dict(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> DatasetDict:
        """
        Converts DataFrames to a Hugging Face DatasetDict format for model training.

        Args:
            train_df (pd.DataFrame): Training dataset.
            val_df (pd.DataFrame): Validation dataset.
            test_df (pd.DataFrame): Test dataset.

        Returns:
            DatasetDict: A dictionary containing the datasets in Hugging Face's format.
        """
        logger.info('Converting DataFrames to Hugging Face DatasetDict')
        dataset_dict = DatasetDict(
            {
                'train': Dataset.from_pandas(train_df),
                'validation': Dataset.from_pandas(val_df),
                'test': Dataset.from_pandas(test_df),
            }
        )

        # Remove unnecessary index columns
        for split in dataset_dict:
            if '__index_level_0__' in dataset_dict[split].column_names:
                dataset_dict[split] = dataset_dict[split].remove_columns(
                    ['__index_level_0__']
                )

        logger.info('DatasetDict conversion complete.')
        return dataset_dict

    def preprocess_function(self, df: pd.DataFrame) -> dict:
        """
        Tokenizes dataset text using a pre-trained BERT tokenizer.

        Args:
            df (pd.DataFrame): The dataset containing preprocessed text.

        Returns:
            dict: Tokenized input representations suitable for transformer models.
        """
        return self.tokenizer(df['title_prepared'], padding=True, truncation=True)

    def process(self) -> None:
        """
        Runs the full data processing pipeline, including:
        - Downloading and extracting the dataset.
        - Loading data into a pandas DataFrame.
        - Preprocessing data.
        - Splitting data into train, validation, and test sets.
        - Converting data to Hugging Face DatasetDict format.
        - Tokenizing the dataset.
        - Uploading the processed dataset to the Hugging Face Hub.
        """
        logger.info('Starting full data processing pipeline')
        self.download_and_extract_dataset()
        news_df = self.load_data()
        news_df = self.preprocess_data(news_df)
        train_df, val_df, test_df = self.split_data(news_df)
        dataset_dict = self.convert_to_dataset_dict(train_df, val_df, test_df)

        logger.info('Tokenizing dataset')
        tokenized_dataset_dict = dataset_dict.map(
            self.preprocess_function, batched=True
        )

        logger.info('Pushing dataset to Hugging Face Hub')
        tokenized_dataset_dict.push_to_hub(config.hf_dataset_registry)

        logger.info('Data processing pipeline completed successfully')


if __name__ == '__main__':
    load_dotenv('settings.env')
    pipeline = DataProcessingPipeline(config=config)
    pipeline.process()
