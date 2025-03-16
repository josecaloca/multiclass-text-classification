import re

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text: str) -> str:
    """
    Cleans text by removing links, tags, punctuation, and stopwords. Ensures the text is lowercase
    Args:
        text (str): The input text.
    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()
    return text
