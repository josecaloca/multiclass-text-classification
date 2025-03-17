import re

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("wordnet")


def clean_text(text: str) -> str:
    """
    Cleans a given text string by performing the following operations:
    - Converts text to lowercase.
    - Removes URLs and HTML tags.
    - Removes punctuation and non-alphanumeric characters.
    - Eliminates English stopwords.
    - Normalizes whitespace.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text, stripped of unnecessary elements.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text
