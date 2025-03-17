import requests
from loguru import logger

NEWS_CLASSIFICATION_API = 'http://127.0.0.1:8000/predict'
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines?country=us&apiKey=65db60009cc54c419769177c84e7eb24'


def classify_news(title: str):
    """Send a news title to the classification API and return the response."""
    payload = {'title': title}
    try:
        response = requests.post(NEWS_CLASSIFICATION_API, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f'Error sending request to classification API: {e}')
        return None


def fetch_news_articles():
    """Fetch top news headlines from the News API."""
    try:
        response = requests.get(NEWS_API_URL)
        response.raise_for_status()
        data = response.json()
        return [
            article['title']
            for article in data.get('articles', [])
            if 'title' in article
        ]
    except requests.exceptions.RequestException as e:
        logger.error(f'Error fetching news articles: {e}')
        return []


def main():
    """Fetch news headlines and classify them using the model API."""
    titles = fetch_news_articles()

    if not titles:
        logger.warning('No news articles found.')
        return

    for title in titles:
        logger.info(f'Classifying title: {title}')
        result = classify_news(title)
        if result:
            logger.info(f'Prediction: {result}')
        else:
            logger.warning('Failed to classify title.')


if __name__ == '__main__':
    main()
