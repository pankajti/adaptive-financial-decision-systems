import yfinance as yf
from datetime import datetime, timedelta, timezone


def get_historical_news(ticker, days=30, limit=100):
    """
    Fetch historical news for a given ticker symbol.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        days (int): Number of days of historical news to retrieve
        limit (int): Max number of news items to fetch

    Returns:
        list: List of normalized news articles
    """
    try:
        stock = yf.Ticker(ticker)

        # Fetch news (yfinance returns list of dicts)
        news = stock.get_news() or []

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        normalized_news = []

        for article in news[:limit]:
            # yfinance typically uses 'providerPublishTime' (epoch seconds)
            ts = article.get("providerPublishTime")

            if ts is None:
                continue

            published_dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            if published_dt < cutoff_date:
                continue

            normalized_news.append({
                "ticker": ticker,
                "title": article.get("title"),
                "publisher": article.get("publisher"),
                "link": article.get("link"),
                "published_time": published_dt,
                "type": article.get("type"),
                "uuid": article.get("uuid")
            })

        return normalized_news

    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []


def print_news(ticker, days=30):
    """Print formatted historical news."""
    news_list = get_historical_news(ticker, days)

    print(f"\n{'='*60}")
    print(f"Recent news for {ticker} (last {days} days)")
    print(f"{'='*60}\n")

    if not news_list:
        print("No news found.\n")
        return

    for i, article in enumerate(news_list[:10], 1):
        print(f"{i}. {article.get('title', 'N/A')}")
        print(f"   Source: {article.get('publisher', 'N/A')}")
        print(f"   Time  : {article.get('published_time')}")
        print(f"   Link  : {article.get('link', 'N/A')}\n")


if __name__ == "__main__":
    print_news("AAPL", days=30)