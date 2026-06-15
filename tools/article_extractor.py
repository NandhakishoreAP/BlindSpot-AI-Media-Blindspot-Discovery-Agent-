# tools/article_extractor.py

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from models.data_models import ArticleData
from utils.logger import get_logger
from config import Config


class ArticleExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("article_extractor")

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

        self.timeout = 15

        self.category_titles = {
            "news",
            "world",
            "business",
            "sport",
            "sports",
            "technology",
            "politics",
            "health",
            "entertainment",
        }

    def validate_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)

            if parsed.scheme in ("http", "https") and parsed.netloc:
                return True

            self.logger.warning(
                f"URL validation failed (invalid scheme or missing netloc): {url}"
            )
            return False

        except Exception as e:
            self.logger.warning(
                f"URL validation failed due to parsing error: {url} | {e}"
            )
            return False

    def _extract_title(self, soup: BeautifulSoup) -> str:
        meta = soup.find("meta", property="og:title")

        if meta and meta.get("content"):
            title = " ".join(meta["content"].split()).strip()

            if title:
                return title

        h1 = soup.find("h1")

        if h1 and h1.get_text(strip=True):
            return " ".join(h1.get_text().split()).strip()

        title_tag = soup.find("title")

        if title_tag and title_tag.get_text(strip=True):
            return " ".join(title_tag.get_text().split()).strip()

        return "Unknown Title"

    def _extract_author(self, soup: BeautifulSoup) -> str:
        possible_authors = []

        selectors = [
            ("meta", {"name": "author"}),
            ("meta", {"property": "article:author"}),
        ]

        for tag_name, attrs in selectors:
            tag = soup.find(tag_name, attrs=attrs)

            if tag and tag.get("content"):
                possible_authors.append(tag["content"].strip())

        rel_author = soup.find(attrs={"rel": "author"})

        if rel_author:
            possible_authors.append(rel_author.get_text(strip=True))

        byline = soup.find(
            class_=lambda c: c and "byline" in str(c).lower()
        )

        if byline:
            possible_authors.append(byline.get_text(strip=True))

        author_el = soup.find(
            class_=lambda c: c and "author" in str(c).lower()
        )

        bbc_author = soup.find(
            attrs={
                "data-testid": lambda x: x and "byline" in str(x).lower()
            }
        )

        if bbc_author:
            possible_authors.append(
                bbc_author.get_text(" ", strip=True)
            )

        if author_el:
            possible_authors.append(author_el.get_text(strip=True))

        for author in possible_authors:

            if not author:
                continue

            author = author.strip()

            if author.startswith("http"):
                continue

            if any(
                social in author.lower()
                for social in [
                    "facebook",
                    "twitter",
                    "instagram",
                    "linkedin",
                    "youtube",
                ]
            ):
                continue

            return author

        return "Unknown"

    def _extract_date(self, soup: BeautifulSoup) -> str:
        meta_fields = [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"name": "publication_date"}),
            ("meta", {"name": "pubdate"}),
            ("meta", {"property": "og:updated_time"}),
        ]

        for tag_name, attrs in meta_fields:
            tag = soup.find(tag_name, attrs=attrs)

            if tag and tag.get("content"):
                return tag["content"].strip()

        time_tag = soup.find("time")

        if time_tag and time_tag.get("datetime"):
            return time_tag["datetime"].strip()

        return "Unknown"

    def _extract_content(self, soup: BeautifulSoup) -> str:
        content_soup = BeautifulSoup(str(soup), "lxml")

        for tag in content_soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
                "iframe",
                "noscript",
            ]
        ):
            tag.decompose()

        main_element = (
    content_soup.find("article")
    or content_soup.find("main")
    or content_soup.find(attrs={"role": "main"})
    or content_soup.find(
        class_=lambda c: c and "article-body" in str(c).lower()
    )
    or content_soup.find(
        class_=lambda c: c and "article-content" in str(c).lower()
    )
    or content_soup.find(
        class_=lambda c: c and "story-body" in str(c).lower()
    )
    or content_soup.find(
        class_=lambda c: c and "post-content" in str(c).lower()
    )
    or content_soup.find(
        class_=lambda c: c and "entry-content" in str(c).lower()
    )
)

        if not main_element:
            main_element = content_soup.find("body")

        if not main_element:
            main_element = content_soup

        paragraphs = []

        for p in main_element.find_all("p"):
            text = p.get_text(" ", strip=True)

            cleaned = re.sub(r"\s+", " ", text).strip()

            if len(cleaned) > 40:
                paragraphs.append(cleaned)

        return "\n\n".join(paragraphs)

    def extract(self, url: str) -> ArticleData:
        if not self.validate_url(url):
            raise ValueError(f"Invalid URL: {url}")

        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
            )

            self.logger.debug(
                f"HTTP {response.status_code} | {response.url}"
            )

            response.raise_for_status()

        except requests.HTTPError:

            if response.status_code == 401:
                raise PermissionError(
                    "Website blocked automated access."
                )

            if response.status_code == 403:
                raise PermissionError(
                    "Access forbidden by website."
                )

            if response.status_code == 404:
                raise ValueError(
                    "Article not found."
                )

            raise

        except requests.RequestException as e:
            raise ConnectionError(
                f"Network request failed for URL '{url}': {e}"
            ) from e

        soup = BeautifulSoup(response.text, "lxml")
        author = self._extract_author(soup)
        publication_date = self._extract_date(soup)
        content = self._extract_content(soup)

        title = self._extract_title(soup)
        title = " ".join(title.split())
        word_count = len(content.split())

        if (
    title.lower().strip() in self.category_titles
    and word_count < 300
):
            raise ValueError(
                "URL appears to be a category page, not a news article."
            )

        

        paywall_words = [
    "subscribe",
    "subscription",
    "sign in",
    "register to continue",
]
        lower_content = content.lower()

        if any(word in lower_content for word in paywall_words):
            self.logger.warning(
                "Possible paywall detected."
            )

        word_count = len(content.split())

        if word_count < 80:
            raise ValueError(
                "Insufficient article content extracted."
            )

        article = ArticleData(
            url=url,
            title=title,
            author=author,
            publication_date=publication_date,
            content=content,
        )

        self.logger.info(
            f"Successfully extracted article: "
            f"'{title}' "
            f"(word count: {article.word_count})"
        )

        return article


if __name__ == "__main__":
    import sys

    try:
        config = Config.from_env()

        extractor = ArticleExtractor(config)

        test_url = (
            "https://www.bbc.com/sport/football/articles/c0ly17770lpo"
        )

        print(f"Extracting article from: {test_url}")

        article = extractor.extract(test_url)

        print("\n--- Extraction Success ---")
        print(f"Title:      {article.title}")
        print(f"Author:     {article.author}")
        print(f"Date:       {article.publication_date}")
        print(f"Word Count: {article.word_count}")
        print(
            f"Content (first 300 chars):\n"
            f"{article.content[:300]}..."
        )
        print("--------------------------")

    except Exception as e:
        print(f"\nError occurred during extraction: {e}")
        sys.exit(1)

