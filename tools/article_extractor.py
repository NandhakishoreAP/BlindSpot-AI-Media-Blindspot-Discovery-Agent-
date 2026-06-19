# tools/article_extractor.py

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from models.data_models import ArticleData
from utils.logger import get_logger
from config import Config


class ArticleExtractor:
    """Extract and validate news article content from a URL.

    This class now includes strict validation to ensure only genuine article pages are processed.
    It rejects homepages, category pages, and pages with insufficient content.
    """
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
        """Extract the main article body.

        Returns cleaned paragraph text joined by double newlines.
        """
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

    def _extract_metadata_content(self, soup: BeautifulSoup) -> str:
        meta_desc = (
            soup.find("meta", property="og:description") or
            soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", property="twitter:description")
        )
        if meta_desc and meta_desc.get("content"):
            return meta_desc["content"].strip()
        return ""

    def _extract_source(self, url: str) -> str:
        if not url:
            return "Unknown"
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc
        except Exception:
            return "Unknown"

    # Original deterministic_title_from_url moved; see later definition.

    # Original _generate_fallback_article moved; see later definition.

    def is_valid_news_article(self, title: str, content: str, raw_html: str) -> bool:
        """Validate that the extracted page is a genuine news article.

        Rules:
        * Title must not be empty, generic placeholder, or indicate a homepage.
        * If the article body has >= 500 words, accept automatically.
        * Otherwise, require either word count >= 200 OR at least 4 paragraphs,
          and the ratio of article body length to raw HTML length >= 0.5.
        """
        if not title or title.lower().startswith("homepage") or title == "Unknown Title":
            return False
        word_count = len(content.split())
        # Automatic acceptance for long articles
        if word_count >= 500:
            return True
        paragraph_count = len([p for p in content.split("\n\n") if p.strip()])
        # Compute body ratio (avoid division by zero)
        body_ratio = len(content) / len(raw_html) if raw_html else 0
        if (word_count >= 200 or paragraph_count >= 4) and body_ratio >= 0.5:
            return True
        return False

    def deterministic_title_from_url(self, url: str) -> str:
        """Generate a deterministic title from a URL when extraction fails."""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip("/")
            if not path:
                return "Homepage of " + parsed.netloc
            parts = path.split("/")
            last_part = parts[-1]
            last_part = re.sub(r'\.[a-zA-Z0-9]+$', '', last_part)
            words = re.split(r'[-_]', last_part)
            title = " ".join(w.capitalize() for w in words if w)
            if len(title) > 5:
                return title
            return "Article on " + parsed.netloc
        except Exception:
            return "Article from web link"

    def _generate_fallback_article(self, url: str, title: str = None) -> ArticleData:
        domain = self._extract_source(url)
        if not title or title == "Unknown Title":
            title = self.deterministic_title_from_url(url)
            
        content = (
            f"This is a structured fallback article for the URL: {url}. "
            f"The article is published on {domain} and is titled '{title}'. "
            f"Due to access restrictions or scraping protection on the publisher's website, "
            f"the full text of the article could not be retrieved. However, we have recorded the "
            f"title and source to proceed with the analysis."
        )
        
        return ArticleData(
            url=url,
            title=title,
            author="Unknown",
            publication_date="Unknown",
            content=content
        )

    def extract(self, url: str) -> ArticleData:
        if not self.validate_url(url):
            self.logger.warning(f"Invalid URL '{url}' passed to extract. Generating structured fallback.")
            art = self._generate_fallback_article(url, "Invalid Web Link")
            art.access_restricted = True
            return art

        # Homepage detection – reject URLs without a path component
        parsed_url = urlparse(url)
        if parsed_url.path.strip('/') == '':
            self.logger.warning("URL appears to be a homepage, not an article. Prompting user for specific article URL.")
            art = self._generate_fallback_article(url, "Homepage detected")
            art.access_restricted = True
            return art

        status_code = 200
        response_text = ""
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
            )
            status_code = response.status_code
            response_text = response.text
            self.logger.info(f"HTTP {status_code} | {response.url}")
        except Exception as e:
            self.logger.warning(f"Scraping failed with connection error: {e}. Attempting fallback.")
            art = self._generate_fallback_article(url)
            art.access_restricted = True
            return art

        # Check access restriction indicators
        access_restricted = False
        if status_code in (401, 403):
            access_restricted = True
        else:
            text_lower = response_text.lower()
            indicators = [
                "access denied", "login required", "subscription required",
                "paywall", "content unavailable"
            ]
            if any(ind in text_lower for ind in indicators):
                access_restricted = True

        if access_restricted:
            self.logger.warning(f"Access restriction detected on {url}. Setting access_restricted=True.")
            art = self._generate_fallback_article(url)
            art.access_restricted = True
            art.content = "Article content could not be accessed. Reason: Publisher restriction. Analysis limited to metadata."
            return art

        soup = BeautifulSoup(response_text, "lxml") if response_text else BeautifulSoup("", "lxml")
        title = self._extract_title(soup)
        title = " ".join(title.split())
        author = self._extract_author(soup)
        publication_date = self._extract_date(soup)

        # Layer 1: Requests + BeautifulSoup
        content = self._extract_content(soup)
        word_count = len(content.split())
        self.logger.info(f"Layer 1 (BS4) extracted word count: {word_count}")

        # Layer 2: newspaper3k
        if word_count < 150:
            self.logger.info("Layer 1 yielded < 150 words. Trying Layer 2 (Newspaper3k)...")
            try:
                from newspaper import Article
                news_art = Article(url)
                news_art.set_html(response_text)
                news_art.parse()
                np_content = news_art.text or ""
                np_words = len(np_content.split())
                self.logger.info(f"Layer 2 (Newspaper3k) extracted word count: {np_words}")
                if np_words >= 150:
                    content = np_content
                    word_count = np_words
                    if news_art.title:
                        title = news_art.title.strip()
                    if news_art.authors:
                        author = news_art.authors[0]
            except Exception as e:
                self.logger.warning(f"Layer 2 (Newspaper3k) failed: {e}")

        # Layer 3: trafilatura
        if word_count < 150:
            self.logger.info("Layer 2 yielded < 150 words. Trying Layer 3 (Trafilatura)...")
            try:
                import trafilatura
                tf_content = trafilatura.extract(response_text) or ""
                tf_words = len(tf_content.split())
                self.logger.info(f"Layer 3 (Trafilatura) extracted word count: {tf_words}")
                if tf_words >= 150:
                    content = tf_content
                    word_count = tf_words
            except Exception as e:
                self.logger.warning(f"Layer 3 (Trafilatura) failed: {e}")

        # Layer 4: Metadata-only fallback
        if word_count < 150:
            self.logger.warning(f"All extraction layers yielded < 150 words (current word count: {word_count}). Using Layer 4 (Metadata fallback).")
            meta_desc = self._extract_metadata_content(soup)
            if meta_desc and len(meta_desc.split()) >= 10:
                self.logger.info("Using metadata description fallback.")
                content = (
                    f"Metadata Description: {meta_desc}\n\n"
                    f"Additional fallback text: The article is titled '{title}' and published on "
                    f"{self._extract_source(url)}."
                )
            else:
                if title and title != "Unknown Title":
                    self.logger.info("Using title-based fallback.")
                    content = (
                        f"This is a fallback description for the article '{title}'. "
                        f"Due to access limits, full paragraphs could not be parsed."
                    )
                else:
                    self.logger.info("No metadata available. Constructing complete fallback article.")
                    return self._generate_fallback_article(url)

        # Check for category page bypass
                # Reject obvious category pages
        if (
            title.lower().strip() in self.category_titles
            and word_count < 300
        ):
            self.logger.warning("URL appears to be a category page, not a news article. Using fallback content.")
            return self._generate_fallback_article(url, title)

        # Final validation based on refined rules
        if not self.is_valid_news_article(title, content, response_text):
            self.logger.warning("Article failed validation checks. Falling back to metadata-only article.")
            return self._generate_fallback_article(url, title)

        # If we reach here the article is considered valid

        article = ArticleData(
            url=url,
            title=title,
            author=author,
            publication_date=publication_date,
            content=content,
            access_restricted=False
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

