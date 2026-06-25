import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from .logger import get_logger


class ArticlePageValidator:
    """
    Validates that a URL + HTML represents a single article page,
    not a tag/category/archive/search/listing page.
    """

    def __init__(self):
        self.logger = get_logger("article_validator")

    def validate(self, url: str, html: str, title: str, content: str) -> dict:
        """
        Returns {"valid": True/False, "reason": "..."}
        """
        reason = None

        # 1. URL pattern checks
        parsed = urlparse(url)
        path = parsed.path.lower()
        url_patterns = [
            r'/tag/',
            r'/tags/',
            r'/category/',
            r'/categories/',
            r'/archive/',
            r'/archives/',
            r'/author/',
            r'/authors/',
            r'/page/\d+',
            r'\?page=',
            r'/search',
            r'\?s=',
            r'/topic/',
            r'/topics/',
            r'/section/',
            r'/sections/',
            r'/taxonomy/',
        ]
        for pat in url_patterns:
            if re.search(pat, path):
                reason = "URL matches tag/category/archive pattern"
                break

        if reason:
            self.logger.info(f"Article validation failed: {reason} | URL: {url}")
            return {"valid": False, "reason": reason}

        if not html:
            return {"valid": True, "reason": ""}

        soup = BeautifulSoup(html, "lxml")

        # 2. Multiple h1 check (category/listing pages often have >1 h1)
        h1_tags = soup.find_all("h1")
        if len(h1_tags) > 1:
            h1_texts = [h.get_text(strip=True).lower() for h in h1_tags if h.get_text(strip=True)]
            # If h1s are all different, likely a listing page
            if len(set(h1_texts)) > 1:
                reason = f"Multiple distinct h1 tags found ({len(h1_tags)}), indicating listing page"
                self.logger.info(f"Article validation failed: {reason}")
                return {"valid": False, "reason": reason}

        # 3. Article element check
        article_tag = soup.find("article")
        if not article_tag and len(content.split()) < 100:
            reason = "No <article> tag and short content, likely non-article page"
            self.logger.info(f"Article validation failed: {reason}")
            return {"valid": False, "reason": reason}

        # 4. Publication date check
        date_indicators = 0
        for tag in soup.find_all(["time", "meta"]):
            if tag.name == "time" and tag.get("datetime"):
                date_indicators += 1
            if tag.name == "meta" and tag.get("property") in (
                "article:published_time", "article:modified_time"
            ):
                date_indicators += 1

        # 5. Author check
        author_found = bool(
            soup.find("meta", attrs={"name": "author"}) or
            soup.find(attrs={"rel": "author"}) or
            soup.find(class_=lambda c: c and "author" in str(c).lower())
        )

        if not date_indicators and not author_found and len(content.split()) < 150:
            reason = "No publication date or author and short content, likely non-article"
            self.logger.info(f"Article validation failed: {reason}")
            return {"valid": False, "reason": reason}

        return {"valid": True, "reason": ""}
