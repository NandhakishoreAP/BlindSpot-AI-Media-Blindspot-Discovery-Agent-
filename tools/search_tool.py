from typing import List
import time
import json
import re
from pathlib import Path
from urllib.parse import urlparse

# pyrefly: ignore [missing-import]
from ddgs import DDGS

from models.data_models import SearchResult
from utils.logger import get_logger
from config import Config

class SearchTool:
    """
    Retrieves and cleans search results from DuckDuckGo to gather evidence for analysis.
    """

    def __init__(self, config: Config) -> None:
        """
        Store config, create search_tool logger, and store max_results.
        """
        self.config: Config = config
        self.logger = get_logger("search_tool")
        self.max_results: int = config.MAX_SEARCH_RESULTS
        self.logger.info(f"SearchTool initialized. Max results: {self.max_results}")

    def _get_cache_path(self) -> Path:
        reports_dir = getattr(self.config, 'REPORTS_DIR', 'reports')
        return Path(reports_dir) / "search_cache.json"

    def _load_cache(self) -> dict:
        path = self._get_cache_path()
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self, cache: dict) -> None:
        path = self._get_cache_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _normalize_query(self, query: str) -> str:
        if not query:
            return ""
        q = query.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)
        return " ".join(q.split())

    def _check_cache(self, normalized_query: str) -> List[SearchResult]:
        cache = self._load_cache()
        entry = cache.get(normalized_query)
        if entry:
            timestamp = entry.get("timestamp", 0)
            # 24 hours TTL
            if time.time() - timestamp < 86400:
                self.logger.info(f"Cache hit for query: '{normalized_query}'")
                results = []
                for r in entry.get("results", []):
                    results.append(SearchResult(**r))
                return results
            else:
                self.logger.info(f"Cache expired (TTL exceeded) for query: '{normalized_query}'")
        self.logger.info(f"Cache miss for query: '{normalized_query}'")
        return None

    def _write_to_cache(self, normalized_query: str, results: List[SearchResult]) -> None:
        cache = self._load_cache()
        serialized_results = [r.model_dump() for r in results]
        cache[normalized_query] = {
            "timestamp": time.time(),
            "results": serialized_results
        }
        self._save_cache(cache)

    def _clean_snippet(self, text: str) -> str:
        """
        Clean raw text snippet by stripping whitespace, replacing multiple spaces,
        truncating to 300 characters, and appending '...' if truncated.
        """
        if text is None or not str(text).strip():
            return ""
        
        cleaned = str(text).strip()
        cleaned = " ".join(cleaned.split())
        
        if len(cleaned) > 300:
            cleaned = cleaned[:300] + "..."
            
        return cleaned

    def _extract_source(self, url: str) -> str:
        """
        Extract domain using urlparse, removing 'www.' prefix if present.
        """
        if not url:
            return "Unknown"
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc
            if not netloc:
                return "Unknown"
            
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc
        except Exception:
            return "Unknown"

    def _is_valid_url(self, url: str) -> bool:
        """
        Returns True if the url starts with http:// or https://.
        """
        if not url:
            return False
        return url.startswith("http://") or url.startswith("https://")

    @staticmethod
    def classify_source_type(url: str) -> str:
        """
        Classifies evidence source url into news, government, academic, research, think tank,
        industry, social media, forums, blog, or other.
        """
        if not url:
            return "other"
        url_lower = url.lower()
        try:
            parsed = urlparse(url_lower)
            netloc = parsed.netloc
            if netloc.startswith("www."):
                netloc = netloc[4:]
        except Exception:
            netloc = url_lower
            parsed = None

        government_indicators = [".gov", "who.int", "un.org", "worldbank.org", "imf.org"]
        academic_indicators = [".edu", "arxiv", "pubmed", "doi.org"]
        research_indicators = ["researchgate.net", "researchgate", "scholar.google", "springer", "sciencedirect", "ieee.org", "wiley.com", "nature.com", "science.org", "ssrn.com"]
        think_tank_indicators = ["cato.org", "brookings.edu", "heritage.org", "rand.org", "csis.org", "pewresearch.org", "chathamhouse.org", "cfr.org"]
        social_media_indicators = ["twitter.com", "x.com", "facebook.com", "linkedin.com", "youtube.com", "instagram.com"]
        forum_indicators = ["forum", "board", "quora.com", "reddit.com", "reddit", "stackexchange.com", "stackoverflow.com", "discussions", "group"]
        blog_indicators = ["blogspot.com", "medium.com", "substack.com", "wordpress.com", "blog."]
        news_domains = [
            "bbc.com", "bbc.co.uk", "nytimes.com", "reuters.com", "apnews.com",
            "bloomberg.com", "cnn.com", "theguardian.com", "guardian.co.uk",
            "economist.com", "wsj.com", "forbes.com", "npr.org", "dw.com",
            "aljazeera.com", "france24.com", "ft.com"
        ]
        industry_indicators = ["mckinsey.com", "gartner.com", "deloitte.com", "accenture.com", "pwc.com", "ey.com", "industry", "marketwatch.com", "nasdaq.com"]

        if any(ind in netloc for ind in government_indicators):
            return "government"
        if any(ind in netloc for ind in academic_indicators):
            return "academic"
        if any(ind in netloc for ind in research_indicators):
            return "research"
        if any(ind in netloc for ind in think_tank_indicators):
            return "think_tank"
        if any(ind in netloc for ind in social_media_indicators):
            return "social_media"
        if any(ind in netloc for ind in forum_indicators):
            return "forums"
        if any(ind in netloc for ind in blog_indicators) or (parsed and "blog" in parsed.path):
            return "blog"
        if any(news in netloc for news in news_domains):
            return "news"
        if any(ind in netloc for ind in industry_indicators):
            return "industry"

        return "other"

    def _score_result(self, result: SearchResult) -> float:
        score = 0.0
        source_type = self.classify_source_type(result.url)
        if source_type == "government":
            score += 50.0
        elif source_type == "academic":
            score += 50.0
        elif source_type == "research":
            score += 45.0
        elif source_type == "think_tank":
            score += 40.0
        elif source_type == "news":
            score += 30.0
        elif source_type == "industry":
            score += 20.0
        elif source_type == "blog":
            score -= 20.0
        elif source_type == "forums":
            score -= 40.0
        elif source_type == "social_media":
            score -= 40.0
            
        # Additional checks
        url_lower = result.url.lower()
        
        # Deprioritize forums and social/blog platforms not covered by classify_source_type
        forum_indicators = ["forum", "board", "quora.com", "reddit", "stackexchange", "stackoverflow", "discussions", "group"]
        if any(ind in url_lower for ind in forum_indicators) and source_type != "forums":
            score -= 40.0
            
        content_farm_indicators = ["slideshare", "pinterest", "ehow", "wikihow", "about.com", "hubpages", "answers.com"]
        if any(ind in url_lower for ind in content_farm_indicators):
            score -= 40.0
            
        return score

    def _is_cacheable(self, query: str) -> bool:
        if not query:
            return False
        q_lower = query.lower()
        generic_terms = {
            "placeholder", "query", "search", "google", "website", "duckduckgo",
            "url", "link", "error", "access denied", "login required", "paywall",
            "forbidden", "content unavailable", "there was a problem"
        }
        if any(t in q_lower for t in generic_terms):
            return False
        if len(q_lower.split()) < 3:
            return False
        return True

    def search(self, query: str) -> List[SearchResult]:
        """
        Executes search using DuckDuckGo and returns a list of cleaned SearchResult objects.
        """
        normalized_q = self._normalize_query(query)
        cacheable = self._is_cacheable(normalized_q)
        
        if cacheable:
            cached_res = self._check_cache(normalized_q)
            if cached_res is not None:
                return cached_res

        try:
            self.logger.info(f"Searching: {query}")
            
            results = []
            filtered_count = 0
            seen_snippets = set()
            seen_titles = set()
            
            with DDGS() as ddgs:
                ddg_results = ddgs.text(
                    query,
                    max_results=self.max_results * 5
                )
                
                if ddg_results:
                    for r in ddg_results:
                        title = r.get("title", "")
                        url = r.get("href", "")
                        body = r.get("body", "")
                        
                        title_str = str(title).strip() if title is not None else ""
                        url_str = str(url).strip() if url is not None else ""
                        body_str = str(body).strip() if body is not None else ""
                        
                        if not title_str or not body_str or not url_str:
                            filtered_count += 1
                            continue
                            
                        if not self._is_valid_url(url_str):
                            filtered_count += 1
                            continue
                            
                        # Reject duplicate content
                        norm_body = " ".join(body_str.lower().split())[:100]
                        norm_title = " ".join(title_str.lower().split())
                        if norm_body in seen_snippets or norm_title in seen_titles:
                            filtered_count += 1
                            continue
                            
                        # Reject thin content (less than 5 words)
                        if len(body_str.split()) < 5:
                            filtered_count += 1
                            continue
                            
                        # Reject spam
                        spam_terms = ["sponsored", "advertisement", "click here", "buy now", "pinterest.com", "slideshare"]
                        if any(term in title_str.lower() or term in body_str.lower() or term in url_str.lower() for term in spam_terms):
                            filtered_count += 1
                            continue

                        # Reject unrelated pages (must share at least one keyword with the query)
                        stop_words = {"criticism", "expert", "opinion", "implementation", "challenges", "unintended", "consequences", "stakeholder", "response", "academic", "analysis", "historical", "outcomes", "and", "the", "for", "with", "about"}
                        query_words = {w.lower() for w in re.findall(r'\b\w+\b', query) if w.lower() not in stop_words and len(w) > 2}
                        res_words = {w.lower() for w in re.findall(r'\b\w+\b', title_str + " " + body_str)}
                        if query_words and not query_words.intersection(res_words):
                            filtered_count += 1
                            continue
                            
                        url_lower = url_str.lower()
                        reject_patterns = [
                            "login", "signin", "signup", "/tag/",
                            "/category/", "/search", "/author/", "/page/"
                        ]
                        if any(pat in url_lower for pat in reject_patterns):
                            filtered_count += 1
                            continue
                            
                        seen_snippets.add(norm_body)
                        seen_titles.add(norm_title)
                        
                        cleaned_snippet = self._clean_snippet(body_str)
                        domain = self._extract_source(url_str)
                        
                        res_obj = SearchResult(
                            query=query,
                            title=title_str,
                            url=url_str,
                            snippet=cleaned_snippet,
                            source=domain
                        )
                        results.append(res_obj)
                        
            # Score and sort results descending by score
            results.sort(key=self._score_result, reverse=True)
            
            # Limit same-domain results to at most 2
            domain_counts = {}
            diverse_results = []
            for r in results:
                domain = r.source
                count = domain_counts.get(domain, 0)
                if count < 2:
                    diverse_results.append(r)
                    domain_counts[domain] = count + 1
                else:
                    filtered_count += 1
            
            if filtered_count > 0:
                self.logger.info(f"Filtered {filtered_count} irrelevant/duplicate/thin/spam/same-domain results")
                
            final_results = diverse_results[:self.max_results]
            if cacheable:
                self._write_to_cache(normalized_q, final_results)
            self.logger.info(f"Found {len(final_results)} results for: {query}")
            return final_results
            
        except Exception as e:
            self.logger.exception(f"Search failed for query: {query}")
            return []

    def search_multiple(
        self,
        queries: List[str],
        delay: float = 0.5
    ) -> List[SearchResult]:
        """
        Execute multiple searches sequentially avoiding duplicates and rate limiting.
        """
        seen_urls = set()
        all_results = []
        domain_counts = {}
        
        for query in queries:
            query_results = self.search(query)
            
            for res in query_results:
                norm_url = res.url.rstrip("/")
                if norm_url not in seen_urls:
                    domain = res.source
                    count = domain_counts.get(domain, 0)
                    if count < 2:
                        seen_urls.add(norm_url)
                        all_results.append(res)
                        domain_counts[domain] = count + 1
            
            time.sleep(delay)
            
        self.logger.info(f"Multi-search complete. Total unique results: {len(all_results)}")
        return all_results

    def format_results_for_llm(
        self,
        results: List[SearchResult]
    ) -> str:
        """
        Format SearchResult objects into a simple text block without URLs.
        """
        if not results:
            return "No search results found."
            
        formatted_parts = []
        for idx, res in enumerate(results, 1):
            part = (
                f"Result {idx}\n"
                f"Title: {res.title}\n"
                f"Source: {res.source}\n"
                f"Summary: {res.snippet}"
            )
            formatted_parts.append(part)
            
        return "\n\n".join(formatted_parts)

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        # Load config
        config = Config.from_env()
        
        # Create tool
        search_tool = SearchTool(config)
        
        # Run single search
        single_query = "carbon tax economic impact peer reviewed studies"
        print("==================================================")
        print(f"RUNNING SINGLE SEARCH FOR: '{single_query}'")
        print("==================================================")
        results = search_tool.search(single_query)
        
        for idx, res in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(f"Title: {res.title}")
            print(f"Source: {res.source}")
            print(f"Snippet (First 100 chars): {res.snippet[:100]}")
            print("-" * 30)
            
        # Run multi-search
        multi_queries = [
            "carbon tax benefits environment",
            "carbon tax negative effects businesses"
        ]
        print("\n==================================================")
        print("RUNNING MULTI-SEARCH")
        print("==================================================")
        multi_results = search_tool.search_multiple(multi_queries, delay=1.5)
        print(f"\nTotal unique results: {len(multi_results)}")
        
        # Run format results for LLM
        print("\n==================================================")
        print("FORMATTED RESULTS FOR LLM")
        print("==================================================")
        formatted = search_tool.format_results_for_llm(multi_results)
        print(formatted)
        
    except Exception as err:
        print(f"\nError during search tool execution: {err}")
