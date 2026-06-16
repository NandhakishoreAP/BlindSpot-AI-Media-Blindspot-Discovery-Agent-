from typing import List
import time
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

    def search(self, query: str) -> List[SearchResult]:
        """
        Executes search using DuckDuckGo and returns a list of cleaned SearchResult objects.
        """
        try:
            self.logger.info(f"Searching: {query}")
            
            results = []
            with DDGS() as ddgs:
                ddg_results = ddgs.text(
                    query,
                    max_results=self.max_results
                )
                
                if ddg_results:
                    for r in ddg_results:
                        title = r.get("title", "")
                        url = r.get("href", "")
                        body = r.get("body", "")
                        
                        title_str = str(title).strip() if title is not None else ""
                        url_str = str(url).strip() if url is not None else ""
                        body_str = str(body).strip() if body is not None else ""
                        
                        if not self._is_valid_url(url_str):
                            continue
                            
                        if not title_str and not body_str:
                            continue
                            
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
                        
            final_results = results[:self.max_results]
            self.logger.info(f"Found {len(final_results)} results for: {query}")
            return final_results
            
        except Exception as e:
            self.logger.exception(f"Search failed for query: {query}")
            return []

    def search_multiple(
        self,
        queries: List[str],
        delay: float = 1.5
    ) -> List[SearchResult]:
        """
        Execute multiple searches sequentially avoiding duplicates and rate limiting.
        """
        seen_urls = set()
        all_results = []
        
        for query in queries:
            query_results = self.search(query)
            
            for res in query_results:
                norm_url = res.url.rstrip("/")
                if norm_url not in seen_urls:
                    seen_urls.add(norm_url)
                    all_results.append(res)
            
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
