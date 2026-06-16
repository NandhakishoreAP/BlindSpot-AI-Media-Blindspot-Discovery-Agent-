from typing import List
from urllib.parse import urlparse

from models.data_models import (
    ArticleData,
    ClaimAnalysis,
    Blindspot,
    SearchResult,
    Evidence
)
from llm.ollama_client import OllamaClient
from utils.logger import get_logger
from config import Config

class EvidenceEvaluator:
    """
    Evaluates external search results to determine their relationship to original article claims.
    """

    def __init__(self, ollama_client: OllamaClient, config: Config) -> None:
        """
        Initializes the EvidenceEvaluator with an OllamaClient and Config.
        """
        self.ollama_client: OllamaClient = ollama_client
        self.config: Config = config
        self.logger = get_logger("evidence_evaluator")

    def _build_system_prompt(self) -> str:
        """
        Return exactly the requested system prompt.
        """
        return """
You are an expert fact-checker and research analyst.
Your job is to evaluate search results and determine:

* Whether each result supports, contradicts, or adds context to an article's claims
* The quality of each source (High, Medium, Low)
* The key insight each result provides
  Focus on finding perspectives and evidence that were MISSING from the original article.
  You must respond ONLY with valid JSON. No explanations outside the JSON.
  """.strip()

    def _build_evaluation_prompt(
        self,
        article: ArticleData,
        claims: ClaimAnalysis,
        blindspots: List[Blindspot],
        search_results: List[SearchResult]
    ) -> str:
        """
        Constructs the user evaluation prompt outlining original claims, detected blindspots,
        and retrieved search results to evaluate.
        """
        key_claims_str = ""
        for idx, claim in enumerate(claims.key_claims, 1):
            key_claims_str += f"{idx}. {claim}\n"

        blindspots_str = ""
        for idx, bs in enumerate(blindspots, 1):
            blindspots_str += f"[{idx}] {bs.category}: {bs.description}\n"

        search_results_str = ""
        for idx, res in enumerate(search_results, 1):
            search_results_str += (
                f"[{idx}] Title: {res.title}\n"
                f"Source: {res.source}\n"
                f"URL: {res.url}\n"
                f"Summary: {res.snippet}\n\n"
            )

        prompt = (
            "ARTICLE\n\n"
            f"Title: {article.title}\n\n"
            f"Main Topic: {claims.main_topic}\n\n"
            f"Author Stance: {claims.author_stance}\n\n"
            f"Tone: {claims.tone}\n\n"
            "KEY CLAIMS\n\n"
            f"{key_claims_str}\n"
            "BLINDSPOTS\n\n"
            f"{blindspots_str}\n"
            "SEARCH RESULTS\n\n"
            f"{search_results_str}\n"
            "INSTRUCTIONS\n\n"
            "Evaluate every search result.\n\n"
            "For each result determine:\n\n"
            "1. relevance\n"
            "   Allowed values:\n"
            "   * Supports\n"
            "   * Contradicts\n"
            "   * Adds Context\n\n"
            "2. quality\n"
            "   Allowed values:\n"
            "   * High\n"
            "   * Medium\n"
            "   * Low\n\n"
            "3. key_insight\n"
            "One sentence describing the value of the result.\n\n"
            "4. include\n"
            "true if useful enough to include.\n\n"
            "Respond with ONLY this JSON format:\n\n"
            "{\n"
            '  "evaluations": [\n'
            "    {\n"
            '      "result_index": 1,\n'
            '      "relevance": "Supports",\n'
            '      "quality": "High",\n'
            '      "key_insight": "...",\n'
            '      "include": true\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "No markdown.\n\n"
            "No explanations.\n\n"
            "Only JSON."
        )
        return prompt

    def _infer_source_quality(self, result: SearchResult) -> str:
        """
        Determine source quality using source credibility indicators.
        """
        source = str(result.source).lower()
        url = str(result.url).lower()

        # High credibility indicators
        high_indicators = [
            ".gov", ".edu", "arxiv", "pubmed", "researchgate",
            "who.int", "un.org", "worldbank.org"
        ]
        
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            domain = ""

        for ind in high_indicators:
            if ind in source or ind in url or (domain and ind in domain):
                return "High"

        # Medium credibility (major news websites)
        medium_news = [
            "bbc.com", "bbc.co.uk", "nytimes.com", "reuters.com", "apnews.com",
            "bloomberg.com", "cnn.com", "theguardian.com", "guardian.co.uk",
            "economist.com", "wsj.com", "forbes.com", "npr.org", "scientificamerican.com",
            "nature.com", "dw.com", "aljazeera.com", "france24.com", "ft.com"
        ]
        for news in medium_news:
            if news in source or news in url or (domain and news in domain):
                return "Medium"

        return "Low"

    def _normalize_quality(
        self,
        llm_quality: str,
        inferred_quality: str
    ) -> str:
        """
        Normalizes the LLM provided quality against the inferred domain credibility,
        ensuring the final quality score does not exceed the inferred credibility.
        """
        ranking = {"Low": 1, "Medium": 2, "High": 3}
        llm_val = ranking.get(llm_quality, 1)
        inf_val = ranking.get(inferred_quality, 1)
        
        final_val = min(llm_val, inf_val)
        
        reverse_ranking = {1: "Low", 2: "Medium", 3: "High"}
        return reverse_ranking[final_val]

    def evaluate(
        self,
        article: ArticleData,
        claims: ClaimAnalysis,
        blindspots: List[Blindspot],
        search_results: List[SearchResult]
    ) -> List[Evidence]:
        """
        Evaluates search results and extracts evidence objects using Ollama.
        """
        try:
            # Empty search result protection
            if not search_results:
                self.logger.warning("No search results available for evaluation")
                return []

            self.logger.info(f"Evaluating {len(search_results)} search results")

            system_prompt = self._build_system_prompt()
            user_prompt = self._build_evaluation_prompt(article, claims, blindspots, search_results)

            self.logger.debug("Sending evidence evaluation request to Ollama")
            # Call Ollama Client with correct parameter names
            response = self.ollama_client.generate_json_with_retry(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2
            )

            self.logger.info("Received evaluation response")

            # Defensive JSON parsing for Cases A, B, and C
            evaluations = []
            if isinstance(response, dict):
                if "evaluations" in response and isinstance(response["evaluations"], list):
                    evaluations = response["evaluations"]
                elif "results" in response and isinstance(response["results"], list):
                    evaluations = response["results"]
                else:
                    # Generic lookup for any list value
                    for key, val in response.items():
                        if isinstance(val, list):
                            evaluations = val
                            break
            elif isinstance(response, list):
                evaluations = response

            self.logger.debug(f"Parsed {len(evaluations)} evaluation records")

            if not isinstance(evaluations, list):
                self.logger.warning("Evaluation response did not contain a valid list of evaluations.")
                return []

            evidence_items = []
            for item in evaluations:
                if not isinstance(item, dict):
                    continue

                # Filter items with include = False
                include_val = item.get("include")
                if include_val is False:
                    continue

                # Protection against missing/invalid result_index
                result_index = item.get("result_index")
                if result_index is None:
                    self.logger.warning("Evaluation missing result_index, skipping entry.")
                    continue

                try:
                    index = int(result_index) - 1
                    if index < 0 or index >= len(search_results):
                        self.logger.warning(
                            f"Evaluation result_index {result_index} out of range [1, {len(search_results)}], skipping entry."
                        )
                        continue
                except (ValueError, TypeError) as e:
                    self.logger.warning(
                        f"Evaluation result_index {result_index} could not be parsed as integer: {e}, skipping entry."
                    )
                    continue

                # Safely compile fields with default fallbacks
                relevance = item.get("relevance")
                if relevance not in ["Supports", "Contradicts", "Adds Context"]:
                    relevance = "Adds Context"

                inferred_quality = self._infer_source_quality(search_results[index])
                llm_quality = item.get("quality")
                if llm_quality not in ["High", "Medium", "Low"]:
                    llm_quality = inferred_quality
                quality = self._normalize_quality(llm_quality, inferred_quality)

                key_insight = item.get("key_insight", "")
                key_insight_str = str(key_insight).strip() if key_insight is not None else ""

                if not search_results[index].url:
                    self.logger.warning("Skipping evidence with empty URL")
                    continue

                evidence_obj = Evidence(
                    search_result=search_results[index],
                    relevance=relevance,
                    quality=quality,
                    key_insight=key_insight_str
                )
                evidence_items.append(evidence_obj)

            self.logger.info(f"Selected {len(evidence_items)} evidence items from {len(search_results)} search results")
            return evidence_items

        except Exception as error:
            self.logger.error(f"Evidence evaluation failed: {error}")
            return []

    def get_confidence_details(
        self,
        blindspots: List[Blindspot],
        evidence: List[Evidence]
    ) -> dict:
        """
        Calculates confidence score and detailed coverage analytics.
        """
        if not blindspots:
            return {
                "score": 0,
                "coverage_ratio": 0.0,
                "blindspot_count": 0,
                "evidence_count": len(evidence),
                "high_quality_count": 0,
                "medium_quality_count": 0,
                "low_quality_count": 0,
                "contradicting_count": 0
            }

        high_quality_count = 0
        medium_quality_count = 0
        low_quality_count = 0
        contradicting_count = 0
        base_score = 0

        for ev in evidence:
            q = ev.quality
            if q == "High":
                high_quality_count += 1
                base_score += 15
            elif q == "Medium":
                medium_quality_count += 1
                base_score += 8
            elif q == "Low":
                low_quality_count += 1
                base_score += 3

            if ev.relevance == "Contradicts":
                contradicting_count += 1
                base_score += 10

        coverage_ratio = min(
            len(evidence) / max(len(blindspots) * 2, 1),
            1.0
        )

        score = min(
            int(base_score * coverage_ratio) + len(evidence) * 2,
            100
        )

        return {
            "score": score,
            "coverage_ratio": coverage_ratio,
            "blindspot_count": len(blindspots),
            "evidence_count": len(evidence),
            "high_quality_count": high_quality_count,
            "medium_quality_count": medium_quality_count,
            "low_quality_count": low_quality_count,
            "contradicting_count": contradicting_count
        }

    def calculate_confidence(
        self,
        blindspots: List[Blindspot],
        evidence: List[Evidence]
    ) -> int:
        """
        Estimate confidence that enough external evidence has been collected.
        """
        details = self.get_confidence_details(blindspots, evidence)
        score = details["score"]
        self.logger.info(f"Confidence score: {score}%")
        self.logger.debug(f"Confidence details: {details}")
        return score

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        # Load config
        config = Config.from_env()

        # Create OllamaClient and EvidenceEvaluator
        ollama_client = OllamaClient(config)
        evaluator = EvidenceEvaluator(ollama_client, config)

        # Import SearchTool only here
        from tools.search_tool import SearchTool

        # Create dummy ArticleData, ClaimAnalysis, and Blindspot
        url = "https://test.com/carbon-tax"
        title = "Government Announces New Carbon Tax Policy"
        author = "Staff Reporter"
        publication_date = "2025-01-01"
        content = (
            "The government today officially announced the introduction of a new carbon tax policy. "
            "Under the plan, carbon emissions will be taxed starting next fiscal year to fight climate change."
        )

        article = ArticleData(
            url=url,
            title=title,
            author=author,
            publication_date=publication_date,
            content=content
        )

        claims = ClaimAnalysis(
            main_topic="Carbon tax policy announcement",
            key_claims=[
                "Government will introduce carbon tax",
                "Tax will reduce emissions by 30%",
                "Environmental groups support the policy"
            ],
            author_stance="Pro-regulation",
            tone="Optimistic",
            framing_summary="The article focuses on climate benefits and environmental groups' support."
        )

        blindspots = [
            Blindspot(
                category="Economic Impact",
                description="Missing economic impact data on small businesses.",
                importance="High",
                suggested_search_query="carbon tax economic impact small businesses"
            )
        ]

        # Run search query
        search_tool = SearchTool(config)
        search_query = "carbon tax economic impact small businesses"
        
        print("==================================================")
        print(f"RUNNING SEARCH FOR: '{search_query}'")
        print("==================================================")
        search_results = search_tool.search(search_query)
        print(f"Number of results found: {len(search_results)}")

        # Run evaluator
        print("\n==================================================")
        print("EVALUATING SEARCH RESULTS")
        print("==================================================")
        evidence_list = evaluator.evaluate(article, claims, blindspots, search_results)

        for idx, ev in enumerate(evidence_list, 1):
            print(f"\nEvidence {idx}:")
            print(f"Relevance: {ev.relevance}")
            print(f"Quality: {ev.quality}")
            print(f"Key Insight: {ev.key_insight}")
            print(f"Source URL: {ev.search_result.url}")
            print("-" * 30)

        # Run calculate_confidence
        print("\n==================================================")
        print("CALCULATING CONFIDENCE SCORE")
        print("==================================================")
        confidence = evaluator.calculate_confidence(blindspots, evidence_list)
        print(f"Confidence Score: {confidence}%")

        # Run get_confidence_details
        print("\n==================================================")
        print("CONFIDENCE ANALYTICS")
        print("====================")
        details = evaluator.get_confidence_details(blindspots, evidence_list)
        for key, val in details.items():
            print(f"{key}: {val}")

    except Exception as e:
        print(f"\nError during evidence evaluator execution: {e}")
