from typing import List
from urllib.parse import urlparse
import re
import json
import traceback

try:
    from rapidfuzz import fuzz
except ImportError:
    class FuzzFallback:
        @staticmethod
        def partial_ratio(s1: str, s2: str) -> float:
            import difflib
            s1 = s1.lower()
            s2 = s2.lower()
            if s1 in s2:
                return 100.0
            len_s1 = len(s1)
            len_s2 = len(s2)
            if len_s1 == 0 or len_s2 == 0:
                return 0.0
            if len_s1 > len_s2:
                s1, s2 = s2, s1
                len_s1, len_s2 = len_s2, len_s1
            max_ratio = 0.0
            for i in range(len_s2 - len_s1 + 1):
                sub = s2[i:i+len_s1]
                r = difflib.SequenceMatcher(None, s1, sub).ratio() * 100.0
                if r > max_ratio:
                    max_ratio = r
            return max_ratio
    fuzz = FuzzFallback()

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
        claims: ClaimAnalysis,
        blindspots: List[Blindspot],
        search_results: List[SearchResult]
    ) -> str:
        """
        Constructs a simplified evaluation prompt outlining original claims, detected blindspots,
        and retrieved search results to evaluate.
        """
        key_claims_str = ""
        for idx, claim in enumerate(claims.key_claims, 1):
            key_claims_str += f"- {claim}\n"

        blindspots_str = ""
        for idx, bs in enumerate(blindspots, 1):
            blindspots_str += f"- {bs.category}: {bs.description}\n"

        search_results_str = ""
        for idx, res in enumerate(search_results, 1):
            search_results_str += (
                f"Result {idx}:\n"
                f"Title: {res.title}\n"
                f"URL: {res.url}\n"
                f"Source: {res.source}\n"
                f"Snippet: {res.snippet}\n\n"
            )

        prompt = (
            f"Main Topic: {claims.main_topic if claims else 'Unknown'}\n\n"
            "Key Claims:\n"
            f"{key_claims_str}\n"
            "Blindspots:\n"
            f"{blindspots_str}\n"
            "Search Results:\n"
            f"{search_results_str}\n"
            "INSTRUCTIONS:\n"
            "Evaluate each search result. Respond ONLY as a JSON list matching this format:\n"
            "[\n"
            "  {\n"
            '    "relevance": "Supports | Contradicts | Adds Context",\n'
            '    "relevance_score": 0-100 integer score (High if it addresses a blindspot, contradicts framing, or provides missing context; Low if it repeats article claims or is irrelevant),\n'
            '    "quality": "High | Medium | Low",\n'
            '    "key_insight": "1-sentence summary",\n'
            '    "evidence_summary": "1-2 sentence explanation of how this search result relates to the article and the blindspot"\n'
            "  }\n"
            "]\n"
            "IMPORTANT: Keep all text values extremely concise. "
            "No explanations or markdown blocks."
        )
        return prompt

    def heuristic_evaluate(
        self,
        article: ArticleData,
        claims: ClaimAnalysis,
        blindspots: List[Blindspot],
        search_results: List[SearchResult]
    ) -> List[Evidence]:
        """
        Rule-based keyword/sentiment matcher to programmatically construct evidence structures.
        Bypasses LLM request entirely.
        """
        self.logger.info("Executing rule-based heuristic evidence evaluation fallback")
        evidence_items = []
        
        for idx, res in enumerate(search_results):
            # 1. Quality estimation based on source credibility
            quality = self._infer_source_quality(res)
            
            # 2. Key insight: truncate snippet or title to a clean concise sentence
            insight = res.snippet[:120] if res.snippet else "Relevant search result data"
            if len(insight) < 15 and res.title:
                insight = res.title[:120]
                
            # 3. Determine relevance: check for contradiction keywords
            text_lower = (res.title + " " + res.snippet).lower()
            contradict_indicators = [
                "criticism", "criticized", "oppose", "concern", "fail", 
                "drawback", "downside", "negative", "limitation", "challenges",
                "disagree", "dispute", "protest", "restrict", "harm"
            ]
            
            relevance = "Adds Context"
            if any(indicator in text_lower for indicator in contradict_indicators):
                relevance = "Contradicts"
            else:
                support_indicators = ["support", "confirm", "prove", "benefit", "improve", "successful"]
                if any(indicator in text_lower for indicator in support_indicators):
                    relevance = "Supports"
            
            # 4. Relevance score logic matching the main pipeline guidelines
            score = 50
            if relevance == "Contradicts":
                score += 30
            elif relevance == "Adds Context":
                score += 10
                
            # Relaxed triple match: topic AND (claim OR blindspot)
            tm_valid, tm_reason, matched_claim_text, matched_bs_cat, match_score = self._triple_match(
                evidence_insight=insight,
                search_result=res,
                claims=claims,
                blindspots=blindspots
            )
            if not tm_valid:
                self.logger.info(f"Heuristic evidence discarded ({tm_reason}): URL={res.url}")
                continue
                
            score += 20
            score = max(0, min(100, score))
            
            ev = Evidence(
                search_result=res,
                relevance=relevance,
                quality=quality,
                key_insight=insight,
                relevance_score=score,
                evidence_summary=f"External source {relevance.lower()} on the topic. Key insight: {insight}",
                related_blindspot=matched_bs_cat,
                match_score=match_score,
                linked_claim=matched_claim_text,
                linked_blindspot=matched_bs_cat,
                matched_claim=matched_claim_text,
                matched_blindspot=matched_bs_cat
            )
            self.logger.info(
                f"Heuristic Evidence created: quality={ev.quality}, relevance={ev.relevance}, score={score}"
            )
            evidence_items.append(ev)
            
        return evidence_items

    def safe_json_parse(self, text: str) -> List[dict]:
        """
        Parses a JSON string containing a list of dicts, or extracts individual dicts
        using brace matching and regular expressions, repairing quotes and commas.
        Collects successfully parsed records in a list to prevent a single malformed
        object from corrupting the entire set.
        """
        if not text:
            return []
        
        # Clean thinking blocks
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        
        parsed_records = []
        
        # Try direct JSON parsing
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        parsed_records.append(item)
                if parsed_records:
                    return parsed_records
            elif isinstance(parsed, dict):
                return [parsed]
        except Exception as e:
            self.logger.warning(f"JSON parse failed (evidence_evaluator.py:231): {traceback.format_exc()}")
            
        # Try finding markdown code block
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if markdown_match:
            candidate_text = markdown_match.group(1).strip()
            try:
                parsed = json.loads(candidate_text)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            parsed_records.append(item)
                    if parsed_records:
                        return parsed_records
                elif isinstance(parsed, dict):
                    return [parsed]
            except Exception as e:
                self.logger.warning(f"JSON regex extract failed (evidence_evaluator.py:248): {traceback.format_exc()}")
        else:
            candidate_text = text
            
        # Extract balanced curly brace blocks
        blocks = []
        stack = []
        start_idx = -1
        for i, char in enumerate(candidate_text):
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append('{')
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx != -1:
                        blocks.append(candidate_text[start_idx:i+1])
                        start_idx = -1
                        
        if not blocks:
            # Fallback to simple regex if balanced scanner found nothing
            blocks = re.findall(r"\{.*?\}", candidate_text, re.DOTALL)
            
        for block in blocks:
            cleaned_block = block.strip()
            
            # Try to load directly
            try:
                parsed_obj = json.loads(cleaned_block)
                if isinstance(parsed_obj, dict):
                    parsed_records.append(parsed_obj)
                    continue
            except Exception as e:
                self.logger.warning(f"JSON clean extract failed (evidence_evaluator.py:282): {traceback.format_exc()}")
                
            # Attempt repairs on individual block
            try:
                rep = cleaned_block.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
                rep = rep.replace('`', '')
                rep = re.sub(r',\s*}', '}', rep)
                parsed_obj = json.loads(rep)
                if isinstance(parsed_obj, dict):
                    parsed_records.append(parsed_obj)
            except Exception as e:
                self.logger.warning(f"Repair JSON parse failed (evidence_evaluator.py:294): {e}")
                # Use repair_json_string from ollama_client if possible
                try:
                    repaired = self.ollama_client.repair_json_string(cleaned_block)
                    parsed_obj = json.loads(repaired)
                    if isinstance(parsed_obj, dict):
                        parsed_records.append(parsed_obj)
                except Exception as e:
                    self.logger.warning(f"Nested JSON parse failed (evidence_evaluator.py:293): {traceback.format_exc()}")
                    
        return parsed_records

    def _score_evidence_item(self, relevance: str, quality: str, key_insight: str, url: str) -> int:
        """
        Scores evidence (0-100) based on relevance, credibility, specificity, and recency.
        """
        # 1. Relevance base score
        rel_lower = relevance.lower()
        if "contradict" in rel_lower:
            score = 60
        elif "context" in rel_lower:
            score = 50
        elif "mixed" in rel_lower:
            score = 45
        else:  # Supports
            score = 40
            
        # 2. Credibility (Quality + Source Type)
        if quality == "High":
            score += 15
        elif quality == "Medium":
            score += 8
            
        from tools.search_tool import SearchTool
        stype = SearchTool.classify_source_type(url)
        if stype in ("government", "academic", "research"):
            score += 10
        elif stype in ("news", "think_tank"):
            score += 5
            
        # 3. Specificity
        # Check if contains figures (numbers/percentages/years)
        if re.search(r'\b\d+(?:\.\d+)?%?\b', key_insight):
            score += 10
        elif len(key_insight.split()) > 12:
            score += 5
            
        # 4. Recency
        # Check if contains recent years (2023, 2024, 2025, 2026) in URL or insight
        recent_years = ["2023", "2024", "2025", "2026"]
        if any(yr in url or yr in key_insight for yr in recent_years):
            score += 10
            
        return max(0, min(100, score))

    def _triple_match(
        self,
        evidence_insight: str,
        search_result: SearchResult,
        claims: "ClaimAnalysis",
        blindspots: "List[Blindspot]"
    ) -> tuple:
        """
        Relaxed triple-match: evidence must match topic AND (claim OR blindspot).
        At least two of three must pass (topic + at least one other).
        Returns (is_valid, rejection_reason, linked_claim, linked_blindspot, match_score)
        with individual scores logged.
        """
        import re

        if not evidence_insight or not search_result:
            return (False, "empty_evidence", "", "", 0.0)

        insight_lower = evidence_insight.lower()
        snippet_lower = (search_result.snippet or "").lower()
        combined = insight_lower + " " + snippet_lower
        combined_words = set(w for w in re.findall(r'\b\w+\b', combined) if len(w) > 3)

        def overlap_ratio(words_a: set) -> float:
            if not words_a or not combined_words:
                return 0.0
            overlap = words_a.intersection(combined_words)
            denom = max(min(len(words_a), len(combined_words)), 1)
            return len(overlap) / denom

        linked_claim = ""
        linked_blindspot = ""
        topic_score = 0.0
        claim_score = 0.0
        blindspot_score = 0.0

        # Topic match
        topic = getattr(claims, "main_topic", "")
        if topic:
            topic_words = set(w for w in re.findall(r'\b\w+\b', topic.lower()) if len(w) > 3)
            topic_score = overlap_ratio(topic_words)

        # Claim match — pick best
        best_claim_ratio = 0.0
        if claims and claims.key_claims:
            for claim in claims.key_claims:
                claim_words = set(w for w in re.findall(r'\b\w+\b', claim.lower()) if len(w) > 3)
                ratio = overlap_ratio(claim_words)
                if ratio > best_claim_ratio:
                    best_claim_ratio = ratio
                    linked_claim = claim
        claim_score = best_claim_ratio

        # Blindspot match — pick best
        best_bs_ratio = 0.0
        if blindspots:
            for bs in blindspots:
                bs_lower = (bs.category + " " + bs.description).lower()
                bs_words = set(w for w in re.findall(r'\b\w+\b', bs_lower) if len(w) > 3)
                ratio = overlap_ratio(bs_words)
                if ratio > best_bs_ratio:
                    best_bs_ratio = ratio
                    linked_blindspot = bs.category
        blindspot_score = best_bs_ratio

        # Relaxed: topic AND (claim OR blindspot)
        topic_ok = topic_score >= 0.25
        claim_ok = claim_score >= 0.25
        blindspot_ok = blindspot_score >= 0.25

        is_valid = topic_ok and (claim_ok or blindspot_ok)
        total_score = (topic_score + max(claim_score, blindspot_score)) / 2.0

        if not is_valid:
            failed = []
            if not topic_ok: failed.append(f"topic={topic_score:.2f}")
            if not claim_ok and not blindspot_ok: failed.append(f"claim={claim_score:.2f} blindspot={blindspot_score:.2f}")
            reason = "relaxed_match_failed: " + ", ".join(failed)
            self.logger.info(
                f"Evidence match REJECTED: {reason} | insight='{evidence_insight[:60]}...'"
            )
        else:
            reason = ""

        return (is_valid, reason, linked_claim, linked_blindspot, total_score)

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
        except Exception as e:
            self.logger.warning(f"Credibility assessment failed (evidence_evaluator.py:452): {e}")
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
        Returns the LLM quality directly to trust LLM evaluations for production hardening.
        """
        return llm_quality

    def _classify_source_type_local(self, url: str) -> str:
        """
        Classifies evidence source url into news, government, academic, think tank,
        industry, social media, blog, spam, or other.
        """
        if not url:
            return "other"
        url_lower = url.lower()
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url_lower)
            netloc = parsed.netloc
            if netloc.startswith("www."):
                netloc = netloc[4:]
        except Exception as e:
            self.logger.warning(f"Source type classification failed (evidence_evaluator.py:496): {e}")
            netloc = url_lower

        government_indicators = [".gov", "who.int", "un.org", "worldbank.org", "imf.org"]
        think_tank_indicators = ["cato.org", "brookings.edu", "heritage.org", "rand.org", "csis.org", "pewresearch.org", "chathamhouse.org", "cfr.org"]
        academic_indicators = [".edu", "arxiv", "pubmed", "researchgate", "doi.org"]
        social_media_indicators = ["twitter.com", "x.com", "facebook.com", "reddit.com", "linkedin.com", "youtube.com", "instagram.com"]
        blog_indicators = ["blogspot.com", "medium.com", "substack.com", "wordpress.com", "blog.", "opinion"]
        news_domains = [
            "bbc.com", "bbc.co.uk", "nytimes.com", "reuters.com", "apnews.com",
            "bloomberg.com", "cnn.com", "theguardian.com", "guardian.co.uk",
            "economist.com", "wsj.com", "forbes.com", "npr.org", "dw.com",
            "aljazeera.com", "france24.com", "ft.com"
        ]
        spam_indicators = ["seofarm", "contentfarm", "spamblog", "coupon", "promo", "deal", "clickbait", "cheap"]

        if any(ind in netloc for ind in spam_indicators):
            return "spam"
        if any(ind in netloc for ind in government_indicators):
            return "government"
        if any(ind in netloc for ind in think_tank_indicators):
            return "think_tank"
        if any(ind in netloc for ind in academic_indicators):
            return "academic"
        if any(ind in netloc for ind in social_media_indicators):
            return "social_media"
        if any(ind in netloc for ind in blog_indicators):
            return "blog"
        if any(news in netloc for news in news_domains):
            return "news"

        return "other"

    def _get_source_quality_weight(self, url: str) -> int:
        """
        Calculates a priority weight based on source types.
        government +30, academic/research +30, thinktank +20, major_news +15, blog -20, spam -50.
        """
        stype = self._classify_source_type_local(url)
        if stype == "government":
            return 30
        if stype in ("academic", "research"):
            return 30
        if stype == "think_tank":
            return 20
        if stype == "news":
            return 15
        if stype == "blog":
            return -20
        if stype == "spam":
            return -50
        return 0

    def check_semantic_coverage(self, category: str, insight: str) -> bool:
        if not category or not insight:
            return False
            
        cat_lower = category.lower().strip()
        ins_lower = insight.lower().strip()
        
        # Suffix-stripping stemmer
        def stem(w: str) -> str:
            w = w.lower().strip()
            if len(w) <= 3:
                return w
            suffixes = ["ing", "ed", "es", "ly", "tion", "ment", "al", "ity", "ive", "ic", "s"]
            for suff in suffixes:
                if w.endswith(suff):
                    return w[:-len(suff)]
            return w

        import re
        stop_words = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "by", "with",
            "is", "are", "was", "were", "will", "would", "should", "can", "could", "about",
            "from", "this", "that", "these", "those", "how", "why", "what", "which", "policy", "rules", "new",
            "impact", "challenges", "outcomes", "opinions", "perspective", "concerns"
        }
        
        def get_stemmed_tokens(text: str):
            words = re.findall(r'\b\w+\b', text.lower())
            return {stem(w) for w in words if w not in stop_words and len(w) > 2}

        cat_tokens = get_stemmed_tokens(cat_lower)
        ins_tokens = get_stemmed_tokens(ins_lower)
        
        if not cat_tokens:
            return False
            
        # 1. Category keyword overlap
        overlap = cat_tokens.intersection(ins_tokens)
        has_cat_overlap = len(overlap) >= 2 or (len(cat_tokens) == 1 and len(overlap) >= 1)
        
        # 2. Synonym / Semantic keyword overlap
        synonyms = {
            "rural": ["village", "remote", "agriculture", "farmer", "villages", "distant"],
            "teacher": ["educator", "salary", "salaries", "compensation", "faculty", "staff"],
            "school": ["education", "classroom", "institution", "infrastructure", "building", "facilities"],
            "funding": ["budget", "allocation", "expenditure", "monetary", "financial", "spent", "spending"],
            "barrier": ["obstacle", "challenge", "hurdle", "difficulty", "inaccessibility", "inequality"]
        }
        
        has_semantic_overlap = False
        for cat_word in cat_tokens:
            for base, syn_list in synonyms.items():
                stemmed_syns = {stem(s) for s in syn_list}
                stemmed_syns.add(stem(base))
                if stem(cat_word) in stemmed_syns:
                    if ins_tokens.intersection(stemmed_syns):
                        has_semantic_overlap = True
                        break
            if has_semantic_overlap:
                break
                
        # 3. Stakeholder match
        stakeholders = {
            "student": ["student", "students", "learner", "learners", "examinee", "examinees", "youth", "child", "children"],
            "teacher": ["teacher", "teachers", "educator", "educators", "faculty", "staff"],
            "motorist": ["motorist", "motorists", "driver", "drivers", "owner", "owners", "public", "citizen", "citizens"],
            "business": ["business", "businesses", "merchant", "merchants", "industry", "industries", "employer", "employers", "company", "companies"]
        }
        
        has_stakeholder_match = False
        cat_words_raw = re.findall(r'\b\w+\b', cat_lower)
        ins_words_raw = re.findall(r'\b\w+\b', ins_lower)
        
        for group, terms in stakeholders.items():
            cat_has_group = any(t in cat_words_raw for t in terms)
            ins_has_group = any(t in ins_words_raw for t in terms)
            if cat_has_group and ins_has_group:
                has_stakeholder_match = True
                break

        is_covered = has_cat_overlap or has_semantic_overlap or has_stakeholder_match
        
        if is_covered:
            self.logger.info(
                f"Coverage match reason: category='{category}' matched insight='{insight}' "
                f"(cat_overlap={has_cat_overlap}, semantic_overlap={has_semantic_overlap}, stakeholder_match={has_stakeholder_match})"
            )
        else:
            self.logger.info(f"Rejected coverage: category='{category}' lacks overlap with insight='{insight}'")
            
        return is_covered

    def _is_evidence_relevant(
        self,
        evidence_insight: str,
        result: SearchResult,
        claims: ClaimAnalysis,
        blindspots: List[Blindspot]
    ) -> bool:
        """
        Validates if evidence is relevant: must share topic concepts, address a blindspot, or address a claim.
        """
        import re
        stop_words = {
            "new", "rule", "in", "without", "a", "valid", "may", "be", "denied", 
            "to", "for", "on", "of", "and", "or", "with", "the", "an", "at", "by", 
            "from", "2026", "2025", "about", "how", "why", "what", "is", "are", "was",
            "were", "will", "would", "should", "can", "could", "article", "report",
            "some", "any", "no", "not", "but", "yes", "this", "that", "these", "those"
        }
        
        def get_tokens(text: str):
            clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
            return {w for w in clean.split() if w not in stop_words and len(w) > 2}

        insight_tokens = get_tokens(evidence_insight)
        snippet_tokens = get_tokens(result.snippet)
        title_tokens = get_tokens(result.title)
        evidence_tokens = insight_tokens.union(snippet_tokens).union(title_tokens)

        for bs in blindspots:
            if self.check_semantic_coverage(bs.category, evidence_insight) or self.check_semantic_coverage(bs.category, result.snippet):
                return True

        if claims and claims.main_topic:
            topic_tokens = get_tokens(claims.main_topic)
            if topic_tokens.intersection(evidence_tokens):
                return True

        if claims and claims.key_claims:
            for claim in claims.key_claims:
                claim_tokens = get_tokens(claim)
                if claim_tokens.intersection(evidence_tokens):
                    return True

        return False

    def evaluate(
        self,
        article: ArticleData,
        claims: ClaimAnalysis,
        blindspots: List[Blindspot],
        search_results: List[SearchResult]
    ):
        """
        Evaluates search results and extracts evidence objects using Ollama.
        
        Returns:
            Tuple[List[Evidence], bool]: evidence items and whether fallback was used.
        """
        import time
        
        if not search_results:
            self.logger.warning("No search results available for evaluation")
            return [], False

        search_results = sorted(search_results, key=lambda r: self._get_source_quality_weight(r.url), reverse=True)

        unique_search_results = []
        seen_urls = set()
        seen_snippets = set()
        for res in search_results[:8]:
            url_norm = res.url.lower().rstrip('/')
            snippet_norm = " ".join(res.snippet.lower().split())
            if not url_norm or url_norm in seen_urls:
                self.logger.debug(f"Deduplicated URL: {res.url}")
                continue
            if snippet_norm in seen_snippets:
                self.logger.debug(f"Deduplicated snippet content: {res.url}")
                continue
            if len(res.snippet.split()) < 5:
                self.logger.debug(f"Rejected thin content: {res.url}")
                continue
            unique_search_results.append(res)
            seen_urls.add(url_norm)
            seen_snippets.add(snippet_norm)
        
        search_results = unique_search_results

        stop_words = {
            "new", "rule", "in", "without", "a", "valid", "may", "be", "denied", 
            "to", "for", "on", "of", "and", "or", "with", "the", "an", "at", "by", 
            "from", "2026", "2025", "about", "how", "why", "what", "is", "are", "was",
            "were", "will", "would", "should", "can", "could", "article", "report",
            "some", "any", "no", "not", "but", "yes", "this", "that", "these", "those"
        }
        combined_context = f"{article.title} {claims.main_topic if claims else ''} {' '.join(claims.key_claims if claims else [])}"
        context_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', combined_context)
        context_words = {w.lower() for w in context_clean.split() if w.strip().lower() not in stop_words and len(w) > 2}

        relevant_results = []
        for res in search_results:
            res_text = f"{res.title} {res.snippet} {res.url}"
            res_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', res_text)
            res_words = {w.lower() for w in res_clean.split() if w.strip()}
            
            overlap_score = len(context_words.intersection(res_words))
            if overlap_score > 0:
                relevant_results.append(res)
            else:
                self.logger.info(f"Rejected irrelevant evidence: URL={res.url} Reason=no keyword overlap")

        search_results_sliced = []
        for res in relevant_results[:4]:
            res_copy = res.model_copy()
            if res_copy.snippet:
                res_copy.snippet = res_copy.snippet[:500]
            if res_copy.title:
                res_copy.title = res_copy.title[:150]
            search_results_sliced.append(res_copy)

        if not search_results_sliced:
            self.logger.warning("No relevant search results left after overlap filtering")
            return [], False

        target_model = self.ollama_client.model
        if not self.ollama_client.pre_call_health_check(target_model):
            self.logger.warning("Ollama pre-call health check failed. Bypassing LLM and using heuristic evaluation.")
            result = self.heuristic_evaluate(article, claims, blindspots, search_results_sliced)
            return result, True

        self.logger.info(f"Evaluating {len(search_results_sliced)} search results")
        self.logger.info("Evidence evaluation started")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_evaluation_prompt(claims, blindspots, search_results_sliced)

        # 4. Evaluation: 1 attempt, 20s timeout
        response_text = ""
        evaluations = []
        max_attempts = 1
        success = False

        for attempt in range(1, max_attempts + 1):
            self.logger.info(f"Sending evidence evaluation request to Ollama (Attempt {attempt}/{max_attempts})")
            start_time = time.time()
            try:
                response_text = self.ollama_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    num_predict=256,
                    timeout=10.0
                )
                elapsed = time.time() - start_time
                self.logger.info(f"Ollama request attempt {attempt} finished in {int(elapsed)} seconds")

                if response_text:
                    evaluations = self.safe_json_parse(response_text)
                    if evaluations:
                        valid_items = [
                            item for item in evaluations
                            if isinstance(item, dict) and any(k in item for k in ["relevance", "quality", "key_insight"])
                        ]
                        if valid_items:
                            self.logger.info(f"Successfully parsed {len(valid_items)} valid evaluation records from attempt {attempt}. STOPPING RETRIES.")
                            evaluations = valid_items
                            success = True
                            break
            except Exception as e:
                self.logger.warning(f"Attempt {attempt} failed with error: {e}")

        if not success or not evaluations:
            self.logger.warning("LLM evidence evaluation timed out or failed. Switching to heuristic evaluation immediately.")
            result = self.heuristic_evaluate(article, claims, blindspots, search_results_sliced)
            return result, True

        self.logger.debug(f"Parsed {len(evaluations)} evaluation records")

        candidate_evidence = []
        for index, item in enumerate(evaluations):
            if index >= len(search_results_sliced):
                break
            if not isinstance(item, dict):
                continue

            # Key insight cleaning
            key_insight = item.get("key_insight", "")
            key_insight_str = str(key_insight).strip() if key_insight is not None else ""

            # 1. Insight length check (Reject if < 15 chars)
            if len(key_insight_str) < 15:
                self.logger.info(f"Rejected evidence reason: insight too short ('{key_insight_str}') for URL={search_results_sliced[index].url}")
                continue
                
            # 2. Generic statement check
            ins_lower = key_insight_str.lower().strip()
            generic_phrases = [
                "this is important",
                "experts disagree",
                "some concerns exist",
                "more research is needed",
                "this remains controversial",
                "opinions are divided",
                "no consensus exists",
                "details are unclear"
            ]
            is_generic = False
            for phrase in generic_phrases:
                if phrase in ins_lower or ins_lower == phrase:
                    is_generic = True
                    break
            if is_generic:
                self.logger.info(f"Rejected evidence reason: generic statement ('{key_insight_str}') for URL={search_results_sliced[index].url}")
                continue
                
            # 3. Duplicate check against candidate_evidence (ratio > 0.80)
            from difflib import SequenceMatcher
            is_duplicate = False
            for existing in candidate_evidence:
                if existing.search_result.url.lower().rstrip('/') == search_results_sliced[index].url.lower().rstrip('/'):
                    is_duplicate = True
                    break
                ratio = SequenceMatcher(None, existing.key_insight.lower(), key_insight_str.lower()).ratio()
                if ratio > 0.80:
                    is_duplicate = True
                    break
            if is_duplicate:
                self.logger.info(f"Rejected evidence reason: duplicates existing evidence for URL={search_results_sliced[index].url}")
                continue

            # Relevance cleaning
            relevance = item.get("relevance")
            relevance_val = None
            if isinstance(relevance, str) and relevance.strip():
                relevance_clean = relevance.strip().lower()
                if "support" in relevance_clean:
                    relevance_val = "Supports"
                elif "contradict" in relevance_clean:
                    relevance_val = "Contradicts"
                elif "mixed" in relevance_clean:
                    relevance_val = "Mixed"
                elif "context" in relevance_clean:
                    relevance_val = "Adds Context"
                    
            if relevance_val is None:
                key_insight_lower = key_insight_str.lower()
                contradict_keywords = ["criticism", "concern", "challenge", "drawback", "downside", "opposition"]
                support_keywords = ["supports", "confirms", "evidence shows", "study finds"]
                mixed_keywords = ["mixed", "partially", "some evidence", "however", "though"]
                
                if any(kw in key_insight_lower for kw in contradict_keywords):
                    relevance_val = "Contradicts"
                elif any(kw in key_insight_lower for kw in support_keywords):
                    relevance_val = "Supports"
                elif any(kw in key_insight_lower for kw in mixed_keywords):
                    relevance_val = "Mixed"
                else:
                    relevance_val = "Adds Context"
                    
            relevance = relevance_val

            # Quality cleaning
            inferred_quality = self._infer_source_quality(search_results_sliced[index])
            llm_quality = item.get("quality", "Medium")
            if isinstance(llm_quality, str):
                llm_quality_clean = llm_quality.strip().lower()
                if "high" in llm_quality_clean:
                    llm_quality = "High"
                elif "low" in llm_quality_clean:
                    llm_quality = "Low"
                else:
                    llm_quality = "Medium"
            else:
                llm_quality = inferred_quality
            quality = self._normalize_quality(llm_quality, inferred_quality)

            if not search_results_sliced[index].url:
                self.logger.warning("Skipping evidence with empty URL")
                continue

            # Duplication & Value filtering
            repeats_claims = False
            claims_text = " ".join(claims.key_claims).lower() if (claims and claims.key_claims) else ""
            claims_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', claims_text)
            insight_clean_str = re.sub(r'[^a-zA-Z0-9\s]', ' ', key_insight_str.lower())
            claims_words = set(w for w in claims_clean.split() if len(w) > 4)
            insight_words = set(w for w in insight_clean_str.split() if len(w) > 4)
            if claims_words and insight_words:
                overlap = len(claims_words.intersection(insight_words)) / len(insight_words)
                if overlap > 0.6:
                    repeats_claims = True
                    
            # Determine relevance score using multi-dimensional scoring function
            relevance_score = self._score_evidence_item(
                relevance=relevance,
                quality=quality,
                key_insight=key_insight_str,
                url=search_results_sliced[index].url
            )

            # Reject ONLY if duplicates claim AND adds no context AND provides no support value
            adds_no_context = (relevance != "Adds Context")
            provides_no_support_value = (relevance != "Supports" or relevance_score < 30)
            
            if repeats_claims and adds_no_context and provides_no_support_value:
                self.logger.info(f"Rejected duplicate/low-value evidence: URL={search_results_sliced[index].url}")
                continue

            # 1. Check required fields
            url_val = search_results_sliced[index].url
            try:
                domain_val = urlparse(url_val).netloc
                if domain_val.startswith("www."):
                    domain_val = domain_val[4:]
            except Exception as e:
                self.logger.warning(f"Domain extraction failed (evidence_evaluator.py:954): {e}")
                domain_val = ""
                
            evidence_summary = str(item.get("evidence_summary", "")).strip()
            if not evidence_summary:
                evidence_summary = key_insight_str
                
            if not url_val or not domain_val or not key_insight_str or not relevance or not evidence_summary:
                self.logger.warning(f"Discarding evidence due to missing required fields: URL={url_val}")
                continue

            # Relaxed triple match: topic AND (claim OR blindspot)
            tm_valid, tm_reason, matched_claim_text, matched_bs_cat, match_score = self._triple_match(
                evidence_insight=key_insight_str,
                search_result=search_results_sliced[index],
                claims=claims,
                blindspots=blindspots
            )
            if not tm_valid:
                self.logger.info(f"Discarding evidence ({tm_reason}): URL={search_results_sliced[index].url}")
                continue

            evidence_obj = Evidence(
                search_result=search_results_sliced[index],
                relevance=relevance,
                quality=quality,
                key_insight=key_insight_str,
                relevance_score=relevance_score,
                evidence_summary=evidence_summary,
                match_score=match_score,
                linked_claim=matched_claim_text,
                linked_blindspot=matched_bs_cat,
                matched_claim=matched_claim_text,
                matched_blindspot=matched_bs_cat
            )
            self.logger.info(
                f"Evidence created: quality={evidence_obj.quality}, relevance={evidence_obj.relevance}, score={relevance_score}"
            )
            candidate_evidence.append(evidence_obj)

        # Sort by relevance_score DESC, quality DESC, credibility DESC
        def get_sort_key(ev: Evidence) -> tuple:
            q_val = 3 if ev.quality == "High" else (2 if ev.quality == "Medium" else 1)
            url = ev.search_result.url
            from tools.search_tool import SearchTool
            stype = SearchTool.classify_source_type(url)
            if stype in ("government", "academic", "research"):
                cred = 4
            elif stype == "news":
                cred = 3
            elif stype == "industry":
                cred = 2
            else:
                cred = 1
            return (ev.relevance_score, q_val, cred)

        evidence_items = []
        domain_counts = {}
        candidates = sorted(candidate_evidence, key=get_sort_key, reverse=True)
        
        for ev in candidates:
            try:
                parsed = urlparse(ev.search_result.url)
                domain = parsed.netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]
            except Exception as e:
                self.logger.warning(f"Domain parse failed (evidence_evaluator.py:1020): {e}")
                domain = "unknown"
                
            if domain_counts.get(domain, 0) < 2:
                evidence_items.append(ev)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            else:
                self.logger.info(f"Rejected evidence (domain limit exceeded for {domain}): URL={ev.search_result.url}")

        if not evidence_items:
            self.logger.warning("No evidence items generated. Bypassing and calling heuristic evaluation.")
            evidence_items = self.heuristic_evaluate(article, claims, blindspots, search_results_sliced)
            self.logger.info(f"Selected {len(evidence_items)} evidence items from {len(search_results_sliced)} search results")
            return evidence_items, True

        self.logger.info(f"Selected {len(evidence_items)} evidence items from {len(search_results_sliced)} search results")
        return evidence_items, False

    def get_confidence_details(
        self,
        blindspots: List[Blindspot],
        evidence: List[Evidence],
        llm_failures: int = 0,
        llm_calls: int = 0
    ) -> dict:
        """
        Continuous confidence scoring: no caps, floors, buckets, or artificial penalties.
        Score emerges from data via weighted continuous formula.
        Final confidence is multiplied by system reliability factor.
        """
        from tools.search_tool import SearchTool

        if not evidence:
            return {
                "score": 0,
                "coverage_ratio": 0.0,
                "blindspot_count": len(blindspots),
                "evidence_count": 0,
                "high_quality_count": 0,
                "medium_quality_count": 0,
                "low_quality_count": 0,
                "contradict_count": 0,
                "unique_domains": 0,
                "academic_sources": 0,
                "government_sources": 0,
                "news_sources": 0,
                "score_breakdown": {},
                "reliability_factor": 0.0
            }

        n = len(evidence)

        # 1. Evidence Quality (weight 0.30)
        quality_values = {"High": 1.0, "Medium": 0.6, "Low": 0.2}
        raw_quality = sum(quality_values.get(ev.quality, 0.3) for ev in evidence)
        quality_component = (raw_quality / max(n, 1)) * 30.0

        # 2. Blindspot Coverage (weight 0.25)
        total_blindspots = len(blindspots)
        covered_blindspots = 0
        for bs in blindspots:
            for ev in evidence:
                if self.check_semantic_coverage(bs.category, ev.key_insight):
                    covered_blindspots += 1
                    break
        coverage_ratio = covered_blindspots / max(total_blindspots, 1)
        coverage_component = coverage_ratio * 25.0

        # 3. Claim Linkage (weight 0.20)
        match_scores = [getattr(ev, "match_score", 0.0) for ev in evidence]
        avg_match = sum(match_scores) / max(n, 1)
        linkage_component = avg_match * 20.0

        # 4. Source Diversity (weight 0.15)
        domain_counts = {}
        for ev in evidence:
            d = ev.search_result.source.lower().strip()
            domain_counts[d] = domain_counts.get(d, 0) + 1
        unique_domains_count = len(domain_counts)
        diversity_component = 15.0 * min(unique_domains_count / max(n, 1), 1.0)

        # 5. Source Reliability (weight 0.10)
        reliability_map = {
            "academic": 1.0, "government": 1.0, "think_tank": 0.8,
            "news": 0.6, "industry": 0.4, "blog": 0.2, "social_media": 0.1, "other": 0.3
        }
        total_reliability = 0.0
        academic_sources = 0
        government_sources = 0
        news_sources = 0
        for ev in evidence:
            url = getattr(ev.search_result, "url", "")
            st = SearchTool.classify_source_type(url)
            total_reliability += reliability_map.get(st, 0.3)
            if st == "academic": academic_sources += 1
            elif st == "government": government_sources += 1
            elif st == "news": news_sources += 1
        source_reliability_component = (total_reliability / max(n, 1)) * 10.0

        # Continuous score from evidence data only — no caps, no floors, no artificial penalties
        raw_score = (
            quality_component +
            coverage_component +
            linkage_component +
            diversity_component +
            source_reliability_component
        )

        # System reliability factor: successful_llm_calls / total_llm_calls
        total_calls = max(llm_calls, 1)  # avoid division by zero
        successful_calls = total_calls - llm_failures
        reliability_factor = max(0.0, successful_calls / total_calls)

        # Final score = data-based score * reliability factor
        score_val = raw_score * reliability_factor

        contradict_count = sum(1 for ev in evidence if getattr(ev, "relevance", "").lower() == "contradicts")
        high_q_count = sum(1 for ev in evidence if ev.quality == "High")
        medium_q_count = sum(1 for ev in evidence if ev.quality == "Medium")
        low_q_count = sum(1 for ev in evidence if ev.quality == "Low")

        score_breakdown = {
            "quality_component": round(quality_component, 2),
            "coverage_component": round(coverage_component, 2),
            "linkage_component": round(linkage_component, 2),
            "diversity_component": round(diversity_component, 2),
            "source_reliability_component": round(source_reliability_component, 2),
            "reliability_factor": round(reliability_factor, 2),
            "coverage_ratio": coverage_ratio,
            "unique_domains": unique_domains_count
        }

        self.logger.info(
            f"Confidence analytics: score={score_val:.1f}, coverage_ratio={coverage_ratio:.2f}, "
            f"evidence_count={n}, reliability={reliability_factor:.2f}, "
            f"high_q={high_q_count}, unique_domains={unique_domains_count}"
        )

        return {
            "score": int(round(score_val)),
            "coverage_ratio": coverage_ratio,
            "blindspot_count": total_blindspots,
            "evidence_count": n,
            "high_quality_count": high_q_count,
            "medium_quality_count": medium_q_count,
            "low_quality_count": low_q_count,
            "contradict_count": contradict_count,
            "unique_domains": unique_domains_count,
            "academic_sources": academic_sources,
            "government_sources": government_sources,
            "news_sources": news_sources,
            "reliability_factor": reliability_factor,
            "score_breakdown": score_breakdown
        }

    def calculate_confidence(
        self,
        blindspots: List[Blindspot],
        evidence: List[Evidence],
        llm_failures: int = 0,
        llm_calls: int = 0
    ) -> int:
        """
        Estimate confidence based purely on evidence data weighted by system reliability.
        """
        details = self.get_confidence_details(blindspots, evidence, llm_failures=llm_failures, llm_calls=llm_calls)
        score = details["score"]
        self.logger.info(f"Confidence score: {score}%")
        self.logger.debug(f"Confidence details: {details}")
        return score

    def analyze_blindspot_coverage(
        self,
        blindspots: List[Blindspot],
        evidence: List[Evidence]
    ) -> dict:
        """
        Determines evidence coverage level for each blindspot category.
        Returns a dict mapping category name to "covered", "partial", or "missing".
        """
        coverage = {}
        if not blindspots:
            return coverage

        for bs in blindspots:
            category = bs.category
            coverage[category] = "missing"
            
            covered = False
            for ev in evidence:
                if self.check_semantic_coverage(category, ev.key_insight):
                    covered = True
                    break
            
            if covered:
                coverage[category] = "covered"

        self.logger.info(f"Analyzed blindspot coverage: {coverage}")
        return coverage


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
        evidence_list, _ = evaluator.evaluate(article, claims, blindspots, search_results)

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
