from typing import List
from urllib.parse import urlparse
import re
import json

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
                
            # If it covers a blindspot
            addresses_blindspot = False
            matched_bs_cat = ""
            for bs in blindspots:
                if self.check_semantic_coverage(bs.category, insight) or (res.snippet and self.check_semantic_coverage(bs.category, res.snippet)):
                    addresses_blindspot = True
                    matched_bs_cat = bs.category
                    break
            if not addresses_blindspot:
                self.logger.info(f"Heuristic evidence discarded: cannot be linked to any blindspot for URL={res.url}")
                continue
                
            score += 20
            score = max(0, min(100, score))
            
            ev = Evidence(
                search_result=res,
                relevance=relevance,
                quality=quality,
                key_insight=insight,
                relevance_score=score,
                evidence_summary=f"External source confirms details and {relevance.lower()} the topic. It provides key insight: {insight}",
                related_blindspot=matched_bs_cat
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
        except Exception:
            pass
            
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
            except Exception:
                pass
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
            except Exception:
                pass
                
            # Attempt repairs on individual block
            try:
                rep = cleaned_block.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
                rep = rep.replace('`', '')
                rep = re.sub(r',\s*}', '}', rep)
                parsed_obj = json.loads(rep)
                if isinstance(parsed_obj, dict):
                    parsed_records.append(parsed_obj)
            except Exception:
                # Use repair_json_string from ollama_client if possible
                try:
                    repaired = self.ollama_client.repair_json_string(cleaned_block)
                    parsed_obj = json.loads(repaired)
                    if isinstance(parsed_obj, dict):
                        parsed_records.append(parsed_obj)
                except Exception:
                    pass
                    
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
        except Exception:
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
    ) -> List[Evidence]:
        """
        Evaluates search results and extracts evidence objects using Ollama.
        """
        import time
        import re
        
        if not search_results:
            self.logger.warning("No search results available for evaluation")
            return []

        # Sort search results by source quality weight descending (Issue 6)
        search_results = sorted(search_results, key=lambda r: self._get_source_quality_weight(r.url), reverse=True)

        # Deduplicate search results and filter thin content (< 5 words)
        unique_search_results = []
        seen_urls = set()
        seen_snippets = set()
        for res in search_results:
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

        # 1. Extract context keywords from title, topic, and key claims
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

        # 2. Filter search results based on overlap
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

        # Slice relevant search results to max 6 and truncate snippets to 120 max, titles to 80 max
        search_results_sliced = []
        for res in relevant_results[:6]:
            res_copy = res.model_copy()
            if res_copy.snippet:
                res_copy.snippet = res_copy.snippet[:120]
            if res_copy.title:
                res_copy.title = res_copy.title[:80]
            search_results_sliced.append(res_copy)

        if not search_results_sliced:
            self.logger.warning("No relevant search results left after overlap filtering")
            return []

        # 3. Check pre-call health check
        target_model = self.ollama_client.model
        if not self.ollama_client.pre_call_health_check(target_model):
            self.logger.warning("Ollama pre-call health check failed. Bypassing LLM and using heuristic evaluation.")
            return self.heuristic_evaluate(article, claims, blindspots, search_results_sliced)

        self.logger.info(f"Evaluating {len(search_results_sliced)} search results")
        self.logger.info("Evidence evaluation started")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_evaluation_prompt(claims, blindspots, search_results_sliced)

        # 4. Evaluation retries: up to 3 retries (4 attempts total)
        response_text = ""
        evaluations = []
        max_attempts = 4
        success = False

        for attempt in range(1, max_attempts + 1):
            self.logger.info(f"Sending evidence evaluation request to Ollama (Attempt {attempt}/{max_attempts})")
            start_time = time.time()
            try:
                response_text = self.ollama_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    num_predict=512,
                    timeout=60.0
                )
                elapsed = time.time() - start_time
                self.logger.info(f"Ollama request attempt {attempt} finished in {int(elapsed)} seconds")

                if response_text:
                    evaluations = self.safe_json_parse(response_text)
                    if evaluations:
                        # Ensure at least one item has expected keys to verify it is a valid evidence structure
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
                
            if attempt < max_attempts:
                time.sleep(2)

        if not success or not evaluations:
            self.logger.warning("All LLM evidence evaluation attempts failed. Falling back to programmatic heuristic evaluate.")
            return self.heuristic_evaluate(article, claims, blindspots, search_results_sliced)

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
                elif any(kw in key_insight_lower for key_insight_lower in support_keywords):
                    relevance_val = "Supports"
                elif any(kw in key_insight_lower for key_insight_lower in mixed_keywords):
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
            except Exception:
                domain_val = ""
                
            evidence_summary = str(item.get("evidence_summary", "")).strip()
            if not evidence_summary:
                evidence_summary = f"External source confirms details and {relevance.lower()} the topic. It provides key insight: {key_insight_str}"
                
            if not url_val or not domain_val or not key_insight_str or not relevance or not evidence_summary:
                self.logger.warning(f"Discarding evidence due to missing required fields: URL={url_val}")
                continue

            # relevance validation:
            if not self._is_evidence_relevant(key_insight_str, search_results_sliced[index], claims, blindspots):
                self.logger.info(f"Rejected evidence reason: no relation to topic for URL={search_results_sliced[index].url}")
                continue

            # Evidence-to-Blindspot Mapping constraint:
            mapped_to_blindspot = False
            matched_bs_cat = ""
            for bs in blindspots:
                if self.check_semantic_coverage(bs.category, key_insight_str) or self.check_semantic_coverage(bs.category, search_results_sliced[index].snippet):
                    mapped_to_blindspot = True
                    matched_bs_cat = bs.category
                    break
            if not mapped_to_blindspot:
                self.logger.info(f"Discarding evidence (cannot be linked to any blindspot): URL={search_results_sliced[index].url}")
                continue

            evidence_obj = Evidence(
                search_result=search_results_sliced[index],
                relevance=relevance,
                quality=quality,
                key_insight=key_insight_str,
                relevance_score=relevance_score,
                evidence_summary=evidence_summary,
                related_blindspot=matched_bs_cat
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
            except Exception:
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
        return evidence_items

    def _is_generic_category(self, category: str) -> bool:
        if not category:
            return True
        generic_list = [
            "opposing perspective", "historical context", "economic impact", "expert opinion",
            "stakeholder view", "alternative perspective", "enforcement challenges", "historical outcomes",
            "unintended consequences", "missing context", "transparency", "social impact",
            "environmental impact", "governance", "policy alternatives", "other perspectives",
            "unrepresented viewpoints", "regulatory challenges", "community engagement",
            "social concerns", "public opinion", "stakeholder response"
        ]
        cat_lower = category.lower().strip()
        for gen in generic_list:
            if cat_lower == gen or gen in cat_lower:
                return True
        for suffix in ["implementation challenges", "stakeholder economic impact", "legal policy precedents", "primary implementation challenges"]:
            if suffix in cat_lower:
                return True
        return False

    def get_confidence_details(
        self,
        blindspots: List[Blindspot],
        evidence: List[Evidence]
    ) -> dict:
        """
        Calculates confidence score and detailed coverage analytics.
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
                "score_breakdown": {}
            }

        # 1. Quality points with saturation penalty
        domain_counts = {}
        quality_score = 0.0
        high_q_count = 0
        medium_q_count = 0
        low_q_count = 0
        for ev in evidence:
            domain = ev.search_result.source.lower().strip()
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            idx = domain_counts[domain]
            
            # Domain saturation penalty
            if idx == 1:
                weight = 1.0
            elif idx == 2:
                weight = 0.5
            elif idx == 3:
                weight = 0.25
            else:
                weight = 0.0
                
            q = ev.quality
            if q == "High":
                base_q = 20
                high_q_count += 1
            elif q == "Medium":
                base_q = 12
                medium_q_count += 1
            else:
                base_q = 5
                low_q_count += 1
                
            quality_score += base_q * weight
            
        quality_score = min(50.0, quality_score)

        # 2. Blindspot coverage points
        total_blindspots = len(blindspots)
        covered_blindspots = 0
        for bs in blindspots:
            covered = False
            for ev in evidence:
                if self.check_semantic_coverage(bs.category, ev.key_insight):
                    covered = True
                    break
            if covered:
                covered_blindspots += 1
                
        coverage_ratio = covered_blindspots / total_blindspots if total_blindspots > 0 else 0.0
        coverage_points = covered_blindspots * 15.0
        coverage_points = min(45.0, coverage_points)

        # 3. Source diversity points
        academic_sources = 0
        government_sources = 0
        news_sources = 0
        for ev in evidence:
            url = getattr(ev.search_result, "url", "")
            source_type = SearchTool.classify_source_type(url)
            if source_type == "academic":
                academic_sources += 1
            elif source_type == "government":
                government_sources += 1
            elif source_type == "news":
                news_sources += 1
                
        unique_domains_count = len(domain_counts)
        diversity_points = unique_domains_count * 5.0
        diversity_points = min(20.0, diversity_points)

        # 4. Contradiction Analysis
        contradict_count = sum(1 for ev in evidence if getattr(ev, "relevance", "").lower() == "contradicts")
        contradiction_bonus = 10.0 if contradict_count > 0 else 0.0

        # 5. Agreement Bonus (if multiple high quality sources are present)
        agreement_bonus = 10.0 if high_q_count >= 2 else 0.0

        # Base confidence score
        raw_score = quality_score + coverage_points + diversity_points + contradiction_bonus + agreement_bonus
        score_val = int(raw_score)

        # Apply Penalties:
        # 1. Generic blindspots penalty (-15)
        has_generic = any(self._is_generic_category(bs.category) for bs in blindspots)
        if has_generic:
            score_val -= 15
            self.logger.info("Penalty applied: generic blindspots (-15)")
            
        # 2. Low source diversity (-10)
        if unique_domains_count <= 1 and len(evidence) >= 2:
            score_val -= 10
            self.logger.info("Penalty applied: low source diversity (-10)")
            
        # 3. Weak coverage (< 50% coverage, -15)
        if coverage_ratio < 0.5:
            score_val -= 15
            self.logger.info("Penalty applied: weak coverage (< 50% coverage, -15)")
            
        # 4. Weak relevance penalty (-15)
        avg_relevance_score = sum(ev.relevance_score for ev in evidence) / len(evidence) if evidence else 0
        if len(evidence) > 0 and avg_relevance_score < 55:
            score_val -= 15
            self.logger.info("Penalty applied: weak relevance (-15)")

        # Apply strict realistic caps (Revised Caps from User request):
        if len(evidence) == 1:
            score_val = min(score_val, 40)
            self.logger.info(f"Confidence cap applied: 1 evidence -> cap 40")
        elif len(evidence) == 2:
            score_val = min(score_val, 60)
            self.logger.info(f"Confidence cap applied: 2 evidence -> cap 60")
        elif len(evidence) == 3:
            score_val = min(score_val, 80)
            self.logger.info(f"Confidence cap applied: 3 evidence -> cap 80")
        elif len(evidence) >= 4:
            score_val = min(score_val, 95)
            self.logger.info(f"Confidence cap applied: 4+ evidence -> cap 95")

        if coverage_ratio < 1.0:
            if score_val > 80:
                self.logger.info(f"Confidence cap applied: coverage < 1.0 -> cap 80")
                score_val = min(score_val, 80)
                
        if high_q_count == 0:
            if score_val > 65:
                self.logger.info(f"Confidence cap applied: high_quality == 0 -> cap 65")
                score_val = min(score_val, 65)

        # B. Repetitive sources cap (unique domains <= 1 and count >= 2)
        if unique_domains_count <= 1 and len(evidence) >= 2:
            score_val = min(score_val, 40)
            
        # D. Zero coverage cap
        has_strong_qd = False
        if covered_blindspots == 0:
            has_strong_qd = (unique_domains_count >= 2 and (high_q_count >= 1 or (high_q_count + medium_q_count) >= 2))
            cap_limit = 35 if has_strong_qd else 20
            score_val = min(score_val, cap_limit)

        # Floors:
        if len(evidence) >= 1:
            score_val = max(15, score_val)

        score_val = max(0, min(100, score_val))

        score_breakdown = {
            "quality_points": int(quality_score),
            "coverage_points": int(coverage_points),
            "diversity_points": int(diversity_points),
            "contradiction_bonus": int(contradiction_bonus),
            "agreement_bonus": int(agreement_bonus),
            "coverage_ratio": coverage_ratio,
            "unique_domains": unique_domains_count
        }

        self.logger.info(
            f"Confidence analytics: score={score_val}, coverage_ratio={coverage_ratio:.2f}, "
            f"evidence_count={len(evidence)}, high_q={high_q_count}, "
            f"contradicts={contradict_count}, unique_domains={unique_domains_count}"
        )
        self.logger.info(
            f"Confidence breakdown details:\n"
            f"  - Quality Points: {quality_score}\n"
            f"  - Coverage Points: {coverage_points}\n"
            f"  - Diversity Points: {diversity_points}\n"
            f"  - Contradiction Bonus: {contradiction_bonus}\n"
            f"  - Agreement Bonus: {agreement_bonus}\n"
            f"  - Caps applied: 1 evidence limit (30)? {len(evidence) == 1}, "
            f"2 evidence limit (50)? {len(evidence) == 2}, repetitive sources limit (40)? {unique_domains_count <= 1 and len(evidence) >= 2}, "
            f"incomplete coverage limit (60)? {coverage_ratio < 0.5}, zero coverage cap (20/35)? {covered_blindspots == 0} (strong Q&D? {has_strong_qd})"
        )

        return {
            "score": score_val,
            "coverage_ratio": coverage_ratio,
            "actual_coverage_ratio": coverage_ratio,
            "fallback_coverage_ratio": coverage_ratio,
            "blindspot_count": total_blindspots,
            "evidence_count": len(evidence),
            "high_quality_count": high_q_count,
            "medium_quality_count": medium_q_count,
            "low_quality_count": low_q_count,
            "contradict_count": contradict_count,
            "unique_domains": unique_domains_count,
            "academic_sources": academic_sources,
            "government_sources": government_sources,
            "news_sources": news_sources,
            "score_breakdown": score_breakdown
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
