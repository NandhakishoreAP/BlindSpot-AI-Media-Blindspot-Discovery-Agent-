from typing import List, Optional
import re

from models.data_models import (
    AgentState,
    PlannerDecision,
    QueryIntent,
    Blindspot,
    Evidence,
    ArticleData,
    ClaimAnalysis,
    SearchResult
)

from llm.ollama_client import OllamaClient
from utils.logger import get_logger
from config import Config

class Planner:
    """
    The decision-making brain of the agent that determines search steps and when to finalize research.
    """

    def __init__(self, ollama_client: OllamaClient, config: Config) -> None:
        """
        Initializes the Planner with an OllamaClient and Config.
        """
        self.ollama_client: OllamaClient = ollama_client
        self.config: Config = config
        self.logger = get_logger("planner")
        self.confidence_threshold: int = config.CONFIDENCE_THRESHOLD
        self.max_search_attempts: int = config.MAX_SEARCH_ATTEMPTS

    def _build_system_prompt(self) -> str:
        """
        Return EXACTLY the requested system prompt.
        """
        return """
You are the planning module of an AI media analysis agent.

Your job is to decide the next action based on the current state of research.

You must choose one of three actions:

* SEARCH: Start initial searches for missing perspectives

* SEARCH_MORE: Continue searching because evidence is insufficient

* GENERATE_REPORT: Enough evidence has been gathered, generate the final report

When generating search queries, make them specific and diverse.

Avoid repeating previous searches.

Prioritize blindspots that have not yet been sufficiently investigated.

Prefer academic, government, expert, and high-quality sources when confidence is low.

You must respond ONLY with valid JSON.

No explanations outside the JSON.
""".strip()

    def _get_evidence_summary(self, evidence: List[Evidence]) -> dict:
        """
        Tallies counts of quality tiers and relevance categories across all evidence gathered.
        """
        summary = {
            "high_quality_count": 0,
            "medium_quality_count": 0,
            "low_quality_count": 0,
            "supports_count": 0,
            "contradicts_count": 0,
            "context_count": 0
        }
        
        for ev in evidence:
            q = getattr(ev, "quality", "Low")
            if q == "High":
                summary["high_quality_count"] += 1
            elif q == "Medium":
                summary["medium_quality_count"] += 1
            else:
                summary["low_quality_count"] += 1
                
            rel = getattr(ev, "relevance", "Adds Context")
            if rel == "Supports":
                summary["supports_count"] += 1
            elif rel == "Contradicts":
                summary["contradicts_count"] += 1
            else:
                summary["context_count"] += 1
                
        return summary

    def _check_semantic_coverage_local(self, category: str, insight: str) -> bool:
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

        return has_cat_overlap or has_semantic_overlap or has_stakeholder_match

    def _estimate_blindspot_coverage(self, state: AgentState) -> dict:
        """
        Estimates which blindspots already have gathered evidence.
        Returns a dictionary mapping category names to boolean coverage.
        """
        coverage = {}
        blindspots = getattr(state, "blindspots", [])
        evidence_list = getattr(state, "evidence", [])
        
        for bs in blindspots:
            category = bs.category
            coverage[category] = False
            
            for ev in evidence_list:
                if self._check_semantic_coverage_local(category, ev.key_insight):
                    coverage[category] = True
                    break
                    
        return coverage


    def _build_planning_prompt(self, state: AgentState) -> str:
        """
        Builds a simplified planning prompt focusing only on blindspots, confidence,
        evidence summary, and used queries.
        """
        blindspots = getattr(state, "blindspots", [])
        evidence_list = getattr(state, "evidence", [])
        used_queries = getattr(state, "search_queries_used", [])
        confidence_score = getattr(state, "confidence_score", 0)
        
        blindspots_str = "\n".join(f"- {bs.category}: {bs.description}" for bs in blindspots)
        
        summary = self._get_evidence_summary(evidence_list)
        evidence_str = (
            f"Total evidence count: {len(evidence_list)}\n"
            f"High-quality: {summary['high_quality_count']}\n"
            f"Medium-quality: {summary['medium_quality_count']}\n"
            f"Low-quality: {summary['low_quality_count']}\n"
            f"Contradicting: {summary['contradicts_count']}"
        )
        
        used_queries_str = "\n".join(f"- {q}" for q in used_queries) if used_queries else "None."

        prompt = (
            f"Current confidence: {confidence_score}%\n"
            f"Confidence threshold: {self.confidence_threshold}%\n\n"
            "BLINDSPOTS TO INVESTIGATE\n"
            f"{blindspots_str}\n\n"
            "EVIDENCE SUMMARY\n"
            f"{evidence_str}\n\n"
            "PREVIOUS QUERIES USED\n"
            f"{used_queries_str}\n\n"
            "INSTRUCTIONS\n"
            "Decide the next action based on research progress. Choose one of: SEARCH_MORE or GENERATE_REPORT.\n"
            "If action is SEARCH_MORE, generate 2-4 search queries that are specific, search-oriented, and avoid repeating previous queries.\n"
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "action": "SEARCH_MORE",\n'
            '  "queries": ["query 1", "query 2"],\n'
            '  "reasoning": "..."\n'
            "}"
        )
        return prompt

    def _generate_intent_queries_for_blindspot(self, topic: str, blindspot: Blindspot) -> List[str]:
        """
        Generates queries for a blindspot using: topic + blindspot + intent.
        Formats:
        - "{topic} {blindspot} criticism"
        - "{topic} {blindspot} impact"
        - "{topic} {blindspot} expert opinion"
        """
        import re
        topic_clean = re.sub(r'[^\w\s\-]', ' ', topic).strip()
        topic_phrase = " ".join(topic_clean.split())
        
        cat_clean = re.sub(r'[^\w\s\-]', ' ', blindspot.category).strip()
        cat_phrase = " ".join(cat_clean.split())
        
        # De-duplicate topic words in category to avoid redundancy
        topic_words_lower = {w.lower() for w in topic_phrase.split()}
        unique_cat_words = [w for w in cat_phrase.split() if w.lower() not in topic_words_lower]
        cat_phrase_filtered = " ".join(unique_cat_words) if unique_cat_words else cat_phrase

        intents = [
            "criticism",
            "impact",
            "expert opinion",
            "implementation challenges"
        ]
        
        queries = []
        for intent in intents[:3]:
            query = f"{topic_phrase} {cat_phrase_filtered} {intent}"
            query_clean = " ".join(query.split())
            queries.append(query_clean)
            
        return queries

    def _construct_high_value_query(self, topic: str, blindspot: Blindspot) -> str:
        """
        Constructs a human-readable natural language search query using intent format.
        """
        queries = self._generate_intent_queries_for_blindspot(topic, blindspot)
        return queries[0] if queries else f"{topic} {blindspot.category} criticism"

    def _generate_initial_queries(self, state: AgentState) -> List[QueryIntent]:
        """
        Generate query_intent objects — each with query_text, target_entity, target_claim, target_blindspot.
        Uses article entities for entity grounding.
        """
        from models.data_models import QueryIntent

        topic = state.claims.main_topic if (state.claims and state.claims.main_topic) else ""
        if not topic or topic == "Unknown":
            topic = state.article.title if state.article else "Policy"
        topic = topic.strip().rstrip(".?!:;, ")

        import re
        topic_clean = re.sub(r'[^\w\s\-]', ' ', topic).strip()
        topic_phrase = " ".join(topic_clean.split())

        # Pull entities from state
        entities = getattr(state, "article_entities", {})
        all_entity_values = []
        for cat_list in entities.values():
            for e in cat_list:
                if isinstance(e, str) and e.strip():
                    all_entity_values.append(e.strip())

        # Pull claims
        claims_list = []
        if state.claims and state.claims.key_claims:
            claims_list = state.claims.key_claims

        blindspots = state.blindspots or []
        coverage = self._estimate_blindspot_coverage(state)
        uncovered_bs = [bs for bs in blindspots if not coverage.get(bs.category, False)]
        covered_bs = [bs for bs in blindspots if coverage.get(bs.category, False)]

        all_intents: List[QueryIntent] = []
        seen_texts = set()

        def add_intent(qt: str, entity: str, claim: str, bs: str):
            text = " ".join(qt.split())
            if text and text.lower() not in seen_texts:
                seen_texts.add(text.lower())
                all_intents.append(QueryIntent(
                    query_text=text,
                    target_entity=entity,
                    target_claim=claim,
                    target_blindspot=bs
                ))

        # Build queries per blindspot × entity × claim
        for bs in uncovered_bs + covered_bs:
            cat = bs.category
            cat_clean = re.sub(r'[^\w\s\-]', ' ', cat).strip()

            # If entities exist, build entity-targeted queries
            if all_entity_values:
                for ent in all_entity_values[:3]:
                    for claim in claims_list[:2]:
                        add_intent(
                            f"{ent} {cat_clean} {claim}",
                            entity=ent, claim=claim, bs=cat
                        )
                    add_intent(
                        f"{ent} {cat_clean} analysis",
                        entity=ent, claim="", bs=cat
                    )

            # Claim + blindspot queries (no specific entity)
            for claim in claims_list[:3]:
                add_intent(
                    f"{topic_phrase} {cat_clean} {claim}",
                    entity="", claim=claim, bs=cat
                )

            # Intent-aspect queries
            for aspect in ["criticism", "impact", "expert perspective"]:
                add_intent(
                    f"{topic_phrase} {cat_clean} {aspect}",
                    entity="", claim="", bs=cat
                )

        # Fallback if no intents
        if not all_intents:
            fallback_text = f"{topic_phrase} policy implementation details and analysis"
            add_intent(fallback_text, entity="", claim="", bs="")

        # Validate query_texts (extract strings, validate, rewrap)
        article_keywords = []
        if state.article and state.article.title:
            article_keywords = [w.lower() for w in re.findall(r'\b\w+\b', state.article.title)]

        raw_texts = [qi.query_text for qi in all_intents]
        valid_texts = self._validate_queries(raw_texts, article_keywords, state)
        valid_set = set(v.lower() for v in valid_texts)
        valid_intents = [qi for qi in all_intents if qi.query_text.lower() in valid_set]

        # De-duplicate by query_text
        seen_final = set()
        deduped = []
        for qi in valid_intents:
            key = qi.query_text.lower()
            if key not in seen_final:
                seen_final.add(key)
                deduped.append(qi)

        # Exclude already-used queries
        used_set = set(str(q).lower().strip() for q in getattr(state, "search_queries_used", []))
        unused = [qi for qi in deduped if qi.query_text.lower().strip() not in used_set]

        return unused[:3] if unused else deduped[:3]

    def _generate_fallback_queries(self, state: AgentState) -> List[str]:
        """
        Generates alternative search queries as raw strings.
        """
        intents = self._generate_initial_queries(state)
        return [qi.query_text for qi in intents]

    def _validate_queries(self, queries: List[str], article_keywords: List[str], state: Optional[AgentState] = None) -> List[str]:
        valid_queries = []
        
        stop_words = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "by", "with",
            "is", "are", "was", "were", "will", "would", "should", "can", "could", "about",
            "from", "this", "that", "these", "those", "how", "why", "what", "which"
        }
        
        filler_terms = {
            "query", "search", "google", "website", "placeholder", "duckduckgo", "url", "link",
            "find", "lookup", "results", "evidence", "articles", "news", "report"
        }

        # Build list of article-specific keywords from title, topic, and blindspots
        valid_keywords = set()
        keyword_stop_words = stop_words.union({"policy", "rules", "new", "announces", "announced", "update", "updates"})
        
        if state:
            if state.article and state.article.title:
                valid_keywords.update(
                    w.lower() for w in re.findall(r'\b\w+\b', state.article.title)
                    if w.lower() not in keyword_stop_words and len(w) > 2
                )
            if state.claims and state.claims.main_topic:
                valid_keywords.update(
                    w.lower() for w in re.findall(r'\b\w+\b', state.claims.main_topic)
                    if w.lower() not in keyword_stop_words and len(w) > 2
                )
            if state.blindspots:
                for bs in state.blindspots:
                    valid_keywords.update(
                        w.lower() for w in re.findall(r'\b\w+\b', bs.category)
                        if w.lower() not in keyword_stop_words and len(w) > 2
                    )

        for q in queries:
            q_clean = q.strip().rstrip(".?!:;, ")
            q_clean = re.sub(r'\s+', ' ', q_clean)
            
            # De-duplicate words inside the query string
            q_clean = " ".join(dict.fromkeys(q_clean.split()))
            
            # 1. Check for generic filler terms
            if any(term in q_clean.lower() for term in filler_terms):
                self.logger.info(f"Query rejected (contains filler terms): '{q_clean}'")
                continue

            # 2. Check for article-specific keywords
            query_words = {w.lower() for w in re.findall(r'\b\w+\b', q_clean)}
            if valid_keywords and not query_words.intersection(valid_keywords):
                self.logger.info(f"Query rejected (lacks article-specific keywords): '{q_clean}'")
                continue

            # 3. Check for meaningful word count
            meaningful = [w for w in q_clean.split() if w.lower() not in stop_words]
            if len(meaningful) < 4:
                # Attempt query expansion using valid keywords rather than discarding
                expanded_words = list(q_clean.split())
                added = 0
                for kw in sorted(valid_keywords):
                    if kw not in q_clean.lower():
                        expanded_words.insert(0, kw.capitalize())
                        meaningful.append(kw)
                        added += 1
                        if len(meaningful) >= 4 or added >= 3:
                            break
                q_clean = " ".join(expanded_words)
                meaningful = [w for w in q_clean.split() if w.lower() not in stop_words]
                
                if len(meaningful) < 4:
                    self.logger.info(f"Query rejected (too short: {len(meaningful)} meaningful words): '{q_clean}'")
                    continue

            valid_queries.append(q_clean)
            
        return valid_queries

    def decide(self, state: AgentState) -> PlannerDecision:
        """
        Applies rules and logic to decide on the next research action.
        Avoids LLM calls whenever possible.
        """
        import re
        try:
            search_attempts = getattr(state, "search_attempts", 0)
            confidence_score = getattr(state, "confidence_score", 0)
            evidence_list = getattr(state, "evidence", [])
            stagnation_count = getattr(state, "stagnation_count", 0)
            
            article = getattr(state, "article", None)
            title = getattr(article, "title", "") if article else ""
            
            # Extract title keywords for validation
            stop_words = {
                "new", "rule", "in", "without", "a", "valid", "may", "be", "denied", 
                "to", "for", "on", "of", "and", "or", "with", "the", "an", "at", "by", 
                "from", "2026", "2025", "about", "how", "why", "what", "is", "are", "was",
                "were", "will", "would", "should", "can", "could", "article", "report",
                "some", "any", "no", "not", "but", "yes", "this", "that", "these", "those"
            }
            title_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', title)
            title_keywords = [w for w in title_clean.split() if w.strip().lower() not in stop_words and len(w) > 2]

            # Extract article keywords (combining title and topic keywords) for validation
            topic = state.claims.main_topic if (state.claims and state.claims.main_topic) else ""
            topic_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', topic)
            topic_keywords = [w for w in topic_clean.split() if w.strip().lower() not in stop_words and len(w) > 2]
            article_keywords = list(dict.fromkeys(title_keywords + topic_keywords))

            self.logger.info(
                f"Planner deciding — attempts: {search_attempts}, "
                f"confidence: {confidence_score}%, "
                f"evidence: {len(evidence_list)}, "
                f"stagnation: {stagnation_count}"
            )

            # Rule 1: confidence >= threshold -> GENERATE_REPORT
            if confidence_score >= self.confidence_threshold:
                self.logger.info("Rule 1: Confidence threshold met, generating report")
                return PlannerDecision(
                    action="GENERATE_REPORT",
                    queries=[],
                    reasoning=f"Confidence score {confidence_score}% meets threshold {self.confidence_threshold}%"
                )

            # Rule 2: search_attempts >= max_attempts -> GENERATE_REPORT
            if search_attempts >= self.max_search_attempts:
                self.logger.info("Rule 2: Max search attempts reached, generating report")
                return PlannerDecision(
                    action="GENERATE_REPORT",
                    queries=[],
                    reasoning=f"Maximum search attempts ({self.max_search_attempts}) reached"
                )

            # Rule 3: stagnation_count >= 2 -> GENERATE_REPORT
            if stagnation_count >= 2:
                self.logger.info("Rule 3: Confidence stagnation detected, generating report")
                return PlannerDecision(
                    action="GENERATE_REPORT",
                    queries=[],
                    reasoning="Confidence delta stagnated <= 5 for 2 consecutive loops"
                )

            # Rule 4: evidence_count == 0 -> SEARCH
            if len(evidence_list) == 0:
                self.logger.info("Rule 4: No evidence gathered yet, starting initial searches")
                initial_intents = self._generate_initial_queries(state)
                query_texts = [qi.query_text for qi in initial_intents]
                valid_queries = self._validate_queries(query_texts, article_keywords, state)
                if not valid_queries:
                    valid_queries = query_texts[:3]
                return PlannerDecision(
                    action="SEARCH",
                    queries=valid_queries,
                    query_intents=initial_intents,
                    reasoning="Starting initial evidence gathering"
                )

            # If none of the rule-based shortcuts trigger, call the LLM planner.
            if not self.ollama_client.pre_call_health_check(self.config.OLLAMA_MODEL):
                self.logger.warning("Ollama pre-call health check failed. Skipping LLM planner decision and using fallback queries.")
                fallback_intents = self._generate_initial_queries(state)
                fb_texts = [qi.query_text for qi in fallback_intents]
                return PlannerDecision(
                    action="SEARCH_MORE",
                    queries=fb_texts[:3],
                    query_intents=fallback_intents[:3],
                    reasoning="Bypassed LLM planner due to health check failure"
                )

            self.logger.info("Calling LLM planner to decide next search queries")
            system_prompt = self._build_system_prompt()
            planning_prompt = self._build_planning_prompt(state)

            response = self.ollama_client.generate_json_with_retry(
                prompt=planning_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                num_predict=128,
                max_retries=0,
                timeout=10.0
            )

            decision_action = response.get("action", "SEARCH_MORE")
            reasoning = response.get("reasoning", "")
            queries = response.get("queries", [])

            if decision_action not in ["SEARCH", "SEARCH_MORE", "GENERATE_REPORT"]:
                decision_action = "SEARCH_MORE"

            if not isinstance(queries, list):
                queries = []

            # Use LLM queries as primary; supplement with programmatic only when LLM returns empty
            if decision_action != "GENERATE_REPORT":
                if queries:
                    valid_queries = queries[:3]
                    # Wrap LLM query strings in QueryIntent (no entity/claim/blindstop provenance available)
                    valid_intents = [
                        QueryIntent(query_text=q, target_entity="", target_claim="", target_blindspot="")
                        for q in valid_queries
                    ]
                else:
                    self.logger.info("LLM returned empty queries; falling back to programmatic generation.")
                    fallback_intents = self._generate_initial_queries(state)
                    valid_queries = [qi.query_text for qi in fallback_intents][:3]
                    valid_intents = fallback_intents[:3]
            else:
                valid_queries = []
                valid_intents = []

            decision = PlannerDecision(
                action=decision_action,
                queries=valid_queries,
                query_intents=valid_intents,
                reasoning=reasoning
            )
            self.logger.info(f"LLM Planner decision: {decision_action} — {reasoning} with queries: {valid_queries}")
            return decision

        except Exception as error:
            self.logger.error(f"Planner failed: {error}")
            fallback_intents = self._generate_initial_queries(state)
            fb_texts = [qi.query_text for qi in fallback_intents]
            return PlannerDecision(
                action="SEARCH_MORE",
                queries=fb_texts[:3],
                query_intents=fallback_intents[:3],
                reasoning="Fallback due to planner error"
            )

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        # Load config
        config = Config.from_env()

        # Create components
        client = OllamaClient(config)
        planner = Planner(client, config)

        # 1. SCENARIO 1 Setup
        article = ArticleData(
            url="https://test.com/policy",
            title="Government Announces New Carbon Tax Policy",
            content="The government plans to introduce carbon tax to fight climate change."
        )

        claims = ClaimAnalysis(
            main_topic="Carbon tax policy",
            key_claims=[
                "Government will introduce carbon tax",
                "Tax will reduce emissions"
            ],
            author_stance="Neutral",
            tone="Informative",
            framing_summary="Simple announcement framing"
        )

        blindspots = [
            Blindspot(
                category="Economic Impact",
                description="Lacks coverage of economic costs and job displacement.",
                importance="High",
                suggested_search_query="carbon tax economic impact small businesses"
            ),
            Blindspot(
                category="Opposing Perspective",
                description="No opposing viewpoints from trade groups.",
                importance="Medium",
                suggested_search_query="industry opposition to carbon tax"
            )
        ]

        state = AgentState(
            article=article,
            claims=claims,
            blindspots=blindspots,
            evidence=[],
            search_attempts=0,
            confidence_score=0
        )

        print("==================================================")
        print("SCENARIO 1: INITIAL DECISION")
        print("==================================================")
        decision_1 = planner.decide(state)
        print(f"Action: {decision_1.action}")
        print(f"Queries: {decision_1.queries}")
        print(f"Reasoning: {decision_1.reasoning}\n")

        # 2. SCENARIO 2 Setup
        res_high = SearchResult(
            query="carbon tax economic impact small businesses",
            title="SBA Report on Carbon Taxing",
            url="https://sba.gov/reports/carbon-tax",
            snippet="Detailed analysis of small business economic transitions.",
            source="sba.gov"
        )
        ev_high = Evidence(
            search_result=res_high,
            relevance="Adds Context",
            quality="High",
            key_insight="Highlights transitions for small businesses."
        )

        res_low = SearchResult(
            query="industry opposition to carbon tax",
            title="Blog: Why Taxes Suck",
            url="https://blogspot.com/tax-suck",
            snippet="A blog post complaining about government pricing policies.",
            source="blogspot.com"
        )
        ev_low = Evidence(
            search_result=res_low,
            relevance="Contradicts",
            quality="Low",
            key_insight="Opinion piece showing opposition on blogs."
        )

        state.search_attempts = 1
        state.confidence_score = 45
        state.evidence = [ev_high, ev_low]
        state.search_queries_used = ["carbon tax economic impact small businesses", "industry opposition to carbon tax"]

        print("==================================================")
        print("SCENARIO 2: MID-RESEARCH DECISION")
        print("==================================================")
        decision_2 = planner.decide(state)
        print(f"Action: {decision_2.action}")
        print(f"Queries: {decision_2.queries}")
        print(f"Reasoning: {decision_2.reasoning}\n")

        # 3. SCENARIO 3 Setup
        state.confidence_score = 75
        
        print("==================================================")
        print("SCENARIO 3: METRICS MET (GENERATE_REPORT)")
        print("==================================================")
        decision_3 = planner.decide(state)
        print(f"Action: {decision_3.action}")
        print(f"Queries: {decision_3.queries}")
        print(f"Reasoning: {decision_3.reasoning}\n")

    except Exception as e:
        print(f"\nError during planner execution: {e}")
