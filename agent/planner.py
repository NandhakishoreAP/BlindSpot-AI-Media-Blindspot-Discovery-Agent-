from typing import List

from models.data_models import (
    AgentState,
    PlannerDecision,
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
            
            category_lower = category.lower()
            cat_words = set(w for w in category_lower.split() if len(w) > 3)
            
            suggested_lower = bs.suggested_search_query.lower()
            suggested_words = set(w for w in suggested_lower.split() if len(w) > 3)
            
            for ev in evidence_list:
                query_lower = ev.search_result.query.lower()
                insight_lower = ev.key_insight.lower()
                
                # Check if category keywords are in the query or key insight
                if any(w in query_lower or w in insight_lower for w in cat_words):
                    coverage[category] = True
                    break
                    
                # Or check if there's high overlap with the suggested query
                query_words = set(w for w in query_lower.split() if len(w) > 3)
                if suggested_words and query_words:
                    shared = suggested_words.intersection(query_words)
                    if len(shared) / len(suggested_words) >= 0.5:
                        coverage[category] = True
                        break
                        
        return coverage

    def _build_planning_prompt(self, state: AgentState) -> str:
        """
        Builds a planning prompt using current state details, coverage, and evidence summaries.
        """
        article = getattr(state, "article", None)
        title = getattr(article, "title", "Unknown") if article else "Unknown"
        search_attempts = getattr(state, "search_attempts", 0)
        confidence_score = getattr(state, "confidence_score", 0)
        evidence_list = getattr(state, "evidence", [])
        blindspots = getattr(state, "blindspots", [])
        
        summary = self._get_evidence_summary(evidence_list)
        coverage = self._estimate_blindspot_coverage(state)
        
        coverage_str = ""
        for cat, covered in coverage.items():
            status = "Covered" if covered else "Not Covered"
            coverage_str += f"- {cat} → {status}\n"
            
        claims = getattr(state, "claims", None)
        key_claims_str = ""
        main_topic = "Unknown"
        author_stance = "Unknown"
        tone = "Unknown"
        framing_summary = "Unknown"
        
        if claims:
            main_topic = getattr(claims, "main_topic", "Unknown")
            author_stance = getattr(claims, "author_stance", "Unknown")
            tone = getattr(claims, "tone", "Unknown")
            framing_summary = getattr(claims, "framing_summary", "Unknown")
            for idx, claim in enumerate(getattr(claims, "key_claims", []), 1):
                key_claims_str += f"{idx}. {claim}\n"
                
        blindspots_list_str = ""
        for idx, bs in enumerate(blindspots, 1):
            blindspots_list_str += f"[{idx}] {bs.category} (Importance: {bs.importance}): {bs.description}\n"
            
        used_queries = getattr(state, "search_queries_used", [])
        if used_queries:
            used_queries_str = "\n".join(f"- {q}" for q in used_queries)
        else:
            used_queries_str = "No previous searches."

        prompt = (
            "ARTICLE\n\n"
            f"Title: {title}\n"
            f"Main Topic: {main_topic}\n"
            f"Author Stance: {author_stance}\n"
            f"Tone: {tone}\n"
            f"Framing Summary: {framing_summary}\n\n"
            "SEARCH STATUS\n\n"
            f"Current search attempts: {search_attempts}\n"
            f"Maximum search attempts: {self.max_search_attempts}\n\n"
            "CONFIDENCE\n\n"
            f"Current confidence score: {confidence_score}%\n"
            f"Configured threshold: {self.confidence_threshold}%\n\n"
            "EVIDENCE SUMMARY\n\n"
            f"Number of evidence items: {len(evidence_list)}\n"
            f"High-quality evidence: {summary['high_quality_count']}\n"
            f"Medium-quality evidence: {summary['medium_quality_count']}\n"
            f"Low-quality evidence: {summary['low_quality_count']}\n"
            f"Contradicting evidence: {summary['contradicts_count']}\n"
            f"Supporting evidence: {summary['supports_count']}\n"
            f"Context evidence: {summary['context_count']}\n\n"
            "BLINDSPOT COVERAGE\n\n"
            f"{coverage_str}\n"
            "BLINDSPOTS\n\n"
            f"{blindspots_list_str}\n"
            "PREVIOUS SEARCHES\n\n"
            f"{used_queries_str}\n\n"
            "INSTRUCTIONS\n\n"
            "Determine:\n"
            "1. Is evidence sufficient?\n"
            "2. Are important blindspots still uncovered?\n"
            "3. Are opposing viewpoints missing?\n"
            "4. Is more research required?\n\n"
            "If evidence quality is low, prioritize searches targeting:\n"
            "* Peer reviewed studies\n"
            "* Academic research\n"
            "* Government reports\n"
            "* Expert analysis\n"
            "* Institutional publications\n"
            "instead of general news coverage.\n\n"
            "Return ONLY a JSON object with this exact structure:\n\n"
            "{\n"
            '  "action": "SEARCH_MORE",\n'
            '  "queries": [\n'
            '    "query 1",\n'
            '    "query 2"\n'
            '  ],\n'
            '  "reasoning": "..."\n'
            "}\n\n"
            "Requirements:\n"
            "If action is SEARCH or SEARCH_MORE:\n"
            "Return 2–4 queries.\n"
            "Queries must:\n"
            "* Avoid previous searches\n"
            "* Target uncovered blindspots\n"
            "* Seek stronger evidence\n"
            "* Seek opposing viewpoints when contradicting evidence is low\n\n"
            "Respond ONLY with the JSON object."
        )
        return prompt

    def _generate_initial_queries(self, state: AgentState) -> List[str]:
        """
        Generate initial queries without using the LLM.
        """
        queries = []
        blindspots = getattr(state, "blindspots", [])
        article = getattr(state, "article", None)
        title = getattr(article, "title", "Unknown") if article else "Unknown"
        
        for bs in blindspots[:4]:
            q = getattr(bs, "suggested_search_query", "")
            if q and q.strip():
                queries.append(q.strip())
            else:
                queries.append(f"{title} {bs.category}")
                
        # Deduplicate
        unique_queries = []
        for q in queries:
            if q not in unique_queries:
                unique_queries.append(q)
        return unique_queries

    def _generate_fallback_queries(self, state: AgentState) -> List[str]:
        """
        Generates alternative search queries based on blindspots, ensuring queries are 
        non-redundant and have low keyword overlap with each other and used queries.
        """
        used_set = set()
        for q in getattr(state, "search_queries_used", []):
            used_set.add(" ".join(str(q).lower().split()))

        queries = []
        article = getattr(state, "article", None)
        title = getattr(article, "title", "Unknown") if article else "Unknown"
        claims = getattr(state, "claims", None)
        topic = getattr(claims, "main_topic", title) if claims else title
        
        blindspots = getattr(state, "blindspots", [])
        if not blindspots:
            queries.append(f"{topic} missing perspectives")
            queries.append(f"{topic} alternative viewpoints")

        for bs in blindspots:
            category = bs.category
            candidates = [
                f"{topic} {category} perspective",
                f"{topic} {category} research studies",
                f"{topic} {category} government report",
                f"{topic} {category} expert analysis"
            ]
            
            for candidate in candidates:
                norm = " ".join(candidate.lower().split())
                if norm in used_set:
                    continue
                    
                skip = False
                candidate_words = set(w for w in norm.split() if len(w) > 3)
                
                for existing in queries:
                    existing_norm = " ".join(existing.lower().split())
                    existing_words = set(w for w in existing_norm.split() if len(w) > 3)
                    if candidate_words and existing_words:
                        shared = candidate_words.intersection(existing_words)
                        overlap = len(shared) / max(len(candidate_words), len(existing_words))
                        if overlap > 0.6:
                            skip = True
                            break
                            
                for used_query in used_set:
                    used_words = set(w for w in used_query.split() if len(w) > 3)
                    if candidate_words and used_words:
                        shared = candidate_words.intersection(used_words)
                        overlap = len(shared) / max(len(candidate_words), len(used_words))
                        if overlap > 0.6:
                            skip = True
                            break
                            
                if not skip:
                    queries.append(candidate)
                    if len(queries) >= 4:
                        break
            if len(queries) >= 4:
                break
                
        return queries[:4] if len(queries) >= 2 else (queries + [f"{topic} expert analysis", f"{topic} alternative studies"])[:4]

    def decide(self, state: AgentState) -> PlannerDecision:
        """
        Applies rules and logic to decide on the next research action.
        """
        try:
            search_attempts = getattr(state, "search_attempts", 0)
            confidence_score = getattr(state, "confidence_score", 0)
            evidence_list = getattr(state, "evidence", [])
            
            self.logger.info(
                f"Planner deciding — attempts: {search_attempts}, "
                f"confidence: {confidence_score}%, "
                f"evidence: {len(evidence_list)}"
            )

            # RULE 1: Max search attempts
            if search_attempts >= self.max_search_attempts:
                self.logger.info("Max search attempts reached, forcing report generation")
                decision = PlannerDecision(
                    action="GENERATE_REPORT",
                    queries=[],
                    reasoning="Maximum search attempts reached"
                )
                self.logger.debug(f"Decision details: action={decision.action} queries={len(decision.queries)}")
                return decision

            # Tally counts for RULE 2 and Saturation checks
            summary = self._get_evidence_summary(evidence_list)
            high_quality_count = summary["high_quality_count"]
            evidence_count = len(evidence_list)

            # Saturation detection: weak evidence only
            is_saturated_with_weak = (evidence_count >= 5 and high_quality_count == 0)

            # RULE 2: Confidence threshold
            if (confidence_score >= self.confidence_threshold 
                    and high_quality_count > 0 
                    and not is_saturated_with_weak):
                self.logger.info("Confidence threshold met, generating report")
                decision = PlannerDecision(
                    action="GENERATE_REPORT",
                    queries=[],
                    reasoning=f"Confidence score {confidence_score}% meets threshold with high-quality evidence"
                )
                self.logger.debug(f"Decision details: action={decision.action} queries={len(decision.queries)}")
                return decision

            # RULE 3: Initial search
            if search_attempts == 0 and len(evidence_list) == 0:
                decision = PlannerDecision(
                    action="SEARCH",
                    queries=self._generate_initial_queries(state),
                    reasoning="Starting initial evidence gathering"
                )
                self.logger.debug(f"Decision details: action={decision.action} queries={len(decision.queries)}")
                return decision

            # RULE 4: Ollama Planner call
            system_prompt = self._build_system_prompt()
            planning_prompt = self._build_planning_prompt(state)

            response = self.ollama_client.generate_json_with_retry(
                prompt=planning_prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )

            decision_action = response.get("action", "SEARCH_MORE")
            reasoning = response.get("reasoning", "")
            queries = response.get("queries", [])

            if decision_action not in ["SEARCH", "SEARCH_MORE", "GENERATE_REPORT"]:
                decision_action = "SEARCH_MORE"

            if not isinstance(queries, list):
                queries = []

            # Post-processing override: weak evidence saturation
            if decision_action == "GENERATE_REPORT" and is_saturated_with_weak:
                self.logger.info("Overriding decision to SEARCH_MORE due to weak-evidence saturation.")
                decision_action = "SEARCH_MORE"
                reasoning = "Overridden to SEARCH_MORE because gathered evidence is low quality; seeking higher-quality sources."
                queries = self._generate_fallback_queries(state)

            if decision_action != "GENERATE_REPORT" and not queries:
                queries = self._generate_fallback_queries(state)

            decision = PlannerDecision(
                action=decision_action,
                queries=queries,
                reasoning=reasoning
            )
            self.logger.info(f"Planner decision: {decision_action} — {reasoning}")
            self.logger.debug(f"Decision details: action={decision.action} queries={len(decision.queries)}")
            return decision

        except Exception as error:
            # RULE 5: Fallback due to errors
            self.logger.error(f"Planner failed: {error}")
            decision = PlannerDecision(
                action="SEARCH_MORE",
                queries=self._generate_fallback_queries(state),
                reasoning="Fallback due to planner error"
            )
            self.logger.debug(f"Decision details: action={decision.action} queries={len(decision.queries)}")
            return decision

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
