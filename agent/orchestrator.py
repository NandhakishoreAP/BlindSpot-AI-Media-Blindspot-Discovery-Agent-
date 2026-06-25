from datetime import datetime, timezone
import urllib.parse
from typing import List

from models.data_models import (
    AgentState,
    BlindspotReport,
    ArticleData,
    Blindspot,
    Evidence,
    ClaimAnalysis
)


from tools.article_extractor import ArticleExtractor
from tools.search_tool import SearchTool
from llm.ollama_client import OllamaClient
from llm.claim_analyzer import ClaimAnalyzer
from llm.blindspot_detector import BlindspotDetector
from llm.evidence_evaluator import EvidenceEvaluator

# Gracefully handle ReportGenerator if not yet fully implemented
try:
    from llm.report_generator import ReportGenerator
except ImportError:
    ReportGenerator = None

from agent.planner import Planner
from utils.logger import get_logger
from utils.diagnostic_logger import DiagnosticLogger
from utils.content_cleaner import clean_article_content, extract_main_article_body
from utils.entity_extractor import EntityExtractor
from utils.understanding_score import calculate_understanding_score
from config import Config

class MediaBlindspotAgent:
    """
    The main orchestrator that coordinates all pipeline components to perform media blindspot analysis.
    """

    def __init__(self, config: Config) -> None:
        """
        Stores config, instantiates logger, runs health check, and instantiates components.
        """
        self.config: Config = config
        self.logger = get_logger("orchestrator")
        
        # Instantiate Ollama Client first (triggers model warmup with fallback)
        self.ollama_client = OllamaClient(config)
        self.ollama_available = self.ollama_client.available
        
        if self.ollama_available:
            self.logger.info(f"Ollama available | Active model: {self.ollama_client.model}")
        else:
            self.logger.error("No model available after fallback attempts. OllamaClient unavailable.")
            self.ollama_failure_reason = (
                "Could not initialize any language model. "
                "Ollama may not be running, or no compatible model is installed. "
                "Please ensure Ollama is running and at least one of the following "
                "models is installed: " + ", ".join(
                    [config.OLLAMA_MODEL] + ["qwen3:4b", "qwen3:1.7b", "phi4-mini"]
                )
            )
            # Remaining components won't be instantiated; analyze() will abort immediately.

        # Instantiate remaining components (only if model is available)
        self.article_extractor = ArticleExtractor(config)
        self.search_tool = SearchTool(config)
        self.entity_extractor = EntityExtractor()
        self.diag_logger = DiagnosticLogger()
        
        if self.ollama_available:
            self.claim_analyzer = ClaimAnalyzer(self.ollama_client, config)
            self.blindspot_detector = BlindspotDetector(self.ollama_client, config)
            self.evidence_evaluator = EvidenceEvaluator(self.ollama_client, config)
            self.planner = Planner(self.ollama_client, config)
            
            if ReportGenerator is not None:
                try:
                    self.report_generator = ReportGenerator(self.ollama_client, config)
                except Exception as e:
                    self.logger.warning(f"Failed to instantiate ReportGenerator: {e}")
                    self.report_generator = None
            else:
                self.report_generator = None
        else:
            self.claim_analyzer = None
            self.blindspot_detector = None
            self.evidence_evaluator = None
            self.planner = None
            self.report_generator = None

        self.logger.info(f"Agent initialized (model available: {self.ollama_available})")

    def _update_state_status(self, state: AgentState, status: str) -> None:
        """
        Updates agent lifecycle state status and logs transition.
        """
        old_status = state.status
        state.status = status
        self.logger.info(f"Transitioning phase: {old_status} -> {status}")
        if not state.phase_timestamps:
            state.phase_timestamps = {}
        state.phase_timestamps[status] = datetime.now(timezone.utc).isoformat()

    def _calculate_source_diversity(self, evidence: List[Evidence]) -> dict:
        """
        Calculates domain and category diversity of the collected evidence.
        """
        diversity = {
            "unique_domains": 0,
            "academic": 0,
            "government": 0,
            "news": 0,
            "think_tank": 0,
            "industry": 0,
            "social_media": 0,
            "blog": 0,
            "other": 0
        }
        if not evidence:
            return diversity
            
        unique_domains = set()
        
        for ev in evidence:
            url = ""
            if hasattr(ev, "search_result") and ev.search_result:
                url = getattr(ev.search_result, "url", "")
            if not url:
                continue
            try:
                parsed = urllib.parse.urlparse(url)
                netloc = parsed.netloc.lower()
                if netloc.startswith("www."):
                    netloc = netloc[4:]
            except Exception as e:
                self.logger.warning(f"Domain parse failed ({orchestrator.py}:121): {e}")
                netloc = ""
                
            if netloc:
                unique_domains.add(netloc)
                
            cat = SearchTool.classify_source_type(url)
            if cat in diversity:
                diversity[cat] += 1
            else:
                diversity["other"] += 1
                
        diversity["unique_domains"] = len(unique_domains)
        return diversity

    def _run_search_loop(self, state: AgentState, overall_start_time: float, progress_callback=None) -> None:
        """
        Runs the iterative autonomous research loop querying the planner and gathering evidence.
        """
        import time
        max_loop_iterations = self.config.MAX_SEARCH_ATTEMPTS + 2
        loop_count = 0
        previous_confidence = getattr(state, "confidence_score", 0)
        state.stagnation_count = 0
        state.confidence_history = [previous_confidence]
        
        research_start = time.time()
        
        total_queries_run = 0
        round_count = 0
        
        research_budget = 90.0  # 90s for research loop (enforced below)

        while True:
            try:
                # Phase budget: research loop has own budget independent of extraction/analysis
                elapsed_research = time.time() - research_start
                if elapsed_research >= research_budget:
                    self.logger.warning(f"Research phase budget exceeded: {elapsed_research:.2f}s (limit {research_budget:.0f}s). Stopping search loop.")
                    state.completion_reason = "research_budget_exceeded"
                    break
                    
                # Search rounds budget check
                if round_count >= 2:
                    self.logger.info("Max search rounds (2) reached. Stopping search loop.")
                    state.completion_reason = "max_attempts_reached"
                    break

                # Query planner
                # If circuit breaker is triggered (llm_failures >= 3), bypass planner LLM
                llm_failures = self.ollama_client.failures_count
                if llm_failures >= 3:
                    self.logger.warning("Circuit breaker triggered: skipping planner LLM call and requesting programmatic report.")
                    state.completion_reason = "max_attempts_reached"
                    break
                    
                decision = self.planner.decide(state)
                
                # Record LLM failures count in state
                state.llm_failures = self.ollama_client.failures_count

                self.logger.info(f"Planner decision: action={decision.action}, reasoning={decision.reasoning}, queries={decision.queries}")
                
                # Record planner decision
                if not state.planner_decisions:
                    state.planner_decisions = []
                state.planner_decisions.append(decision)

                if decision.action == "GENERATE_REPORT":
                    if state.confidence_score >= self.config.CONFIDENCE_THRESHOLD:
                        state.completion_reason = "confidence_threshold_reached"
                    else:
                        state.completion_reason = "planner_requested_report"
                    break
                    
                if not decision.queries:
                    self.logger.warning("Planner returned no queries. Generating fallback queries from blindspots.")
                    # Guarantee at least one query per blindspot
                    fallback_queries = []
                    for bs in state.blindspots:
                        sq = getattr(bs, "suggested_search_query", "").strip()
                        if sq:
                            fallback_queries.append(sq)
                    if not fallback_queries:
                        # Last resort: generate from blindspot categories + topic
                        topic = getattr(state.claims, "main_topic", "") if state.claims else ""
                        for bs in state.blindspots[:3]:
                            cat = getattr(bs, "category", "")
                            q = f"{topic} {cat}".strip()
                            if len(q.split()) >= 3:
                                fallback_queries.append(q)
                    if fallback_queries:
                        decision.queries = fallback_queries[:3]
                        self.logger.info(f"Generated {len(decision.queries)} fallback queries: {decision.queries}")
                    else:
                        state.completion_reason = "planner_requested_report"
                        break
                    
                # Slice queries to max 3 per round
                queries = decision.queries[:3]
                
                # Propagate query_intents to state for downstream tracking
                if decision.query_intents:
                    state.query_intents.extend(decision.query_intents[:3])
                
                # Total query budget limit check
                if total_queries_run + len(queries) > 6:
                    allowed_count = 6 - total_queries_run
                    if allowed_count <= 0:
                        self.logger.info("Total query budget (6) reached. Stopping search loop.")
                        state.completion_reason = "max_attempts_reached"
                        break
                    queries = queries[:allowed_count]
                    
                # Execute searches
                self._update_state_status(state, "searching")
                if progress_callback:
                    progress_callback("searching", state)
                    
                self.logger.info(f"Executing queries: {queries}")
                
                # Extract article and blindspot keywords
                import re
                stop_words = {
                    "new", "rule", "in", "without", "a", "valid", "may", "be", "denied", 
                    "to", "for", "on", "of", "and", "or", "with", "the", "an", "at", "by", 
                    "from", "2026", "2025", "about", "how", "why", "what", "is", "are", "was",
                    "were", "will", "would", "should", "can", "could", "article", "report",
                    "some", "any", "no", "not", "but", "yes", "this", "that", "these", "those"
                }
                article_text = f"{state.article.title} {state.claims.main_topic if state.claims else ''}"
                article_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', article_text)
                article_keywords = {w.lower() for w in article_clean.split() if w.strip().lower() not in stop_words and len(w) > 2}
                
                blindspot_text = " ".join(f"{bs.category} {bs.description}" for bs in state.blindspots)
                blindspot_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', blindspot_text)
                blindspot_keywords = {w.lower() for w in blindspot_clean.split() if w.strip().lower() not in stop_words and len(w) > 2}

                start_time_search = time.time()
                search_results = self.search_tool.search_multiple(
                    queries,
                    article_keywords=article_keywords,
                    blindspot_keywords=blindspot_keywords
                )
                self.logger.info(f"TIMING: Search execution completed in {time.time() - start_time_search:.2f} seconds")
                
                total_queries_run += len(queries)
                round_count += 1
                
                # Store query -> evidence traceability
                if not state.query_history:
                    state.query_history = {}
                for q in queries:
                    if q not in state.query_history:
                        state.query_history[q] = []
                for res in search_results:
                    q = res.query
                    url = res.url
                    if q in state.query_history and url not in state.query_history[q]:
                        state.query_history[q].append(url)
                
                # Metrics updates
                state.metrics["searches_executed"] += 1
                state.metrics["results_collected"] += len(search_results)
                
                # Keep queries in search_queries_used
                search_queries_used = getattr(state, "search_queries_used", [])
                for q in queries:
                    if q not in search_queries_used:
                        search_queries_used.append(q)
                state.search_queries_used = search_queries_used
                
                # Increment search attempts
                state.search_attempts = getattr(state, "search_attempts", 0) + 1
                
                # Persist search results in state
                state.search_results.extend(search_results)
                
                # Search failure recovery logic
                if not search_results:
                    self.logger.warning("No search results returned")
                    if state.search_attempts >= self.config.MAX_SEARCH_ATTEMPTS:
                        state.completion_reason = "max_attempts_reached"
                        break
                    else:
                        continue
                        
                # Deduplicate incoming search results before calling evidence evaluator
                existing_urls = set(getattr(ev.search_result, "url", "").lower().rstrip("/") for ev in getattr(state, "evidence", []))
                unique_search_results = []
                seen_in_current_batch = set()
                for res in search_results:
                    url = getattr(res, "url", "").lower().rstrip("/")
                    if not url:
                        continue
                    if url not in existing_urls and url not in seen_in_current_batch:
                        unique_search_results.append(res)
                        seen_in_current_batch.add(url)
                    else:
                        self.logger.info(f"Skipping duplicate search result URL before evaluation: {url}")
                
                if not unique_search_results:
                    self.logger.info("All search results in this loop iteration were duplicate URLs.")
                    if state.search_attempts >= self.config.MAX_SEARCH_ATTEMPTS:
                        state.completion_reason = "max_attempts_reached"
                        break
                    continue
                    
                # Evaluate evidence
                self._update_state_status(state, "evaluating_evidence")
                if progress_callback:
                    progress_callback("evaluating_evidence", state)
                    
                start_time_eval = time.time()
                
                # Evidence evaluation: try LLM with 25s timeout, fall back to heuristic
                used_fallback_evidence_round = False
                try:
                    new_evidence, used_fallback_evidence_round = self.evidence_evaluator.evaluate(
                        state.article,
                        state.claims,
                        state.blindspots,
                        unique_search_results
                    )
                except Exception as eval_err:
                    self.logger.warning(f"Evidence evaluation error: {eval_err}. Using heuristic fallback.")
                    used_fallback_evidence_round = True
                    new_evidence = self.evidence_evaluator.heuristic_evaluate(
                        state.article,
                        state.claims,
                        state.blindspots,
                        unique_search_results[:4]
                    )
                if used_fallback_evidence_round:
                    state.used_fallback_evidence = True
                self.logger.info(f"TIMING: Evidence evaluation completed in {time.time() - start_time_eval:.2f} seconds")
                
                # Record LLM failures count in state
                state.llm_failures = self.ollama_client.failures_count

                # Empty evidence recovery
                if not new_evidence:
                    self.logger.info("No useful evidence found")
                    if state.search_attempts >= self.config.MAX_SEARCH_ATTEMPTS:
                        state.completion_reason = "max_attempts_reached"
                        break
                    continue
                    
                # Extend evidence
                state_evidence = getattr(state, "evidence", [])
                state_evidence.extend(new_evidence)
                state.evidence = state_evidence
                
                # Update blindspot coverage
                state.blindspot_coverage = self.evidence_evaluator.analyze_blindspot_coverage(
                    state.blindspots,
                    state.evidence
                )
                
                # Metrics update
                state.metrics["evidence_collected"] += len(new_evidence)
                
                # Update confidence score
                old_confidence = state.confidence_score
                state.confidence_score = self.evidence_evaluator.calculate_confidence(
                    state.blindspots,
                    state.evidence
                )
                state.confidence_history.append(state.confidence_score)
                self.logger.info(f"Confidence score updated: {old_confidence}% -> {state.confidence_score}%")
                
                # Store confidence details with reliability
                llm_failures = getattr(state, "llm_failures", 0)
                llm_calls = max(llm_failures + 1, 1)
                details = self.evidence_evaluator.get_confidence_details(
                    state.blindspots,
                    state.evidence,
                    llm_failures=llm_failures,
                    llm_calls=llm_calls
                )
                setattr(state, "confidence_details", details)
                
                # Source diversity calculation
                diversity = self._calculate_source_diversity(state.evidence)
                state.source_diversity = diversity
                
                unique_domains_list = list(set(ev.search_result.source.lower().strip() for ev in state.evidence))
                state.unique_domains = unique_domains_list
                
                confidence_delta = state.confidence_score - previous_confidence
                state.last_confidence_delta = confidence_delta
                
                if confidence_delta <= 5:
                    state.stagnation_count += 1
                else:
                    state.stagnation_count = 0
                
                # Enhanced Logging
                delta_str = f"+{confidence_delta}%" if confidence_delta > 0 else f"{confidence_delta}%"
                self.logger.info(
                    f"Loop {loop_count}:\n"
                    f"Evidence count: {len(state.evidence)}\n"
                    f"Confidence: {state.confidence_score}%\n"
                    f"Delta: {delta_str}\n"
                    f"Unique domains: {diversity['unique_domains']}\n"
                    f"Time elapsed: {int(elapsed_total)} seconds"
                )
                
                if state.stagnation_count >= 2:
                    self.logger.info("Research stagnation detected")
                    state.completion_reason = "research_saturation"
                    break
                    
                previous_confidence = state.confidence_score
                
                if state.search_attempts >= self.config.MAX_SEARCH_ATTEMPTS:
                    state.completion_reason = "max_attempts_reached"
                    break
            except Exception as loop_err:
                self.logger.error(f"Error in search loop iteration: {loop_err}. Incrementing attempts and continuing.")
                state.search_attempts = getattr(state, "search_attempts", 0) + 1
                if state.search_attempts >= self.config.MAX_SEARCH_ATTEMPTS:
                    state.completion_reason = "error_in_search_loop"
                    break

    def _abort_with_report(self, article: ArticleData, reason: str) -> BlindspotReport:
        """Create an aborted report with consistent formatting."""
        state = AgentState(article=article)
        state.started_at = datetime.now(timezone.utc).isoformat()
        state.completed_at = datetime.now(timezone.utc).isoformat()
        state.completion_reason = reason
        state.confidence_score = 0
        state.research_duration_seconds = 0
        state.source_diversity = {}
        state.metrics = {"searches_executed": 0, "results_collected": 0, "evidence_collected": 0}
        state.phase_timestamps = {"article_extraction": datetime.now(timezone.utc).isoformat()}
        self._update_state_status(state, "complete")
        self.diag_logger.flush(article.url)
        return BlindspotReport(
            article_title=article.title or "Unknown",
            article_url=article.url or "",
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            main_topic="Analysis could not be completed.",
            key_claims=[],
            blindspots=[],
            evidence_count=0,
            blindspot_score=0,
            score_reasoning=f"Analysis could not be completed. Reason: {reason}.",
            missing_categories=[],
            balanced_conclusion=f"Analysis could not be completed. Reason: {reason}",
            search_attempts=0,
            confidence_score=0,
            confidence_label="Very Low Confidence",
            evidence_strength="None",
            completion_reason=reason,
            research_duration=0,
            source_diversity={},
            report_version="1.0",
            search_metrics={},
            evidence_balance={},
            quality_distribution={},
            query_history={},
            phase_timestamps={},
            blindspot_coverage={}
        )

    def analyze(self, url: str) -> BlindspotReport:
        """
        Executes the entire research pipeline from article URL extraction to final report generation.
        """
        import time
        try:
            # Lifecycle log
            self.logger.info(f"Starting BlindSpot AI analysis for: {url}")
            
            # Abort immediately if no model available
            if not self.ollama_available:
                self.logger.error("Cannot analyze: no language model available.")
                dummy = ArticleData(
                    url=url, title="Unknown", content="",
                    access_restricted=False, is_metadata_only=False
                )
                return self._abort_with_report(dummy, getattr(self, "ollama_failure_reason", "model_unavailable"))
            
            overall_start_time = time.time()
            
            # Phase budgets
            BUDGET_EXTRACTION = 60.0
            BUDGET_ANALYSIS = 60.0
            BUDGET_RESEARCH = 60.0
            BUDGET_REPORT = 30.0

            # Phase 1: Article Extraction
            start_extract = time.time()
            article = self.article_extractor.extract(url)
            if time.time() - start_extract > BUDGET_EXTRACTION:
                self.logger.warning("Extraction phase exceeded budget")
            elapsed_extract = time.time() - start_extract
            self.logger.info(f"TIMING: Extraction completed in {elapsed_extract:.2f} seconds")
            self.logger.info(f"Article extracted: {article.title} ({article.word_count} words, quality: {article.article_quality})")
            self.diag_logger.log("article_extraction", duration=elapsed_extract, outputs={
                "title": article.title, "url": article.url, "word_count": article.word_count,
                "paragraph_count": article.paragraph_count,
                "access_restricted": getattr(article, "access_restricted", False),
                "is_metadata_only": getattr(article, "is_metadata_only", False),
                "article_quality": getattr(article, "article_quality", "High")
            })

            # Diagnostic validation output
            self.logger.info(
                f"ARTICLE VALIDATION: "
                f"word_count={article.word_count}, "
                f"paragraph_count={article.paragraph_count}, "
                f"metadata_only={article.is_metadata_only}, "
                f"validation_reason={'metadata_only' if article.is_metadata_only else 'access_restricted' if article.access_restricted else 'passed'}, "
                f"article_quality={article.article_quality}"
            )

            # Phase 2: Abort only for metadata-only, restricted, or empty content
            has_content = bool(article.content and article.content.strip())
            is_restricted = getattr(article, "access_restricted", False)
            is_metadata = getattr(article, "is_metadata_only", False)
            
            if is_metadata or is_restricted or not has_content:
                reason = "extraction_failed"
                if not has_content:
                    reason = "empty_content"
                    self.logger.warning("Article has no content. Aborting analysis.")
                elif is_metadata:
                    reason = "metadata_only"
                    self.logger.warning("Article is metadata-only. Aborting analysis.")
                elif is_restricted:
                    reason = "access_restricted"
                    self.logger.warning("Article is access-restricted. Aborting analysis.")
                return self._abort_with_report(article, reason)
            
            # Phase 2: Initialize State
            state = AgentState(article=article)
            state.search_results = []
            state.started_at = datetime.now(timezone.utc).isoformat()
            state.phase_timestamps = {
                "article_extraction": datetime.now(timezone.utc).isoformat()
            }
            state.metrics = {
                "searches_executed": 0,
                "results_collected": 0,
                "evidence_collected": 0
            }
            self._update_state_status(state, "extraction")
            
            # Phase 2.5: Entity Extraction
            state.phase_timestamps["entity_extraction"] = datetime.now(timezone.utc).isoformat()
            self._update_state_status(state, "extracting_entities")
            start_entities = time.time()
            entities = self.entity_extractor.extract(
                article.content, article.title
            )
            if time.time() - start_entities > BUDGET_ANALYSIS * 0.3:
                self.logger.warning("Entity extraction exceeded phase budget")
            elapsed_entities = time.time() - start_entities
            state.article_entities = entities
            entity_count = sum(len(v) for v in entities.values())
            self.logger.info(f"Entity extraction complete: {entity_count} entities in {sum(1 for v in entities.values() if v)} categories")
            self.diag_logger.log("entity_extraction", duration=elapsed_entities, outputs={
                "total_entities": entity_count,
                "categories": {k: len(v) for k, v in entities.items() if v}
            })

            # Phase 2: Content Cleaning (before claim analysis)
            cleaned = clean_article_content(state.article.content)
            cleaned = extract_main_article_body(cleaned)
            state.article.content = cleaned

            # Phase 3: Claim Analysis
            state.phase_timestamps["claim_analysis"] = datetime.now(timezone.utc).isoformat()
            self._update_state_status(state, "analyzing_claims")
            
            start_claims = time.time()
            claims, used_fallback_claims = self.claim_analyzer.analyze(article)
            elapsed_claims = time.time() - start_claims
            self.logger.info(f"TIMING: Claims analysis completed in {elapsed_claims:.2f} seconds")
            
            state.used_fallback_claims = used_fallback_claims
            if used_fallback_claims:
                self.logger.warning("Claim analysis used fallback (LLM unavailable or failed)")
            
            # If LLM failed, return early with explicit reason
            if not claims or claims.main_topic == "Unknown" or len(claims.key_claims) == 0:
                self.logger.warning("Claim analysis returned no valid claims. Unable to determine reliably.")
                state.claims = claims if claims else None
                state.blindspots = []
                state.evidence = []
                state.completion_reason = "no_valid_claims"
                state.confidence_score = 0
                state.completed_at = datetime.now(timezone.utc).isoformat()
                state.research_duration_seconds = 0
                state.source_diversity = self._calculate_source_diversity([])
                state.phase_timestamps["analysis_complete"] = datetime.now(timezone.utc).isoformat()
                self._update_state_status(state, "complete")
                self.diag_logger.flush(state.article.url)
                from models.data_models import BlindspotReport
                return BlindspotReport(
                    article_title=state.article.title or "Unknown",
                    article_url=state.article.url or "",
                    analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                    main_topic="Unable to determine - claim analysis failed",
                    key_claims=[],
                    blindspots=[],
                    evidence_count=0,
                    blindspot_score=0,
                    score_reasoning="Claim analysis could not produce reliable claims from this article.",
                    missing_categories=[],
                    balanced_conclusion="Analysis could not be completed. Unable to determine reliably - claim analysis failed to identify valid claims.",
                    search_attempts=0,
                    confidence_score=0,
                    confidence_label="Low Confidence",
                    evidence_strength="None",
                    completion_reason="no_valid_claims",
                    research_duration=0,
                    source_diversity={},
                    report_version="1.0",
                    search_metrics={},
                    evidence_balance={},
                    quality_distribution={},
                    query_history={},
                    phase_timestamps=state.phase_timestamps,
                    blindspot_coverage={}
                )
            state.claims = claims
            self.logger.info("Claims analyzed")
            self.diag_logger.log("claim_analysis", duration=elapsed_claims if 'elapsed_claims' in dir() else None, outputs={
                "main_topic": claims.main_topic,
                "key_claims_count": len(claims.key_claims),
                "key_claims": claims.key_claims[:3]
            })
            
            # Record LLM failures count in state
            state.llm_failures = self.ollama_client.failures_count

            # Phase 3.5: Article Understanding Check
            score_result = calculate_understanding_score(
                article, claims, state.article_entities,
                used_fallback_claims=used_fallback_claims,
                metadata_only_article=getattr(article, "is_metadata_only", False)
            )
            state.understanding_score = score_result["score"]
            self.diag_logger.log("understanding_score", outputs={
                "score": score_result["score"],
                "decision": score_result["decision"],
                "components": score_result.get("components", {})
            })

            # Only abort on understanding score for metadata-only articles
            if score_result["decision"] == "abort" and getattr(article, "is_metadata_only", False):
                self.logger.warning(
                    f"Understanding score {score_result['score']}/100 — "
                    f"aborting analysis (metadata-only). Components: {score_result['components']}"
                )
                state.blindspots = []
                state.evidence = []
                state.completion_reason = "low_understanding"
                state.confidence_score = 0
                state.completed_at = datetime.now(timezone.utc).isoformat()
                state.research_duration_seconds = 0
                state.source_diversity = self._calculate_source_diversity([])
                state.phase_timestamps["analysis_complete"] = datetime.now(timezone.utc).isoformat()
                self._update_state_status(state, "complete")
                self.diag_logger.flush(state.article.url)
                from models.data_models import BlindspotReport
                return BlindspotReport(
                    article_title=state.article.title,
                    article_url=state.article.url,
                    analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                    main_topic=state.claims.main_topic if state.claims else "Unknown",
                    key_claims=state.claims.key_claims if state.claims else [],
                    blindspots=[],
                    evidence_count=0,
                    blindspot_score=0,
                    score_reasoning="Article understanding too low for reliable blindspot detection.",
                    missing_categories=[],
                    balanced_conclusion="Analysis could not be completed. The article content was insufficient for reliable blindspot detection.",
                    search_attempts=0,
                    confidence_score=0,
                    confidence_label="Low Confidence",
                    evidence_strength="None",
                    completion_reason="low_understanding",
                    research_duration=0,
                    source_diversity={},
                    report_version="1.0",
                    search_metrics={},
                    evidence_balance={},
                    quality_distribution={},
                    query_history={},
                    phase_timestamps=state.phase_timestamps,
                    blindspot_coverage={}
                )
            elif score_result["decision"] == "abort":
                # Non-metadata article with low score: proceed with warning
                self.logger.warning(
                    f"Understanding score {score_result['score']}/100 — "
                    f"proceeding with reduced confidence. Components: {score_result['components']}"
                )

            # Phase 4: Blindspot Detection
            state.phase_timestamps["blindspot_detection"] = datetime.now(timezone.utc).isoformat()
            self._update_state_status(state, "detecting_blindspots")
            
            start_bs = time.time()
            blindspots, used_fallback_blindspots = self.blindspot_detector.detect(article, state.claims)
            elapsed_bs = time.time() - start_bs
            self.logger.info(f"TIMING: Blindspot detection completed in {elapsed_bs:.2f} seconds")
            
            if time.time() - start_bs > BUDGET_ANALYSIS * 0.4:
                self.logger.warning("Blindspot detection exceeded phase budget")

            # Validate each blindspot against article entities and claims
            validated_blindspots = []
            for bs in blindspots:
                is_valid, reason, linked_entities, linked_claims = (
                    self.blindspot_detector.validate_blindspot(bs, state.article_entities, state.claims)
                )
                if is_valid:
                    bs.related_entities = linked_entities
                    bs.related_claims = linked_claims
                    validated_blindspots.append(bs)
                else:
                    self.logger.info(f"Rejected blindspot: '{bs.category}' reason={reason}")

            state.blindspots = validated_blindspots
            state.used_fallback_blindspots = used_fallback_blindspots
            
            if not state.blindspots:
                self.logger.info("No reliable blindspots identified after validation. Skipping research loop.")
                state.evidence = []
                state.completion_reason = "no_blindspots"
                state.confidence_score = 0
                state.completed_at = datetime.now(timezone.utc).isoformat()
                state.research_duration_seconds = 0
                state.source_diversity = self._calculate_source_diversity([])
                state.phase_timestamps["analysis_complete"] = datetime.now(timezone.utc).isoformat()
                self._update_state_status(state, "complete")
                self.diag_logger.flush(state.article.url)
                from models.data_models import BlindspotReport
                return BlindspotReport(
                    article_title=state.article.title or "Unknown",
                    article_url=state.article.url or "",
                    analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                    main_topic=state.claims.main_topic if state.claims else "Unknown",
                    key_claims=state.claims.key_claims if state.claims else [],
                    blindspots=[],
                    evidence_count=0,
                    blindspot_score=0,
                    score_reasoning="No reliable blindspots identified. The article appears to cover its topic without major omissions.",
                    missing_categories=[],
                    balanced_conclusion="Analysis complete. No reliable blindspots identified. The article covers key perspectives within its scope.",
                    search_attempts=0,
                    confidence_score=0,
                    confidence_label="Low Confidence",
                    evidence_strength="None",
                    completion_reason="no_blindspots",
                    research_duration=0,
                    source_diversity={},
                    report_version="1.0",
                    search_metrics={},
                    evidence_balance={},
                    quality_distribution={},
                    query_history={},
                    phase_timestamps=state.phase_timestamps,
                    blindspot_coverage={}
                )
            self.logger.info(f"Found {len(state.blindspots)} validated blindspots to investigate")
            self.diag_logger.log("blindspot_detection", duration=elapsed_bs if 'elapsed_bs' in dir() else None, outputs={
                "total_detected": len(blindspots),
                "validated": len(validated_blindspots),
                "categories": [bs.category for bs in state.blindspots]
            })
            
            # Initialize coverage mapping
            state.blindspot_coverage = self.evidence_evaluator.analyze_blindspot_coverage(
                state.blindspots,
                state.evidence
            )
            
            # Phase 5: Agentic Search Loop
            self.logger.info("Research started")
            self._update_state_status(state, "researching")
            state.phase_timestamps["research_start"] = datetime.now(timezone.utc).isoformat()
            
            start_search = time.time()
            self._run_search_loop(state, overall_start_time)
            elapsed_search = time.time() - start_search
            self.logger.info(f"TIMING: Search loop completed in {elapsed_search:.2f} seconds")
            


            state.phase_timestamps["research_end"] = datetime.now(timezone.utc).isoformat()
            self.diag_logger.log("search_loop", duration=elapsed_search if 'elapsed_search' in dir() else None, outputs={
                "search_attempts": state.search_attempts,
                "evidence_count": len(state.evidence),
                "confidence_score": state.confidence_score,
                "queries_used": getattr(state, "search_queries_used", [])[:5]
            })
            
            # Calculate final research timeline before generating report
            state.completed_at = datetime.now(timezone.utc).isoformat()
            start_time_dt = datetime.fromisoformat(state.started_at)
            end_time_dt = datetime.fromisoformat(state.completed_at)
            state.research_duration_seconds = int((end_time_dt - start_time_dt).total_seconds())

            # Recalculate understanding score with all fallback flags
            final_score_result = calculate_understanding_score(
                state.article, state.claims, state.article_entities,
                used_fallback_claims=state.used_fallback_claims,
                used_fallback_blindspots=state.used_fallback_blindspots,
                used_fallback_evidence=getattr(state, "used_fallback_evidence", False),
                metadata_only_article=getattr(state.article, "is_metadata_only", False)
            )
            state.understanding_score = final_score_result["score"]
            self.logger.info(f"Final understanding score: {state.understanding_score}/100 (decision: {final_score_result['decision']})")

            # Adjust score reasoning for reduced-quality articles
            article_quality = getattr(state.article, "article_quality", "High")
            state.metrics["article_quality"] = article_quality
            if article_quality == "Medium":
                self.logger.info("Article quality: Medium (150-250 words). Analysis proceeds with reduced confidence expectations.")
            elif article_quality == "Low":
                self.logger.info("Article quality: Low (<150 words but valid). Analysis proceeds with minimum confidence expectations.")

            # Phase 6: Report Generation
            state.phase_timestamps["report_generation"] = datetime.now(timezone.utc).isoformat()
            self._update_state_status(state, "generating_report")
            self.logger.info("Starting report generation...")

            report = None
            start_report = time.time()
            try:
                if self.report_generator is None:
                    raise RuntimeError("ReportGenerator not instantiated")
                report = self.report_generator.generate(state)
                if time.time() - start_report > BUDGET_REPORT:
                    self.logger.warning("Report generation exceeded phase budget")
            except Exception as e:
                self.logger.error(f"Report generation failed: {e}. Returning report with failure reason.")
                from models.data_models import BlindspotReport
                report = BlindspotReport(
                    article_title=getattr(state.article, "title", "Unknown"),
                    article_url=getattr(state.article, "url", ""),
                    analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                    main_topic="Report Generation Failed",
                    key_claims=getattr(state.claims, "key_claims", []) if state.claims else [],
                    blindspots=getattr(state, "blindspots", []),
                    evidence_count=len(getattr(state, "evidence", [])),
                    blindspot_score=0,
                    score_reasoning=f"Report generation failed: {e}",
                    missing_categories=[],
                    balanced_conclusion=f"Report could not be generated. Reason: {e}",
                    search_attempts=getattr(state, "search_attempts", 0),
                    confidence_score=getattr(state, "confidence_score", 0),
                    confidence_label="Very Low Confidence",
                    evidence_strength="None",
                    completion_reason=getattr(state, "completion_reason", "report_error"),
                    research_duration=getattr(state, "research_duration_seconds", 0),
                    source_diversity=getattr(state, "source_diversity", {}),
                    report_version="1.0",
                    search_metrics=getattr(state, "metrics", {}),
                    evidence_balance={},
                    quality_distribution={},
                    query_history={},
                    phase_timestamps=getattr(state, "phase_timestamps", {}),
                    blindspot_coverage=getattr(state, "blindspot_coverage", {})
                )
            elapsed_report = time.time() - start_report
            self.logger.info(f"TIMING: Report generation completed in {elapsed_report:.2f} seconds")

            state.phase_timestamps["analysis_complete"] = datetime.now(timezone.utc).isoformat()
            self._update_state_status(state, "complete")
            self.logger.info(f"Analysis complete. Blindspot score: {report.blindspot_score}/100")

            self.diag_logger.log("report_generation", duration=elapsed_report,
                inputs={"confidence_score": state.confidence_score, "evidence_count": len(state.evidence)},
                outputs={"blindspot_score": report.blindspot_score, "completion_reason": state.completion_reason}
            )
            self.diag_logger.flush(state.article.url)
            
            return report
            
        except Exception as err:
            self._update_state_status(state, "error") if 'state' in locals() else None
            if 'state' in locals():
                state.completion_reason = "error"
            self.logger.error(f"Media Blindspot Agent analysis failed: {err}")
            title = "Unknown Article"
            try:
                if 'article' in locals() and article:
                    title = article.title
            except Exception as e:
                self.logger.error(f"Error accessing article title in error handler ({orchestrator.py}:847): {e}")
                
            from models.data_models import BlindspotReport
            return BlindspotReport(
                article_title=title,
                article_url=url,
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                main_topic="Error - Analysis Failed",
                key_claims=[],
                blindspots=[],
                evidence_count=0,
                blindspot_score=0,
                score_reasoning=f"Analysis failed: {err}",
                missing_categories=[],
                balanced_conclusion=f"Analysis could not be completed. Reason: {err}",
                search_attempts=0,
                confidence_score=0,
                confidence_label="Very Low Confidence",
                evidence_strength="None",
                completion_reason="system_error",
                research_duration=0,
                source_diversity={},
                report_version="1.0",
                search_metrics={},
                evidence_balance={},
                quality_distribution={},
                query_history={},
                phase_timestamps={},
                blindspot_coverage={}
            )

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        config = Config.from_env()
        agent = MediaBlindspotAgent(config)
        print("Agent initialized successfully")
        print(f"Ollama available: {agent.ollama_available}")
    except ImportError:
        print("ReportGenerator not yet implemented — will be added in Step 12")
    except Exception as e:
        print(f"Error during orchestrator initialization: {e}")
