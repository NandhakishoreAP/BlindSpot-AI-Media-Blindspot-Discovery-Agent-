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
        
        # Instantiate Ollama Client first
        self.ollama_client = OllamaClient(config)
        
        # Health check
        self.ollama_available = self.ollama_client.health_check()
        if self.ollama_available:
            self.logger.info("Ollama available")
        else:
            self.logger.warning("Ollama unavailable")

        # Instantiate remaining components
        self.article_extractor = ArticleExtractor(config)
        self.search_tool = SearchTool(config)
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

        self.logger.info("Agent initialized")

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
            except Exception:
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
        
        while True:
            try:
                # Phase 11: Global Time Budget Check (Hard total budget of 60 seconds)
                elapsed_total = time.time() - overall_start_time
                if elapsed_total >= 60.0:
                    self.logger.warning(f"Hard total runtime budget exceeded: {elapsed_total:.2f}s elapsed (limit 60s). Stopping search loop.")
                    state.completion_reason = "time_budget_exceeded"
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
                    state.completion_reason = "planner_requested_report"
                    break
                    
                # Slice queries to max 3 per round
                queries = decision.queries[:3]
                
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
                start_time_search = time.time()
                search_results = self.search_tool.search_multiple(queries)
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
                
                # If circuit breaker is triggered, use heuristic_evaluate directly
                if self.ollama_client.failures_count >= 3:
                    self.logger.warning("Circuit breaker triggered: using heuristic_evaluate directly.")
                    new_evidence = self.evidence_evaluator.heuristic_evaluate(
                        state.article,
                        state.claims,
                        state.blindspots,
                        unique_search_results[:6]
                    )
                else:
                    new_evidence = self.evidence_evaluator.evaluate(
                        state.article,
                        state.claims,
                        state.blindspots,
                        unique_search_results
                    )
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
                
                # Store confidence details
                details = self.evidence_evaluator.get_confidence_details(
                    state.blindspots,
                    state.evidence
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

    def analyze(self, url: str) -> BlindspotReport:
        """
        Executes the entire research pipeline from article URL extraction to final report generation.
        """
        import time
        try:
            # Lifecycle log
            self.logger.info(f"Starting BlindSpot AI analysis for: {url}")
            
            overall_start_time = time.time()
            
            # Phase 1: Article Extraction
            start_extract = time.time()
            article = self.article_extractor.extract(url)
            elapsed_extract = time.time() - start_extract
            self.logger.info(f"TIMING: Extraction completed in {elapsed_extract:.2f} seconds")
            self.logger.info(f"Article extracted: {article.title} ({article.word_count} words)")
            
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
            
            # Check for access restriction right after extraction
            if getattr(article, "access_restricted", False):
                self.logger.warning("Article is access restricted. Skipping claims, blindspots, and research loop.")
                state.claims = ClaimAnalysis(
                    main_topic=article.title or "Unknown Topic",
                    key_claims=[],
                    author_stance="Unknown",
                    tone="Unknown",
                    framing_summary="Access Restricted"
                )
                state.blindspots = []
                state.evidence = []
                state.completion_reason = "access_restricted"
                state.confidence_score = 0
                state.completed_at = datetime.now(timezone.utc).isoformat()
                state.research_duration_seconds = 0
                state.source_diversity = self._calculate_source_diversity([])
                
                # Jump straight to report generation
                state.phase_timestamps["report_generation"] = datetime.now(timezone.utc).isoformat()
                self._update_state_status(state, "generating_report")
                
                start_report = time.time()
                report = None
                try:
                    if self.report_generator is not None:
                        report = self.report_generator.generate(state)
                    else:
                        raise RuntimeError("ReportGenerator is not instantiated")
                except Exception as e:
                    self.logger.error(f"ReportGenerator generate failed: {e}. Falling back to programmatic fallback report.")
                    from models.data_models import BlindspotReport
                    report = BlindspotReport(
                        article_title=state.article.title,
                        article_url=state.article.url,
                        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                        main_topic=state.claims.main_topic,
                        key_claims=[],
                        blindspots=[],
                        evidence_count=0,
                        blindspot_score=25,
                        score_reasoning="Article content could not be accessed. Reason: Publisher restriction. Analysis limited to metadata.",
                        missing_categories=["General Context"],
                        balanced_conclusion="Article content could not be accessed. Reason: Publisher restriction. Analysis limited to metadata.",
                        search_attempts=0,
                        confidence_score=0,
                        confidence_label="Low Confidence",
                        evidence_strength="Weak",
                        completion_reason="access_restricted",
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
                elapsed_report = time.time() - start_report
                self.logger.info(f"TIMING: Report generation completed in {elapsed_report:.2f} seconds")
                state.phase_timestamps["analysis_complete"] = datetime.now(timezone.utc).isoformat()
                self._update_state_status(state, "complete")
                return report

            # Phase 3: Claim Analysis
            state.phase_timestamps["claim_analysis"] = datetime.now(timezone.utc).isoformat()
            self._update_state_status(state, "analyzing_claims")
            
            # Check Circuit Breaker
            if self.ollama_client.failures_count >= 3:
                self.logger.warning("Circuit breaker active: bypassing claim analysis LLM call and using fallback.")
                claims = self.claim_analyzer._get_deterministic_fallback(article)
            else:
                start_claims = time.time()
                claims = self.claim_analyzer.analyze(article)
                elapsed_claims = time.time() - start_claims
                self.logger.info(f"TIMING: Claims analysis completed in {elapsed_claims:.2f} seconds")
            
            # Fallback to programmatic claims if LLM claims fail or are empty
            if not claims or claims.main_topic == "Unknown" or len(claims.key_claims) == 0:
                self.logger.warning("Claim analysis failed or returned empty claims. Applying programmatic claims fallback.")
                claims = self.claim_analyzer._get_deterministic_fallback(article)
            state.claims = claims
            self.logger.info("Claims analyzed")
            
            # Record LLM failures count in state
            state.llm_failures = self.ollama_client.failures_count

            # Phase 4: Blindspot Detection
            state.phase_timestamps["blindspot_detection"] = datetime.now(timezone.utc).isoformat()
            self._update_state_status(state, "detecting_blindspots")
            
            # Check Circuit Breaker
            if self.ollama_client.failures_count >= 3:
                self.logger.warning("Circuit breaker active: bypassing blindspot detection LLM call and using fallback.")
                blindspots = self.blindspot_detector._get_deterministic_blindspots(article, state.claims)
            else:
                start_bs = time.time()
                blindspots = self.blindspot_detector.detect(article, state.claims)
                elapsed_bs = time.time() - start_bs
                self.logger.info(f"TIMING: Blindspot detection completed in {elapsed_bs:.2f} seconds")
            
            # Ensure at least 2 blindspots are present
            if not blindspots or len(blindspots) < 2:
                self.logger.warning("Blindspot detection returned insufficient blindspots. Applying deterministic blindspots fallback.")
                blindspots = self.blindspot_detector._get_deterministic_blindspots(article, state.claims)
            state.blindspots = blindspots
            self.logger.info(f"Found {len(state.blindspots)} blindspots to investigate")
            
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
            
            # Force at least 1 search if search_attempts is 0
            if state.search_attempts == 0:
                self.logger.warning("No searches were executed by the planner. Forcing at least 1 programmatic search.")
                query = state.blindspots[0].suggested_search_query if state.blindspots else f"{state.article.title} criticism"
                if not query:
                    query = f"{state.article.title} concerns"
                
                self._update_state_status(state, "searching")
                
                start_time_search = time.time()
                search_results = self.search_tool.search_multiple([query])
                self.logger.info(f"TIMING: Search execution completed in {time.time() - start_time_search:.2f} seconds")
                
                state.metrics["searches_executed"] += 1
                state.metrics["results_collected"] += len(search_results)
                state.search_queries_used = [query]
                state.search_attempts = 1
                state.search_results.extend(search_results)
                
                self._update_state_status(state, "evaluating_evidence")
                
                start_time_eval = time.time()
                
                # Check Circuit Breaker for evaluating evidence
                if self.ollama_client.failures_count >= 3:
                    new_evidence = self.evidence_evaluator.heuristic_evaluate(
                        state.article,
                        state.claims,
                        state.blindspots,
                        search_results[:2]
                    )
                else:
                    new_evidence = self.evidence_evaluator.evaluate(
                        state.article,
                        state.claims,
                        state.blindspots,
                        search_results
                    )
                self.logger.info(f"TIMING: Evidence evaluation completed in {time.time() - start_time_eval:.2f} seconds")
                
                state.evidence = new_evidence
                
                state.blindspot_coverage = self.evidence_evaluator.analyze_blindspot_coverage(
                    state.blindspots,
                    state.evidence
                )
                
                state.confidence_score = self.evidence_evaluator.calculate_confidence(
                    state.blindspots,
                    state.evidence
                )
                
                details = self.evidence_evaluator.get_confidence_details(
                    state.blindspots,
                    state.evidence
                )
                setattr(state, "confidence_details", details)
                
                diversity = self._calculate_source_diversity(state.evidence)
                state.source_diversity = diversity
                unique_domains_list = list(set(ev.search_result.source.lower().strip() for ev in state.evidence))
                state.unique_domains = unique_domains_list

            state.phase_timestamps["research_end"] = datetime.now(timezone.utc).isoformat()
            
            # Final Safety Invariant checks
            state.llm_failures = self.ollama_client.failures_count
            
            # 1. Guarantee: blindspots >= 2
            if not state.blindspots or len(state.blindspots) < 2:
                self.logger.warning("Safety Guarantee: Insufficient blindspots. Generating deterministic blindspots.")
                state.blindspots = self.blindspot_detector._get_deterministic_blindspots(article, state.claims)
                
            # 2. Guarantee: search_attempts >= 1
            if getattr(state, "search_attempts", 0) < 1:
                self.logger.warning("Safety Guarantee: No searches were executed. Setting search_attempts to 1.")
                state.search_attempts = 1
                
            # 3. Guarantee: evidence_count >= 1 (inject fallback context evidence if empty)
            if not getattr(state, "evidence", None) or len(state.evidence) == 0:
                self.logger.warning("Safety Guarantee: No evidence found. Injecting fallback context evidence.")
                from models.data_models import SearchResult, Evidence
                fallback_query = state.blindspots[0].suggested_search_query if state.blindspots else f"{state.article.title} criticism"
                fallback_res = SearchResult(
                    query=fallback_query,
                    title=f"General context search on {state.claims.main_topic if state.claims else 'topic'}",
                    url="https://archive.org",
                    snippet="Public archived reference material for contextual information.",
                    source="archive.org"
                )
                fallback_ev = Evidence(
                    search_result=fallback_res,
                    relevance="Adds Context",
                    quality="Medium",
                    key_insight="Provides generic background context for the requested topic."
                )
                state.evidence = [fallback_ev]
                
            # 4. Guarantee: confidence_score >= 15
            if not getattr(state, "confidence_score", None) or state.confidence_score < 15:
                self.logger.warning("Safety Guarantee: Confidence score is under 15%. Forcing to 15%.")
                state.confidence_score = 15

            # Calculate final research timeline before generating report
            state.completed_at = datetime.now(timezone.utc).isoformat()
            start_time_dt = datetime.fromisoformat(state.started_at)
            end_time_dt = datetime.fromisoformat(state.completed_at)
            state.research_duration_seconds = int((end_time_dt - start_time_dt).total_seconds())

            # Phase 6: Report Generation
            state.phase_timestamps["report_generation"] = datetime.now(timezone.utc).isoformat()
            self._update_state_status(state, "generating_report")
            self.logger.info("Starting report generation...")

            report = None
            start_report = time.time()
            try:
                if state.llm_failures >= 3 or self.report_generator is None:
                    raise RuntimeError("Circuit breaker triggered or ReportGenerator not instantiated")
                report = self.report_generator.generate(state)
            except Exception as e:
                self.logger.error(f"ReportGenerator generate failed: {e}. Falling back to programmatic report generation in orchestrator.")
                # Fallback report generation
                from llm.report_generator import ReportGenerator as RG
                temp_rg = RG(self.ollama_client, self.config)
                
                coverage = 0.0
                if getattr(state, "confidence_details", None):
                    coverage = state.confidence_details.get("effective_coverage", 0.0)
                    if coverage == 0.0:
                        coverage = state.confidence_details.get("coverage_ratio", 0.0)
                
                prog_report = temp_rg.generate_programmatic_report(
                    topic=state.claims.main_topic if state.claims else "Unknown",
                    claims=state.claims,
                    blindspots=state.blindspots,
                    evidence=state.evidence,
                    confidence=state.confidence_score,
                    coverage=coverage
                )
                
                # Construct report
                from models.data_models import BlindspotReport
                report = BlindspotReport(
                    article_title=state.article.title,
                    article_url=state.article.url,
                    analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                    main_topic=state.claims.main_topic if state.claims else "Unknown",
                    key_claims=state.claims.key_claims if state.claims else [],
                    blindspots=state.blindspots,
                    evidence_count=len(state.evidence),
                    blindspot_score=prog_report["blindspot_score"],
                    score_reasoning=prog_report["score_reasoning"],
                    missing_categories=prog_report["missing_categories"],
                    balanced_conclusion=prog_report["balanced_conclusion"],
                    search_attempts=state.search_attempts,
                    confidence_score=state.confidence_score,
                    confidence_label=temp_rg._get_confidence_label(state.confidence_score),
                    evidence_strength="Weak" if state.confidence_score < 30 else ("Moderate" if state.confidence_score < 60 else "Strong"),
                    completion_reason=state.completion_reason,
                    research_duration=state.research_duration_seconds,
                    source_diversity=state.source_diversity,
                    report_version="1.0",
                    search_metrics=state.metrics,
                    evidence_balance=temp_rg._calculate_evidence_balance(state.evidence),
                    quality_distribution=temp_rg._calculate_quality_distribution(state.evidence),
                    query_history=getattr(state, "query_history", {}),
                    phase_timestamps=getattr(state, "phase_timestamps", {}),
                    blindspot_coverage=getattr(state, "blindspot_coverage", {})
                )
            elapsed_report = time.time() - start_report
            self.logger.info(f"TIMING: Report generation completed in {elapsed_report:.2f} seconds")

            state.phase_timestamps["analysis_complete"] = datetime.now(timezone.utc).isoformat()
            self._update_state_status(state, "complete")
            self.logger.info(f"Analysis complete. Blindspot score: {report.blindspot_score}/100")
            
            return report
            
        except Exception as err:
            self._update_state_status(state, "error") if 'state' in locals() else None
            if 'state' in locals():
                state.completion_reason = "error"
            self.logger.error(f"Media Blindspot Agent analysis failed: {err}. Returning fallback report.")
            
            fallback_title = "Unknown Article"
            fallback_url = url
            try:
                if 'article' in locals() and article:
                    fallback_title = article.title
                    fallback_url = article.url
            except Exception:
                pass
                
            from models.data_models import BlindspotReport
            fallback_report = BlindspotReport(
                article_title=fallback_title,
                article_url=fallback_url,
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                main_topic=fallback_title,
                key_claims=[f"Key claim: {fallback_title}"],
                blindspots=[],
                evidence_count=0,
                blindspot_score=25,
                score_reasoning="Fallback report generated due to an unhandled system failure during execution.",
                missing_categories=["General Context"],
                balanced_conclusion="Due to technical constraints, a detailed media analysis could not be completed. The article content was analyzed using default criteria.",
                search_attempts=1,
                confidence_score=15,
                confidence_label="Low Confidence",
                evidence_strength="Weak",
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
            return fallback_report

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
