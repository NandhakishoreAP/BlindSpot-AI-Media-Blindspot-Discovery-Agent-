from datetime import datetime
import urllib.parse
from typing import List

from models.data_models import (
    AgentState,
    BlindspotReport,
    ArticleData,
    Blindspot,
    Evidence
)

# Configure AgentState to allow extra attributes dynamically for dynamic tracking fields
if hasattr(AgentState, "model_config"):
    AgentState.model_config["extra"] = "allow"
    if hasattr(AgentState, "model_rebuild"):
        try:
            AgentState.model_rebuild(force=True)
        except Exception:
            pass
elif hasattr(AgentState, "Config"):
    setattr(AgentState.Config, "extra", "allow")

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
        state.status = status
        self.logger.info(f"Agent status: {status}")

    def _calculate_source_diversity(self, evidence: List[Evidence]) -> dict:
        """
        Calculates domain and category diversity of the collected evidence.
        """
        diversity = {
            "unique_domains": 0,
            "academic": 0,
            "government": 0,
            "news": 0,
            "other": 0
        }
        if not evidence:
            return diversity
            
        unique_domains = set()
        
        academic_indicators = [".edu", "arxiv", "pubmed", "researchgate"]
        government_indicators = [".gov", "who.int", "un.org", "worldbank.org"]
        news_domains = [
            "bbc.com", "bbc.co.uk", "nytimes.com", "reuters.com", "apnews.com",
            "bloomberg.com", "cnn.com", "theguardian.com", "guardian.co.uk",
            "economist.com", "wsj.com", "forbes.com", "npr.org", "scientificamerican.com",
            "nature.com", "dw.com", "aljazeera.com", "france24.com", "ft.com"
        ]
        
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
                
            if not netloc:
                continue
                
            unique_domains.add(netloc)
            
            is_academic = any(ind in netloc for ind in academic_indicators)
            is_gov = any(ind in netloc for ind in government_indicators)
            is_news = any(news in netloc for news in news_domains)
            
            if is_academic:
                diversity["academic"] += 1
            elif is_gov:
                diversity["government"] += 1
            elif is_news:
                diversity["news"] += 1
            else:
                diversity["other"] += 1
                
        diversity["unique_domains"] = len(unique_domains)
        return diversity

    def _run_search_loop(self, state: AgentState, progress_callback=None) -> None:
        """
        Runs the iterative autonomous research loop querying the planner and gathering evidence.
        """
        max_loop_iterations = self.config.MAX_SEARCH_ATTEMPTS + 2
        loop_count = 0
        previous_confidence = getattr(state, "confidence_score", 0)
        state.stagnation_count = 0
        
        while True:
            loop_count += 1
            if loop_count > max_loop_iterations:
                self.logger.warning("Safety limit reached. Stopping search loop.")
                state.completion_reason = "max_attempts_reached"
                break
                
            # Query planner
            decision = self.planner.decide(state)
            
            if decision.action == "GENERATE_REPORT":
                if state.confidence_score >= self.config.CONFIDENCE_THRESHOLD:
                    state.completion_reason = "confidence_threshold_reached"
                else:
                    state.completion_reason = "planner_requested_report"
                break
                
            if not decision.queries:
                state.completion_reason = "planner_requested_report"
                break
                
            # Execute searches
            self._update_state_status(state, "searching")
            if progress_callback:
                progress_callback("searching", state)
                
            self.logger.info(f"Executing queries: {decision.queries}")
            search_results = self.search_tool.search_multiple(decision.queries)
            
            # Metrics updates
            state.metrics["searches_executed"] += 1
            state.metrics["results_collected"] += len(search_results)
            
            # Keep queries in search_queries_used
            search_queries_used = getattr(state, "search_queries_used", [])
            for q in decision.queries:
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
                    state.completion_reason = "no_search_results"
                    break
                else:
                    continue
                    
            # Evaluate evidence
            self._update_state_status(state, "evaluating_evidence")
            if progress_callback:
                progress_callback("evaluating_evidence", state)
                
            new_evidence = self.evidence_evaluator.evaluate(
                state.article,
                state.claims,
                state.blindspots,
                search_results
            )
            
            # Empty evidence recovery
            if not new_evidence:
                self.logger.info("No useful evidence found")
                if state.search_attempts >= self.config.MAX_SEARCH_ATTEMPTS:
                    state.completion_reason = "max_attempts_reached"
                    break
                continue
                
            # Deduplicate evidence before extending
            existing_urls = set(getattr(ev.search_result, "url", "").rstrip("/") for ev in getattr(state, "evidence", []))
            unique_new_evidence = []
            duplicate_count = 0
            for ev in new_evidence:
                url = getattr(ev.search_result, "url", "").rstrip("/")
                if url not in existing_urls:
                    unique_new_evidence.append(ev)
                    existing_urls.add(url)
                else:
                    duplicate_count += 1
            
            if duplicate_count > 0:
                self.logger.info(f"Skipped {duplicate_count} duplicate evidence items")
                
            state_evidence = getattr(state, "evidence", [])
            state_evidence.extend(unique_new_evidence)
            state.evidence = state_evidence
            
            # Metrics update: Evidence collected
            state.metrics["evidence_collected"] += len(unique_new_evidence)
            
            # Update confidence score
            state.confidence_score = self.evidence_evaluator.calculate_confidence(
                state.blindspots,
                state.evidence
            )
            
            # Store confidence details
            details = self.evidence_evaluator.get_confidence_details(
                state.blindspots,
                state.evidence
            )
            setattr(state, "confidence_details", details)
            
            # Source diversity calculation
            diversity = self._calculate_source_diversity(state.evidence)
            state.source_diversity = diversity
            
            # Confidence progress tracking
            confidence_delta = state.confidence_score - previous_confidence
            state.last_confidence_delta = confidence_delta
            
            if confidence_delta <= 0:
                state.stagnation_count += 1
            else:
                state.stagnation_count = 0
            
            # Enhanced Logging
            delta_str = f"+{confidence_delta}%" if confidence_delta > 0 else f"{confidence_delta}%"
            self.logger.info(
                f"Loop {loop_count} complete:\n"
                f"Evidence={len(state.evidence)}\n"
                f"Confidence={state.confidence_score}%\n"
                f"Delta={delta_str}\n"
                f"Unique Domains={diversity['unique_domains']}"
            )
            
            # Research saturation detection
            if state.stagnation_count >= 2:
                self.logger.info("Research appears saturated. Confidence is no longer improving.")
                state.completion_reason = "research_saturation"
                break
                
            previous_confidence = state.confidence_score
            
            if state.search_attempts >= self.config.MAX_SEARCH_ATTEMPTS:
                state.completion_reason = "max_attempts_reached"
                break

    def analyze(self, url: str) -> BlindspotReport:
        """
        Executes the entire research pipeline from article URL extraction to final report generation.
        """
        try:
            # Lifecycle log
            self.logger.info(f"Starting BlindSpot AI analysis for: {url}")
            
            # Phase 1: Article Extraction
            article = self.article_extractor.extract(url)
            self.logger.info(f"Article extracted: {article.title} ({article.word_count} words)")
            
            # Phase 2: Initialize State
            state = AgentState(article=article)
            if getattr(state, "__pydantic_extra__", None) is None:
                try:
                    object.__setattr__(state, "__pydantic_extra__", {})
                except AttributeError:
                    pass
            setattr(state, "search_results", [])
            state.started_at = datetime.utcnow().isoformat()
            state.metrics = {
                "searches_executed": 0,
                "results_collected": 0,
                "evidence_collected": 0
            }
            self._update_state_status(state, "analyzing_claims")
            
            # Phase 3: Claim Analysis
            claims = self.claim_analyzer.analyze(article)
            state.claims = claims
            self.logger.info("Claims analyzed")
            self._update_state_status(state, "detecting_blindspots")
            
            # Phase 4: Blindspot Detection
            blindspots = self.blindspot_detector.detect(article, state.claims)
            self.logger.info("Blindspots detected")
            
            if not blindspots:
                self.logger.warning("No blindspots detected. Using default fallback blindspot.")
                fallback_bs = Blindspot(
                    category="Alternative Perspectives",
                    description="Investigate missing perspectives, counterarguments, expert critiques, historical context, economic implications, and demographic impacts.",
                    importance="Medium",
                    suggested_search_query=f"{article.title} alternative perspective analysis"
                )
                state.blindspots = [fallback_bs]
            else:
                state.blindspots = blindspots
                
            self.logger.info(f"Found {len(state.blindspots)} blindspots to investigate")
            
            # Phase 5: Agentic Search Loop
            self.logger.info("Research started")
            self._update_state_status(state, "researching")
            self._run_search_loop(state)
            self.logger.info("Research completed")
            
            # Phase 6: Report Generation
            self._update_state_status(state, "generating_report")
            
            # Report Generator safety check
            if self.report_generator is None:
                raise RuntimeError("ReportGenerator not available")
                
            report = self.report_generator.generate(state)
            self.logger.info("Report generated")
            
            # Timeline completion tracking
            state.completed_at = datetime.utcnow().isoformat()
            start_time = datetime.fromisoformat(state.started_at)
            end_time = datetime.fromisoformat(state.completed_at)
            state.research_duration_seconds = int((end_time - start_time).total_seconds())
            
            self._update_state_status(state, "complete")
            self.logger.info(f"Analysis complete. Blindspot score: {report.blindspot_score}/100")
            
            return report
            
        except Exception as err:
            self._update_state_status(state, "error") if 'state' in locals() else None
            if 'state' in locals():
                state.completion_reason = "error"
            self.logger.error(f"Evidence evaluation failed: {err}")
            raise

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
