import json
import urllib.parse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Any

from models.data_models import AgentState, BlindspotReport, Blindspot, Evidence
from llm.ollama_client import OllamaClient
from utils.logger import get_logger
from config import Config

REPORT_VERSION = "1.0"

class ReportGenerator:
    """
    Synthesizes the final report on discovered blindspots, combining evidence balance,
    source quality analytics, and LLM-driven explanation.
    """

    def __init__(self, ollama_client: OllamaClient, config: Config) -> None:
        """
        Initializes the ReportGenerator.
        """
        self.ollama_client = ollama_client
        self.config = config
        self.logger = get_logger("report_generator")

    def _calculate_evidence_balance(self, evidence: List[Evidence]) -> dict:
        """
        Calculates counts of evidence that support, contradict, or add context to claims.
        """
        balance = {
            "supports": 0,
            "contradicts": 0,
            "adds_context": 0
        }
        for ev in evidence:
            rel = getattr(ev, "relevance", "").lower()
            if "support" in rel:
                balance["supports"] += 1
            elif "contradict" in rel:
                balance["contradicts"] += 1
            else:
                # Default fallback is adds_context
                balance["adds_context"] += 1
        return balance

    def _calculate_quality_distribution(self, evidence: List[Evidence]) -> dict:
        """
        Calculates counts of low, medium, and high quality evidence sources.
        """
        distribution = {
            "high": 0,
            "medium": 0,
            "low": 0
        }
        for ev in evidence:
            qual = getattr(ev, "quality", "").lower()
            if "high" in qual:
                distribution["high"] += 1
            elif "medium" in qual:
                distribution["medium"] += 1
            else:
                # Default fallback is low
                distribution["low"] += 1
        return distribution

    def _classify_source_type(self, url: str) -> str:
        """
        Classifies evidence source url into news, government, academic, think tank,
        industry, social media, blog, or other.
        """
        if not url:
            return "other"
        url_lower = url.lower()
        try:
            parsed = urllib.parse.urlparse(url_lower)
            netloc = parsed.netloc
            if netloc.startswith("www."):
                netloc = netloc[4:]
        except Exception as e:
            self.logger.warning(f"Source type classification failed ({report_generator.py}:82): {e}")
            netloc = url_lower
            parsed = None

        government_indicators = [".gov", "who.int", "un.org", "worldbank.org", "imf.org"]
        think_tank_indicators = ["cato.org", "brookings.edu", "heritage.org", "rand.org", "csis.org", "pewresearch.org", "chathamhouse.org", "cfr.org"]
        academic_indicators = [".edu", "arxiv", "pubmed", "researchgate", "doi.org"]
        social_media_indicators = ["twitter.com", "x.com", "facebook.com", "reddit.com", "linkedin.com", "youtube.com", "instagram.com"]
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
        if any(ind in netloc for ind in think_tank_indicators):
            return "think_tank"
        if any(ind in netloc for ind in academic_indicators):
            return "academic"
        if any(ind in netloc for ind in social_media_indicators):
            return "social_media"
        if any(ind in netloc for ind in blog_indicators) or (parsed and "blog" in parsed.path):
            return "blog"
        if any(news in netloc for news in news_domains):
            return "news"
        if any(ind in netloc for ind in industry_indicators):
            return "industry"

        return "other"

    def _get_confidence_label(self, score: int) -> str:
        """
        Continuous label: directly reflects the score value.
        """
        if score >= 70:
            return "High Confidence"
        elif score >= 40:
            return "Moderate Confidence"
        elif score >= 15:
            return "Low Confidence"
        return "Very Low Confidence"

    def _calculate_fallback_score(self, state: AgentState) -> int:
        # Deprecated — kept to avoid AttributeError only
        return 40

    def _format_claims(self, state: AgentState) -> str:
        """
        Format claims section for the prompt.
        """
        if not state.claims or not state.claims.key_claims:
            return "None."
        return "\n".join(f"- {claim}" for claim in state.claims.key_claims)

    def _format_blindspots(self, state: AgentState) -> str:
        """
        Format blindspots section for the prompt.
        """
        if not state.blindspots:
            return "None."
        return "\n".join(
            f"- Category: {bs.category}\n  Description: {bs.description}"
            for bs in state.blindspots
        )

    def _format_evidence(self, state: AgentState) -> str:
        """
        Format evidence section for the prompt.
        """
        if not state.evidence:
            return "None."
        return "\n".join(
            f"- Relevance: {ev.relevance}\n  Insight: {ev.key_insight}"
            for ev in state.evidence
        )


    def _build_prompt(self, state: AgentState, heuristic_score: int, balance: dict, confidence: int) -> str:
        """
        Builds a simplified prompt layout containing ONLY the requested elements.
        """
        article_title = state.article.title
        article_topic = state.claims.main_topic if state.claims else "Unknown"
        
        claims_str = self._format_claims(state)
        blindspots_str = self._format_blindspots(state)
        evidence_str = self._format_evidence(state)
        analytics_str = f"Confidence Score: {confidence}%"
        
        instructions = (
            "You are a neutral media analyst. Output the final report JSON explaining the article's blindspots.\n"
            "The blindspot_score (0-100) represents the amount of missing context in the article.\n"
            "Be fair. Do NOT use terms like 'biased' or 'misleading'. Use neutral, evidence-based language.\n"
            "EVERY sentence in every section MUST reference a specific claim, blindspot, or piece of evidence. Never generate generic statements.\n"
            "Your score_reasoning must be highly specific, referencing the actual key claims, identified blindspots, evidence, confidence, and coverage. Keep it under 50 words.\n"
            "Each of the conclusion sections (conclusion_covers, conclusion_missing, conclusion_evidence, conclusion_uncertainty, conclusion_assessment) MUST contain exactly 1 or 2 bullet points (as list of strings) describing the respective content. Do NOT use paragraphs.\n"
            "The balanced_conclusion MUST be formatted using exactly these 5 headings:\n"
            "What The Article Covers\n"
            "What May Be Missing\n"
            "Evidence Findings\n"
            "Remaining Uncertainty\n"
            "Overall Assessment\n"
            "Each section in balanced_conclusion must contain the corresponding bullet points. Ensure the tone is objective, balanced, and strictly neutral."
        )

        schema = {
            "blindspot_score": "refined integer score between 0 and 100 representing missing context volume and import",
            "score_reasoning": "highly specific explanation referencing the key claims, detected blindspots, quality and count of gathered evidence, confidence score, and coverage",
            "missing_categories": ["list of specific categories of omitted perspectives"],
            "conclusion_covers": ["1-2 bullet points (list of strings) explaining what the article covers"],
            "conclusion_missing": ["1-2 bullet points (list of strings) explaining what the article leaves out"],
            "conclusion_evidence": ["1-2 bullet points (list of strings) summarizing the gathered external evidence findings"],
            "conclusion_uncertainty": ["1-2 bullet points (list of strings) identifying remaining uncertainties or research gaps"],
            "conclusion_assessment": ["1-2 bullet points (list of strings) giving an overall balanced assessment"],
            "balanced_conclusion": "a synthesized conclusion formatted with 5 exact headings: 'What The Article Covers', 'What May Be Missing', 'Evidence Findings', 'Remaining Uncertainty', 'Overall Assessment', each containing up to 2 bullet points."
        }

        schema_str = json.dumps(schema, indent=2)

        return (
            f"System Instructions:\n{instructions}\n\n"
            f"JSON Schema:\n{schema_str}\n\n"
            f"Article:\n"
            f"Title: {article_title}\n"
            f"Topic: {article_topic}\n\n"
            f"Claims:\n{claims_str}\n\n"
            f"Blindspots:\n{blindspots_str}\n\n"
            f"Evidence:\n{evidence_str}\n\n"
            f"Analytics:\n{analytics_str}"
        )

    def make_conclusion_neutral(self, text: str, confidence_score: int) -> str:
        """
        Converts strong, accusatory statements to neutral, evidence-based language.
        No canned template sentences are injected.
        """
        # Soften strong terms
        replacements = [
            (r'\bthe article is biased\b', 'the article focuses on specific perspectives'),
            (r'\bthe article is misleading\b', 'the article omits some context'),
            (r'\bbiased\b', 'focused on specific perspectives'),
            (r'\bmisleading\b', 'omits some context'),
            (r'\bfails to address\b', 'does not appear to discuss'),
            (r'\bfail to address\b', 'do not appear to discuss'),
            (r'\blacks\b', 'does not appear to discuss'),
            (r'\black\b', 'do not appear to discuss'),
            (r'\bomitted key\b', 'may have excluded additional'),
            (r'\bcompletely lacks\b', 'exhibits limited coverage of'),
            (r'\bfails to\b', 'does not fully'),
            (r'\bfail to\b', 'do not fully'),
            (r'\bnot balanced\b', 'focused on specific perspectives'),
            (r'\bunsubstantiated\b', 'not fully verified by available evidence'),
            (r'\bis false\b', 'is not conclusively proven by available evidence'),
            (r'\bis incorrect\b', 'may lack supporting evidence in secondary sources'),
            (r'\bcompletely wrong\b', 'not fully verified'),
            (r'\bwrong\b', 'lacking consensus'),
            (r'\blies\b', 'unverified assertions'),
            (r'\blie\b', 'unverified assertion')
        ]
        
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
            
        return text

    def _parse_conclusion_section(self, text: str, header: str) -> List[str]:
        """
        Parses bullet points under a specific header from a text block.
        Headers are plain text (no markdown hashes).
        """
        normalized_header = header.lower().strip()
        known_headers = ["what the article covers", "what may be missing", "evidence findings",
                         "remaining uncertainty", "overall assessment"]
        lines = text.splitlines()
        
        start_idx = -1
        for idx, line in enumerate(lines):
            clean_line = line.lower().strip()
            if normalized_header in clean_line:
                start_idx = idx + 1
                break
                
        if start_idx == -1:
            return []
            
        section_lines = []
        for line in lines[start_idx:]:
            clean_l = line.lower().strip()
            # Check if this line is a known section header (stop before next section)
            is_next_header = any(
                kh in clean_l and len(clean_l) < 50
                for kh in known_headers if kh != normalized_header
            )
            if is_next_header:
                break
            section_lines.append(line.strip())
            
        bullets = []
        for line in section_lines:
            m = re.match(r'^[-*\u2022]\s+(.*)$', line)
            if m:
                bullets.append(m.group(1).strip())
            elif re.match(r'^\d+\.\s+(.*)$', line):
                m = re.match(r'^\d+\.\s+(.*)$', line)
                bullets.append(m.group(1).strip())
            elif line.strip() and not line.strip().startswith("#"):
                bullets.append(line.strip())
                
        return [b for b in bullets if b][:2]

    def _validate_report_response(self, response: dict, confidence_score: int = 0, heuristic_score: int = 50) -> dict:
        """
        Validates the raw LLM response dictionary, applies safe defaults, logs warnings
        on discrepancies, and automatically appends confidence-aware caveats.
        """
        validated = {}

        # 1. Validate blindspot_score
        score = response.get("blindspot_score")
        if not isinstance(score, int):
            try:
                score = int(score)
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid blindspot_score '{score}' received; falling back to heuristic: {heuristic_score}")
                score = heuristic_score
        validated["blindspot_score"] = max(0, min(100, score))

        # 2. Validate score_reasoning
        score_reasoning = response.get("score_reasoning")
        if not isinstance(score_reasoning, str) or not score_reasoning.strip():
            self.logger.warning("Empty or invalid score_reasoning received, using default fallback.")
            validated["score_reasoning"] = f"Assigned based on heuristic score reference ({heuristic_score}) refined by evidence metrics."
        else:
            validated["score_reasoning"] = score_reasoning.strip()

        # 3. Validate missing_categories
        missing_categories = response.get("missing_categories")
        if not isinstance(missing_categories, list):
            self.logger.warning("Invalid missing_categories received, using default empty list.")
            validated["missing_categories"] = []
        else:
            validated["missing_categories"] = [str(c).strip() for c in missing_categories if c]

        # 4. Get raw balanced_conclusion
        balanced_conclusion = response.get("balanced_conclusion")
        if not isinstance(balanced_conclusion, str):
            balanced_conclusion = ""
        else:
            balanced_conclusion = balanced_conclusion.strip()

        # 5. Validate/parse structured conclusion list fields
        for field, header in [
            ("conclusion_covers", "What The Article Covers"),
            ("conclusion_missing", "What May Be Missing"),
            ("conclusion_evidence", "Evidence Findings"),
            ("conclusion_uncertainty", "Remaining Uncertainty"),
            ("conclusion_assessment", "Overall Assessment")
        ]:
            val = response.get(field)
            if isinstance(val, list) and val:
                validated[field] = [str(x).strip() for x in val if str(x).strip()][:2]
            else:
                parsed = self._parse_conclusion_section(balanced_conclusion, header)
                validated[field] = parsed[:2] if parsed else []

        # Assemble balanced_conclusion directly from neutralized lists to guarantee exact headers & format
        conclusion_parts = []
        for field, header in [
            ("conclusion_covers", "What The Article Covers"),
            ("conclusion_missing", "What May Be Missing"),
            ("conclusion_evidence", "Evidence Findings"),
            ("conclusion_uncertainty", "Remaining Uncertainty"),
            ("conclusion_assessment", "Overall Assessment")
        ]:
            bullets = validated[field]
            neutral_bullets = [self.make_conclusion_neutral(b, confidence_score) for b in bullets]
            validated[field] = neutral_bullets
            part = f"{header}\n" + "\n".join(f"- {b}" for b in neutral_bullets)
            conclusion_parts.append(part)

        validated["balanced_conclusion"] = "\n\n".join(conclusion_parts)
        return validated

    def generate_programmatic_report(
        self,
        topic: str,
        claims: Any,
        blindspots: List[Blindspot],
        evidence: List[Evidence],
        confidence: int,
        coverage: float
    ) -> dict:
        """
        Builds reasoning, missing categories, balanced conclusion, and blindspot score
        programmatically without any LLM calls.
        """
        if hasattr(claims, "key_claims"):
            key_claims_list = claims.key_claims
            framing_summary = getattr(claims, "framing_summary", "")
        elif isinstance(claims, list):
            key_claims_list = claims
            framing_summary = ""
        else:
            key_claims_list = []
            framing_summary = ""

        num_blindspots = len(blindspots) if blindspots else 0
        evidence_count = len(evidence) if evidence else 0

        # Build highly specific reasonings using actual claims and blindspots
        claims_summary = ", ".join(f"'{c}'" for c in key_claims_list[:2]) if key_claims_list else "no claims extracted"
        blindspots_summary = ", ".join(f"'{b.category}'" for b in blindspots[:2]) if blindspots else "no blindspots identified"
        
        score_reasoning_str = (
            f"The analysis for the topic '{topic}' evaluated claims such as {claims_summary}. "
            f"With confidence {confidence}%, blindspots include {blindspots_summary}. "
            f"{evidence_count} evidence items collected."
        )

        # Continuous score derived from evidence quality and coverage
        total_quality = 0
        for ev in (evidence or []):
            qual = getattr(ev, "quality", "").lower()
            if "high" in qual: total_quality += 1.0
            elif "medium" in qual: total_quality += 0.5
            else: total_quality += 0.2
        avg_quality = total_quality / max(len(evidence or []), 1)

        coverage_ratio_bs = 0.0
        if blindspots:
            covered = sum(1 for bs in blindspots if any(
                bs.category.lower() in (ev.key_insight or "").lower() for ev in (evidence or [])
            ))
            coverage_ratio_bs = covered / len(blindspots)

        source_types = len(set(
            self._classify_source_type(getattr(ev.search_result, "url", ""))
            for ev in (evidence or [])
        ))

        heuristic_score = (avg_quality * 40) + (coverage_ratio_bs * 35) + (min(source_types / 4.0, 1.0) * 25)
        heuristic_score = max(0, min(100, int(round(heuristic_score))))

        original_framing = framing_summary if (framing_summary and framing_summary != "Unknown") else "the main topic claims"
        
        supporting_insights = []
        contradictory_insights = []
        context_insights = []
        for ev in (evidence or []):
            insight = ev.key_insight.strip().rstrip(".")
            rel = getattr(ev, "relevance", "Adds Context").lower()
            if "support" in rel:
                supporting_insights.append(insight)
            elif "contradict" in rel:
                contradictory_insights.append(insight)
            else:
                context_insights.append(insight)

        # Build 5 sections with up to 2 bullet points each — data-driven, no canned sentences
        missing_cats = [bs.category for bs in blindspots] if blindspots else []
        missing_desc = [bs.description.strip().rstrip('.') for bs in blindspots] if blindspots else []

        covers_lines = [f"- Reports on {c.strip('.')}" for c in key_claims_list[:2]] if key_claims_list else []
        if not covers_lines:
            covers_lines = [f"- Addresses the topic of {topic.strip('.')}"]

        missing_lines = []
        for i, cat in enumerate(missing_cats[:2]):
            desc = missing_desc[i] if i < len(missing_desc) else ""
            missing_lines.append(f"- Does not discuss {cat}" + (f": {desc}" if desc else ""))

        evidence_lines = []
        if supporting_insights:
            evidence_lines.append(f"- Supporting: {'; and '.join(supporting_insights[:2])}.")
        if contradictory_insights:
            evidence_lines.append(f"- Contradicting: {'; and '.join(contradictory_insights[:2])}.")
        if not evidence_lines:
            evidence_lines.append("- No external evidence collected.")

        uncertainty_lines = []
        for cat in missing_cats[:2]:
            uncertainty_lines.append(f"- Implications of {cat} remain unverified")

        assessment_lines = [
            f"- Research confidence is {confidence}% across {evidence_count} evidence items covering {num_blindspots} blindspot categories."
        ]

        balanced_conclusion = (
            "What The Article Covers\n"
            + "\n".join(covers_lines) + "\n\n"
            "What May Be Missing\n"
            + "\n".join(missing_lines) + "\n\n"
            "Evidence Findings\n"
            + "\n".join(evidence_lines[:2]) + "\n\n"
            "Remaining Uncertainty\n"
            + "\n".join(uncertainty_lines) + "\n\n"
            "Overall Assessment\n"
            + "\n".join(assessment_lines)
        )
        balanced_conclusion = self.make_conclusion_neutral(balanced_conclusion, confidence)

        missing_cats = [bs.category for bs in blindspots] if blindspots else []

        return {
            "blindspot_score": max(0, min(100, heuristic_score)),
            "score_reasoning": score_reasoning_str,
            "missing_categories": missing_cats,
            "balanced_conclusion": balanced_conclusion,
            "conclusion_covers": [l.lstrip("- ").strip() for l in covers_lines],
            "conclusion_missing": [l.lstrip("- ").strip() for l in missing_lines],
            "conclusion_evidence": [l.lstrip("- ").strip() for l in evidence_lines],
            "conclusion_uncertainty": [l.lstrip("- ").strip() for l in uncertainty_lines],
            "conclusion_assessment": [l.lstrip("- ").strip() for l in assessment_lines]
        }

    def generate(self, state: AgentState) -> BlindspotReport:
        """
        Generates the BlindspotReport, validates, and stores a JSON file containing metadata.
        """
        # Validate inputs first
        is_restricted = getattr(state.article, "access_restricted", False)
        has_invalid_claims = not state.claims or not state.claims.main_topic or state.claims.main_topic == "Unknown"
        is_invalid = is_restricted or has_invalid_claims
        
        if is_invalid:
            self.logger.warning(f"Generating programmatic fallback report due to restriction/invalid inputs (restricted={is_restricted}, invalid_claims={has_invalid_claims})")
            reason = "Publisher restriction" if is_restricted else "Invalid claims or topic"
            reasoning_text = f"Article content could not be accessed. Reason: {reason}. Analysis limited to metadata."
            
            report = BlindspotReport(
                article_title=state.article.title or "Unknown Title",
                article_url=state.article.url or "",
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                main_topic=state.claims.main_topic if state.claims else "Unknown",
                key_claims=state.claims.key_claims if state.claims else [],
                blindspots=[],
                evidence_count=0,
                blindspot_score=25,
                score_reasoning=reasoning_text,
                missing_categories=["General Context"],
                balanced_conclusion=reasoning_text,
                search_attempts=0,
                confidence_score=0,
                confidence_label="Low Confidence",
                evidence_strength="Weak",
                completion_reason=state.completion_reason or "access_restricted",
                research_duration=0,
                source_diversity={},
                report_version=REPORT_VERSION,
                search_metrics={},
                evidence_balance={},
                quality_distribution={},
                query_history={},
                phase_timestamps=getattr(state, "phase_timestamps", {}),
                blindspot_coverage={}
            )
            
            # Save it to disk as well
            self._save_report_to_disk(report, state)
            return report

        confidence = int(round(getattr(state, "confidence_score", 0)))
        conf_label = self._get_confidence_label(confidence)
        self.logger.info(f"Generating report. Research confidence: {confidence}% ({conf_label})")

        # Derive evidence_strength based on confidence (Phase 10 rules)
        if confidence < 30:
            evidence_strength = "Weak"
        elif confidence < 60:
            evidence_strength = "Moderate"
        else:
            evidence_strength = "Strong"

        balance = {"supports": 0, "contradicts": 0, "adds_context": 0}
        quality_dist = {"high": 0, "medium": 0, "low": 0}
        heuristic_score = 40
        score = 40
        validated = {}

        try:
            # 1. Analytics and Heuristics
            balance = self._calculate_evidence_balance(state.evidence)
            quality_dist = self._calculate_quality_distribution(state.evidence)
            heuristic_score = self._calculate_fallback_score(state)
            article_topic = state.claims.main_topic if state.claims else "Unknown"

            coverage = 0.0
            if getattr(state, "confidence_details", None):
                coverage = state.confidence_details.get("effective_coverage", 0.0)
                if coverage == 0.0:
                    coverage = state.confidence_details.get("coverage_ratio", 0.0)

            # Check programmatic trigger conditions
            llm_failures = getattr(state, "llm_failures", 0)
            use_programmatic = (
                confidence == 0 or
                llm_failures > 0 or
                not self.ollama_client.pre_call_health_check(self.ollama_client.model)
            )

            if use_programmatic:
                self.logger.info(f"Triggering programmatic report generation (confidence={confidence}, llm_failures={llm_failures})")
                validated = self.generate_programmatic_report(
                    topic=article_topic,
                    claims=state.claims,
                    blindspots=state.blindspots or [],
                    evidence=state.evidence or [],
                    confidence=confidence,
                    coverage=coverage
                )
                score = int(round(validated["blindspot_score"]))
            else:
                # 2. Build Prompts
                system_prompt = "/no_think\n\nReturn ONLY valid JSON."
                user_prompt = self._build_prompt(state, heuristic_score, balance, confidence)

                # Call LLM with num_predict=512
                response = {}
                try:
                    response = self.ollama_client.generate_json_with_retry(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=0.2,
                        num_predict=512,
                        max_retries=0,
                        timeout=10.0
                    )
                    self.logger.info("LLM report received")
                except Exception as e:
                    self.logger.error(f"Failed to generate report from LLM, using fallbacks: {e}")

                # 4. Fallback if response is invalid or empty
                if not response or not isinstance(response, dict) or response == {}:
                    self.logger.warning("LLM report generation failed or returned empty. Using programmatic generator.")
                    validated = self.generate_programmatic_report(
                        topic=article_topic,
                        claims=state.claims,
                        blindspots=state.blindspots or [],
                        evidence=state.evidence or [],
                        confidence=confidence,
                        coverage=coverage
                    )
                else:
                    # 5. Report Quality Validation
                    validated = self._validate_report_response(response, confidence_score=confidence, heuristic_score=heuristic_score)
                    
                    # Ensure all required headers are present in the conclusion
                    required_headers = [
                        "What The Article Covers",
                        "What May Be Missing",
                        "Evidence Findings",
                        "Remaining Uncertainty",
                        "Overall Assessment"
                    ]
                    conclusion_lower = validated.get("balanced_conclusion", "").lower()
                    has_headers = all(h.lower() in conclusion_lower for h in required_headers)
                    if not has_headers:
                        self.logger.warning("LLM balanced_conclusion is missing required section headers. Substituting with programmatic conclusion fallback.")
                        prog_report = self.generate_programmatic_report(
                            topic=article_topic,
                            claims=state.claims,
                            blindspots=state.blindspots or [],
                            evidence=state.evidence or [],
                            confidence=confidence,
                            coverage=coverage
                        )
                        validated["balanced_conclusion"] = prog_report["balanced_conclusion"]
                        for field in ["conclusion_covers", "conclusion_missing", "conclusion_evidence", "conclusion_uncertainty", "conclusion_assessment"]:
                            validated[field] = prog_report[field]
                
                score = int(round(validated["blindspot_score"]))
                self.logger.info("Report validated")

        except Exception as outer_e:
            self.logger.error(f"Error during report calculation/generation steps: {outer_e}. Generating safe fallback report.", exc_info=True)
            topic = getattr(getattr(state, "claims", None), "main_topic", "Unknown")
            claims = getattr(state, "claims", [])
            blindspots = getattr(state, "blindspots", [])
            evidence = getattr(state, "evidence", [])
            coverage = 0.0
            if getattr(state, "confidence_details", None):
                coverage = state.confidence_details.get("effective_coverage", 0.0)
                if coverage == 0.0:
                    coverage = state.confidence_details.get("coverage_ratio", 0.0)
            validated = self.generate_programmatic_report(
                topic=topic,
                claims=claims,
                blindspots=blindspots,
                evidence=evidence,
                confidence=confidence,
                coverage=coverage
            )
            score = int(round(validated["blindspot_score"]))

        evidence_count = len(state.evidence) if getattr(state, "evidence", None) else 0
        blindspot_count = len(state.blindspots) if getattr(state, "blindspots", None) else 0

        # 5. Build Final BlindspotReport Model
        try:
            report = BlindspotReport(
                article_title=state.article.title,
                article_url=state.article.url,
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                main_topic=state.claims.main_topic if state.claims else "Unknown",
                key_claims=state.claims.key_claims if state.claims else [],
                blindspots=state.blindspots,
                evidence_count=len(state.evidence),
                blindspot_score=score,
                score_reasoning=validated.get("score_reasoning", "Fallback reasoning"),
                missing_categories=validated.get("missing_categories", []),
                balanced_conclusion=validated.get("balanced_conclusion", "Fallback conclusion"),
                search_attempts=state.search_attempts,
                confidence_score=confidence,
                confidence_label=conf_label,
                evidence_strength=evidence_strength,
                completion_reason=state.completion_reason,
                research_duration=state.research_duration_seconds,
                source_diversity=state.source_diversity,
                # Structured conclusion lists:
                conclusion_covers=validated.get("conclusion_covers", []),
                conclusion_missing=validated.get("conclusion_missing", []),
                conclusion_evidence=validated.get("conclusion_evidence", []),
                conclusion_uncertainty=validated.get("conclusion_uncertainty", []),
                conclusion_assessment=validated.get("conclusion_assessment", []),
                # Enhanced fields:
                report_version=REPORT_VERSION,
                search_metrics=state.metrics,
                evidence_balance=balance,
                quality_distribution=quality_dist,
                query_history=getattr(state, "query_history", {}),
                phase_timestamps=getattr(state, "phase_timestamps", {}),
                blindspot_coverage=getattr(state, "blindspot_coverage", {})
            )
        except Exception as model_e:
            self.logger.error(f"Failed to instantiate BlindspotReport model: {model_e}", exc_info=True)
            report = BlindspotReport(
                article_title=getattr(getattr(state, "article", None), "title", "Unknown"),
                article_url=getattr(getattr(state, "article", None), "url", ""),
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                main_topic=getattr(getattr(state, "claims", None), "main_topic", "Unknown"),
                key_claims=getattr(getattr(state, "claims", None), "key_claims", []),
                blindspots=getattr(state, "blindspots", []),
                evidence_count=len(getattr(state, "evidence", [])),
                blindspot_score=40,
                score_reasoning="Critical fallback reasoning due to model initialization failure.",
                missing_categories=[],
                balanced_conclusion="Critical fallback conclusion due to model initialization failure.",
                search_attempts=getattr(state, "search_attempts", 0),
                confidence_score=confidence,
                confidence_label=conf_label,
                evidence_strength=evidence_strength,
                completion_reason=getattr(state, "completion_reason", "error"),
                research_duration=getattr(state, "research_duration_seconds", 0),
                source_diversity=getattr(state, "source_diversity", {}),
                # Structured conclusion lists:
                conclusion_covers=validated.get("conclusion_covers", []),
                conclusion_missing=validated.get("conclusion_missing", []),
                conclusion_evidence=validated.get("conclusion_evidence", []),
                conclusion_uncertainty=validated.get("conclusion_uncertainty", []),
                conclusion_assessment=validated.get("conclusion_assessment", []),
                report_version=REPORT_VERSION,
                search_metrics=getattr(state, "metrics", {}),
                evidence_balance=balance,
                quality_distribution=quality_dist
            )

        self._save_report_to_disk(report, state)
        return report

    def _save_report_to_disk(self, report: BlindspotReport, state: AgentState) -> None:
        """
        Saves the report to a JSON file in the reports directory.
        """
        try:
            confidence = report.confidence_score
            conf_label = report.confidence_label
            evidence_strength = report.evidence_strength
            
            report_summary = {
                "blindspots_found": len(getattr(state, "blindspots", [])),
                "evidence_items": len(getattr(state, "evidence", [])),
                "search_attempts": getattr(state, "search_attempts", 0),
                "confidence_score": confidence
            }
            
            credibility_breakdown = {
                "academic": 0, "government": 0, "news": 0, "think_tank": 0,
                "industry": 0, "blog": 0, "social_media": 0, "other": 0
            }
            for ev in getattr(state, "evidence", []):
                url = getattr(ev.search_result, "url", "")
                cat = self._classify_source_type(url)
                if cat in credibility_breakdown:
                    credibility_breakdown[cat] += 1
                else:
                    credibility_breakdown["other"] += 1

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            title_str = report.article_title or "Unknown"
            safe_title = "".join(c for c in title_str if c.isalnum() or c in (" ", "_", "-")).strip()
            safe_title = safe_title.replace(" ", "_")[:50]
            filename = f"report_{safe_title}_{timestamp}.json"

            reports_dir = Path(self.config.REPORTS_DIR)
            reports_dir.mkdir(parents=True, exist_ok=True)
            filepath = reports_dir / filename

            json_data = {
                "report": report.model_dump(),
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "confidence_score": confidence,
                    "confidence_label": conf_label,
                    "evidence_strength": evidence_strength,
                    "completion_reason": getattr(state, "completion_reason", ""),
                    "research_duration_seconds": getattr(state, "research_duration_seconds", 0),
                    "source_diversity": getattr(state, "source_diversity", {}),
                    "metrics": getattr(state, "metrics", {}),
                    "report_summary": report_summary,
                    "report_version": REPORT_VERSION,
                    "search_queries_used": getattr(state, "search_queries_used", []),
                    "evidence": [ev.model_dump() for ev in getattr(state, "evidence", [])],
                    "blindspots": [bs.model_dump() for bs in getattr(state, "blindspots", [])],
                    "phase_timestamps": getattr(state, "phase_timestamps", {}),
                    "blindspot_coverage": getattr(state, "blindspot_coverage", {}),
                    "query_history": getattr(state, "query_history", {}),
                    "search_metrics": getattr(state, "metrics", {}),
                    "evidence_balance": report.evidence_balance,
                    "quality_distribution": report.quality_distribution,
                    "raw_search_results": [res.model_dump() for res in getattr(state, "search_results", [])],
                    "planner_decisions": [
                        dec.model_dump() if hasattr(dec, "model_dump") else dec 
                        for dec in getattr(state, "planner_decisions", [])
                    ],
                    "source_credibility_breakdown": credibility_breakdown
                }
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Report saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to write report to disk: {e}")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        from models.data_models import ArticleData, ClaimAnalysis, Blindspot, SearchResult, Evidence
        config = Config.from_env()
        client = OllamaClient(config)
        rg = ReportGenerator(client, config)

        # 1. Create ArticleData
        article = ArticleData(
            url="https://example.com/mock-article",
            title="Mock Article on Renewable Energy",
            author="John Doe",
            publication_date="2026-06-17",
            content="Renewable energy is growing. Solar and wind power are the future. Solar panels are cheap."
        )

        # 2. Create ClaimAnalysis
        claims = ClaimAnalysis(
            main_topic="Renewable Energy Growth",
            key_claims=["Solar power is growing", "Wind power is the future"],
            author_stance="Pro-renewables",
            tone="Informative",
            framing_summary="Focuses on growth and benefits of solar/wind power."
        )

        # 3. Create 2 Blindspots
        bs1 = Blindspot(
            category="Grid Reliability",
            description="Omitted discussion on grid instability due to intermittent solar/wind supply.",
            importance="High",
            suggested_search_query="grid reliability intermittent renewable energy solar wind"
        )
        bs2 = Blindspot(
            category="Rare Earth Metals Cost",
            description="No mention of rare earth metals mining costs and environmental impact.",
            importance="Medium",
            suggested_search_query="rare earth metals environmental cost solar panels wind turbines"
        )

        # 4. Create 2 SearchResults
        res1 = SearchResult(
            query="grid reliability intermittent renewable energy solar wind",
            title="Managing Grid Reliability with High Renewable Penetration",
            url="https://example.gov/grid-reliability",
            snippet="DOE report showing that grid updates are critical to manage intermittency."
        )
        res2 = SearchResult(
            query="rare earth metals environmental cost solar panels wind turbines",
            title="The Environmental Footprint of Green Energy Technologies",
            url="https://example.edu/environmental-footprint",
            snippet="Academic study highlighting mining impacts for lithium, cobalt, and rare earth metals."
        )

        # 5. Create 2 Evidence objects
        ev1 = Evidence(
            search_result=res1,
            relevance="Adds Context",
            quality="High",
            key_insight="Grid adjustments are necessary to prevent outages when solar/wind are intermittent."
        )
        ev2 = Evidence(
            search_result=res2,
            relevance="Contradicts",
            quality="High",
            key_insight="Rare earth metals mining creates major ecological hazards not mentioned in clean energy pieces."
        )

        # 6. Create Full AgentState
        state = AgentState(
            article=article,
            claims=claims,
            blindspots=[bs1, bs2],
            evidence=[ev1, ev2],
            search_attempts=1,
            confidence_score=25,  # Selected under 30 to verify the warning caveat appends
            search_queries_used=["grid reliability intermittent renewable energy solar wind", "rare earth metals environmental cost solar panels wind turbines"],
            status="generating_report",
            search_results=[res1, res2],
            metrics={"searches_executed": 2, "results_collected": 2, "evidence_collected": 2},
            source_diversity={"unique_domains": 2, "government": 1, "academic": 1},
            research_duration_seconds=120,
            completion_reason="confidence_threshold_reached",
            phase_timestamps={"article_extraction": "2026-06-17T10:00:00Z", "analysis_complete": "2026-06-17T10:02:00Z"},
            query_history={"grid reliability intermittent renewable energy solar wind": ["https://example.gov/grid-reliability"], "rare earth metals environmental cost solar panels wind turbines": ["https://example.edu/environmental-footprint"]},
            blindspot_coverage={"Grid Reliability": "partial", "Rare Earth Metals Cost": "partial"}
        )

        print("==================================================")
        print("GENERATING REPORT FROM MOCK STATE")
        print("==================================================")
        report = rg.generate(state)

        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print(f"Article Title:      {report.article_title}")
        print(f"Blindspot Score:    {report.blindspot_score}/100")
        print(f"Confidence Score:   {report.confidence_score}% ({report.confidence_label})")
        print(f"Missing Categories: {report.missing_categories}")
        print(f"Conclusion:\n{report.balanced_conclusion}")
        print("-" * 60)

        # Check if file exists
        reports_dir = Path(config.REPORTS_DIR)
        saved_files = list(reports_dir.glob("report_*.json"))
        if saved_files:
            latest_file = max(saved_files, key=lambda p: p.stat().st_mtime)
            print(f"Saved JSON Path:    {latest_file.resolve()}")
            assert latest_file.exists(), "Saved report file does not exist!"
            print("Verification: SUCCESS (Report file successfully created on disk).")
        else:
            print("Verification: FAILED (No report file found in reports directory).")

    except Exception as e:
        print(f"Error during ReportGenerator testing: {e}")
        import traceback
        traceback.print_exc()
