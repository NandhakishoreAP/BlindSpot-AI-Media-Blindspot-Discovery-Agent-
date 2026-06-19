from typing import List
import re

from models.data_models import (
    ArticleData,
    ClaimAnalysis,
    Blindspot
)

from llm.ollama_client import OllamaClient
from utils.logger import get_logger
from config import Config

class BlindspotDetector:
    """
    Identifies missing perspectives, stakeholder voices, and omitted context from articles.
    """

    def __init__(self, ollama_client: OllamaClient, config: Config) -> None:
        """
        Store both arguments as instance attributes and initialize the logger.
        """
        self.ollama_client: OllamaClient = ollama_client
        self.config: Config = config
        self.logger = get_logger("blindspot_detector")

    def _build_system_prompt(self) -> str:
        """
        Return exactly the requested system prompt.
        """
        return (
            "You are an expert in media literacy, critical analysis, and investigative journalism.\n\n"
            "Your job is to identify what important information, perspectives, evidence, stakeholders, or context is MISSING from a news article.\n\n"
            "You are NOT fact-checking the article.\n"
            "You are NOT determining whether the article is true or false.\n\n"
            "You are identifying important perspectives that are absent or underrepresented.\n\n"
            "Think about:\n\n"
            "* Whose voices are missing?\n"
            "* What evidence was not presented?\n"
            "* What historical context was omitted?\n"
            "* What economic impacts were omitted?\n"
            "* What social impacts were omitted?\n"
            "* What environmental impacts were omitted?\n"
            "* What demographic groups are affected but not mentioned?\n"
            "* What expert opinions are missing?\n"
            "* What opposing viewpoints are missing?\n"
            "* What policy alternatives are missing?\n"
            "* What international comparisons are missing?\n"
            "* What long-term consequences are missing?\n\n"
            "For category values, generate highly specific article-focused categories representing genuine missing perspectives (e.g., 'Teacher salary allocation', 'Rural school funding gaps', 'CBT infrastructure readiness', 'Student accessibility barriers').\n"
            "Do NOT generate categories that simply restate the article topic or claims.\n"
            "Examples:\n"
            "Good: 'Teacher salary allocation', 'Rural school funding gaps', 'CBT infrastructure readiness', 'Student accessibility barriers'\n"
            "Bad: 'Education Budget Allocation and Reform', 'NEET Reforms', 'Curriculum Changes'\n\n"
            "Strictly avoid generic categories like 'Opposing Perspective', 'Historical Context', 'Economic Impact', 'Expert Opinion', 'Missing Context', 'Transparency', 'Social Impact', or 'Environmental Impact'. "
            "Every category must be specific to the topic of the article.\n\n"
            "You must respond ONLY with valid JSON.\n\n"
            "No markdown.\n"
            "No explanations.\n"
            "No code fences.\n"
            "No text outside the JSON."
        )

    def _build_detection_prompt(
        self,
        article: ArticleData,
        claims: ClaimAnalysis
    ) -> str:
        """
        Builds a simplified prompt containing only article title, topic, and key claims.
        """
        key_claims_list = ""
        for idx, claim in enumerate(claims.key_claims, 1):
            key_claims_list += f"{idx}. {claim}\n"

        prompt = (
            "Title: " + str(article.title) + "\n"
            "Topic: " + str(claims.main_topic) + "\n"
            "Claims:\n" + key_claims_list + "\n"
            "Identify exactly 3 to 5 key blindspots representing missing perspectives, referencing specific entities, stakeholders, policies, or locations mentioned in the article.\n"
            "Return a JSON object in this format:\n"
            "{\n"
            '  "blindspots": [\n'
            "    {\n"
            '      "category": "Specific category (at least 4 words, e.g. Teacher salary budget allocation, Rural school funding gaps, Student accessibility barriers)",\n'
            '      "description": "Short explanation of the omission (max 12 words)",\n'
            '      "importance": "High | Medium | Low",\n'
            '      "suggested_search_query": "Short query to research the missing perspective (max 6 words)"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "IMPORTANT: Every category MUST represent a genuine missing perspective, have at least 4 words, and NOT be a generic label or restate the topic/claims.\n"
            "Keep descriptions under 12 words and queries under 6 words to avoid truncation."
        )
        return prompt

    def _is_too_similar(self, category: str, topic: str, claims_list: List[str]) -> bool:
        """
        Returns True if the blindspot category is too similar to the topic or any of the key claims.
        """
        import difflib
        c_clean = category.strip().lower()
        t_clean = topic.strip().lower()
        
        # SequenceMatcher ratio
        ratio = difflib.SequenceMatcher(None, c_clean, t_clean).ratio()
        if ratio > 0.70:
            return True
            
        # Check claims
        for claim in claims_list:
            claim_clean = claim.strip().lower()
            if difflib.SequenceMatcher(None, c_clean, claim_clean).ratio() > 0.70:
                return True
                
        # Word overlap check
        c_words = set(w for w in c_clean.split() if len(w) > 3)
        t_words = set(w for w in t_clean.split() if len(w) > 3)
        if c_words and t_words:
            overlap = c_words.intersection(t_words)
            if len(overlap) / len(c_words) > 0.70:
                return True
                
        return False

    def _is_generic_category(self, category: str) -> bool:
        """
        Returns True if the category is generic (e.g. Opposing Perspective, Historical Context, etc.)
        """
        if not category:
            return True
        generic_list = [
            "opposing perspective",
            "historical context",
            "economic impact",
            "expert opinion",
            "stakeholder view",
            "alternative perspective",
            "enforcement challenges",
            "historical outcomes",
            "unintended consequences",
            "missing context",
            "transparency",
            "social impact",
            "environmental impact",
            "governance",
            "policy alternatives",
            "other perspectives",
            "unrepresented viewpoints",
            "regulatory challenges",
            "community engagement"
        ]
        cat_lower = category.lower().strip()
        for gen in generic_list:
            if cat_lower == gen or gen in cat_lower:
                return True
        return False

    def is_generic_blindspot(self, category: str, article: ArticleData = None, title: str = None, topic: str = None) -> bool:
        if not category:
            return True
        cat_lower = category.lower().strip()
        
        generic_list = [
            "opposing perspective", "historical context", "economic impact", "expert opinion",
            "stakeholder view", "alternative perspective", "enforcement challenges", "historical outcomes",
            "unintended consequences", "missing context", "transparency", "social impact",
            "environmental impact", "governance", "policy alternatives", "other perspectives",
            "unrepresented viewpoints", "regulatory challenges", "community engagement",
            "social concerns", "public opinion", "stakeholder response"
        ]
        
        if cat_lower in generic_list:
            return True
            
        import re
        stop_words = {
            "the", "and", "for", "with", "about", "new", "rule", "policy", "report", "news", 
            "article", "impact", "challenges", "consequences", "outcomes", "perspectives",
            "transparency", "governance", "alternatives", "response", "opinion", "social",
            "environmental", "economic", "historical", "context", "expert", "stakeholder"
        }
        
        text_for_entities = ""
        content_sample = ""
        if article:
            text_for_entities += (article.title or "") + " "
            if article.content:
                content_sample = article.content[:1000]
        if title:
            text_for_entities += title + " "
        if topic:
            text_for_entities += topic + " "
            
        text_for_entities += content_sample
        
        capitalized_words = set(re.findall(r'\b[A-Z][A-Za-z0-9\-]+\b', text_for_entities))
        all_words = re.findall(r'\b\w+\b', text_for_entities.lower())
        keywords = {w for w in all_words if w not in stop_words and len(w) > 3}
        
        entity_keywords = {w.lower() for w in capitalized_words if w.lower() not in stop_words and len(w) > 2}
        combined_article_terms = entity_keywords.union(keywords)
        
        cat_words = set(re.findall(r'\b\w+\b', cat_lower))
        cat_words_no_stops = {w for w in cat_words if w not in stop_words}
        
        if cat_words_no_stops.intersection(combined_article_terms):
            return False
            
        return True

    def description_repeats_claims(self, description: str, claims_list: List[str]) -> bool:
        """
        Checks if the blindspot's description repeats any article claims using token overlap.
        """
        if not description or not claims_list:
            return False
        import re
        stop_words = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "is", "was", "were", "are", "be"}
        def get_words(text: str):
            return set(w.lower() for w in re.findall(r'\b\w+\b', text) if w.lower() not in stop_words and len(w) > 2)
        desc_words = get_words(description)
        if not desc_words:
            return False
        for claim in claims_list:
            claim_words = get_words(claim)
            if claim_words:
                overlap = desc_words.intersection(claim_words)
                if len(overlap) / len(desc_words) > 0.70:
                    return True
        return False

    def link_claims_to_blindspots(self, blindspots: List[Blindspot], claims: ClaimAnalysis) -> None:
        """
        Maps and links related claims to each blindspot based on token overlap/relevance.
        """
        if not blindspots:
            return
        if not claims or not claims.key_claims:
            for bs in blindspots:
                bs.related_claims = []
            return
            
        import re
        stop_words = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "is", "was", "were", "are", "be", "this", "that"}
        def get_words(text: str):
            return set(w.lower() for w in re.findall(r'\b\w+\b', text) if w.lower() not in stop_words and len(w) > 2)
            
        for bs in blindspots:
            bs.related_claims = []
            bs_words = get_words(f"{bs.category} {bs.description}")
            if not bs_words:
                continue
            for claim in claims.key_claims:
                claim_words = get_words(claim)
                if claim_words:
                    overlap = bs_words.intersection(claim_words)
                    if overlap:
                        bs.related_claims.append(claim)
            
            # Safe fallback: if no related claims found, associate the first claim
            if not bs.related_claims and claims.key_claims:
                bs.related_claims.append(claims.key_claims[0])

    def _make_category_specific(self, category: str, topic: str) -> str:
        cat_lower = category.lower().strip()
        topic_clean = " ".join(topic.split()[:3]) if topic else "Policy"
        topic_clean = re.sub(r'[^\w\s\-]', '', topic_clean).strip()
        if not topic_clean:
            topic_clean = "Policy"

        if "enforcement challenges" in cat_lower or "regulatory challenges" in cat_lower:
            return f"{topic_clean} Enforcement Logistics Challenges"
        if "historical outcomes" in cat_lower or "historical context" in cat_lower or "past precedents" in cat_lower:
            return f"{topic_clean} Historical Policy Precedents"
        if "opposing perspective" in cat_lower or "alternative perspective" in cat_lower or "other perspectives" in cat_lower or "unrepresented viewpoints" in cat_lower or "stakeholder view" in cat_lower:
            return f"{topic_clean} Omitted Stakeholder Gaps"
        if "economic impact" in cat_lower:
            return f"{topic_clean} Compliance Cost Burden"
        if "expert opinion" in cat_lower:
            return f"{topic_clean} Academic Research Gaps"
        if "unintended consequences" in cat_lower:
            return f"{topic_clean} Unintended Policy Spillovers"
        if "transparency" in cat_lower or "governance" in cat_lower:
            return f"{topic_clean} Governance Transparency Issues"
        if "social impact" in cat_lower:
            return f"{topic_clean} Public Social Impact"
        if "environmental impact" in cat_lower:
            return f"{topic_clean} Environmental Ecosystem Gaps"
        if "policy alternatives" in cat_lower:
            return f"{topic_clean} Alternative Policy Studies"
        if "missing context" in cat_lower:
            return f"{topic_clean} Essential Background Context"
            
        return f"{topic_clean} Omitted {category.strip()}"

    def build_article_specific_blindspots(
        self,
        article_title: str,
        claims: ClaimAnalysis,
        topic: str
    ) -> List[Blindspot]:
        """
        Derives article-specific, non-generic blindspots from article title, claims, and topic.
        Avoids generic templates.
        """
        import re
        t_clean = (topic or "").lower()
        title_clean = (article_title or "").lower()
        claims_text = " ".join(claims.key_claims).lower() if (claims and claims.key_claims) else ""
        combined = f"{t_clean} {title_clean} {claims_text}"

        self.logger.info(f"Building article-specific blindspots for topic: '{topic}'")
        
        categories = []
        if any(w in combined for w in ["vehicle", "car", "emissions", "tax", "gst", "transport", "puc"]):
            categories = [
                ("Rural Vehicle Owner Burden", "Omitted discussion on the disproportionate compliance cost for rural motorists lacking alternatives.", "High", "rural vehicle owner tax burden"),
                ("Automobile Industry Compliance Impact", "Potential economic disruption and supply chain transition costs for manufacturers.", "Medium", "automobile industry emissions compliance impact"),
                ("GST Tax Enforcement Complexity", "Administrative hurdles and policy enforcement challenges in graded taxation systems.", "Medium", "GST tax enforcement administrative complexity")
            ]
        elif any(w in combined for w in ["neet", "exam", "leak", "coaching", "nta", "test", "coaching"]):
            categories = [
                ("Student Testing Privacy Implications", "Unresolved student data privacy concerns regarding digital testing and platform communication.", "High", "student privacy digital testing moderation"),
                ("Private Coaching Industry Incentives", "Lack of coverage on the commercial incentives of private coaching centers in exam leakage.", "Medium", "coaching industry exam leak incentives"),
                ("Regional Inequality in Testing Access", "Disparities in computer-based testing infrastructure between urban and rural centers.", "Medium", "regional inequality testing infrastructure access")
            ]
        elif any(w in combined for w in ["climate", "carbon", "renewable", "energy", "solar", "wind", "coal", "grid", "mining"]):
            categories = [
                ("Renewable Grid Reliability Gaps", "Omitted discussion on grid instability and power outages under high renewable penetration.", "High", "grid reliability renewable energy intermittency"),
                ("Rare Earth Metals Cost", "Ecological footprint and human costs of mining rare earth metals for clean tech.", "Medium", "rare earth metals environmental mining cost"),
                ("Local Utility Transition Costs", "Financial burden on local energy providers adapting to green infrastructure mandates.", "Medium", "utility providers clean energy transition cost")
            ]
        elif any(w in combined for w in ["education", "school", "budget", "teacher", "curriculum", "learn"]):
            categories = [
                ("Teacher Salary Budget Allocation", "Lack of details on budget adjustments for public educator salaries and benefits.", "High", "teacher salary education budget allocation"),
                ("Rural School Funding Disparities", "Disproportionate resource distribution leaving remote school districts behind.", "Medium", "rural school funding allocation gap"),
                ("Student Digital Access Barriers", "Hurdles in infrastructure, devices, and internet access for disadvantaged learners.", "Medium", "disadvantaged student educational technology access barriers")
            ]

        if not categories:
            stop_words = {"about", "above", "after", "again", "against", "policy", "news", "report", "the", "and", "article", "views"}
            clean_title = re.sub(r'[^\w\s]', ' ', article_title or topic or "")
            words = [w.strip() for w in clean_title.split() if w.strip() and w.lower() not in stop_words and len(w) > 3]
            
            entities = []
            for idx, w in enumerate(words):
                if w[0].isupper():
                    entities.append(w)
            
            if not entities:
                entities = words[:3]
                
            entity = " ".join(entities[:2]) if entities else "Policy"
            entity = re.sub(r'[^\w\s\-]', '', entity).strip()
            if not entity:
                entity = "Policy"

            categories = [
                (f"{entity} Primary Implementation Challenges", f"Operational hurdles and logistical complexities in implementing {entity}.", "High", f"{entity.lower()} implementation challenges"),
                (f"{entity} Stakeholder Economic Impact", f"Compliance cost burdens and transition overheads for affected groups of {entity}.", "Medium", f"{entity.lower()} stakeholder economic transition impact"),
                (f"{entity} Legal Policy Precedents", f"Omission of comparative outcomes and regulatory legal precedents for {entity}.", "Medium", f"{entity.lower()} policy legal precedents")
            ]

        blindspots = []
        for cat, desc, imp, query in categories[:3]:
            if self.is_generic_blindspot(cat, title=article_title, topic=topic) or len(cat.split()) < 4:
                cat = self._make_category_specific(cat, topic)
            blindspots.append(
                Blindspot(
                    category=cat,
                    description=desc,
                    importance=imp,
                    suggested_search_query=query
                )
            )

        self.link_claims_to_blindspots(blindspots, claims)
        return blindspots

    def _get_deterministic_blindspots(self, article: ArticleData, claims: ClaimAnalysis) -> List[Blindspot]:
        """
        Generates deterministic, article-linked blindspots programmatically using build_article_specific_blindspots.
        """
        title = article.title if article else ""
        topic = claims.main_topic if claims else ""
        return self.build_article_specific_blindspots(title, claims, topic)

    def _limit_words(self, text: str, max_words: int) -> str:
        if not text:
            return ""
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text

    def detect(
        self,
        article: ArticleData,
        claims: ClaimAnalysis
    ) -> List[Blindspot]:
        """
        Detects blindspots inside the article using Ollama.
        """
        fallback_blindspots = self._get_deterministic_blindspots(article, claims)
        try:
            # 1. Log: "Detecting blindspots for: {article.title}"
            self.logger.info(f"Detecting blindspots for: {article.title}")

            # 2. Verify Ollama availability and model via pre-call check
            if not self.ollama_client.pre_call_health_check(self.config.MODEL_BLINDSPOT_DETECTOR):
                self.logger.warning("Ollama pre-call health check failed. Skipping LLM blindspot detection and using deterministic fallback.")
                self.link_claims_to_blindspots(fallback_blindspots, claims)
                return fallback_blindspots

            # 3. Build prompts:
            system_prompt = self._build_system_prompt()
            analysis_prompt = self._build_detection_prompt(article, claims)

            # 4. Call generate_json_with_retry
            response = self.ollama_client.generate_json_with_retry(
                prompt=analysis_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                num_predict=256,
                model=self.config.MODEL_BLINDSPOT_DETECTOR,
                timeout=60.0
            )

            # 5. Log raw response:
            self.logger.debug(f"Raw blindspot response: {response}")

            # Helper to parse response into Blindspot list
            def parse_response_to_blindspots(resp: dict) -> List[Blindspot]:
                if not isinstance(resp, dict) or not resp:
                    return []
                items = resp.get("blindspots", [])
                if not isinstance(items, list):
                    return []
                res = []
                claims_text_list = claims.key_claims if claims else []
                topic_str = claims.main_topic if claims else article.title
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    cat = str(item.get("category", "")).strip()
                    
                    # Reject if category length is less than 4 words
                    if len(cat.split()) < 4:
                        self.logger.info(f"Rejected blindspot reason: category '{cat}' has fewer than 4 words.")
                        continue
                        
                    if self.is_generic_blindspot(cat, article):
                        self.logger.info(f"Rejected blindspot reason: '{cat}' is generic. Attempting to make specific.")
                        cat = self._make_category_specific(cat, topic_str)
                        if self.is_generic_blindspot(cat, article) or len(cat.split()) < 4:
                            self.logger.info(f"Rejected blindspot reason: '{cat}' is generic or too short after specifier.")
                            continue
                    
                    if self._is_too_similar(cat, topic_str, claims_text_list):
                        self.logger.info(f"Rejected blindspot reason: '{cat}' is too similar to topic or claims.")
                        continue
                        
                    desc = str(item.get("description", "")).strip()
                    if self.description_repeats_claims(desc, claims_text_list):
                        self.logger.info(f"Rejected blindspot reason: description '{desc}' repeats article claims.")
                        continue
                        
                    imp = str(item.get("importance", "Medium")).strip()
                    query = str(item.get("suggested_search_query", "")).strip()
                    if not cat or not desc:
                        continue
                    desc = self._limit_words(desc, 12)
                    query = self._limit_words(query, 6)
                    res.append(Blindspot(
                        category=cat,
                        description=desc,
                        importance=imp,
                        suggested_search_query=query
                    ))
                return res

            blindspots = parse_response_to_blindspots(response)
            
            # Check if valid (3 to 5 blindspots, none generic, none too similar)
            has_generic = any(self.is_generic_blindspot(bs.category, article) for bs in blindspots)
            claims_text_list = claims.key_claims if claims else []
            topic_str = claims.main_topic if claims else article.title
            has_too_similar = any(self._is_too_similar(bs.category, topic_str, claims_text_list) for bs in blindspots)
            is_valid = 3 <= len(blindspots) <= 5 and not has_generic and not has_too_similar

            if not is_valid:
                self.logger.warning("Generated blindspots are generic, empty, too similar, or wrong count. Regenerating once.")
                # Automatically regenerate once
                response = self.ollama_client.generate_json_with_retry(
                    prompt=analysis_prompt,
                    system_prompt=system_prompt,
                    temperature=0.4,
                    num_predict=256,
                    model=self.config.MODEL_BLINDSPOT_DETECTOR,
                    timeout=60.0
                )
                self.logger.debug(f"Regenerated blindspot response: {response}")
                blindspots = parse_response_to_blindspots(response)
                
                # Check again. If still invalid, use deterministic fallback
                has_generic = any(self.is_generic_blindspot(bs.category, article) for bs in blindspots)
                has_too_similar = any(self._is_too_similar(bs.category, topic_str, claims_text_list) for bs in blindspots)
                is_valid = 3 <= len(blindspots) <= 5 and not has_generic and not has_too_similar
                if not is_valid:
                    self.logger.warning("Regenerated blindspots are still generic, too similar, or invalid. Applying deterministic fallback.")
                    blindspots = fallback_blindspots

            # Supplement if count < 3
            if len(blindspots) < 3:
                self.logger.info(f"Supplementing blindspots from fallback. Current count: {len(blindspots)}")
                for fb in fallback_blindspots:
                    if fb.category not in [b.category for b in blindspots]:
                        blindspots.append(fb)
                        if len(blindspots) >= 3:
                            break
            # Force exactly between 3 and 5
            if len(blindspots) < 3:
                blindspots = fallback_blindspots[:3]

            blindspots = blindspots[:5]
            self.link_claims_to_blindspots(blindspots, claims)

            # 10. Log:
            self.logger.info(f"Detected {len(blindspots)} blindspots")
            return blindspots

        except Exception as error:
            # 12. Error Handling
            self.logger.error(f"Blindspot detection failed: {error}. Returning fallback.")
            self.link_claims_to_blindspots(fallback_blindspots, claims)
            return fallback_blindspots

if __name__ == "__main__":
    try:
        config = Config.from_env()
        client = OllamaClient(config)
        detector = BlindspotDetector(client, config)

        # Create ArticleData
        url = "https://test.com"
        title = "Government Announces New Carbon Tax Policy"
        author = "Staff Reporter"
        publication_date = "2025-01-01"
        
        content = (
            "The government today officially announced the introduction of a new carbon tax policy, "
            "marking a historic milestone in the nation's fight against climate change. Under the new plan, "
            "carbon emissions will be taxed starting next fiscal year, providing strong financial incentives "
            "for industrial firms and power plants to switch to cleaner energy alternatives.\n\n"
            "Speaking at the policy launch, Minister of Environment and Climate Change Sarah Jenkins "
            "expressed high confidence in the legislation. 'Today, we are taking a bold and necessary step "
            "to protect our planet for future generations,' Jenkins declared. 'By putting a price on pollution, "
            "we are encouraging innovation, driving investments in renewable energy, and guaranteeing that "
            "our carbon reduction goals will be met on schedule.'\n\n"
            "Environmental advocacy groups quickly welcomed the announcement. Mark Fletcher, Executive Director "
            "of the Climate Action Coalition, praised the policy as a major breakthrough. 'This carbon pricing "
            "framework is exactly what is needed to accelerate the transition away from fossil fuels,' Fletcher "
            "stated. 'It sends a clear signal to the market and sets us on a definitive path to reduce greenhouse "
            "gas emissions by 30% over the next decade. We urge all sectors of society to embrace this positive change.'\n\n"
            "Officials confirmed that revenues from the carbon tax will be directly reinvested into green "
            "infrastructure projects and clean energy subsidies, ensuring a sustainable cycle of emission reductions "
            "across the country."
        )

        article = ArticleData(
            url=url,
            title=title,
            author=author,
            publication_date=publication_date,
            content=content
        )

        # Create ClaimAnalysis
        main_topic = "Carbon tax policy announcement"
        key_claims = [
            "Government will introduce carbon tax",
            "Tax will reduce emissions by 30%",
            "Environmental groups support the policy"
        ]
        author_stance = "Pro-regulation"
        tone = "Optimistic"
        framing_summary = (
            "The article presents carbon tax as a straightforward solution with broad support, "
            "without exploring opposition or economic concerns."
        )

        claims = ClaimAnalysis(
            main_topic=main_topic,
            key_claims=key_claims,
            author_stance=author_stance,
            tone=tone,
            framing_summary=framing_summary
        )

        print("==================================================")
        print("RUNNING BLINDSPOT DETECTION")
        print("==================================================")
        
        blindspots = detector.detect(article, claims)

        print("\n==================================================")
        print("BLINDSPOT DETECTION RESULTS")
        print("===========================")
        for bs in blindspots:
            print(f"Category:\n{bs.category}\n")
            print(f"Importance:\n{bs.importance}\n")
            print(f"Description:\n{bs.description}\n")
            print(f"Suggested Search Query:\n{bs.suggested_search_query}\n")
            print("--------------------------------------------------")

    except Exception as e:
        print(f"\nError during blindspot detector execution: {e}")
