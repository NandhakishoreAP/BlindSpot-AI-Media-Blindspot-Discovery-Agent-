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
            "For category values, choose ONLY from:\n\n"
            "* Expert Opinion\n"
            "* Historical Context\n"
            "* Economic Impact\n"
            "* Scientific Evidence\n"
            "* Affected Demographics\n"
            "* Policy Alternatives\n"
            "* Opposing Perspective\n"
            "* International Comparison\n"
            "* Long-Term Consequences\n"
            "* Stakeholder Voice\n\n"
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
        Builds a detailed prompt containing article information, claim analysis, key claims,
        and article content (truncated to 2000 characters).
        """
        key_claims_list = ""
        for idx, claim in enumerate(claims.key_claims, 1):
            key_claims_list += f"{idx}. {claim}\n"

        truncated_content = article.content[:2000]

        prompt = (
            "ARTICLE INFORMATION\n\n"
            f"* Title: {article.title}\n"
            f"* Author: {article.author}\n"
            f"* Publication Date: {article.publication_date}\n\n"
            "CLAIM ANALYSIS\n\n"
            f"* Main Topic: {claims.main_topic}\n"
            f"* Author Stance: {claims.author_stance}\n"
            f"* Tone: {claims.tone}\n"
            f"* Framing Summary: {claims.framing_summary}\n\n"
            "KEY CLAIMS\n\n"
            f"{key_claims_list}\n"
            "ARTICLE CONTENT\n\n"
            f"{truncated_content}\n\n"
            "Identify the 3 to 5 most important blindspots in this article.\n\n"
            "Focus on information that would significantly improve a reader's understanding of the issue.\n\n"
            "Return a JSON object with EXACTLY this structure:\n\n"
            "{\n"
            '  "blindspots": [\n'
            "    {\n"
            '      "category": "",\n'
            '      "description": "",\n'
            '      "importance": "",\n'
            '      "suggested_search_query": ""\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Field Requirements:\n\n"
            "category:\n"
            "Must be ONE of:\n"
            "* Expert Opinion\n"
            "* Historical Context\n"
            "* Economic Impact\n"
            "* Scientific Evidence\n"
            "* Affected Demographics\n"
            "* Policy Alternatives\n"
            "* Opposing Perspective\n"
            "* International Comparison\n"
            "* Long-Term Consequences\n"
            "* Stakeholder Voice\n\n"
            "description:\n"
            "Explain specifically what information is missing and why it matters.\n\n"
            "importance:\n"
            "Must be:\n"
            "High\n"
            "Medium\n"
            "Low\n\n"
            "suggested_search_query:\n\n"
            "Requirements:\n"
            "* 5 to 15 words\n"
            "* specific\n"
            "* research-oriented\n"
            "* suitable for DuckDuckGo search\n"
            "* should help discover the missing information\n\n"
            "Good example:\n"
            "economic impact carbon tax manufacturing sector study\n\n"
            "Bad example:\n"
            "find more economic impacts\n\n"
            "Respond with ONLY the JSON object containing the blindspots list."
        )
        return prompt

    def detect(
        self,
        article: ArticleData,
        claims: ClaimAnalysis
    ) -> List[Blindspot]:
        """
        Detects blindspots inside the article using Ollama.
        """
        try:
            # 1. Log: "Detecting blindspots for: {article.title}"
            self.logger.info(f"Detecting blindspots for: {article.title}")

            # 2. Verify Ollama health:
            if not self.ollama_client.health_check():
                raise RuntimeError("Ollama server unavailable")

            # 3. Build prompts:
            system_prompt = self._build_system_prompt()
            analysis_prompt = self._build_detection_prompt(article, claims)

            # 4. Call generate_json_with_retry
            response = self.ollama_client.generate_json_with_retry(
                prompt=analysis_prompt,
                system_prompt=system_prompt,
                temperature=0.4
            )

            # 5. Log raw response:
            self.logger.debug(f"Raw blindspot response: {response}")

            # 6. Validate response:
            if not isinstance(response, dict):
                raise RuntimeError("Blindspot detector returned invalid response")

            # 7. Extract:
            blindspot_items = response.get("blindspots", [])
            if not isinstance(blindspot_items, list):
                blindspot_items = []

            # 8. Convert to Blindspot objects
            blindspots = []
            for item in blindspot_items:
                if not isinstance(item, dict):
                    continue

                category = item.get("category", "Unknown")
                description = item.get("description", "")
                importance = item.get("importance", "Medium")
                suggested_search_query = item.get("suggested_search_query", "")

                category_str = str(category).strip() if category is not None else "Unknown"
                description_str = str(description).strip() if description is not None else ""
                importance_str = str(importance).strip() if importance is not None else "Medium"
                query_str = str(suggested_search_query).strip() if suggested_search_query is not None else ""

                # 9. Filter invalid entries:
                if description_str == "":
                    continue

                blindspot_obj = Blindspot(
                    category=category_str,
                    description=description_str,
                    importance=importance_str,
                    suggested_search_query=query_str
                )
                blindspots.append(blindspot_obj)

            # 10. Log:
            self.logger.info(f"Detected {len(blindspots)} blindspots")

            # 11. Return:
            return blindspots

        except Exception as error:
            # 12. Error Handling
            self.logger.error(f"Blindspot detection failed: {error}")
            raise RuntimeError(f"Blindspot detection failed: {error}") from error

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
