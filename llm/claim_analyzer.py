import json
import time
from models.data_models import ArticleData, ClaimAnalysis
from llm.ollama_client import OllamaClient, ExtractionPreset
from utils.logger import get_logger
from config import Config

class ClaimAnalyzer:
    """
    Analyzes news articles to identify main topics, key claims, stance, tone, and framing.
    """

    def __init__(self, ollama_client: OllamaClient, config: Config) -> None:
        """
        Initializes the ClaimAnalyzer with an OllamaClient and Config.
        """
        self.ollama_client: OllamaClient = ollama_client
        self.config: Config = config
        self.logger = get_logger("claim_analyzer")

    def _build_system_prompt(self) -> str:
        """
        Returns the system prompt for the claim analyzer LLM.
        """
        return "/no_think\n\nReturn ONLY valid JSON."

    def _build_analysis_prompt(self, article: ArticleData) -> str:
        """
        Builds the user prompt requesting JSON schema format and article.
        """
        truncated_content: str = article.content[:3000]
        schema = {
            "main_topic": "short topic label (2-8 words). Examples: Climate Policy, Carbon Taxation, Remote Work, Healthcare Reform",
            "key_claims": ["list of 3 to 6 key claims"],
            "author_stance": "stance label (e.g. Pro-regulation, Neutral, Skeptical)",
            "tone": "tone label (e.g. Balanced, Critical, Alarmist, Optimistic)",
            "framing_summary": "2-3 sentences describing the framing of the issue"
        }
        schema_str = json.dumps(schema, indent=2)
        return (
            f"Schema:\n{schema_str}\n\n"
            f"Article:\n"
            f"Title: {article.title}\n"
            f"Author: {article.author}\n"
            f"Date: {article.publication_date}\n\n"
            f"Content:\n{truncated_content}"
        )

    def analyze(self, article: ArticleData) -> ClaimAnalysis:
        """
        Analyzes the claims of an article using the Ollama LLM client.
        
        Args:
            article (ArticleData): The article metadata and content.
            
        Returns:
            ClaimAnalysis: The structured claims, tone, and framing of the article.
            
        Raises:
            RuntimeError: If Ollama health check or claim analysis fails.
        """
        try:
            # A. Log analysis start
            self.logger.info(f"Analyzing claims for: {article.title}")

            # B. Verify Ollama availability
            if not self.ollama_client.health_check():
                raise RuntimeError("Ollama server unavailable")

            # C. Build prompts
            system_prompt: str = self._build_system_prompt()
            analysis_prompt: str = self._build_analysis_prompt(article)

            # D. Call model and record execution time
            start_time = time.time()
            preset = ExtractionPreset.CLAIM_ANALYZER
            response: dict = self.ollama_client.generate_json_with_retry(
                prompt=analysis_prompt,
                system_prompt=system_prompt,
                max_retries=2,
                temperature=preset["temperature"],
                num_predict=preset["num_predict"],
                num_ctx=preset["num_ctx"],
                preset_name=ExtractionPreset.CLAIM_ANALYZER_NAME
            )
            elapsed = time.time() - start_time
            self.logger.info(f"LLM claim extraction completed in {elapsed:.2f} seconds")

            # E. Log raw response at DEBUG level
            self.logger.debug(f"Raw claim analysis response: {response}")

            # F. Convert response into ClaimAnalysis with safe defaults & validation
            if not isinstance(response, dict):
                response = {}

            main_topic = response.get("main_topic")
            key_claims = response.get("key_claims")
            author_stance = response.get("author_stance")
            tone = response.get("tone")
            framing_summary = response.get("framing_summary")

            # Robust nested/escaped fallback checks for string fields if they returned empty/missing
            def get_fallback_field(field_name: str, current_val: str) -> str:
                if isinstance(current_val, str) and current_val.strip() not in ("", "Unknown"):
                    return current_val
                # Scan all dictionary key-value strings for the field name
                import re
                for k, v in response.items():
                    key_str = str(k)
                    val_str = str(v)
                    if field_name in key_str or field_name in val_str:
                        combined = f"{key_str} : {val_str}"
                        # Match everything after field name and colon, up to the end of string or closing quotes
                        match = re.search(fr'{field_name}["\\]*\s*[:=]\s*["\\]*([\s\S]+?)(?:["\\]+\s*)?$', combined)
                        if match:
                            candidate = match.group(1).strip().rstrip('}').strip().strip('"').strip("'")
                            if candidate:
                                return candidate
                        # Simple split fallback
                        if ":" in combined:
                            parts = combined.split(":", 1)
                            candidate = parts[1].strip().strip('"').strip("'").strip().rstrip('}').strip()
                            if candidate:
                                return candidate
                return "Unknown"

            main_topic = get_fallback_field("main_topic", main_topic)
            author_stance = get_fallback_field("author_stance", author_stance)
            tone = get_fallback_field("tone", tone)
            framing_summary = get_fallback_field("framing_summary", framing_summary)

            # Validate field types and assign fallbacks if invalid/missing/empty
            if not isinstance(key_claims, list):
                key_claims = []
            else:
                key_claims = [str(claim) for claim in key_claims if claim is not None and str(claim).strip() != ""]

            # G. Validation
            if main_topic.strip() == "":
                main_topic = "Unknown"

            if not key_claims:
                self.logger.warning("No claims extracted from article.")

            # Log number of claims extracted at DEBUG level
            self.logger.debug(f"Extracted {len(key_claims)} claims")

            claims = ClaimAnalysis(
                main_topic=main_topic,
                key_claims=key_claims,
                author_stance=author_stance,
                tone=tone,
                framing_summary=framing_summary
            )

            # H. Log completion
            self.logger.info(f"Claim analysis complete. Topic: {claims.main_topic}")

            # Performance benchmark logging
            tokens = getattr(self.ollama_client, "last_eval_count", 0)
            self.logger.info(
                f"Performance Benchmark - Claims: {len(claims.key_claims)} | "
                f"Duration: {elapsed:.2f}s | "
                f"Tokens: {tokens}"
            )

            # I. Return ClaimAnalysis object
            return claims

        except Exception as error:
            # J. Exception Handling
            self.logger.error(f"Claim analysis failed: {error}")
            raise RuntimeError(f"Claim analysis failed: {error}") from error

if __name__ == "__main__":
    try:
        # Load config
        config = Config.from_env()

        # Create OllamaClient
        client = OllamaClient(config)

        # Create ClaimAnalyzer
        analyzer = ClaimAnalyzer(client, config)

        # Create ArticleData with a realistic 200-300 word article
        article = ArticleData(
            url="https://test.com",
            title="Government Announces New Climate Policy",
            author="Staff Reporter",
            publication_date="2025-01-01",
            content=(
                "The federal government has officially announced a sweeping new climate policy that introduces "
                "mandatory carbon taxes across key industrial sectors. According to the official press release, the "
                "primary goal of this legislation is reducing greenhouse gas emissions by forty percent over the next "
                "decade, aligning with global climate accords. The administration believes this is a vital step toward "
                "combating the rising threats of climate change.\n\n"
                "Supporters of the initiative, including various environmental advocacy groups and green energy pioneers, "
                "say the policy helps climate goals by incentivizing innovation in clean technologies. They argue that "
                "putting a price on carbon is the most efficient market-based mechanism to reduce pollution and accelerate "
                "the adoption of solar, wind, and battery storage solutions.\n\n"
                "Conversely, critics say the policy increases business costs, which will ultimately be passed down to "
                "consumers in the form of higher utility bills and fuel prices. Representatives from manufacturing and "
                "transportation lobbies contend that the sudden tax burden could force companies to lay off workers or "
                "relocate operations overseas, thereby harming the national economy.\n\n"
                "Meanwhile, economists are divided on the long-term impacts of the policy. Some analysts forecast that the "
                "tax will stimulate a robust green jobs sector and create new avenues of economic growth. Others, however, "
                "warn that the transition period could lead to persistent inflation and supply chain disruptions. As the "
                "debate intensifies, the policy is set to face legislative votes next month."
            )
        )

        # Call analyze and measure execution time
        start_time = time.time()
        claims = analyzer.analyze(article)
        execution_time = time.time() - start_time

        # Print formatted results
        print("\n==========================================")
        print("ANALYSIS RESULTS")
        print("==========================================\n")
        print("Main Topic:")
        print(claims.main_topic)
        print("\nKey Claims:")
        for claim in claims.key_claims:
            print(f"- {claim}")
        print("\nAuthor Stance:")
        print(claims.author_stance)
        print("\nTone:")
        print(claims.tone)
        print("\nFraming Summary:")
        print(claims.framing_summary)
        print(f"\nExecution Time: {execution_time:.2f} seconds")
        print("\n==========================================")

    except Exception as err:
        print(f"\nError occurred during claim analyzer execution: {err}")
