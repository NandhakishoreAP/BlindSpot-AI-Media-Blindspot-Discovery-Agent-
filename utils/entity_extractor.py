import re
from typing import Dict, List, Optional

from utils.logger import get_logger

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

ENTITY_CATEGORIES = [
    "people", "organizations", "locations", "policies",
    "programs", "exams", "political_parties", "institutions"
]

DEFAULT_ENTITIES = {cat: [] for cat in ENTITY_CATEGORIES}

KNOWN_EXAMS = {"NEET", "JEE", "UPSC", "GATE", "CAT", "GMAT", "GRE", "SAT", "ACT", "CLAT", "AIIMS", "JIPMER"}
KNOWN_PROGRAMS = {"GST", "CBT", "RTE", "POSHAN", "PM-KISAN", "Ayushman", "Digital India", "Make in India"}
KNOWN_PARTIES = {"BJP", "INC", "AAP", "CPI", "CPM", "TMC", "DMK", "AIADMK", "SP", "BSP", "NCP", "JDU", "RJD", "YSRCP", "TDP", "TRS", "BRS", "PDP", "JKNC"}
KNOWN_INSTITUTIONS = {"NTA", "NITI Aayog", "RBI", "SEBI", "ISRO", "DRDO", "ICAR", "UGC", "AICTE", "UNESCO", "WHO", "IMF", "World Bank", "SCo, India", "Supreme Court", "Election Commission"}


class EntityExtractor:
    def __init__(self):
        self.logger = get_logger("entity_extractor")

    def extract(self, text: str, title: str = "") -> dict:
        combined = f"{title} {text}"
        entities = dict(DEFAULT_ENTITIES)

        # 1. LLM-based extraction
        entities = self._llm_extract(combined) or dict(DEFAULT_ENTITIES)

        # 2. Regex fallback for missed entities
        regex_entities = self._regex_extract(combined)
        entities = self._merge(entities, regex_entities)

        # 3. Known entity matching
        known = self._match_known(combined)
        entities = self._merge(entities, known)

        self.logger.info(f"Extracted entities: {sum(len(v) for v in entities.values())} total")
        return entities

    def _llm_extract(self, text: str) -> Optional[dict]:
        try:
            from config import Config
            from llm.client_factory import create_llm_client

            config = Config.from_env()
            client = create_llm_client(config)

            if not client.available:
                self.logger.info("LLM extraction skipped (backend unavailable)")
                return None

            prompt = (
                "Extract named entities from the following text. "
                "Return ONLY a JSON object with these keys: "
                "people, organizations, locations, policies, programs, exams, political_parties, institutions. "
                "Each key must map to a list of strings. "
                "Use the exact entity name as it appears in the text. "
                "If no entities are found for a category, use an empty list.\n\n"
                f"Text: {text[:2000]}"
            )

            response = client.generate_json(
                prompt=prompt,
                system_prompt="/no_think\n\nReturn ONLY valid JSON with entity lists.",
                temperature=0.1,
                num_predict=256
            )

            if isinstance(response, dict):
                for cat in ENTITY_CATEGORIES:
                    if cat not in response or not isinstance(response[cat], list):
                        response[cat] = []
                    response[cat] = [str(e).strip() for e in response[cat] if e and str(e).strip()]
                return response
        except Exception as e:
            self.logger.warning(f"LLM entity extraction failed: {e}")
        return None

    def _regex_extract(self, text: str) -> dict:
        entities = {cat: [] for cat in ENTITY_CATEGORIES}

        # Uppercase multi-word phrases (potential organizations/institutions/programs)
        matches = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+(?:\s+[A-Z][a-zA-Z]+)*\b', text)
        for m in matches:
            m = m.strip()
            if len(m.split()) <= 6 and len(m) > 3:
                entities["organizations"].append(m)

        # Location patterns: "in/during/at/from <Capitalized>"
        loc_pattern = r'(?:in|at|from|to|across|throughout)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        loc_matches = re.findall(loc_pattern, text)
        for m in loc_matches:
            m = m.strip()
            if len(m.split()) <= 3 and len(m) > 2:
                entities["locations"].append(m)

        return entities

    def _match_known(self, text: str) -> dict:
        entities = {cat: [] for cat in ENTITY_CATEGORIES}

        for exam in KNOWN_EXAMS:
            if exam.lower() in text.lower():
                entities["exams"].append(exam)

        for prog in KNOWN_PROGRAMS:
            if prog.lower() in text.lower():
                entities["programs"].append(prog)

        for party in KNOWN_PARTIES:
            if party.lower() in text.lower():
                entities["political_parties"].append(party)

        for inst in KNOWN_INSTITUTIONS:
            if inst.lower() in text.lower():
                entities["institutions"].append(inst)

        return entities

    def _merge(self, primary: dict, secondary: dict) -> dict:
        result = dict(primary)
        for cat in ENTITY_CATEGORIES:
            seen = set(e.lower() for e in result.get(cat, []))
            for entity in secondary.get(cat, []):
                e_lower = entity.lower()
                if e_lower not in seen:
                    is_dup = False
                    if fuzz:
                        for existing in result.get(cat, []):
                            if fuzz.ratio(e_lower, existing.lower()) > 85:
                                is_dup = True
                                break
                    if not is_dup:
                        result[cat].append(entity)
                        seen.add(e_lower)
        return result
