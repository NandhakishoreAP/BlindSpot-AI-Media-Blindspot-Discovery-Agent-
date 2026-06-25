import re
from typing import Optional, Dict, List

from models.data_models import ArticleData, ClaimAnalysis
from utils.logger import get_logger


def calculate_understanding_score(
    article: ArticleData,
    claims: Optional[ClaimAnalysis],
    entities: dict,
    used_fallback_claims: bool = False,
    used_fallback_blindspots: bool = False,
    used_fallback_evidence: bool = False,
    metadata_only_article: bool = False
) -> dict:
    """
    Computes an article understanding score (0-100) based on:
    - Entity quality (0-25)
    - Claim quality (0-25)
    - Claim/entity alignment (0-25)
    - Topic/article alignment (0-25)
    
    Applies penalties when fallback logic was used:
    - metadata_only: force score to 0
    - fallback claims: -40
    - fallback blindspots: -40
    - fallback evidence: -30

    Returns dict with score, component breakdown, and decision.
    """
    logger = get_logger("understanding_score")
    
    # Bug 2: Metadata-only articles must score 0
    if metadata_only_article:
        logger.warning("Metadata-only article: understanding score forced to 0")
        return {
            "score": 0,
            "components": {"entity_quality": 0, "claim_quality": 0, "claim_entity_alignment": 0, "topic_article_alignment": 0},
            "penalties": {"metadata_only": True, "fallback_claims": False, "fallback_blindspots": False, "fallback_evidence": False, "total_penalty": -100},
            "decision": "abort",
            "details": {"entity_count": 0, "claim_count": 0, "entity_claim_match_ratio": 0}
        }

    # 1. Entity quality (0-25) — continuous
    entity_count = sum(len(v) for v in entities.values()) if entities else 0
    entity_score = min(entity_count / 10.0, 1.0) * 25

    # 2. Claim quality (0-25) — continuous
    claim_count = 0
    if claims and claims.key_claims:
        claim_count = len(claims.key_claims)
    claim_score = min(claim_count / 6.0, 1.0) * 25

    # 3. Claim/entity alignment (0-25) — continuous ratio
    all_entities_lower = set()
    if entities:
        for cat_list in entities.values():
            for e in cat_list:
                all_entities_lower.add(e.lower())

    alignment_score = 0
    if claims and claims.key_claims and all_entities_lower:
        entity_claim_count = 0
        for claim in claims.key_claims:
            claim_lower = claim.lower()
            if any(e in claim_lower for e in all_entities_lower):
                entity_claim_count += 1
        ratio = entity_claim_count / max(len(claims.key_claims), 1)
        alignment_score = ratio * 25

    # 4. Topic/article alignment (0-25) — continuous weighted
    topic_score = 0
    if claims and claims.main_topic and article and article.title:
        topic_lower = claims.main_topic.lower()
        title_lower = article.title.lower()
        content_sample = (article.content or "")[:500].lower()

        topic_words = set(w for w in re.findall(r'\b\w+\b', topic_lower) if len(w) > 3)
        title_words = set(w for w in re.findall(r'\b\w+\b', title_lower) if len(w) > 3)
        content_words = set(w for w in re.findall(r'\b\w+\b', content_sample) if len(w) > 3)

        if topic_words:
            title_overlap_ratio = len(topic_words.intersection(title_words)) / max(len(topic_words), 1) if title_words else 0
            content_overlap_ratio = len(topic_words.intersection(content_words)) / max(len(topic_words), 1) if content_words else 0
            entity_in_topic = any(e in topic_lower for e in all_entities_lower)
            topic_score = (title_overlap_ratio * 15) + (content_overlap_ratio * 5) + (5 if entity_in_topic else 0)

    total = entity_score + claim_score + alignment_score + topic_score

    # Apply fallback penalties (Bug 2: updated severity)
    penalty = 0
    if used_fallback_claims:
        penalty -= 40
        logger.warning("Fallback claims penalty: -40")
    if used_fallback_blindspots:
        penalty -= 40
        logger.warning("Fallback blindspots penalty: -40")
    if used_fallback_evidence:
        penalty -= 30
        logger.warning("Fallback evidence penalty: -30")
    
    total = max(0, total + penalty)

    if metadata_only_article:
        decision = "abort"
    elif total >= 70:
        decision = "proceed"
    elif total >= 50:
        decision = "proceed_with_warning"
    else:
        decision = "proceed_with_warning"

    result = {
        "score": total,
        "components": {
            "entity_quality": entity_score,
            "claim_quality": claim_score,
            "claim_entity_alignment": alignment_score,
            "topic_article_alignment": topic_score
        },
        "penalties": {
            "fallback_claims": used_fallback_claims,
            "fallback_blindspots": used_fallback_blindspots,
            "fallback_evidence": used_fallback_evidence,
            "total_penalty": penalty
        },
        "decision": decision,
        "details": {
            "entity_count": entity_count,
            "claim_count": claim_count,
            "entity_claim_match_ratio": round(alignment_score / 25, 2) if alignment_score > 0 else 0
        }
    }

    logger.info(f"Understanding score: {total}/100 ({decision}, penalty={penalty})")
    return result
