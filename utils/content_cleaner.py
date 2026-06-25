import re


BOILERPLATE_PATTERNS = [
    # Navigation / UI
    r'\bToggle\b',
    r'\bSection Navigation\b',
    r'\bPrint\b',
    r'\bShare\b',
    r'\bFollow\b',
    r'\bSubscribe\b',
    r'\bSign ?[Ii]n\b',
    r'\bSign ?[Uu]p\b',
    r'\bNewsletter\b',
    r'\bDonate\b',
    r'\bDonation\b',
    r'\bLog ?[Ii]n\b',
    r'\bLog ?[Oo]ut\b',
    r'\bRegister\b',
    # Article widgets
    r'\bRelated Articles?\b',
    r'\bRecommended Articles?\b',
    r'\bYou May Also Like\b',
    r'\bMore from\b',
    r'\bRead More\b',
    r'\bSee Also\b',
    r'\bTrending\b',
    r'\bMost Read\b',
    r'\bMost Popular\b',
    r'\bSponsored\b',
    r'\bAdvertisement\b',
    # Generic UI
    r'\bSections\b',
    r'\bNavigation\b',
    r'\bMenu\b',
    r'\bClose\b',
    r'\bSkip to content\b',
    r'\bBack to top\b',
    r'\bBack to\b',
    r'\bLoading\b',
]


def clean_article_content(content: str) -> str:
    """
    Removes boilerplate navigation text, widget headers, and repeated UI strings
    from article content before it reaches the claim analyzer.
    """
    if not content:
        return ""

    lines = content.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines that are purely boilerplate
        skip = False
        for pat in BOILERPLATE_PATTERNS:
            if re.search(pat, stripped, re.IGNORECASE):
                # Only skip if the line is short (likely a heading/button text)
                # or the boilerplate is most of the line
                if len(stripped.split()) <= 8:
                    skip = True
                    break
        if not skip:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def extract_main_article_body(content: str) -> str:
    """
    Further reduces content to likely article body paragraphs.
    Removes very short lines (likely metadata/navigation remnants).
    """
    if not content:
        return ""

    paragraphs = content.split("\n")
    valid = []
    for p in paragraphs:
        words = p.split()
        # Skip very short lines or lines with mostly numbers/symbols
        if len(words) < 5:
            continue
        alpha_ratio = sum(1 for w in words if any(c.isalpha() for c in w)) / max(len(words), 1)
        if alpha_ratio < 0.5:
            continue
        valid.append(p)

    return "\n".join(valid)
