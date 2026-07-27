<div align="center">

# BlindSpot AI — Agentic Media Blindspot Discovery System

**An agentic AI system that reads news articles, detects what perspectives are missing, autonomously searches for evidence, and generates a balanced blindspot report — built as a solo end-to-end implementation.**

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-dashboard-FF4B4B.svg)
![Qwen3](https://img.shields.io/badge/Qwen3-8B%20local-6B46C1.svg)
![DuckDuckGo](https://img.shields.io/badge/DuckDuckGo-search-DE5833.svg)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

</div>

> **The Problem**: Modern fact-checking tools answer "Is this true or false?" — but rarely answer "What important information is *missing*?" Even factually accurate articles can create confirmation bias, information bubbles, and polarized opinions through selective framing, omitted expert opinions, missing historical context, or ignored demographic impacts. Readers have no tool to detect what they aren't being told.

---

## What BlindSpot AI Does

BlindSpot AI is a fully agentic media analysis system for any news article URL. It goes beyond summarization or fact-checking:

- Extracts and parses article content from any publicly accessible news URL
- Identifies the author's stance, tone, key claims, and framing strategy using a local LLM
- Detects specific blindspots — missing expert voices, omitted economic impacts, ignored historical context, underrepresented demographics, absent scientific evidence
- Autonomously plans a multi-round web search strategy to investigate each blindspot
- Evaluates retrieved evidence for quality, relevance, and whether it supports, contradicts, or adds context
- Decides whether to search again or generate the final report based on confidence scoring
- Produces a structured report with a blindspot score (0–100), balanced conclusion, and downloadable JSON
- Displays everything in a clean Streamlit dashboard with tabs, expandable cards, and color-coded scoring

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│              User provides any news article URL                  │
│                   via Streamlit dashboard                        │
└──────────────────────────────┬──────────────────────────────────┘
                                │
┌──────────────────────────────▼──────────────────────────────────┐
│                      EXTRACTION LAYER                            │
│         Requests + BeautifulSoup4 — title, author, date,        │
│         content extracted; URL validated before any LLM call     │
└──────────────────────────────┬──────────────────────────────────┘
                                │
┌──────────────────────────────▼──────────────────────────────────┐
│                     INTELLIGENCE LAYER                           │
│                                                                  │
│   ClaimAnalyzer          BlindspotDetector     EvidenceEvaluator │
│   (stance, tone,         (what is missing,     (quality scoring, │
│    key claims,            importance rating,    relevance tags,   │
│    framing)               search queries)       insight extract)  │
│                                                                  │
│                    Qwen3 8B via Ollama (local, free, private)    │
└──────────────────────────────┬──────────────────────────────────┘
                                │
┌──────────────────────────────▼──────────────────────────────────┐
│                       AGENTIC LOOP                               │
│                                                                  │
│   Planner ──► Search ──► Evaluate ──► Confidence Check          │
│      ▲                                       │                   │
│      └───────── Need More Research? ─────────┘                  │
│                         │ No                                     │
│                         ▼                                        │
│                  ReportGenerator                                 │
└──────────────────────────┬──────────────────────────────────────┘
                                │
┌──────────────────────────────▼──────────────────────────────────┐
│                     PRESENTATION LAYER                           │
│         Streamlit dashboard — score card, blindspot cards,       │
│         evidence tabs, balanced conclusion, JSON download        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This Is Agentic AI

Most AI pipelines follow a fixed sequence: input → LLM → output. BlindSpot AI does not.

The agent continuously evaluates its own state and makes decisions:

```
Goal: Find what this article is missing
  │
  ▼
Reason: What perspectives are absent?
  │
  ▼
Plan: Which searches would surface that evidence?
  │
  ▼
Act: Execute web searches via DuckDuckGo
  │
  ▼
Observe: What did the searches return?
  │
  ▼
Evaluate: Is this evidence high quality? Does it support or contradict?
  │
  ▼
Reflect: Is my confidence score above the threshold?
  │
  ├── No → Re-plan with new queries → Loop back to Act
  │
  └── Yes → Generate final balanced report
```

The Planner module uses three hard rules before calling the LLM:

1. If maximum search attempts are reached → force report generation
2. If confidence score exceeds threshold → generate report immediately
3. If no evidence exists yet → use blindspot-derived queries directly (no LLM overhead)

Only for intermediate decisions does the Planner invoke the LLM to reason about what to search next. This avoids unnecessary token usage while keeping the loop genuinely adaptive.

---

## Features

| Component | Description |
|---|---|
| Article Extractor | Fetches any public news URL, strips navigation/ads/scripts, extracts clean article text with title, author, and date |
| Claim Analyzer | Identifies main topic, 3–6 key claims, author stance (e.g. Pro-regulation, Skeptical), tone (e.g. Alarmist, Balanced), and framing strategy |
| Blindspot Detector | Detects 3–5 specific missing perspectives with category labels, importance ratings (High/Medium/Low), and ready-to-use search queries |
| Agentic Search Loop | Planner decides SEARCH / SEARCH\_MORE / GENERATE\_REPORT at each step; DuckDuckGo search tool deduplicates across rounds |
| Evidence Evaluator | Tags each search result as Supports / Contradicts / Adds Context, rates source quality (High/Medium/Low), extracts a one-sentence insight |
| Confidence Scorer | Calculates 0–100 score based on evidence quality, quantity, and coverage ratio against detected blindspots |
| Report Generator | Produces blindspot score, score reasoning, missing categories list, and balanced 3–5 sentence conclusion grounded in gathered evidence |
| JSON Storage | Every report auto-saved to `reports/` with timestamp filename for persistence across sessions |
| Streamlit Dashboard | Score card with color coding, expandable blindspot cards, evidence tabs by relevance type, past report loader from sidebar |

---

## Project Structure

```
blindspot_ai/
├── app.py                        Streamlit UI — full dashboard
├── main.py                       CLI entry point for end-to-end testing
├── config.py                     Config class — loads from .env with defaults
├── requirements.txt              All dependencies
├── .env                          Environment variables (not committed)
├── reports/                      Auto-saved JSON reports (timestamped)
│
├── agent/
│   ├── orchestrator.py           MediaBlindspotAgent — main agentic loop
│   ├── planner.py                Planner — SEARCH / SEARCH_MORE / GENERATE_REPORT
│   └── state.py                  AgentState — working memory across loop iterations
│
├── tools/
│   ├── article_extractor.py      Requests + BeautifulSoup4 article parser
│   └── search_tool.py            DuckDuckGo search with deduplication and formatting
│
├── llm/
│   ├── ollama_client.py          Ollama API client with JSON extraction and retry
│   ├── claim_analyzer.py         LLM module — claims, stance, tone, framing
│   ├── blindspot_detector.py     LLM module — missing perspectives detection
│   ├── evidence_evaluator.py     LLM module — relevance, quality, insight scoring
│   └── report_generator.py       LLM module — final report + JSON persistence
│
├── models/
│   └── data_models.py            All Pydantic models — ArticleData, ClaimAnalysis,
│                                 Blindspot, SearchResult, Evidence, AgentState,
│                                 PlannerDecision, BlindspotReport
│
└── utils/
    └── logger.py                 Shared logger factory with timestamped formatting
```

---

## Technology Stack

| Component | Technology | Reason |
|---|---|---|
| Language | Python 3.11 | Largest AI/ML ecosystem, fastest prototyping |
| LLM | Qwen3 8B via Ollama | Free, local, no API cost, strong reasoning, privacy-preserving |
| Article Extraction | Requests + BeautifulSoup4 | Lightweight, reliable, fully controllable |
| Web Search | DuckDuckGo Search (duckduckgo-search) | Free, no API key, no quota limits |
| Data Validation | Pydantic v2 | Type-safe models with auto-validation and serialization |
| UI | Streamlit | Fastest path to a functional AI dashboard |
| Storage | JSON files | Zero setup, sufficient for report persistence at this scale |
| Environment | python-dotenv | Clean separation of config from code |
| Logging | Python logging | Structured per-module logs with timestamps |

### Why These Choices Over Alternatives

**Qwen3 8B over GPT-4 / Gemini:**
Running locally via Ollama means zero API cost, zero latency spikes from remote calls, and no data leaving the machine. Qwen3 8B has strong instruction-following and JSON output reliability — the two things that matter most for an agentic pipeline where every module expects structured output.

**DuckDuckGo over Google CSE / SerpAPI:**
Both alternatives require API keys and impose quota limits. DuckDuckGo Search via the `duckduckgo-search` Python package is completely free with no registration. For a system that may run 3–5 search rounds per article, hitting a quota mid-analysis would break the agentic loop.

**BeautifulSoup4 over Scrapy / Newspaper3k:**
Scrapy is built for large-scale crawling — overkill for single-URL extraction. Newspaper3k has inconsistent results on modern news sites that heavily use JavaScript rendering. BeautifulSoup4 gives full control over which tags to extract and in what priority order.

**JSON files over SQLite / PostgreSQL:**
Each report is a self-contained document with no relational structure. JSON files with timestamp filenames give instant human-readability, easy portability, and zero setup cost. The Streamlit sidebar loads them directly with `Path.glob()`.

**Streamlit over React / Flask:**
The entire UI including sidebar, tabs, expandable cards, progress bars, and download buttons is implemented in pure Python. A React frontend would require a separate backend API layer, build tooling, and JavaScript — tripling the implementation surface for no functional gain at this scale.

---

## Data Flow — Step by Step

```
1. User pastes URL into Streamlit input field

2. ArticleExtractor
   ├── Validates URL (scheme + netloc check)
   ├── GET request with browser User-Agent header
   ├── BeautifulSoup parses HTML
   ├── Strips <script>, <style>, <nav>, <footer>, <aside>
   ├── Extracts: title (h1 → og:title → <title>)
   │            author (meta[name=author] → class containing "author")
   │            date   (article:published_time → <time datetime>)
   │            content (article → div.content → body → all <p> tags)
   └── Returns ArticleData with word_count auto-calculated

3. ClaimAnalyzer (Qwen3 8B)
   ├── System prompt: expert media analyst role
   ├── User prompt: article title + author + first 3000 chars of content
   ├── Output: JSON with main_topic, key_claims[], author_stance, tone, framing_summary
   └── Returns ClaimAnalysis

4. BlindspotDetector (Qwen3 8B)
   ├── System prompt: media literacy and investigative journalism role
   ├── User prompt: claims + framing + first 2000 chars of content
   ├── Output: JSON with blindspots[] — each has category, description, importance, suggested_search_query
   └── Returns List[Blindspot]

5. AgentState created
   └── Holds article, claims, blindspots, evidence[], confidence_score, search_attempts

6. Agentic Loop begins
   ├── Planner.decide(state)
   │   ├── Rule 1: attempts >= max → GENERATE_REPORT
   │   ├── Rule 2: confidence >= threshold → GENERATE_REPORT
   │   ├── Rule 3: first iteration → SEARCH with blindspot-derived queries
   │   └── Rule 4: subsequent → LLM decides SEARCH_MORE with new queries
   │
   ├── SearchTool.search_multiple(queries)
   │   ├── DuckDuckGo text search per query
   │   ├── 1.5s delay between queries (rate limit protection)
   │   └── URL-based deduplication across all rounds
   │
   ├── EvidenceEvaluator.evaluate(article, claims, blindspots, results)
   │   ├── LLM scores each result: relevance + quality + key_insight + include flag
   │   └── Returns List[Evidence] — only included items
   │
   ├── confidence_score = calculate_confidence(blindspots, evidence)
   │   ├── High quality evidence: +15 pts each
   │   ├── Medium quality: +8 pts each
   │   ├── Low quality: +3 pts each
   │   ├── Contradicting evidence: +10 pts each (opposing views are valuable)
   │   └── Score scaled by coverage ratio: evidence count / blindspot count
   │
   └── Loop continues until GENERATE_REPORT decision

7. ReportGenerator (Qwen3 8B)
   ├── Prompt includes all evidence, blindspots, claims, confidence score
   ├── Output: blindspot_score (0–100), score_reasoning, missing_categories[], balanced_conclusion
   ├── BlindspotReport assembled with all state data
   └── Saved as reports/report_YYYYMMDD_HHMMSS.json

8. Streamlit displays:
   ├── Score card (green ≤40 / orange ≤70 / red >70)
   ├── Metrics row: claims, blindspots, evidence items, search attempts
   ├── Blindspot cards (expandable, with suggested search query)
   ├── Evidence tabs (Supports / Contradicts / Adds Context)
   ├── Balanced conclusion (styled card)
   └── Download button (full JSON report)
```

---

## Blindspot Score Interpretation

| Score Range | Label | Meaning |
|---|---|---|
| 0 – 40 | 🟢 Low Blindspot | Article covers most major perspectives; minor gaps exist |
| 41 – 70 | 🟡 Medium Blindspot | Noticeable omissions in evidence, expert voices, or context |
| 71 – 100 | 🔴 High Blindspot | Significant framing gaps; important perspectives systematically missing |

The score is generated by the LLM grounded in the actual evidence gathered — not a keyword count or sentiment score. The reasoning is always shown alongside the number.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Qwen3 8B model pulled locally

```bash
# Pull the model (one-time, ~5GB)
ollama pull qwen3:8b

# Verify Ollama is running
ollama list
```

### Installation

```bash
# Clone the repository
git clone https://github.com/NandhakishoreAP/blindspot-ai.git
cd blindspot_ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
MAX_SEARCH_ATTEMPTS=5
CONFIDENCE_THRESHOLD=70
MAX_SEARCH_RESULTS=5
REPORTS_DIR=reports
```

### Run the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Run via CLI (for testing)

```bash
python main.py
```

This runs a full end-to-end analysis on a hardcoded BBC article and prints the complete report to the terminal.

---

## API — Internal Module Interface

BlindSpot AI is not a REST API, but each module has a clean programmatic interface:

```python
from config import Config
from agent.orchestrator import MediaBlindspotAgent

config = Config.from_env()
agent = MediaBlindspotAgent(config)

report = agent.analyze("https://www.bbc.com/news/articles/...")

print(report.blindspot_score)       # 0–100
print(report.balanced_conclusion)   # 3–5 sentence synthesis
print(report.missing_categories)    # ['Expert Opinion', 'Economic Impact', ...]
print(report.evidence_count)        # number of evidence items gathered
```

The `BlindspotReport` object is a fully typed Pydantic model and can be serialized to JSON with `report.model_dump()`.

---

## Pydantic Data Models

| Model | Purpose |
|---|---|
| `ArticleData` | Extracted article with auto-calculated word count |
| `ClaimAnalysis` | LLM output — topic, claims, stance, tone, framing |
| `Blindspot` | Single missing perspective with importance and search query |
| `SearchResult` | One DuckDuckGo result with cleaned snippet and extracted source domain |
| `Evidence` | Evaluated search result with relevance tag, quality rating, and key insight |
| `AgentState` | Full working memory — article, claims, blindspots, evidence, confidence, attempts |
| `PlannerDecision` | Planner output — action enum, query list, reasoning string |
| `BlindspotReport` | Final output — all fields needed for display and persistence |

---

## Known Limitations

Stated directly rather than glossed over:

- **Paywalled articles**: BeautifulSoup cannot bypass login walls or JavaScript-rendered content. URLs from sites like The Times, Bloomberg, or WSJ behind hard paywalls will return insufficient content and raise a `ValueError`. The error is surfaced cleanly in the UI.
- **Search rate limits**: DuckDuckGo does not publish rate limit thresholds. Heavy usage (many analyses in quick succession) may result in temporary blocks. The 1.5-second delay between queries mitigates this but does not eliminate the risk.
- **LLM JSON reliability**: Qwen3 8B occasionally wraps JSON output in markdown code fences or adds preamble text. The `OllamaClient` handles this with a three-stage extraction pipeline (direct parse → regex code block → raw object search) and retries up to 3 times before raising.
- **Confidence scoring is approximate**: The confidence formula uses evidence count and quality as proxies for "sufficient research." It cannot verify whether the evidence actually addresses the specific blindspot it was retrieved for — that would require a separate grounding step.
- **Local hardware dependency**: Qwen3 8B requires a machine with sufficient RAM (8GB minimum, 16GB recommended). On CPU-only machines, each LLM call may take 30–90 seconds, making a full analysis 5–10 minutes. A GPU reduces this to under a minute.
- **English-only analysis**: The claim analyzer and blindspot detector prompts are written in English. Articles in other languages will be parsed but the LLM analysis quality will degrade.

---

## Functional Requirements Coverage

| ID | Requirement | Status |
|---|---|---|
| FR1 | User provides article URL | ✅ Streamlit text input with validation |
| FR2 | System validates URL | ✅ scheme + netloc check before any network call |
| FR3 | System extracts article content | ✅ ArticleExtractor with multi-strategy fallback |
| FR4 | System identifies claims and stance | ✅ ClaimAnalyzer via Qwen3 8B |
| FR5 | System detects blindspots | ✅ BlindspotDetector with importance ratings |
| FR6 | System autonomously generates search strategies | ✅ Planner with LLM-driven query generation |
| FR7 | System retrieves external evidence | ✅ SearchTool with deduplication across rounds |
| FR8 | System evaluates evidence quality | ✅ EvidenceEvaluator with quality + relevance scoring |
| FR9 | System decides whether to search again | ✅ Agentic loop with confidence threshold gating |
| FR10 | System generates balanced reports | ✅ ReportGenerator with grounded LLM synthesis |
| FR11 | System calculates blindspot scores | ✅ 0–100 score with reasoning |
| FR12 | System saves reports automatically | ✅ JSON persistence with timestamp filenames |

---

## Acknowledgements

Built solo as a complete end-to-end implementation covering system design, data modeling, agentic orchestration, LLM prompt engineering, web scraping, search integration, evidence evaluation, and frontend dashboard development.

LLM inference: [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B) via [Ollama](https://ollama.com).
Search: [duckduckgo-search](https://github.com/deedy5/duckduckgo_search).
Article parsing: [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/).
Dashboard: [Streamlit](https://streamlit.io).

---

<div align="center">

**Turning news consumption from passive reading into active, evidence-backed understanding.**

</div>