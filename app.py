import logging
import streamlit as st
import json
import os
import time
import traceback
import threading
from pathlib import Path

logger = logging.getLogger(__name__)
from config import Config
from agent.orchestrator import MediaBlindspotAgent
from models.data_models import BlindspotReport

# DictAttrProxy is an ultra-robust dict subclass supporting both attribute-based (getattr)
# and key-based (.get) lookups. Prevents schema and version mismatch crashes.
class DictAttrProxy(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = DictAttrProxy(v)
            elif isinstance(v, list):
                self[k] = [DictAttrProxy(i) if isinstance(i, dict) else i for i in v]

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'DictAttrProxy' object has no attribute '{name}'")

    def model_dump(self):
        return self

    def dict(self):
        return self

def load_safe_report(rep_dict: dict):
    try:
        return BlindspotReport(**rep_dict)
    except Exception as e:
        logger.warning(f"Report load failed ({__file__}:36): {e}")
        return DictAttrProxy(rep_dict)

# Page Config MUST be the first Streamlit command
st.set_page_config(
    page_title="BlindSpot AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional SaaS Custom CSS (Linear/Stripe inspired)
st.markdown(
    """
    <style>
    :root {
        --bg-primary: #08080a;
        --bg-card: #121316;
        --bg-secondary: #17181c;
        --border: #222326;
        --text-primary: #f7f8f8;
        --text-secondary: #8a8f98;
        --success: #09825d;
        --warning: #c97b00;
        --danger: #d33d44;
        --accent: #5e6ad2;
    }
    
    /* Global layout constraint (1400px max width centered) */
    .main .block-container {
        max-width: 1400px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: 0 auto !important;
    }
    
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Dashboard Cards */
    .dashboard-card {
        background-color: var(--bg-card);
        border-radius: 10px;
        padding: 20px 28px;
        border: 1px solid var(--border);
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .dashboard-card:hover {
        border-color: #313339;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin-top: 1.2em;
        margin-bottom: 0.6em;
    }
    h1 { font-size: 2.25rem; }
    h2 { font-size: 1.75rem; }
    h3 { font-size: 1.35rem; }
    h4 { font-size: 1.1rem; }
    
    /* Streamlit heading overrides for consistency */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        margin-top: 1.2em !important;
        margin-bottom: 0.6em !important;
    }
    
    /* Report section headings larger */
    .report-section-heading {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
        color: var(--text-primary);
    }
    
    /* Evidence cards */
    .evidence-card {
        background-color: var(--bg-secondary);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border: 1px solid var(--border);
    }
    
    /* Score display */
    .score-large {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
    }
    .score-unit {
        font-size: 1.2rem;
        color: var(--text-secondary);
        font-weight: 400;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 5px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        border: 1px solid transparent;
    }
    .badge-success { background-color: rgba(9, 130, 93, 0.15); color: #0eb383; border-color: rgba(9, 130, 93, 0.25); }
    .badge-warning { background-color: rgba(201, 123, 0, 0.15); color: #e59c24; border-color: rgba(201, 123, 0, 0.25); }
    .badge-danger { background-color: rgba(211, 61, 68, 0.15); color: #f85c63; border-color: rgba(211, 61, 68, 0.25); }
    .badge-accent { background-color: rgba(94, 106, 210, 0.15); color: #828df2; border-color: rgba(94, 106, 210, 0.25); }
    
    /* Scrollable Recent Reports container */
    .recent-reports-container {
        max-height: 450px;
        overflow-y: auto;
        padding-right: 4px;
    }
    
    /* Custom Scrollbar for Recent Reports */
    .recent-reports-container::-webkit-scrollbar {
        width: 4px;
    }
    .recent-reports-container::-webkit-scrollbar-track {
        background: transparent;
    }
    .recent-reports-container::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 2px;
    }
    .recent-reports-container::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    
    /* Confidence display bar */
    .confidence-bar-container {
        width: 100%;
        height: 8px;
        background-color: var(--bg-secondary);
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "agent" not in st.session_state:
    st.session_state.agent = None
if "report" not in st.session_state:
    st.session_state.report = None
if "error" not in st.session_state:
    st.session_state.error = None
if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = None
if "url_input_value" not in st.session_state:
    st.session_state.url_input_value = ""
if "current_ui_status" not in st.session_state:
    st.session_state.current_ui_status = "Initializing"


# Get Agent Function
def get_agent() -> MediaBlindspotAgent:
    if st.session_state.agent is None:
        try:
            config = Config.from_env()
            st.session_state.agent = MediaBlindspotAgent(config)
        except Exception as e:
            tb = traceback.format_exc()
            st.session_state.error = {
                "message": f"Failed to initialize MediaBlindspotAgent: {e}",
                "traceback": tb
            }
            print(f"Failed to initialize MediaBlindspotAgent:\n{tb}")
    return st.session_state.agent

# Helper to find full report JSON metadata from disk matching report details
def get_full_report_data(report) -> dict:
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return {}
    url_to_match = getattr(report, "article_url", "")
    if not url_to_match:
        return {}
        
    report_files = list(reports_dir.glob("report_*.json"))
    report_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    for rf in report_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)
            rep_obj = data.get("report", {})
            if rep_obj.get("article_url") == url_to_match:
                return data
        except Exception as e:
            logger.warning(f"Report data read failed ({__file__}:250): {e}")
            continue
    return {}

# Callbacks for programmatic widget state updates (safe from StreamlitAPIException)
def handle_clear():
    st.session_state.report = None
    st.session_state.error = None
    st.session_state.url_input_value = ""
    st.session_state.delete_confirm = None

def handle_load_report(rf):
    try:
        with open(rf, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        rep_dict = loaded_data.get("report", {})
        st.session_state.report = load_safe_report(rep_dict)
        st.session_state.error = None
        st.session_state.url_input_value = getattr(st.session_state.report, "article_url", "")
        st.session_state.delete_confirm = None
    except Exception as ex:
        st.session_state.error = {
            "message": f"Error loading report: {ex}",
            "traceback": traceback.format_exc()
        }

def set_sample_url(url):
    st.session_state.url_input_value = url
    st.session_state.report = None
    st.session_state.error = None
    st.session_state.delete_confirm = None

def handle_delete_report(rf, url_to_match):
    try:
        rf.unlink()
        st.session_state.delete_confirm = None
        if st.session_state.report:
            curr_url = getattr(st.session_state.report, "article_url", "")
            if curr_url == url_to_match:
                st.session_state.report = None
                st.session_state.url_input_value = ""
    except Exception as e:
        st.session_state.error = {
            "message": f"Delete failed: {e}",
            "traceback": ""
        }

# Load report files for sidebar
reports_dir = Path("reports")
report_files = []
if reports_dir.exists():
    report_files = list(reports_dir.glob("report_*.json"))
    report_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

# Sidebar setup
with st.sidebar:
    st.markdown("## BlindSpot AI")
    st.markdown("---")
    
    # System Status Section (read-only)
    try:
        config = Config.from_env()
        model = config.OLLAMA_MODEL
        status = "Active" if config.OLLAMA_BASE_URL else "Inactive"
        dev_mode = debug_mode_env = getattr(config, "DEBUG_MODE", False) or os.environ.get("DEBUG_MODE", "").lower() in ("1", "true")
    except Exception as e:
        logger.warning(f"System status check failed ({__file__}:336): {e}")
        dev_mode = False
        
    st.markdown("---")
    
    # Recent Reports Section
    st.markdown("### Recent Reports")
    if not report_files:
        st.info("No recent reports.")
    else:
        st.markdown('<div class="recent-reports-container">', unsafe_allow_html=True)
        # Display only latest 10 reports
        for idx, rf in enumerate(report_files[:10]):
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rep_obj = data.get("report", {})
                title = rep_obj.get("article_title", rf.name)
                score = rep_obj.get("blindspot_score", "N/A")
                date = rep_obj.get("analysis_timestamp", "")[:10]
                url_to_match = rep_obj.get("article_url", "")
            except Exception as e:
                logger.warning(f"Report title read failed ({__file__}:358): {e}")
                title = rf.name
                score = "Error"
                date = ""
                url_to_match = ""
                
            # Score badge class
            if isinstance(score, (int, float)):
                if score <= 40:
                    badge_class = "badge-success"
                elif score <= 70:
                    badge_class = "badge-warning"
                else:
                    badge_class = "badge-danger"
            else:
                badge_class = "badge-accent"
                
            st.markdown(
                f"""
                <div style="border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin-bottom: 8px; background-color: var(--bg-card);">
                    <div style="font-weight: 500; font-size: 0.85rem; color: var(--text-primary); margin-bottom: 4px; line-height: 1.2;">{title[:35]}...</div>
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 8px;">
                        <span>{date}</span>
                        <span class="badge {badge_class}" style="font-size: 0.65rem; padding: 1px 4px;">Score: {score}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Inline Deletion confirmation or buttons
            if st.session_state.delete_confirm == rf:
                st.markdown(
                    "<div style='text-align: center; margin-bottom: 8px;'>\n"
                    "<span style='color: var(--danger); font-size: 0.85rem; display: block; margin-bottom: 4px;'>Confirm delete?</span>\n"
                    "</div>",
                    unsafe_allow_html=True
                )
                st.button("Confirm Delete", key=f"confirm_yes_{idx}_{rf.name}", type="primary", use_container_width=True, on_click=handle_delete_report, args=(rf, url_to_match))
                if st.button("Cancel", key=f"confirm_no_{idx}_{rf.name}", use_container_width=True):
                    st.session_state.delete_confirm = None
                    st.rerun()
            else:
                c_load, c_del = st.columns(2)
                with c_load:
                    st.button("Load", key=f"load_{idx}_{rf.name}", use_container_width=True, on_click=handle_load_report, args=(rf,))
                with c_del:
                    if st.button("Delete", key=f"del_{idx}_{rf.name}", use_container_width=True):
                        st.session_state.delete_confirm = rf
                        st.rerun()
            st.markdown("")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # System Status (clean table, below recent reports)
    st.markdown("### System Status")
    try:
        config_sys = Config.from_env()
        
        # Get active model info from agent if available
        agent_obj = st.session_state.get("agent")
        if agent_obj and hasattr(agent_obj, "ollama_client") and agent_obj.ollama_client:
            active_model = agent_obj.ollama_client.model
            if hasattr(agent_obj, "ollama_available"):
                model_available = agent_obj.ollama_available
            else:
                model_available = False
        else:
            active_model = config_sys.OLLAMA_MODEL
            model_available = False
        
        model_status = "Available" if model_available else "Unavailable"
        model_status_color = "var(--success)" if model_available else "var(--danger)"
        
        search_status = "Active" if config_sys.OLLAMA_BASE_URL else "Inactive"
        runtime_budget = f"{getattr(config_sys, 'MAX_RESEARCH_SECONDS', 90)} sec"
        
        # Last analysis result
        last_report = st.session_state.get("report")
        last_error = st.session_state.get("error")
        if last_report:
            reason = getattr(last_report, "completion_reason", "completed")
            score = getattr(last_report, "blindspot_score", 0)
            if reason in ("system_error", "model_unavailable"):
                last_result = f"Failed ({reason})"
                last_result_color = "var(--danger)"
            elif reason in ("empty_content", "access_restricted", "metadata_only", "extraction_failed"):
                last_result = f"Aborted ({reason})"
                last_result_color = "var(--warning)"
            else:
                last_result = f"Score: {score}/100"
                last_result_color = "var(--success)"
        elif last_error:
            last_result = "Failed"
            last_result_color = "var(--danger)"
        else:
            last_result = "None"
            last_result_color = "var(--text-secondary)"
        
        st.markdown(
            f"""
            <div style="font-size:0.85rem; line-height:1.6;">
                <div style="display:flex; justify-content:space-between; padding:2px 0; border-bottom: 1px solid var(--border);">
                    <span style="color:var(--text-secondary);">Active Model</span>
                    <span style="color:var(--text-primary);">{active_model}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:2px 0; border-bottom: 1px solid var(--border);">
                    <span style="color:var(--text-secondary);">Model Status</span>
                    <span style="color:{model_status_color};">{model_status}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:2px 0; border-bottom: 1px solid var(--border);">
                    <span style="color:var(--text-secondary);">Search Engine</span>
                    <span style="color:var(--accent);">{search_status}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:2px 0; border-bottom: 1px solid var(--border);">
                    <span style="color:var(--text-secondary);">Runtime Budget</span>
                    <span style="color:var(--text-primary);">{runtime_budget}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:2px 0;">
                    <span style="color:var(--text-secondary);">Last Analysis</span>
                    <span style="color:{last_result_color};">{last_result}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        logger.warning(f"System status check failed ({__file__}:336): {e}")
        st.warning("Status unavailable")
    
    st.markdown("---")
    
    # About Section (Collapsed by default)
    with st.expander("About", expanded=False):
        st.markdown(
            "BlindSpot AI analyzes target news articles to identify omitted perspectives "
            "and information coverage gaps by cross-referencing claims against academic, "
            "policy, and media reference sources."
        )

# Main Header (Clean typography, no emojis/gradients)
st.markdown(
    """
    <div style="margin-bottom: 32px; margin-top: 10px;">
        <h1 style="font-size: 2.25rem; font-weight: 600; color: var(--text-primary); margin: 0 0 8px 0; letter-spacing: -0.03em;">BlindSpot AI</h1>
        <p style="font-size: 1.05rem; color: var(--text-secondary); margin: 0;">Discover missing perspectives and framing gaps in news coverage.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# URL Input Section (Aligned in a single horizontal row)
col_input, col_btn, col_clr = st.columns([7, 1.5, 1.5])

with col_input:
    url_input = st.text_input(
        label="URL",
        key="url_input_value",
        placeholder="https://www.bbc.com/news/...",
        label_visibility="collapsed"
    )

with col_btn:
    analyze_button = st.button("Analyze", type="primary", use_container_width=True)

with col_clr:
    clear_button = st.button("Clear", use_container_width=True, on_click=handle_clear)

# Background thread target function (Thread-safe, avoids st.session_state modification inside thread)
def run_analysis_thread(agent, url):
    try:
        agent.current_ui_status = "Initializing"
        agent.current_analysis_report = None
        agent.current_analysis_error = None
        
        report = agent.analyze(url)
        agent.current_analysis_report = report
        agent.current_ui_status = "Completed"
    except Exception as e:
        tb = traceback.format_exc()
        agent.current_analysis_error = {
            "message": str(e),
            "traceback": tb
        }
        agent.current_ui_status = "Failed"
        print(f"Analysis thread failed:\n{tb}")

# Analysis trigger logic
if analyze_button:
    st.session_state.error = None
    st.session_state.report = None
    
    url_stripped = st.session_state.url_input_value.strip()
    if not url_stripped or not (url_stripped.startswith("http://") or url_stripped.startswith("https://")):
        st.warning("Please enter a valid URL starting with http:// or https://")
    else:
        agent = get_agent()
        if agent is not None:
            # Monkey-patch _update_state_status to update the persistent agent state status
            original_update = agent._update_state_status
            def patched_update(state, status):
                original_update(state, status)
                agent.current_ui_status = status
            agent._update_state_status = patched_update
            
            # Start background execution thread
            t = threading.Thread(target=run_analysis_thread, args=(agent, url_stripped))
            t.start()
            
            # 8-state status list panel rendering
            status_steps = [
                ("Initializing", ["initializing"]),
                ("Extracting Article", ["extraction"]),
                ("Analyzing Claims", ["analyzing_claims"]),
                ("Detecting Blindspots", ["detecting_blindspots"]),
                ("Searching Evidence", ["searching", "researching"]),
                ("Evaluating Evidence", ["evaluating_evidence"]),
                ("Generating Report", ["generating_report"]),
                ("Completed", ["complete"])
            ]
            
            # Poll thread status and render status updates in real-time
            status_container = st.empty()
            while t.is_alive():
                curr_status = getattr(agent, "current_ui_status", "Initializing").lower()
                st.session_state.current_ui_status = curr_status
                
                # Find current step index
                curr_idx = 0
                for idx, (label, codes) in enumerate(status_steps):
                    if curr_status in codes or curr_status == label.lower():
                        curr_idx = idx
                        break
                        
                with status_container.container():
                    st.markdown(
                        """
                        <div class="dashboard-card" style="max-width: 500px; margin: 40px auto; padding: 24px;">
                            <h4 style="margin-top:0; margin-bottom:20px; font-size:1.1rem; font-weight:600; text-align:center; color: var(--text-primary);">Analyzing Article Coverage</h4>
                            <div style="display: flex; flex-direction: column; gap: 12px;">
                        """,
                        unsafe_allow_html=True
                    )
                    
                    for idx, (label, codes) in enumerate(status_steps):
                        if idx < curr_idx:
                            # Completed
                            icon = '<span style="color: var(--success); margin-right: 12px; font-weight: bold; font-family: monospace;">✓</span>'
                            style = "color: var(--text-secondary);"
                        elif idx == curr_idx:
                            # Active
                            icon = '<span style="color: var(--accent); margin-right: 12px; font-weight: bold; font-family: monospace;">●</span>'
                            style = "color: var(--text-primary); font-weight: 500;"
                        else:
                            # Pending
                            icon = '<span style="color: var(--border); margin-right: 12px; font-family: monospace; opacity: 0.4;">○</span>'
                            style = "color: var(--text-secondary); opacity: 0.5;"
                            
                        st.markdown(
                            f'<div style="display: flex; align-items: center; font-size: 0.95rem; {style}">{icon} {label}</div>',
                            unsafe_allow_html=True
                        )
                        
                    st.markdown("</div></div>", unsafe_allow_html=True)
                time.sleep(0.5)
                
            # Post-run synchronization from thread results
            st.session_state.report = getattr(agent, "current_analysis_report", None)
            st.session_state.error = getattr(agent, "current_analysis_error", None)
            
            # Clear loading UI and refresh page to render dashboard
            status_container.empty()
            st.rerun()

# User-Friendly Error Banners (Hiding Python tracebacks from main UI)
def display_error(error_dict):
    msg = error_dict.get("message", "").lower()
    
    if "401" in msg or "403" in msg or "forbidden" in msg or "unauthorized" in msg or "paywall" in msg or "access_restricted" in msg:
        friendly_title = "Publisher Restriction"
        friendly_msg = "Content could not be accessed. This article is restricted by the publisher (paywall, login, or bot-blocking rules). Analysis is limited to metadata bounds."
    elif "timeout" in msg:
        friendly_title = "Analysis Timeout"
        friendly_msg = "The request timed out because the model took longer than the timeout threshold. Try again or check local service resources."
    elif "modelnotfound" in msg or "model not found" in msg or "not installed" in msg:
        friendly_title = "Model Unavailable"
        friendly_msg = "The requested LLM model is not installed or available on the local Ollama server."
    elif "connection" in msg or "unreachable" in msg or "refused" in msg or "ollama" in msg:
        friendly_title = "Network Failure"
        friendly_msg = "Could not connect to the local Ollama service. Please verify that Ollama is running and available."
    else:
        friendly_title = "Article Inaccessible"
        friendly_msg = "The system encountered an error loading or parsing this article URL. Please verify the URL and try again."
        
    st.markdown(
        f"""
        <div class="dashboard-card" style="border-left: 4px solid var(--danger); background-color: rgba(211, 61, 68, 0.05); padding: 16px 20px; margin-top: 20px;">
            <h4 style="color: #f85c63; margin-top: 0; margin-bottom: 6px; font-size: 1.05rem;">{friendly_title}</h4>
            <p style="font-size: 0.9rem; line-height: 1.5; color: var(--text-primary); margin: 0;">
                {friendly_msg}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if st.session_state.error:
    display_error(st.session_state.error)

if st.session_state.report is None:
    # Onboarding Empty State Screen
    st.markdown(
        """
        <div style="max-width: 800px; margin: 60px auto; text-align: center;">
            <h2 style="font-size: 1.75rem; font-weight: 600; margin-bottom: 8px; color: var(--text-primary); letter-spacing: -0.02em;">Analyze Any News Article</h2>
            <p style="color: var(--text-secondary); font-size: 1rem; margin-bottom: 32px; line-height: 1.5;">
                Paste a news URL to identify missing perspectives and framing gaps.
            </p>
            <div style="font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px; text-align: left; border-bottom: 1px solid var(--border); padding-bottom: 8px;">Sample Articles to Test</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Render buttons in a centered content layout
    _, c_content, _ = st.columns([1.5, 7, 1.5])
    with c_content:
        c_s1, c_s2, c_s3 = st.columns(3)
        s1_url = "https://www.thedailystar.net/opinion/views/news/promises-and-pitfalls-the-new-education-budget-4199791"
        s2_url = "https://www.bbc.com/sport/football/articles/c0ly17770lpo"
        s3_url = "https://httpstat.us/403"
        
        c_s1.button("Education Budget Opinion", use_container_width=True, on_click=set_sample_url, args=(s1_url,))
        c_s2.button("World Cup Football Article", use_container_width=True, on_click=set_sample_url, args=(s2_url,))
        c_s3.button("Access Restricted Test URL", use_container_width=True, on_click=set_sample_url, args=(s3_url,))

else:
    # ----------------------------------------------------
    # REPORT PAGE VIEW (RESTRUCTURED WORKFLOW)
    # ----------------------------------------------------
    report = st.session_state.report
    
    is_restricted = getattr(report, "completion_reason", "") == "access_restricted"
    score = getattr(report, "blindspot_score", 0)
    confidence = getattr(report, "confidence_score", 0)
    strength = getattr(report, "evidence_strength", "Weak")
    
    # 1. Article Summary Card
    title = getattr(report, 'article_title', 'Unknown Title')
    url = getattr(report, 'article_url', 'No URL')
    url_short = url[:60] if len(url) > 60 else url
    date_str = getattr(report, 'analysis_timestamp', 'N/A')[:10]
    
    # Confidence bar color
    conf_color = "#0eb383" if confidence >= 70 else "#e59c24" if confidence >= 40 else "#f85c63"

    st.markdown(
        f"""
        <div class="dashboard-card" style="margin-top: 20px;">
            <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Target Article</div>
            <div style="font-size: 1.15rem; font-weight: 600; line-height: 1.3; color: var(--text-primary); margin-bottom: 4px;">{title}</div>
            <div style="font-size: 0.8rem; margin-bottom: 16px;"><a href="{url}" target="_blank" style="color: var(--accent); text-decoration: none; word-break: break-all;">{url_short}</a></div>
            <div style="display: flex; gap: 32px; flex-wrap: wrap;">
                <div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">Analysis Date</div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: var(--text-primary);">{date_str}</div>
                </div>
                <div style="flex:1; min-width: 160px;">
                    <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">Confidence: {confidence}%</div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar-fill" style="width: {confidence}%; background-color: {conf_color};"></div>
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">Evidence Strength</div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: var(--text-primary);">{strength}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Access restricted banner
    if is_restricted:
        st.markdown(
            """
            <div class="dashboard-card" style="border-left: 4px solid var(--warning); background-color: rgba(201, 123, 0, 0.05); padding: 16px 20px;">
                <h4 style="color: #e59c24; margin-top: 0; margin-bottom: 6px; font-size: 1.05rem;">Publisher Restriction</h4>
                <p style="font-size: 0.9rem; line-height: 1.5; color: var(--text-primary); margin: 0;">
                    Content could not be accessed. Analysis has been limited to metadata limits.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 2. Blindspot Score Section
    if score <= 40:
        score_color = "#0eb383"  # success green
        score_label = "Minor Perspective Gaps"
    elif score <= 70:
        score_color = "#e59c24"  # warning orange
        score_label = "Moderate Perspective Gaps"
    else:
        score_color = "#f85c63"  # danger red
        score_label = "Significant Perspective Gaps"

    # Extract one-sentence explanation
    reasoning = getattr(report, 'score_reasoning', '')
    one_sentence_reasoning = reasoning.split('.')[0] + '.' if '.' in reasoning else reasoning

    st.markdown(
        f"""
        <div class="dashboard-card" style="border-left: 4px solid {score_color}; display: flex; align-items: center; gap: 24px; padding: 24px;">
            <div style="font-size: 3rem; font-weight: 700; color: var(--text-primary); line-height: 1; min-width: 100px;">
                {score}<span style="font-size: 1.2rem; color: var(--text-secondary); font-weight: 400;">/100</span>
            </div>
            <div style="flex: 1;">
                <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">Blindspot Score</div>
                <h4 style="margin: 0 0 4px 0; font-size: 1.1rem; font-weight: 600; color: {score_color};">{score_label}</h4>
                <p style="margin: 0; font-size: 0.9rem; color: var(--text-secondary); line-height: 1.4;">{one_sentence_reasoning}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. Key Findings Metric Row (Single Card)
    claims_count = len(getattr(report, "key_claims", []))
    blindspots_count = len(getattr(report, "blindspots", []))
    search_metrics = getattr(report, "search_metrics", {})
    searches = search_metrics.get("searches_executed", getattr(report, "search_attempts", 0)) if isinstance(search_metrics, dict) else getattr(report, "search_attempts", 0)
    evidence_count = getattr(report, "evidence_count", 0)

    st.markdown('<div class="report-section-heading">Key Findings</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="dashboard-card" style="padding: 20px; display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 16px; text-align: center;">
            <div style="flex: 1; min-width: 120px;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 4px;">Claims Analyzed</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">{claims_count}</div>
            </div>
            <div style="width: 1px; height: 32px; background-color: var(--border);"></div>
            <div style="flex: 1; min-width: 120px;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 4px;">Gaps Detected</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">{blindspots_count}</div>
            </div>
            <div style="width: 1px; height: 32px; background-color: var(--border);"></div>
            <div style="flex: 1; min-width: 120px;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 4px;">Evidence Evaluated</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">{evidence_count if not is_restricted else 0}</div>
            </div>
            <div style="width: 1px; height: 32px; background-color: var(--border);"></div>
            <div style="flex: 1; min-width: 120px;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 4px;">Search Queries</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">{searches}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Render claims (dimmed list below findings)
    claims_list = getattr(report, "key_claims", [])
    if claims_list:
        with st.expander("Show Extracted Claims List", expanded=False):
            for idx, c in enumerate(claims_list, 1):
                st.markdown(f"**{idx}.** {c}")

    # 4. Missing Perspectives Accordion list
    st.markdown('<div class="report-section-heading">Missing Perspectives</div>', unsafe_allow_html=True)
    blindspots = getattr(report, "blindspots", [])
    if not blindspots:
        st.info("No blindspots identified.")
    else:
        for idx, bs in enumerate(blindspots):
            importance = getattr(bs, "importance", "Medium").strip().capitalize()
            category = getattr(bs, "category", "Uncategorized").strip()
            expander_title = f"{category} — Priority: {importance}"
            
            with st.expander(expander_title, expanded=(idx == 0)):
                st.markdown(f"**Category:** {category}")
                st.markdown(f"**Importance:** {importance}")
                st.markdown(f"**Description:** {getattr(bs, 'description', '')}")
                st.markdown(f"**Suggested Search:** `{getattr(bs, 'suggested_search_query', '')}`")
                
                # Related Claims rendering
                related_claims = getattr(bs, 'related_claims', [])
                if related_claims:
                    st.markdown("**Related Claims:**")
                    for claim in related_claims:
                        st.markdown(f"- {claim}")

    # 5. Evidence Analysis (HTML Table & Badges & Expand Details)
    st.markdown('<div class="report-section-heading">Evidence Analysis</div>', unsafe_allow_html=True)
    full_report_data = get_full_report_data(report)
    metadata_dict = full_report_data.get("metadata", {})
    evidence_list = metadata_dict.get("evidence", [])

    if is_restricted:
        st.info("Evidence analysis is unavailable for restricted articles.")
    else:
        if not evidence_list:
            st.info("No supporting evidence registered.")
        else:
            for idx, ev in enumerate(evidence_list):
                source = ev.get("search_result", {}).get("source", "Reference")
                src_url = ev.get("search_result", {}).get("url", "")
                relevance = ev.get("relevance", "Adds Context")
                quality = ev.get("quality", "Medium")
                insight = ev.get("key_insight", "")
                evidence_sum = ev.get("evidence_summary", "")
                related_bs = ev.get("related_blindspot", "")
                
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                            <div style="font-weight: 600; font-size: 1rem; color: var(--text-primary);">{source}</div>
                            <div style="display: flex; gap: 8px;">
                                <span class="badge badge-accent">{relevance}</span>
                                <span class="badge badge-warning">{quality}</span>
                            </div>
                        </div>
                        <div style="color: var(--text-primary); font-size: 0.9rem; margin-bottom: 6px;"><strong>Key Insight:</strong> {insight}</div>
                        {f'<div style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 4px;"><strong>Summary:</strong> {evidence_sum}</div>' if evidence_sum else ''}
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: var(--text-secondary);">
                            <span>Linked Perspective: {related_bs if related_bs else 'Unlinked'}</span>
                            {f'<a href="{src_url}" target="_blank" style="color: var(--accent); text-decoration: none;">View Source ↗</a>' if src_url else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # 6. Conclusion (Clean panel, readable, no borders)
    st.markdown('<div class="report-section-heading">Conclusion</div>', unsafe_allow_html=True)
    conclusion_text = getattr(report, 'balanced_conclusion', '')
    # Format: ensure headers are bold, no ### visible, consistent spacing
    formatted_conclusion = conclusion_text
    import re
    # Replace plain header lines with styled versions
    headers = ["What The Article Covers", "What May Be Missing", "Evidence Findings",
               "Remaining Uncertainty", "Overall Assessment"]
    for h in headers:
        formatted_conclusion = re.sub(
            rf'^{re.escape(h)}\s*$',
            f'<div style="font-weight:600; font-size:1rem; margin-top:16px; margin-bottom:8px; color:var(--text-primary);">{h}</div>',
            formatted_conclusion,
            flags=re.MULTILINE
        )
    st.markdown(
        f"""
        <div class="dashboard-card" style="font-size: 0.95rem; line-height: 1.7;">
            {formatted_conclusion}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dynamic exports generation for downloads
    try:
        json_data = report.model_dump()
    except AttributeError:
        json_data = report.dict()
    report_json_str = json.dumps(json_data, indent=2, ensure_ascii=False)

    # Build md_summary dynamically to prevent NameError
    md_summary = f"""# Media Blindspot Analysis Report

**Article Title:** {title}
**Article URL:** {url}
**Date of Analysis:** {date_str}

---

## Blindspot Score: {score}/100
*Confidence: {confidence}% ({strength} Evidence)*

### Reasoning
{reasoning}

---

## Discovered Blindspots
"""
    for bs in getattr(report, "blindspots", []):
        md_summary += f"""
### {getattr(bs, 'category', 'Uncategorized')} ({getattr(bs, 'importance', 'Medium')})
*Description:* {getattr(bs, 'description', '')}
*Suggested Query:* `{getattr(bs, 'suggested_search_query', '')}`
"""
    md_summary += f"""
---

## Analysis Summary (Balanced Conclusion)
{getattr(report, 'balanced_conclusion', '')}
"""

    # Build txt_summary dynamically to prevent NameError
    txt_summary = f"""EXECUTIVE MEDIA BLINDSPOT ANALYSIS REPORT
==================================================
Article: {title}
URL: {url}
Timestamp: {getattr(report, 'analysis_timestamp', 'Unknown')}

BLINDSPOT SCORE: {score}/100
CONFIDENCE SCORE: {confidence}% (Strength: {strength})
--------------------------------------------------
EXECUTIVE REASONING:
{reasoning}

KEY OMITTED CATEGORIES:
{', '.join(getattr(report, 'missing_categories', ['None']))}

DETAILED CONCLUSION:
{getattr(report, 'balanced_conclusion', '')}

==================================================
Generated by BlindSpot AI.
"""

    # Download controls
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    dl_col1.download_button("Download JSON", report_json_str, file_name="report.json", mime="application/json", key="dl_json")
    dl_col2.download_button("Download Markdown Summary", md_summary, file_name="report.md", mime="text/markdown", key="dl_md")
    dl_col3.download_button("Download Executive Summary", txt_summary, file_name="executive_summary.txt", mime="text/plain", key="dl_txt")

    # Developer Tools
    debug_mode_env = False
    try:
        debug_mode_env = getattr(Config.from_env(), "DEBUG_MODE", False)
    except Exception as e:
        logger.warning(f"Developer tools failed ({__file__}:915): {e}")
        
    if dev_mode or debug_mode_env:
        st.markdown("---")
        with st.expander("Developer Tools", expanded=False):
            st.json(json_data)
