import sys
from config import Config
from utils.logger import get_logger
from agent.orchestrator import MediaBlindspotAgent

def main():
    # Load configuration from environment
    config = Config.from_env()
    
    # Create the logger
    logger = get_logger("blindspot_ai")
    logger.info("BlindSpot AI starting...")
    
    # Accept URL input interactively or from command-line arguments
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        try:
            url = input("Enter the article URL to analyze: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
            
    if not url:
        print("Error: No URL provided.")
        sys.exit(1)
        
    logger.info(f"Starting analysis for URL: {url}")
    
    try:
        # Instantiate and run the MediaBlindspotAgent
        agent = MediaBlindspotAgent(config)
        report = agent.analyze(url)
        
        # Display Final Polished Output
        print("\n" + "=" * 60)
        print("                 BLINDSPOT AI ANALYSIS REPORT")
        print("=" * 60)
        print(f"Article Title:   {report.article_title}")
        print(f"Article URL:     {report.article_url}")
        print(f"Timestamp:       {report.analysis_timestamp}")
        print(f"Main Topic:      {report.main_topic}")
        print("-" * 60)
        
        print(f"Blindspot Score: {report.blindspot_score}/100")
        print(f"Score Reasoning: {report.score_reasoning}")
        print("-" * 60)
        
        print("Key Claims Evaluated:")
        for idx, claim in enumerate(report.key_claims, 1):
            print(f"  {idx}. {claim}")
        print("-" * 60)
        
        print("Discovered Omissions & Blindspots:")
        for idx, bs in enumerate(report.blindspots, 1):
            print(f"  [{idx}] {bs.category} ({bs.importance} Importance):")
            print(f"      {bs.description}")
        print("-" * 60)
        
        print("Balanced Conclusion Synthesis:")
        print(report.balanced_conclusion)
        print("-" * 60)
        
        # Display Final Research Metrics and Metadata (Requirement 11)
        print("                 FINAL RESEARCH METADATA")
        print("-" * 60)
        print(f"Research Duration:         {report.research_duration} seconds")
        print(f"Completion Reason:         {report.completion_reason}")
        print(f"Confidence Score:          {report.confidence_score}% ({report.confidence_label})")
        print(f"Evidence Strength:         {report.evidence_strength}")
        print(f"Searches Performed:        {report.search_attempts}")
        print(f"Evidence Items Collected:  {report.evidence_count}")
        print("-" * 60)
        
        print("Source Diversity breakdown:")
        sd = report.source_diversity
        print(f"* Unique Domains:          {sd.get('unique_domains', 0)}")
        print(f"* Academic:                {sd.get('academic', 0)}")
        print(f"* Government:              {sd.get('government', 0)}")
        print(f"* News:                    {sd.get('news', 0)}")
        print(f"* Think Tank:              {sd.get('think_tank', 0)}")
        print(f"* Industry:                {sd.get('industry', 0)}")
        print(f"* Social Media:            {sd.get('social_media', 0)}")
        print(f"* Blog:                    {sd.get('blog', 0)}")
        print(f"* Other:                   {sd.get('other', 0)}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"E2E Agent execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
