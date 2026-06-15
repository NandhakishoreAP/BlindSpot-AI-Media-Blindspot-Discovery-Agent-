import sys
from config import Config
from utils.logger import get_logger

def main():
    # Load configuration from environment
    config = Config.from_env()
    
    # Create the logger
    logger = get_logger("blindspot_ai")
    
    # Log starting message
    logger.info("BlindSpot AI starting...")
    
    # Print configuration representation
    print(config)
    
    # Exit cleanly
    sys.exit(0)

if __name__ == "__main__":
    main()
