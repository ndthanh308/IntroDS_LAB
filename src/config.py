"""
Configuration file for arXiv scraper
"""

# Student ID
STUDENT_ID = "23127538"

# arXiv paper range to scrape (user request)
# Paper range: 2307.11657 to 2307.16656
START_YEAR_MONTH = "2307"
START_ID = 11657
END_YEAR_MONTH = "2307"
END_ID = 16656

# API rate limits
ARXIV_API_DELAY = 3.0  # seconds between requests
SEMANTIC_SCHOLAR_DELAY = 1.1  # seconds between requests (1 req/sec limit)

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5.0  # seconds

# Output directories
DATA_DIR = f"./{STUDENT_ID}/{STUDENT_ID}_data"
LOGS_DIR = "./logs"

# File size limits (in bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Semantic Scholar API
SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_FIELDS = "references,references.paperId,references.externalIds,references.title,references.authors,references.publicationDate,references.year"

# Semantic Scholar API key (user-provided)
SEMANTIC_SCHOLAR_API_KEY = "YOUR_API_KEY"

