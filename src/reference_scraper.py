"""
Reference scraper using Semantic Scholar API
"""

import time
import json
import logging
import requests
from typing import Dict, List, Optional

from config import (
    SEMANTIC_SCHOLAR_API_BASE,
    SEMANTIC_SCHOLAR_FIELDS,
    SEMANTIC_SCHOLAR_DELAY,
    MAX_RETRIES,
    RETRY_DELAY,
    SEMANTIC_SCHOLAR_API_KEY
)

logger = logging.getLogger(__name__)


class ReferenceScraper:
    """Scraper for paper references using Semantic Scholar API"""
    
    def __init__(self):
        """Initialize reference scraper"""
        self.api_base = SEMANTIC_SCHOLAR_API_BASE
        self.session = requests.Session()
        # If an API key is provided, set it on the session headers
        if SEMANTIC_SCHOLAR_API_KEY:
            # Semantic Scholar expects the key in the 'x-api-key' header
            self.session.headers.update({'x-api-key': SEMANTIC_SCHOLAR_API_KEY})
        self.stats = {
            'papers_queried': 0,
            'papers_found': 0,
            'papers_not_found': 0,
            'total_references': 0,
            'references_with_arxiv_id': 0,
            'api_errors': 0
        }
    
    def get_paper_references(self, arxiv_id: str) -> Optional[List[Dict]]:
        """
        Get references for a paper from Semantic Scholar
        
        Args:
            arxiv_id: arXiv ID (e.g., "2307.11656")
        
        Returns:
            List of reference dictionaries or None if failed
        """
        self.stats['papers_queried'] += 1
        
        url = f"{self.api_base}/paper/arXiv:{arxiv_id}"
        params = {"fields": SEMANTIC_SCHOLAR_FIELDS}
        
        # Use more retries for rate limit cases
        max_retries_for_rate_limit = MAX_RETRIES * 2  # 6 retries for rate limit
        consecutive_rate_limits = 0
        
        for attempt in range(max_retries_for_rate_limit):
            try:
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    references = data.get("references", [])
                    
                    self.stats['papers_found'] += 1
                    self.stats['total_references'] += len(references)
                    
                    logger.info(f"Found {len(references)} references for {arxiv_id}")
                    time.sleep(SEMANTIC_SCHOLAR_DELAY)
                    return references
                
                elif response.status_code == 404:
                    logger.warning(f"Paper {arxiv_id} not found in Semantic Scholar database")
                    self.stats['papers_not_found'] += 1
                    time.sleep(SEMANTIC_SCHOLAR_DELAY)
                    return None
                
                elif response.status_code == 429:
                    consecutive_rate_limits += 1
                    # Rate limit exceeded - check Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            wait_time = int(retry_after)
                            logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries_for_rate_limit}), waiting {wait_time} seconds (from Retry-After header)...")
                            time.sleep(wait_time)
                        except ValueError:
                            wait_time = 120
                            logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries_for_rate_limit}), waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                    else:
                        # Exponential backoff: wait longer on each retry
                        wait_time = min(60 * (attempt + 1), 300)  # 60s, 120s, 180s, 240s, 300s (max 5 min)
                        logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries_for_rate_limit}), waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    
                    # Continue retrying for rate limit
                    if attempt < max_retries_for_rate_limit - 1:
                        continue
                    else:
                        logger.error(f"Rate limit persists after {max_retries_for_rate_limit} attempts for {arxiv_id}")
                        break
                
                else:
                    logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES}: HTTP {response.status_code}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        break
                
            except requests.exceptions.Timeout:
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES}: Request timeout")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    break
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    break
        
        self.stats['api_errors'] += 1
        if consecutive_rate_limits >= 3:
            logger.error(f"Failed to get references for {arxiv_id} due to persistent rate limiting after {max_retries_for_rate_limit} attempts. "
                        f"Paper may exist in Semantic Scholar but API is currently unavailable. "
                        f"Try again later or check if paper exists at: https://www.semanticscholar.org/paper/arXiv:{arxiv_id}")
        else:
            logger.error(f"Failed to get references for {arxiv_id} after {MAX_RETRIES} attempts")
        return None
    
    def extract_arxiv_references(self, references: List[Dict]) -> Dict[str, Dict]:
        """
        Extract references that have arXiv IDs
        
        Args:
            references: List of reference dictionaries from Semantic Scholar
        
        Returns:
            Dictionary mapping arXiv IDs to metadata
        """
        arxiv_references = {}
        
        for ref in references:
            # Check if reference has arXiv ID
            external_ids = ref.get('externalIds', {})
            if not external_ids or 'ArXiv' not in external_ids:
                continue
            
            arxiv_id = external_ids['ArXiv']
            
            # Convert arXiv ID format if needed (e.g., "2307.11656" -> "2307-11656")
            # Keep original format for consistency
            
            # Extract metadata
            # Required fields: title, authors, submission_date, SemanticScholar ID
            metadata = {
                'title': ref.get('title', ''),
                'authors': [author.get('name', '') for author in ref.get('authors', [])],
                'submission_date': ref.get('publicationDate', ''),
                'semantic_scholar_id': ref.get('paperId', '')  # SemanticScholar ID as required
            }
            
            # Format arxiv_id to match required format (yyyymm-id)
            try:
                if '.' in arxiv_id:
                    parts = arxiv_id.split('.')
                    formatted_id = f"{parts[0]}-{parts[1]}"
                else:
                    formatted_id = arxiv_id
                
                arxiv_references[formatted_id] = metadata
                self.stats['references_with_arxiv_id'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to format arXiv ID {arxiv_id}: {e}")
        
        return arxiv_references
    
    def scrape_references(self, arxiv_id: str, output_path: str) -> bool:
        """
        Scrape references for a paper and save to file
        
        Args:
            arxiv_id: arXiv ID (e.g., "2208.11941")
            output_path: Path to save references.json
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Scraping references for {arxiv_id}...")
        
        try:
            # Get references from Semantic Scholar
            references = self.get_paper_references(arxiv_id)
            
            if references is None:
                # Create empty references file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f, indent=2, ensure_ascii=False)
                logger.warning(f"No references found for {arxiv_id} (paper may not exist in Semantic Scholar or API unavailable), saved empty file")
                return False
            
            # Log total references found
            total_refs = len(references)
            logger.info(f"Found {total_refs} total references for {arxiv_id} from Semantic Scholar")
            
            # Extract only references with arXiv IDs
            arxiv_references = self.extract_arxiv_references(references)
            
            # Log how many have arXiv IDs
            arxiv_refs_count = len(arxiv_references)
            if total_refs > 0 and arxiv_refs_count == 0:
                logger.warning(f"Paper {arxiv_id} has {total_refs} references in Semantic Scholar, but none have arXiv IDs. "
                             f"References may be from journals/conferences only. Saved empty file.")
            elif arxiv_refs_count > 0:
                logger.info(f"Extracted {arxiv_refs_count} references with arXiv IDs from {total_refs} total references")
            
            # Save to file (always create file, even if empty)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(arxiv_references, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(arxiv_references)} arXiv references to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scrape references for {arxiv_id}: {e}")
            # Always create file, even on error (empty dict)
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f, indent=2, ensure_ascii=False)
                logger.warning(f"Created empty references.json due to error for {arxiv_id}")
            except Exception as write_error:
                logger.error(f"Failed to create references.json file for {arxiv_id}: {write_error}")
            return False
    
    def get_stats(self) -> Dict:
        """Get scraping statistics"""
        return self.stats.copy()

