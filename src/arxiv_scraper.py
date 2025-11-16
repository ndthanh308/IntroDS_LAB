"""
arXiv paper scraper module
"""

import os
import time
import json
import logging
import arxiv
import requests
from typing import Dict, List, Optional, Tuple

from utils import (
    format_arxiv_id, format_folder_name, extract_tar_gz,
    process_tex_files, ensure_dir, clean_temp_files, get_directory_size
)
from config import ARXIV_API_DELAY, MAX_RETRIES, RETRY_DELAY

logger = logging.getLogger(__name__)


class ArxivScraper:
    """Scraper for arXiv papers"""
    
    def __init__(self, output_dir: str):
        """
        Initialize arXiv scraper
        
        Args:
            output_dir: Base directory for output
        """
        self.output_dir = output_dir
        self.client = arxiv.Client()
        self.stats = {
            'papers_attempted': 0,
            'papers_successful': 0,
            'papers_failed': 0,
            'versions_downloaded': 0,
            'total_download_time': 0.0,
            'total_processing_time': 0.0
        }
    
    def get_paper_metadata(self, arxiv_id: str) -> Optional[Dict]:
        """
        Get metadata for a paper
        
        Args:
            arxiv_id: arXiv ID (e.g., "2370.11657")
        
        Returns:
            Metadata dictionary or None if failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                search = arxiv.Search(id_list=[arxiv_id])
                paper = next(self.client.results(search))
                
                # Extract metadata
                # Note: revised_dates will be populated later when downloading all versions
                metadata = {
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'submission_date': paper.published.isoformat() if paper.published else None,
                    'revised_dates': [],  # Will be populated from all versions
                    'abstract': paper.summary,
                    'categories': paper.categories,
                    'primary_category': paper.primary_category,
                    'doi': paper.doi,
                    'journal_ref': paper.journal_ref,
                    'arxiv_id': arxiv_id,
                    'pdf_url': paper.pdf_url,
                    'comment': paper.comment
                }
                # Try to detect number of versions if available on paper object
                try:
                    versions = getattr(paper, 'versions', None)
                    if versions:
                        # versions may be a list-like
                        metadata['num_versions'] = len(versions)
                    else:
                        metadata['num_versions'] = None
                except Exception:
                    metadata['num_versions'] = None
                
                logger.info(f"Retrieved metadata for {arxiv_id}: {paper.title}")
                return metadata
                
            except StopIteration:
                logger.warning(f"Paper {arxiv_id} not found")
                return None
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {arxiv_id}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        
        logger.error(f"Failed to get metadata for {arxiv_id} after {MAX_RETRIES} attempts")
        return None
    
    def download_source(self, arxiv_id: str, version: str, output_dir: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Download source files for a specific version
        
        Args:
            arxiv_id: arXiv ID without version
            version: Version string (e.g., "v1")
            output_dir: Directory to save source
        
        Returns:
            Tuple of (success, tar_path, updated_date)
        """
        versioned_id = f"{arxiv_id}{version}"

        for attempt in range(MAX_RETRIES):
            try:
                search = arxiv.Search(id_list=[versioned_id])
                paper = next(self.client.results(search))
                
                # Get updated date for this version
                updated_date = paper.updated.isoformat() if paper.updated else None
                
                # Extract year-month and paper number from arxiv_id for source URL
                # Format: YYMM.nnnnn -> YYMM/nnnnn
                parts = arxiv_id.split('.')
                if len(parts) == 2:
                    year_month = parts[0]  # e.g., "2208"
                    paper_num = parts[1]   # e.g., "12396"
                else:
                    # Fallback: try to extract from versioned_id
                    parts = versioned_id.replace('v', '.').split('.')
                    if len(parts) >= 2:
                        year_month = parts[0]
                        paper_num = parts[1]
                    else:
                        year_month = arxiv_id[:4]
                        paper_num = arxiv_id[5:].replace('v', '')
                
                # Download source
                temp_dir = os.path.join(output_dir, "temp")
                ensure_dir(temp_dir)
                
                tar_filename = f"{versioned_id}.tar.gz"
                tar_path = os.path.join(temp_dir, tar_filename)
                
                logger.info(f"Downloading source for {versioned_id}...")
                start_time = time.time()
                
                # Try using paper.download_source() first
                try:
                    paper.download_source(dirpath=temp_dir, filename=tar_filename)
                except Exception as download_err:
                    error_str = str(download_err)
                    # If download_source fails due to None attribute or other known reasons, try direct download
                    logger.debug(f"download_source() failed for {versioned_id}: {download_err} — trying direct URLs")

                    # Construct source URLs manually - try multiple formats.
                    # Use the e-print endpoint first (serves the source archive directly),
                    # then several /src/... patterns as fallbacks.
                    source_urls = [
                        f"https://arxiv.org/e-print/{versioned_id}",
                        f"https://arxiv.org/e-print/{arxiv_id}",
                        f"https://arxiv.org/src/{year_month}/{paper_num}/{versioned_id}.tar.gz",
                        f"https://arxiv.org/src/{year_month}/{paper_num}/{arxiv_id}.tar.gz",
                        f"https://arxiv.org/src/{year_month}/{paper_num}.tar.gz",
                    ]

                    downloaded = False
                    for source_url in source_urls:
                        logger.debug(f"Attempting direct download from: {source_url}")
                        try:
                            response = requests.get(source_url, timeout=60, stream=True, allow_redirects=True)

                            if response.status_code == 200:
                                # Check if it's actually a tar.gz file or HTML redirect
                                content_type = response.headers.get('Content-Type', '')
                                if 'html' in content_type.lower():
                                    logger.debug(f"Skipping {source_url} (HTML response, likely redirect)")
                                    continue

                                # Check first few bytes to verify it's a tar.gz file
                                first_chunk = next(response.iter_content(chunk_size=10))
                                if not first_chunk.startswith(b'\x1f\x8b'):  # gzip magic number
                                    logger.debug(f"Skipping {source_url} (not a gzip file, got: {first_chunk[:10]})")
                                    response.close()
                                    continue

                                # Save file
                                with open(tar_path, 'wb') as f:
                                    f.write(first_chunk)  # Write first chunk
                                    for chunk in response.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                logger.info(f"Downloaded {versioned_id} via direct URL: {source_url}")
                                downloaded = True
                                break
                            elif response.status_code == 404:
                                logger.debug(f"404 for {source_url}, trying next format...")
                                continue
                            else:
                                logger.debug(f"HTTP {response.status_code} for {source_url}, trying next format...")
                                continue
                        except Exception as url_err:
                            logger.debug(f"Error downloading from {source_url}: {url_err}, trying next format...")
                            continue

                    if not downloaded:
                        # Paper exists (we got a Paper object) but has no source files available -> treat as 'no_source'
                        logger.warning(f"Paper {versioned_id} does not have source files available (only PDF). Marking version as no_source.")
                        time.sleep(ARXIV_API_DELAY)
                        return 'no_source', None, None
                
                download_time = time.time() - start_time
                
                # Verify file was downloaded
                if not os.path.exists(tar_path) or os.path.getsize(tar_path) == 0:
                    raise Exception(f"Downloaded file is empty or doesn't exist: {tar_path}")

                self.stats['total_download_time'] += download_time
                logger.info(f"Downloaded {versioned_id} in {download_time:.2f}s")

                time.sleep(ARXIV_API_DELAY)
                return 'downloaded', tar_path, updated_date
                
            except StopIteration:
                logger.warning(f"Version {versioned_id} not found")
                return 'not_found', None, None
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {versioned_id}: {error_msg}")

                # If it's a NoneType-like error, treat as no_source (paper found but no TeX source)
                if "'NoneType' object has no attribute 'replace'" in error_msg or "NoneType" in error_msg:
                    logger.warning(f"Paper {versioned_id} may not have source files available (only PDF)")
                    return 'no_source', None, None

                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        
        logger.error(f"Failed to download {versioned_id} after {MAX_RETRIES} attempts")
        return 'error', None, None
    
    def scrape_paper(self, arxiv_id: str, paper_dir: str) -> bool:
        """
        Scrape a single paper with all versions
        
        Args:
            arxiv_id: arXiv ID (e.g., "2208.11941")
            paper_dir: Directory for this paper's data
        
        Returns:
            True if successful, False otherwise
        """
        self.stats['papers_attempted'] += 1
        logger.info(f"Scraping paper {arxiv_id}...")
        
        start_time = time.time()
        temp_dir = os.path.join(paper_dir, "temp")
        
        try:
            # Get metadata
            metadata = self.get_paper_metadata(arxiv_id)
            if not metadata:
                self.stats['papers_failed'] += 1
                return False
            
            time.sleep(ARXIV_API_DELAY)
            
            # Create directories
            ensure_dir(paper_dir)
            tex_dir = os.path.join(paper_dir, "tex")
            ensure_dir(tex_dir)
            
            # Try to download versions — detect number of versions if possible
            versions_downloaded = 0
            versions_seen = 0
            revised_dates = []  # Collect all revised dates from versions
            
            # Track sizes before/after image removal for this paper
            paper_size_before = 0
            paper_size_after = 0

            max_versions = metadata.get('num_versions') or 20

            for v in range(1, max_versions + 1):
                version = f"v{v}"
                versions_seen += 1
                status, tar_path, updated_date = self.download_source(arxiv_id, version, paper_dir)

                # Handle status
                if status == 'not_found':
                    if v == 1:
                        # No v1 means paper metadata missing or invalid
                        logger.error(f"No v1 found for {arxiv_id}")
                        self.stats['papers_failed'] += 1
                        return False
                    else:
                        # No more versions available — do not create a folder for non-existent versions
                        break

                if status == 'no_source':
                    # Version exists but no TeX source — create an empty version_dir per requirements
                    folder_name = format_folder_name(arxiv_id)
                    version_dir = os.path.join(tex_dir, f"{folder_name}{version}")
                    ensure_dir(version_dir)
                    logger.info(f"Version {version} exists for {arxiv_id} but has no TeX source; created empty folder {version_dir}")
                    # still collect updated_date if present
                    if updated_date and v > 1:
                        if updated_date not in revised_dates:
                            revised_dates.append(updated_date)
                    # continue to next version
                    continue

                if status == 'downloaded':
                    # Collect revised date (skip v1 as it's the submission date)
                    if updated_date and v > 1:
                        if updated_date not in revised_dates:
                            revised_dates.append(updated_date)

                    # create version dir now that we have a downloaded archive
                    folder_name = format_folder_name(arxiv_id)
                    version_dir = os.path.join(tex_dir, f"{folder_name}{version}")
                    ensure_dir(version_dir)

                    # Extract source into the version_dir
                    if tar_path and extract_tar_gz(tar_path, version_dir):
                        # Compute size before removing images (version-level)
                        v_before = get_directory_size(version_dir)
                        paper_size_before += v_before

                        # Preserve original archive subfolder structure and file names.
                        # Only remove image files and strip includegraphics from .tex files.
                        process_stats = process_tex_files(version_dir)
                        logger.info(f"Processed {process_stats['processed']} TeX files, removed {process_stats['images_removed']} image files for {version}")

                        # Compute size after removing images
                        v_after = get_directory_size(version_dir)
                        paper_size_after += v_after

                        versions_downloaded += 1
                        self.stats['versions_downloaded'] += 1

                    # Clean up tar file immediately
                    if tar_path and os.path.exists(tar_path):
                        try:
                            os.remove(tar_path)
                            logger.debug(f"Removed tar file: {tar_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove tar file {tar_path}: {e}")

                if status == 'error':
                    logger.warning(f"Error while attempting to download {version} for {arxiv_id}; stopping further attempts")
                    break
            
            if versions_seen == 0:
                logger.error(f"No versions detected for {arxiv_id}")
                self.stats['papers_failed'] += 1
                return False
            
            # Update metadata with all revised dates
            metadata['revised_dates'] = sorted(revised_dates)
            # Save size metrics into metadata so the pipeline can report before/after image removal sizes
            try:
                metadata['size_before_bytes'] = paper_size_before
                metadata['size_after_bytes'] = paper_size_after
            except Exception:
                metadata['size_before_bytes'] = metadata.get('size_before_bytes', 0)
                metadata['size_after_bytes'] = metadata.get('size_after_bytes', 0)
            
            # Save metadata
            metadata_path = os.path.join(paper_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['papers_successful'] += 1
            
            logger.info(f"Successfully scraped {arxiv_id} ({versions_downloaded} versions) in {processing_time:.2f}s")
            return True
            
        finally:
            # Always clean temp directory, even if error occurs
            if os.path.exists(temp_dir):
                clean_temp_files(temp_dir)
                logger.debug(f"Cleaned temp directory: {temp_dir}")
    
    def get_stats(self) -> Dict:
        """Get scraping statistics"""
        return self.stats.copy()

