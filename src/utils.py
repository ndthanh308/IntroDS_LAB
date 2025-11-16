"""
Utility functions for arXiv scraper
"""

import os
import re
import tarfile
import gzip
import shutil
import logging
 
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = "./logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "scraper.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def format_arxiv_id(year_month: str, paper_id: int) -> str:
    """
    Format arXiv ID from year-month and paper ID
    
    Args:
        year_month: Year-month string (e.g., "2208")
        paper_id: Paper ID number
    
    Returns:
        Formatted arXiv ID (e.g., "2208.11941")
    """
    return f"{year_month}.{paper_id:05d}"


def format_folder_name(arxiv_id: str) -> str:
    """
    Convert arXiv ID to folder name format
    
    Args:
        arxiv_id: arXiv ID (e.g., "2208.11941")
    
    Returns:
        Folder name (e.g., "2208-11941")
    """
    return arxiv_id.replace(".", "-")


def extract_tar_gz(tar_path: str, extract_dir: str) -> bool:
    """
    Extract .tar.gz file or handle gzip-compressed LaTeX source
    
    Args:
        tar_path: Path to .tar.gz file
        extract_dir: Directory to extract to
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(tar_path):
        logger.error(f"Failed to extract {tar_path}: file does not exist")
        return False
    
    # Try to extract as tar.gz first
    try:
        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(path=extract_dir)
        logger.info(f"Extracted {tar_path} to {extract_dir}")
        return True
    except (tarfile.TarError, tarfile.ReadError):
        # If tar extraction fails, try as gzip-compressed LaTeX source
        try:
            with gzip.open(tar_path, 'rb') as gz_file:
                content = gz_file.read()
            
            # Check if it's LaTeX source (starts with \documentclass or similar)
            if content.startswith(b'\\') or b'\\documentclass' in content[:1000]:
                # Determine filename from tar_path
                base_name = os.path.splitext(os.path.splitext(os.path.basename(tar_path))[0])[0]
                tex_filename = f"{base_name}.tex"
                tex_path = os.path.join(extract_dir, tex_filename)
                
                # Write decompressed content
                with open(tex_path, 'wb') as f:
                    f.write(content)
                
                logger.info(f"Extracted gzip-compressed LaTeX source {tar_path} to {tex_path}")
                return True
            else:
                logger.error(f"Failed to extract {tar_path}: not a valid tar archive or LaTeX source")
                return False
        except (gzip.BadGzipFile, OSError) as gz_error:
            # File is not gzipped, try reading as plain text/LaTeX
            try:
                with open(tar_path, 'rb') as f:
                    content = f.read()
                
                # Check if it's LaTeX source or plain text
                if content.startswith(b'\\') or b'\\documentclass' in content[:1000] or b'%' in content[:100]:
                    # Determine filename from tar_path
                    base_name = os.path.splitext(os.path.splitext(os.path.basename(tar_path))[0])[0]
                    tex_filename = f"{base_name}.tex"
                    tex_path = os.path.join(extract_dir, tex_filename)
                    
                    # Write content as-is
                    with open(tex_path, 'wb') as f:
                        f.write(content)
                    
                    logger.info(f"Extracted plain LaTeX source {tar_path} to {tex_path}")
                    return True
                else:
                    logger.error(f"Failed to extract {tar_path}: not a valid archive or LaTeX source (got: {content[:20]})")
                    return False
            except Exception as e:
                logger.error(f"Failed to extract {tar_path}: {e}")
                return False
        except Exception as e:
            logger.error(f"Failed to extract {tar_path}: {e}")
            return False
    except Exception as e:
        logger.error(f"Failed to extract {tar_path}: {e}")
        return False


def remove_figures_from_tex(tex_content: str) -> str:
    """
    Remove figure environments and includegraphics commands from TeX content
    
    Args:
        tex_content: Original TeX file content
    
    Returns:
        TeX content with figures removed
    """
    # Remove \includegraphics commands
    tex_content = re.sub(
        r'\\includegraphics(\[.*?\])?\{.*?\}',
        r'% Figure removed',
        tex_content
    )
    
    # Remove figure environments
    tex_content = re.sub(
        r'\\begin\{figure\}.*?\\end\{figure\}',
        r'% Figure environment removed',
        tex_content,
        flags=re.DOTALL
    )
    
    # Remove figure* environments
    tex_content = re.sub(
        r'\\begin\{figure\*\}.*?\\end\{figure\*\}',
        r'% Figure environment removed',
        tex_content,
        flags=re.DOTALL
    )
    
    return tex_content


def remove_image_files(directory: str) -> int:
    """
    Remove common image file types from directory
    
    Args:
        directory: Directory to clean
    
    Returns:
        Number of files removed
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.eps', '.svg', '.bmp', '.tiff']
    removed_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    removed_count += 1
                    logger.debug(f"Removed image file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
    
    return removed_count


def process_tex_files(tex_dir: str) -> Dict[str, int]:
    """
    Process all .tex files in directory to remove figures
    
    Args:
        tex_dir: Directory containing .tex files
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'processed': 0,
        'failed': 0,
        'images_removed': 0
    }
    
    # Remove image files first
    stats['images_removed'] = remove_image_files(tex_dir)
    
    # Process .tex files
    for root, dirs, files in os.walk(tex_dir):
        for file in files:
            if file.endswith('.tex'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Remove figures
                    cleaned_content = remove_figures_from_tex(content)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    
                    stats['processed'] += 1
                    logger.debug(f"Processed TeX file: {file_path}")
                except Exception as e:
                    stats['failed'] += 1
                    logger.warning(f"Failed to process {file_path}: {e}")
    
    return stats


def get_directory_size(directory: str) -> int:
    """
    Calculate total size of directory in bytes
    
    Args:
        directory: Directory path
    
    Returns:
        Total size in bytes
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def clean_temp_files(temp_dir: str):
    """
    Remove temporary files and directories
    
    Args:
        temp_dir: Temporary directory to clean
    """
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean {temp_dir}: {e}")


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)

