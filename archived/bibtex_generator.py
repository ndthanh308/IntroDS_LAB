"""
BibTeX generator for arXiv papers (archived)

This file was moved from `src/` because the main pipeline does not import
or use it. Keeping an archived copy here in case it's needed later.
"""

import os
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BibtexGenerator:
    """Generate BibTeX entries for arXiv papers"""
    
    @staticmethod
    def generate_bibtex_entry(metadata: Dict, arxiv_id: str) -> str:
        """
        Generate BibTeX entry from metadata
        
        Args:
            metadata: Paper metadata dictionary
            arxiv_id: arXiv ID
        
        Returns:
            BibTeX entry as string
        """
        # Create citation key
        authors = metadata.get('authors', [])
        year = metadata.get('submission_date', '')[:4] if metadata.get('submission_date') else ''
        
        if authors:
            first_author = authors[0].split()[-1]  # Last name
            cite_key = f"{first_author}{year}"
        else:
            cite_key = f"arxiv{arxiv_id.replace('.', '_')}"
        
        # Build BibTeX entry
        bibtex = f"@article{{{cite_key},\n"
        
        # Title
        title = metadata.get('title', '').replace('{', '\\{').replace('}', '\\}')
        bibtex += f"  title={{{title}}},\n"
        
        # Authors
        if authors:
            author_str = " and ".join(authors)
            bibtex += f"  author={{{author_str}}},\n"
        
        # Year
        if year:
            bibtex += f"  year={{{year}}},\n"
        
        # arXiv specific fields
        bibtex += f"  eprint={{{arxiv_id}}},\n"
        bibtex += f"  archivePrefix={{arXiv}},\n"
        
        # Primary category
        if metadata.get('primary_category'):
            bibtex += f"  primaryClass={{{metadata['primary_category']}}},\n"
        
        # Abstract
        if metadata.get('abstract'):
            abstract = metadata['abstract'].replace('{', '\\{').replace('}', '\\}')
            bibtex += f"  abstract={{{abstract}}},\n"
        
        # Journal reference if exists
        if metadata.get('journal_ref'):
            bibtex += f"  journal={{{metadata['journal_ref']}}},\n"
        
        # DOI if exists
        if metadata.get('doi'):
            bibtex += f"  doi={{{metadata['doi']}}},\n"
        
        bibtex += "}\n"
        
        return bibtex
    
    @staticmethod
    def create_bibtex_file(metadata: Dict, arxiv_id: str, output_path: str) -> bool:
        """
        Create BibTeX file for a paper
        
        Args:
            metadata: Paper metadata
            arxiv_id: arXiv ID
            output_path: Path to save .bib file
        
        Returns:
            True if successful
        """
        try:
            bibtex_entry = BibtexGenerator.generate_bibtex_entry(metadata, arxiv_id)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(bibtex_entry)
            
            logger.info(f"Created BibTeX file: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create BibTeX file {output_path}: {e}")
            return False
