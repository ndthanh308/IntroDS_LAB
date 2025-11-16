"""
Main arXiv scraper script
Student ID: 23127538
"""

import os
import time
import json
import argparse
import logging
import shutil
 
import psutil
import threading
import csv

from config import (
    STUDENT_ID, START_YEAR_MONTH, START_ID,
    END_YEAR_MONTH, END_ID, DATA_DIR, LOGS_DIR
)
from utils import (
    setup_logging, format_arxiv_id, format_folder_name,
    ensure_dir, get_directory_size
)
from arxiv_scraper import ArxivScraper
from reference_scraper import ReferenceScraper

logger = logging.getLogger(__name__)


class ArxivScraperPipeline:
    """Complete arXiv scraping pipeline"""
    
    def __init__(self, output_dir: str):
        """
        Initialize scraper pipeline
        
        Args:
            output_dir: Base output directory
        """
        self.output_dir = output_dir
        ensure_dir(output_dir)
        
        self.arxiv_scraper = ArxivScraper(output_dir)
        self.reference_scraper = ReferenceScraper()
        
        self.stats = {
            'total_papers': 0,
            'successful_papers': 0,
            'failed_papers': 0,
            'total_runtime': 0.0,
            'paper_runtimes': [],
            'paper_sizes_before': [],
            'paper_sizes_after': [],
            'reference_counts': [],
            'memory_samples': [],        # per-paper max memory
            'all_memory_samples': [],    # all sampled memory values across papers
            'disk_sizes': [],            # per-paper max disk observed
            'max_disk_usage_bytes': 0,
            'final_output_size_bytes': 0,
            'paper_rows': []
        }
    
    def generate_paper_ids(self, start_ym: str, start_id: int, 
                          end_ym: str, end_id: int) -> list:
        """
        Generate list of arXiv IDs to scrape
        
        Args:
            start_ym: Start year-month (e.g., "2208")
            start_id: Start paper ID
            end_ym: End year-month
            end_id: End paper ID
        
        Returns:
            List of arXiv IDs
        """
        paper_ids = []
        
        # Convert year-month to comparable integers
        start_ym_int = int(start_ym)
        end_ym_int = int(end_ym)
        
        current_ym = start_ym_int
        
        while current_ym <= end_ym_int:
            ym_str = str(current_ym)
            
            # Determine ID range for this month
            if current_ym == start_ym_int and current_ym == end_ym_int:
                # Same month
                id_start = start_id
                id_end = end_id
            elif current_ym == start_ym_int:
                # First month
                id_start = start_id
                id_end = 99999  # Arbitrary high number
            elif current_ym == end_ym_int:
                # Last month
                id_start = 0
                id_end = end_id
            else:
                # Middle months
                id_start = 0
                id_end = 99999
            
            # For this assignment, we only need papers in the specific range
            if current_ym == start_ym_int:
                for paper_id in range(id_start, min(id_end + 1, 99999)):
                    arxiv_id = format_arxiv_id(ym_str, paper_id)
                    paper_ids.append(arxiv_id)
            
            
            if current_ym == end_ym_int and current_ym != start_ym_int:
                for paper_id in range(0, id_end + 1):
                    arxiv_id = format_arxiv_id(ym_str, paper_id)
                    paper_ids.append(arxiv_id)
            
            # Move to next month
            year = current_ym // 100
            month = current_ym % 100
            if month == 12:
                current_ym = (year + 1) * 100 + 1
            else:
                current_ym = year * 100 + month + 1
        
        return paper_ids

    def _load_existing_csv_ids(self, csv_path: str) -> set:
        """Load existing arXiv IDs from stats.csv to avoid duplicates when appending."""
        ids = set()
        if not os.path.exists(csv_path):
            return ids
        try:
            with open(csv_path, 'r', encoding='utf-8') as cf:
                reader = csv.DictReader(cf)
                for r in reader:
                    aid = r.get('arxiv_id')
                    if aid:
                        ids.add(aid)
        except Exception:
            pass
        return ids

    def _append_row_to_csv(self, csv_path: str, row: dict):
        """Append a single row to CSV, writing header if file doesn't exist."""
        write_header = not os.path.exists(csv_path)
        keys = ['arxiv_id','success','runtime_s','mem_before_rss','mem_after_rss','mem_max_rss','mem_avg_rss','size_before_bytes','size_after_bytes','disk_max_bytes','final_output_size_bytes','references_count']
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as cf:
                writer = csv.DictWriter(cf, fieldnames=keys)
                if write_header:
                    writer.writeheader()
                # ensure only keys in keys
                out = {k: row.get(k, '') for k in keys}
                writer.writerow(out)
        except Exception as e:
            logger.warning(f"Failed to append row to {csv_path}: {e}")

    def _merge_and_save_stats_json(self, stats_file: str):
        """Merge existing scraping_stats.json with current run stats and save atomically."""
        merged = {
            'pipeline': {},
            'arxiv': {},
            'references': {}
        }
        # Load existing
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                existing = None
        else:
            existing = None

        # Start with existing values if any
        if existing:
            merged['pipeline'] = existing.get('pipeline', {})
            merged['arxiv'] = existing.get('arxiv', {})
            merged['references'] = existing.get('references', {})
        else:
            merged['pipeline'] = {}
            merged['arxiv'] = {}
            merged['references'] = {}

        # Merge pipeline stats: update numeric aggregates conservatively
        # For simplicity, we won't deep-merge all fields; save current pipeline under 'pipeline_runs' list
        # Ensure pipeline_runs is a list and append current stats
        pipeline_runs = merged.get('pipeline_runs')
        if not isinstance(pipeline_runs, list):
            pipeline_runs = []
        pipeline_runs.append(self.stats)
        merged['pipeline_runs'] = pipeline_runs

        # Merge arxiv and references by summing sensible numeric fields if present
        arxiv_existing = merged.get('arxiv', {})
        arxiv_current = self.arxiv_scraper.get_stats()
        merged_arxiv = {}
        for key in set(list(arxiv_existing.keys()) + list(arxiv_current.keys())):
            try:
                merged_arxiv[key] = arxiv_existing.get(key, 0) + arxiv_current.get(key, 0)
            except Exception:
                merged_arxiv[key] = arxiv_current.get(key, arxiv_existing.get(key))
        merged['arxiv'] = merged_arxiv

        ref_existing = merged.get('references', {})
        ref_current = self.reference_scraper.get_stats()
        merged_ref = {}
        for key in set(list(ref_existing.keys()) + list(ref_current.keys())):
            try:
                merged_ref[key] = ref_existing.get(key, 0) + ref_current.get(key, 0)
            except Exception:
                merged_ref[key] = ref_current.get(key, ref_existing.get(key))
        merged['references'] = merged_ref

        # Save merged file
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(merged, f, indent=2)
            logger.info(f"Statistics saved to: {stats_file}")
        except Exception as e:
            logger.warning(f"Failed to save merged stats to {stats_file}: {e}")
    
    def scrape_single_paper(self, arxiv_id: str) -> bool:
        """
        Scrape a single paper with all components
        
        Args:
            arxiv_id: arXiv ID
        
        Returns:
            True if successful
        """
        start_time = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing paper {arxiv_id}")
        logger.info(f"{'='*60}")
        
        # Create paper directory
        folder_name = format_folder_name(arxiv_id)
        paper_dir = os.path.join(self.output_dir, folder_name)
        ensure_dir(paper_dir)
        
        # Measure size before
        size_before = get_directory_size(paper_dir) if os.path.exists(paper_dir) else 0
        
        # Step 1: Scrape paper and download sources
        success = self.arxiv_scraper.scrape_paper(arxiv_id, paper_dir)
        
        if not success:
            logger.error(f"Failed to scrape paper {arxiv_id}")
            self.stats['failed_papers'] += 1
            return False
        
        # Measure size after
        size_after = get_directory_size(paper_dir)
        self.stats['paper_sizes_before'].append(size_before)
        self.stats['paper_sizes_after'].append(size_after)
        
        # Step 2: Load metadata for BibTeX and references
        metadata_path = os.path.join(paper_dir, "metadata.json")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.stats['failed_papers'] += 1
            return False
        
        # Step 3: Keep original .bib files in their extracted locations
        # We do NOT copy .bib files to a central references.bib. Original .bib files (if any)
        # remain where they were extracted so their original paths and names are preserved.
        tex_root = os.path.join(paper_dir, "tex")
        if os.path.exists(tex_root):
            logger.info(f"Preserving extracted files under {tex_root}; .bib files (if present) will remain in their original locations.")
        
        # Step 4: Scrape references
        references_path = os.path.join(paper_dir, "references.json")
        self.reference_scraper.scrape_references(arxiv_id, references_path)
        
        # Count references
        try:
            with open(references_path, 'r', encoding='utf-8') as f:
                references = json.load(f)
                self.stats['reference_counts'].append(len(references))
        except:
            self.stats['reference_counts'].append(0)
        
        # Track runtime
        runtime = time.time() - start_time
        self.stats['paper_runtimes'].append(runtime)
        self.stats['successful_papers'] += 1
        
        logger.info(f"Successfully processed {arxiv_id} in {runtime:.2f}s")
        logger.info(f"Paper size: {size_after / 1024:.2f} KB")
        
        return True
    
    def run(self, start_ym: str = None, start_id: int = None,
            end_ym: str = None, end_id: int = None):
        """
        Run complete scraping pipeline
        
        Args:
            start_ym: Start year-month (default from config)
            start_id: Start paper ID (default from config)
            end_ym: End year-month (default from config)
            end_id: End paper ID (default from config)
        """
        # Use defaults from config if not provided
        start_ym = start_ym or START_YEAR_MONTH
        start_id = start_id if start_id is not None else START_ID
        end_ym = end_ym or END_YEAR_MONTH
        end_id = end_id if end_id is not None else END_ID
        
        logger.info("="*80)
        logger.info("arXiv Scraper Pipeline Started")
        logger.info(f"Student ID: {STUDENT_ID}")
        logger.info(f"Range: {start_ym}.{start_id:05d} to {end_ym}.{end_id:05d}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*80)
        
        pipeline_start = time.time()
        
        # Generate paper IDs
        logger.info("\nGenerating paper IDs...")
        entry_start = time.time()
        paper_ids = self.generate_paper_ids(start_ym, start_id, end_ym, end_id)
        # Prepare resume: skip papers that already have metadata.json (considered processed)
        to_process = []
        for aid in paper_ids:
            folder_name = format_folder_name(aid)
            paper_dir = os.path.join(self.output_dir, folder_name)
            metadata_path = os.path.join(paper_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                logger.debug(f"Skipping already-processed paper {aid}")
                continue
            to_process.append(aid)
        self.stats['total_papers'] = len(paper_ids)
        self.stats['entry_discovery_time'] = time.time() - entry_start
        
        logger.info(f"Total papers to scrape: {len(paper_ids)}")
        logger.info(f"First paper: {paper_ids[0]}")
        logger.info(f"Last paper: {paper_ids[-1]}")
        if not to_process:
            logger.info("No new papers to process (all IDs already processed). Exiting.")
            # Still update merged stats file to include current run's minor info
            stats_file = os.path.join(self.output_dir, 'scraping_stats.json')
            self._merge_and_save_stats_json(stats_file)
            return

        # Load existing CSV IDs so we don't duplicate rows when appending
        csv_path = os.path.join(self.output_dir, 'stats.csv')
        existing_csv_ids = self._load_existing_csv_ids(csv_path)
        # Scrape each paper with instrumentation (memory, time, sizes)
        try:
            proc = psutil.Process()
        except Exception:
            proc = None

        for i, arxiv_id in enumerate(to_process, 1):
            logger.info(f"\n[{i}/{len(to_process)}] Processing {arxiv_id}")

            try:
                # Per-paper sampling lists
                local_mem_samples = []
                local_disk_samples = []
                stop_event = threading.Event()

                def sampler():
                    while not stop_event.is_set():
                        try:
                            if proc:
                                local_mem_samples.append(proc.memory_info().rss)
                            local_disk_samples.append(get_directory_size(self.output_dir))
                        except Exception:
                            pass
                        time.sleep(0.5)

                # Start sampler thread
                sampler_thread = threading.Thread(target=sampler, daemon=True)
                sampler_thread.start()

                mem_before = proc.memory_info().rss if proc else 0
                t0 = time.time()
                ok = self.scrape_single_paper(arxiv_id)
                t1 = time.time()
                mem_after = proc.memory_info().rss if proc else 0

                # Stop sampler and wait for it
                stop_event.set()
                sampler_thread.join(timeout=2.0)

                # Compute memory/disk stats from samples
                mem_max = max(local_mem_samples) if local_mem_samples else (mem_after or 0)
                mem_avg = (sum(local_mem_samples) / len(local_mem_samples)) if local_mem_samples else (mem_after or 0)
                disk_max = max(local_disk_samples) if local_disk_samples else get_directory_size(self.output_dir)

                # Update run-level aggregates
                self.stats['memory_samples'].append(mem_max)
                self.stats['all_memory_samples'].extend(local_mem_samples)
                self.stats['disk_sizes'].append(disk_max)
                if disk_max > self.stats['max_disk_usage_bytes']:
                    self.stats['max_disk_usage_bytes'] = disk_max

                # Gather per-paper info
                folder_name = format_folder_name(arxiv_id)
                paper_dir = os.path.join(self.output_dir, folder_name)
                # Prefer metadata-reported before/after sizes (set by arxiv_scraper), fallback to directory size reads
                size_before = 0
                size_after = 0
                metadata_path = os.path.join(paper_dir, 'metadata.json')
                try:
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                            size_before = int(meta.get('size_before_bytes', 0) or 0)
                            size_after = int(meta.get('size_after_bytes', 0) or 0)
                except Exception:
                    pass
                if size_before == 0:
                    size_before = get_directory_size(paper_dir) if os.path.exists(paper_dir) else 0
                if size_after == 0:
                    size_after = get_directory_size(paper_dir) if os.path.exists(paper_dir) else 0

                # references count
                references_path = os.path.join(paper_dir, 'references.json')
                try:
                    with open(references_path, 'r', encoding='utf-8') as f:
                        refs = json.load(f)
                        ref_count = len(refs)
                except Exception:
                    ref_count = 0

                # Build row
                row = {
                    'arxiv_id': arxiv_id,
                    'success': bool(ok),
                    'runtime_s': round(t1 - t0, 2),
                    'mem_before_rss': mem_before,
                    'mem_after_rss': mem_after,
                    'mem_max_rss': mem_max,
                    'mem_avg_rss': int(mem_avg),
                    'size_before_bytes': size_before,
                    'size_after_bytes': size_after,
                    'disk_max_bytes': disk_max,
                    'final_output_size_bytes': get_directory_size(self.output_dir),
                    'references_count': ref_count
                }

                # Append to CSV if not already present
                if arxiv_id not in existing_csv_ids:
                    self._append_row_to_csv(csv_path, row)
                    existing_csv_ids.add(arxiv_id)

                # Save row to in-memory stats for this run
                self.stats['paper_rows'].append(row)

                # Append to aggregates
                self.stats['paper_runtimes'].append(t1 - t0)
                self.stats['paper_sizes_before'].append(size_before)
                self.stats['paper_sizes_after'].append(size_after)
                self.stats['reference_counts'].append(ref_count)

            except Exception as e:
                logger.error(f"Unexpected error processing {arxiv_id}: {e}")
                self.stats['failed_papers'] += 1

            # Progress update every 10 papers
            if i % 10 == 0:
                self.print_progress()
        
        # Final cleanup
        self.cleanup_all_temp_files()
        
        # Final statistics
        self.stats['total_runtime'] = time.time() - pipeline_start
        # Record final output directory size
        try:
            self.stats['final_output_size_bytes'] = get_directory_size(self.output_dir)
        except Exception:
            self.stats['final_output_size_bytes'] = self.stats.get('final_output_size_bytes', 0)
        self.print_final_stats()
        # Write stats outputs (CSV + MD)
        try:
            # Write md summary (aggregated from CSV to include historical runs)
            self.write_stats_outputs()
        except Exception as e:
            logger.warning(f"Failed to write stats outputs: {e}")

        # Merge and save scraping_stats.json with previous runs
        stats_file = os.path.join(self.output_dir, 'scraping_stats.json')
        self._merge_and_save_stats_json(stats_file)

    def write_stats_outputs(self):
        """Write stats.csv and stats.md summary into output_dir"""
        csv_path = os.path.join(self.output_dir, 'stats.csv')
        md_path = os.path.join(self.output_dir, 'stats.md')

        # Read CSV to compute aggregated metrics (includes past runs since CSV is append-only)
        total = 0
        success = 0
        failed = 0
        size_before_list = []
        size_after_list = []
        refs_list = []
        runtimes = []
        try:
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as cf:
                    reader = csv.DictReader(cf)
                    for r in reader:
                        total += 1
                        if str(r.get('success')).lower() in ('1', 'true', 'yes'):
                            success += 1
                        else:
                            failed += 1
                        try:
                            size_before_list.append(int(r.get('size_before_bytes') or 0))
                        except Exception:
                            pass
                        try:
                            size_after_list.append(int(r.get('size_after_bytes') or 0))
                        except Exception:
                            pass
                        try:
                            refs_list.append(int(r.get('references_count') or 0))
                        except Exception:
                            pass
                        try:
                            runtimes.append(float(r.get('runtime_s') or 0.0))
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Failed to read CSV for aggregates: {e}")

        avg_size_before = (sum(size_before_list)/len(size_before_list)) if size_before_list else 0
        avg_size_after = (sum(size_after_list)/len(size_after_list)) if size_after_list else 0
        avg_refs = (sum(refs_list)/len(refs_list)) if refs_list else 0
        avg_time = (sum(runtimes)/len(runtimes)) if runtimes else 0
        # Memory: compute max per-paper observed and average across all sampled points
        max_mem = max(self.stats['memory_samples']) if self.stats['memory_samples'] else 0
        avg_mem = (sum(self.stats['all_memory_samples'])/len(self.stats['all_memory_samples'])) if self.stats.get('all_memory_samples') else 0
        max_disk = self.stats.get('max_disk_usage_bytes', 0)
        final_output_size = self.stats.get('final_output_size_bytes') or get_directory_size(self.output_dir)

        with open(md_path, 'w', encoding='utf-8') as mf:
            mf.write('# Scraper run summary\n\n')
            mf.write(f'- Total papers recorded in CSV: {total}\n')
            mf.write(f'- Successful rows: {success}\n')
            mf.write(f'- Failed rows: {failed}\n')
            mf.write(f'- Success rate: {100.0 * success / max(1,total):.2f}%\n')
            mf.write('\n')
            mf.write('## Data sizes\n')
            mf.write(f'- Average size before processing: {avg_size_before/1024:.2f} KB\n')
            mf.write(f'- Average size after processing: {avg_size_after/1024:.2f} KB\n')
            mf.write('\n')
            mf.write('## References\n')
            mf.write(f'- Average references per paper: {avg_refs:.2f}\n')
            mf.write('\n')
            mf.write('## Performance\n')
            mf.write(f'- Average time per paper: {avg_time:.2f}s\n')
            mf.write(f'- Max memory (rss) observed in this run: {max_mem} bytes\n')
            mf.write(f'- Average memory (rss) during processing: {avg_mem:.0f} bytes\n')
            mf.write(f'- Max disk usage observed during run: {max_disk} bytes\n')
            mf.write(f'- Final output directory size: {final_output_size} bytes\n')
            mf.write(f'- Total pipeline runtime (this run): {self.stats.get("total_runtime", 0):.2f}s\n')
    
    def print_progress(self):
        """Print progress statistics"""
        logger.info("\n" + "="*60)
        logger.info("PROGRESS UPDATE")
        logger.info(f"Successful: {self.stats['successful_papers']}")
        logger.info(f"Failed: {self.stats['failed_papers']}")
        logger.info(f"Success rate: {self.stats['successful_papers']/max(1, self.stats['successful_papers']+self.stats['failed_papers'])*100:.1f}%")
        logger.info("="*60 + "\n")
    
    def cleanup_all_temp_files(self):
        """Clean up all remaining temp directories in output folder"""
        logger.info("\nCleaning up temporary files...")
        temp_cleaned = 0
        
        if not os.path.exists(self.output_dir):
            return
        
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                temp_dir = os.path.join(item_path, "temp")
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        temp_cleaned += 1
                        logger.debug(f"Removed temp directory: {temp_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temp directory {temp_dir}: {e}")
        
        if temp_cleaned > 0:
            logger.info(f"Cleaned {temp_cleaned} temp directories")
        else:
            logger.info("No temp directories found")
    
    def print_final_stats(self):
        """Print final statistics"""
        logger.info("\n" + "="*80)
        logger.info("FINAL STATISTICS")
        logger.info("="*80)
        
        logger.info(f"\nScraping Results:")
        logger.info(f"  Total papers attempted: {self.stats['total_papers']}")
        logger.info(f"  Successful: {self.stats['successful_papers']}")
        logger.info(f"  Failed: {self.stats['failed_papers']}")
        logger.info(f"  Success rate: {self.stats['successful_papers']/max(1, self.stats['total_papers'])*100:.1f}%")
        
        arxiv_stats = self.arxiv_scraper.get_stats()
        logger.info(f"\nArXiv Statistics:")
        logger.info(f"  Versions downloaded: {arxiv_stats['versions_downloaded']}")
        logger.info(f"  Total download time: {arxiv_stats['total_download_time']:.2f}s")
        
        ref_stats = self.reference_scraper.get_stats()
        logger.info(f"\nReference Statistics:")
        logger.info(f"  Papers queried: {ref_stats['papers_queried']}")
        logger.info(f"  Papers found: {ref_stats['papers_found']}")
        logger.info(f"  Total references: {ref_stats['total_references']}")
        logger.info(f"  References with arXiv ID: {ref_stats['references_with_arxiv_id']}")
        
        if self.stats['reference_counts']:
            avg_refs = sum(self.stats['reference_counts']) / len(self.stats['reference_counts'])
            logger.info(f"  Average references per paper: {avg_refs:.2f}")
        
        if self.stats['paper_runtimes']:
            avg_runtime = sum(self.stats['paper_runtimes']) / len(self.stats['paper_runtimes'])
            logger.info(f"\nPerformance:")
            logger.info(f"  Total runtime: {self.stats['total_runtime']:.2f}s")
            logger.info(f"  Average time per paper: {avg_runtime:.2f}s")
        
        if self.stats['paper_sizes_after']:
            avg_size = sum(self.stats['paper_sizes_after']) / len(self.stats['paper_sizes_after'])
            total_size = sum(self.stats['paper_sizes_after'])
            logger.info(f"\nStorage:")
            logger.info(f"  Average paper size: {avg_size/1024:.2f} KB")
            logger.info(f"  Total size: {total_size/1024/1024:.2f} MB")
        
        logger.info("\n" + "="*80)
    
    def save_stats(self):
        """Save statistics to JSON file"""
        stats_file = os.path.join(self.output_dir, "scraping_stats.json")
        
        all_stats = {
            'pipeline': self.stats,
            'arxiv': self.arxiv_scraper.get_stats(),
            'references': self.reference_scraper.get_stats()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2)
        
        logger.info(f"\nStatistics saved to: {stats_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='arXiv Paper Scraper')
    parser.add_argument('--start-ym', type=str, help='Start year-month (e.g., 2307)')
    parser.add_argument('--start-id', type=int, help='Start paper ID')
    parser.add_argument('--end-ym', type=str, help='End year-month (e.g., 2308)')
    parser.add_argument('--end-id', type=int, help='End paper ID')
    parser.add_argument('--output', type=str, default=DATA_DIR, help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(LOGS_DIR)
    
    # Create and run pipeline
    pipeline = ArxivScraperPipeline(args.output)
    pipeline.run(
        start_ym=args.start_ym,
        start_id=args.start_id,
        end_ym=args.end_ym,
        end_id=args.end_id
    )
    
    logger.info("\nScraping completed!")


if __name__ == "__main__":
    main()

