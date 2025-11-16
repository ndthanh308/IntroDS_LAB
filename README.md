# arXiv Scraper — Usage Guide (Student ID: 23127538)

This README provides precise, implementation-focused instructions to set up and run the arXiv scraping project in a CPU-only Google Colab environment (or locally). 

Table of Contents
- [1. Environment Setup Guide](#1-environment-setup-guide)  
  - [1.1 Google Colab (CPU-only)](#11-google-colab-cpu-only)  
  - [1.2 Local environment (optional)](#12-local-environment-optional)  
- [2. Software Version](#2-software-version)  
- [3. Required Packages](#3-required-packages)  
- [4. Instructions to Run the Code](#4-instructions-to-run-the-code)  
  - [4.1 Clone / Upload the project](#41-clone--upload-the-project)  
  - [4.2 Install dependencies](#42-install-dependencies)  
  - [4.3 Configure API key and parameters](#43-configure-api-key-and-parameters)  
  - [4.4 Run the pipeline](#44-run-the-pipeline)  
  - [4.5 Logs and outputs](#45-logs-and-outputs)  
- [5. Specific Configurations](#5-specific-configurations)  
  - [5.1 Input arXiv ID range](#51-input-arxiv-id-range)  
  - [5.2 Scraping rate / delays](#52-scraping-rate--delays)  
  - [5.3 Parallelism](#53-parallelism)  
  - [5.4 Output location](#54-output-location)  
- [6. Example Run (Google Colab)](#6-example-run-google-colab)  
- [7. Folder Structure Explanation](#7-folder-structure-explanation)  
- [8. Notes and Best Practices](#8-notes-and-best-practices)

---

## 1. Environment Setup Guide

### 1.1 Google Colab (CPU-only)
1. Open a new Colab notebook and set Runtime → Change runtime type → Hardware accelerator: None.  
2. Mount Google Drive to persist outputs:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Place or clone the project folder under your Drive (recommended) to persist outputs across sessions. See Example Run for a sample workflow.

### 1.2 Local environment (optional)
- Create and activate a Python virtual environment:
  ```bash
  python -m venv .venv
  # Windows
  .venv\Scripts\activate
  # macOS / Linux
  source .venv/bin/activate
  ```
- Install dependencies (see Section 3).

---

## 2. Software Version

- Python: tested with Python 3.11 (compatible with Python 3.8+).  
- Operating System: any OS with Python 3.8+ is supported; Colab runs on Linux.

---

## 3. Required Packages

All required packages are listed in `src/requirements.txt`. Primary packages include (but may not be limited to):
- arxiv
- requests
- pandas
- psutil
- tqdm

Install all packages with:
```bash
pip install -r src/requirements.txt
```
If a package is missing, install it individually:
```bash
pip install arxiv requests pandas psutil tqdm
```

---

## 4. Instructions to Run the Code

### 4.1 Clone / Upload the project
- From GitHub:
  ```bash
  git clone https://github.com/ndthanh308/IntroDS_LAB.git project
  cd project/23127538
  ```
- Or upload the project into Colab/Drive and cd into the `23127538` directory.

### 4.2 Install dependencies
```bash
pip install -r src/requirements.txt
```
In Colab, prefix commands with `!` inside a cell:
```bash
!pip install -r src/requirements.txt
```

### 4.3 Configure API key and parameters
- Recommended: set the Semantic Scholar API key as an environment variable in the runtime (do not commit keys to the repository):
  ```python
  import os
  os.environ['SEMANTIC_SCHOLAR_API_KEY'] = "<YOUR_API_KEY>"
  ```
- Alternatively, edit `src/config.py` to set:
  - STUDENT_ID
  - START_YEAR_MONTH / START_ID
  - END_YEAR_MONTH / END_ID
  - DATA_DIR (output root)
  - ARXIV_API_DELAY / SEMANTIC_SCHOLAR_DELAY

### 4.4 Run the pipeline
- Default (use values defined in `src/config.py`):
  ```bash
  python src/main.py
  ```
- Override via CLI arguments:
  ```bash
  python src/main.py --start-ym 2307 --start-id 11657 --end-ym 2307 --end-id 16656 --output "./23127538/23127538_data"
  ```
- Resume behavior: rerun the same command to continue from the last processed paper; the pipeline skips papers already containing `metadata.json`.

### 4.5 Logs and outputs
- Logs: stored in `logs/` (path defined by `LOGS_DIR` in `src/config.py`) — review `scraper.log` for progress and error traces.  
- Outputs: root output (default `DATA_DIR`) contains per-paper folders, `stats.csv`, `stats.md`, and `scraping_stats.json`.

---

## 5. Specific Configurations

### 5.1 Input arXiv ID range
- Configure the start/end IDs and months in `src/config.py` or pass them via CLI flags as shown above.

### 5.2 Scraping rate / delays
- Configure delays in `src/config.py`:
  - `ARXIV_API_DELAY` — seconds between arXiv requests (recommended ≥ 3.0).
  - `SEMANTIC_SCHOLAR_DELAY` — seconds between Semantic Scholar calls (recommended ≥ 1.0).
- The pipeline implements basic retry/backoff logic for transient HTTP errors.

### 5.3 Parallelism
- The pipeline runs sequentially by default to respect rate limits and simplify deterministic outputs.  
- If parallel execution is implemented, configure worker count in `src/config.py` (search for `MAX_WORKERS`) and ensure rate-limits are respected.

### 5.4 Output location
- Outputs are organized under a student-id root. Set `DATA_DIR` in `src/config.py` (example default: `./23127538/23127538_data`).

---

## 6. Example Run (Google Colab)

Cells to run in Colab (example):

1) Mount Drive and change working directory:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/<your-folder>/project/23127538
```

2) Install dependencies:
```bash
!pip install -r src/requirements.txt
```

3) Set API key and run:
```python
import os
os.environ['SEMANTIC_SCHOLAR_API_KEY'] = "YOUR_API_KEY"
```
```bash
!python src/main.py
```

4) Inspect outputs:
```bash
!ls -la ./23127538/23127538_data
!head -n 20 ./23127538/23127538_data/stats.md
!tail -n 200 ./logs/scraper.log
```

Notes:
- Break large runs into batches (e.g., 100–500 papers) because Colab sessions can terminate.
- The pipeline supports resume so batched runs are safe.

---

## 7. Folder Structure Explanation

Canonical data layout produced by the pipeline:
```
23127538/                           # Student ID root
 ├─ 2307-11657/                     # One paper
 │   ├─ tex/
 │   │ ├─ 2307-11657v1/
 │   │ │ ├─ *.tex
 │   │ │ ├─ *.bib        <-- .bib files remain inside version folder
 │   │ │ └─ <subfolders> # original archive structure preserved (except images)
 │   ├─ metadata.json
 │   └─ references.json
 ├─ src/                            # Source code
 │  ├─ main.py
 │  ├─ utils.py
 │  ├─ arxiv_scraper.py
 │  ├─ reference_scraper.py
 │  └─ requirements.txt
 ├─ stats.csv
 ├─ stats.md
 └─ scraping_stats.json
```

Important behavior:
- Preserve original relative layout of extracted TeX sources; remove only image files.
- Keep empty version folders for versions with no TeX source.
- `.bib` files (if present) remain in the corresponding version folder under `tex/<yymm-id>v<version>/`.

---

## 8. Notes and Best Practices

- Do not include report content in this README. This file is strictly for reproducibility and execution instructions.  
- Store API keys as environment variables in Colab/session rather than committing to the repository.  
- For long-running, high-volume scraping prefer a stable VM (e.g., GCP/Azure/AWS) rather than Colab. If using Colab, split work into batches and persist outputs to Drive.  
- Monitor `logs/scraper.log` to detect failures or rate-limit responses and adjust delays accordingly.  
- If you want a pre-filled Colab notebook with these steps, request it and a notebook will b