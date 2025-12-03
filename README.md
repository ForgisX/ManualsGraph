# ManualsGraph

**From Manuals to Troubleshooting Graphs: Generative Knowledge Extraction for Industrial Maintenance**

This repository accompanies an academic paper: [ManualsGraph](https://www.overleaf.com/project/692dde1e0ee444c08e101d48)

## Overview

ManualsGraph transforms unstructured industrial machine manuals into structured causal knowledge graphs that enable real-time fault diagnosis and automated repair guidance. We extract "symptom → root cause → fix" relationships from technical documents to power lightweight, interpretable classifiers deployable on edge devices.

The project combines classical parsing algorithms with transformer-based NLP to handle complex, semi-structured manuals at scale (1M+ documents), enabling similarity search, causal reasoning, and transfer learning across similar machines.

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and environment manager
- Python 3.12 (recommended) or 3.10-3.13

**Why uv?** We use uv for dependency management because:
- It automatically handles Python version compatibility issues
- Much faster than pip (10-100x speedup)
- Better dependency resolution prevents version conflicts
- Avoids issues with packages like pandas that need specific Python versions

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd ManualsGraph
    ```

2.  Install uv (if not already installed):
    ```bash
    # Windows
    winget install --id=astral-sh.uv -e
    
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  Create virtual environment with Python 3.12 and install dependencies:
    ```bash
    uv venv --python 3.12
    
    # Activate the environment:
    # Windows PowerShell:
    .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    
    # Install dependencies:
    uv pip install -r requirements.txt
    ```

**Note:** Python 3.14 is currently not supported due to pandas compatibility issues. Use Python 3.12 for best results.

## Documentation

To view the documentation, run:

```bash
mkdocs serve
```

Then open your browser to `http://127.0.0.1:8000`. 

## PDF Processing Pipeline

### Sort and Classify PDFs

The project includes an AI-powered pipeline to sort and classify industrial manufacturing manuals:

```bash
# Process first 100 PDFs (default, cost-effective for testing)
python scripts/sort_pdfs.py

# Process specific number of PDFs
python scripts/sort_pdfs.py --max-pdfs 500

# Process all PDFs (use with caution - can be expensive!)
python scripts/sort_pdfs.py --max-pdfs -1

# Skip AI classification (only digital vs scanned)
python scripts/sort_pdfs.py --skip-ai
```

**What it does:**
- **Stage 1**: Classifies PDFs as digital (text-extractable) or scanned (needs OCR)
- **Stage 2**: Filters digital PDFs by manufacturing relevance using AI
- **Stage 3**: Extracts machine metadata (manufacturer, model, type)
- **Stage 4**: Organizes into structured folders: `manuals/sorted_pdfs/digital/manufacturing/{OEM}/{Machine}/`
- **Stage 5**: Maintains JSON databases in `config-manuals-structure/` for OEMs, factory modules, and machines

The pipeline shows real-time cost tracking and can be safely interrupted and resumed.

### Download from Internet Archive

You can download manuals from internet archive with the following script

```bash
python scripts/download_from_internet_archive.py --count 1000 --output-dir manuals --workers 5 
```
Parameters:
 * **count**: Number of manuals we want to download
 * **output-dir**: Path of output directory
 * **workers**: Number of thread to parallelize the process 

## Code Style and Pre-commit Hooks

This repository uses [pre-commit](https://pre-commit.com/) to enforce code style with Black, isort, and flake8.

### Setup pre-commit

1. Install pre-commit and style tools:
    ```bash
    uv pip install -r requirements.txt
    ```
2. Install the pre-commit hooks:
    ```bash
    pre-commit install
    ```

### Usage
- Hooks will run automatically on `git commit`.
- To run checks manually on all files:
    ```bash
    pre-commit run --all-files
    ``` 

For more details on code style and pre-commit hooks, see the [documentation](docs/index.md).

## Download from Internet Archive

You can download manuals from internet archive with the following script

```bash
python scripts/download_from_internet_archive.py --count 1000 --output-dir manuals --workers 5 
```
Parameters:
 * **count**: Number of manuals we want to download
 * **output-dir**: Path of output directory
 * **workers**: Number of thread to parallelize the process 