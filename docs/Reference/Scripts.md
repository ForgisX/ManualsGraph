# Technical Reference: Scripts

This page provides a detailed reference for all primary scripts in the ManualsGraph repository.

## `scripts/sort_pdfs.py`
The orchestrator for sorting and classifying incoming PDF manuals.

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--manuals-dir` | Directory containing PDFs to process | `manuals/` |
| `--max-pdfs` | Max number of PDFs to process | None (all) |
| `--skip-ai` | Skip AI classification (just move to digital/scanned) | False |
| `--dry-run` | Predict actions without moving files | False |

**Key Classes:**
- `TokenTracker`: Manages Azure OpenAI token usage and cost estimation.
- `ClassificationResult`: A Pydantic model for structured AI output.

---

## `scripts/llamaindex-processing.py`
Extracts faults and root causes using LlamaIndex and Azure OpenAI.

| Argument | Description | Default |
| :--- | :--- | :--- |
| `pdf` | Path to the PDF manual | None |
| `--out` | Output directory for JSON files | `data/llamaindex_outputs/...` |
| `--show-costs` | Print estimated token costs at completion | False |

**Internal Logic:**
- **Chunking**: Splits text into 8192-character nodes with a 400-character overlap.
- **Context Window**: When a fault is found, the script pulls surrounding nodes to find root causes.

---

## `scripts/download_from_internet_archive.py`
Batch downloader for public domain manuals.

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--count` | Number of manuals to download | 100 |
| `--output-dir` | Target directory for downloads | `manuals` |
| `--workers` | Number of parallel download threads | 5 |
