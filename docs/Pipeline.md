# Processing Pipeline Guide

ManualsGraph processes manuals through a 3-stage pipeline: **Download → Sort → Extract**.

## Stage 1: Download from Internet Archive
Script: `scripts/download_from_internet_archive.py`

This script searches the Internet Archive for manuals under Public Domain or Creative Commons licenses.

```bash
python scripts/download_from_internet_archive.py --count 50 --output-dir manuals/external
```
-   **Filter**: Strictly targets machine, vehicle, and military equipment.
-   **Deduplication**: Automatically skips already downloaded files.

## Stage 2: PDF Sorting and Classification
Script: `scripts/sort_pdfs.py`

This is the gatekeeping stage that organizes raw PDFs into a structured hierarchy.

```bash
python scripts/sort_pdfs.py --manuals-dir manuals/external --dry-run
```

### The 5-Stage Sort Logic:
1.  **Digital vs. Scanned**: Checks if text is extractable. Scanned PDFs are moved to a `scanned/` folder for future OCR.
2.  **Quick Gate**: Uses a small AI sample to determine if the document is manufacturing-related.
3.  **Full Extraction**: Identifies the Manufacturer (OEM) and Model.
4.  **Matching**: Checks against `config-manuals-structure/oems.json`.
5.  **Organization**: Moves the PDF to `sorted_manuals/digital/manufacturing/<OEM>/<Model>/`.

## Stage 3: Knowledge Extraction
Script: `scripts/llamaindex-processing.py`

The final stage extracts structured "knowledge triples" from the sorted manuals.

```bash
python scripts/llamaindex-processing.py sorted_manuals/.../your_manual.pdf
```

### Extraction Features:
-   **Machine Metadata**: Extracts manufacturer, model, and series.
-   **Fault Extraction**: Identifies error codes and messages.
-   **Root Cause Analysis**: Correlates faults with their underlying causes and fixing steps.
-   **Output**: Generates `metadata.json`, `fault_db.json`, and `root_cause_db.json`.
