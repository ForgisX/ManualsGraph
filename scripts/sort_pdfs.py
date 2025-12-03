"""
PDF Sorting and Classification Script

Multi-stage pipeline:
1. Digital vs Scanned classification
2. Manufacturing relevance filtering (for digital PDFs)
3. OEM (manufacturer) classification
4. Machine model extraction and organization
5. Factory module categorization

Maintains JSON databases:
- config-manuals-structure/oems.json - Manufacturer registry
- config-manuals-structure/factory_modules.json - Factory component taxonomy
- config-manuals-structure/machines.json - Machine registry with manual links

Folder structure:
manuals/
    ...original sources...
sorted_manuals/
    digital/
        manufacturing/
            {OEM}/
                {Machine}/
                    manual.pdf
        other/
    scanned/
"""

import os
import shutil
import argparse
import re
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader
from tqdm import tqdm
import json
from difflib import SequenceMatcher

# Azure OpenAI imports
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel, Field

load_dotenv()

# --- Configuration Constants ---

# Paths
CONFIG_DIR = Path(__file__).parent.parent / "config-manuals-structure"
OEMS_JSON = CONFIG_DIR / "oems.json"
MODULES_JSON = CONFIG_DIR / "factory_modules.json"
MACHINES_JSON = CONFIG_DIR / "machines.json"

# PDF classification thresholds
MIN_TEXT_RATIO = 0.3  # At least 30% of pages must have extractable text
MIN_CHARS_PER_PAGE = 100  # Minimum characters per page to count as "has text"
MAX_PAGES_TO_SAMPLE = 10  # Sample first N pages for classification
METADATA_TEXT_LENGTH = 4096  # Characters to send to AI for classification (reduced to save tokens)

# Fuzzy matching threshold for OEM names
OEM_MATCH_THRESHOLD = 0.8  # 80% similarity required

# Pricing: dollars per 1M tokens (Standard tier)
MODEL_PRICING_INPUT = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-5-mini": 0.25,
    "gpt-5-nano": 0.05,
}
MODEL_PRICING_OUTPUT = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.60,
    "gpt-5-mini": 2.00,
    "gpt-5-nano": 0.40,
}

# Token estimation: approximate 4 characters per token
def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


class TokenTracker:
    """Track token usage and costs for Azure OpenAI API calls."""
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.events = []

    def record(self, label: str, prompt_text: str, output_text: str):
        in_tok = estimate_tokens_from_text(prompt_text)
        out_tok = estimate_tokens_from_text(output_text)
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.events.append((label, in_tok, out_tok))

    def summary(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }


def current_cost_estimate(tracker: TokenTracker) -> dict:
    """Compute current estimated costs using selected model."""
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    in_price = MODEL_PRICING_INPUT.get(model_name)
    out_price = MODEL_PRICING_OUTPUT.get(model_name)
    if in_price is None or out_price is None:
        return {"model": model_name, "in_cost": None, "out_cost": None, "total_cost": None}
    in_cost = (tracker.input_tokens / 1_000_000) * in_price
    out_cost = (tracker.output_tokens / 1_000_000) * out_price
    return {"model": model_name, "in_cost": in_cost, "out_cost": out_cost, "total_cost": in_cost + out_cost}


# --- Pydantic Models for AI Extraction ---

class ClassificationResult(BaseModel):
    """Single-shot classification result for a PDF."""
    is_manufacturing: bool = Field(description="True if related to industrial manufacturing equipment, False otherwise")
    manufacturer: Optional[str] = Field(default=None, description="OEM/manufacturer name if manufacturing-related, else None")
    is_new_oem: bool = Field(default=False, description="True if this OEM is not in the known list")
    model: Optional[str] = Field(default=None, description="Machine model/number if found")
    is_new_machine: bool = Field(default=False, description="True if this machine model is new (not in existing registry)")
    machine_type: Optional[str] = Field(default=None, description="Type of machine (e.g., 'robot', 'PLC', 'drive')")
    factory_module: Optional[str] = Field(default=None, description="Matched factory module ID from known list, or suggestion if new")
    reasoning: str = Field(description="Brief reasoning for classification")


# --- JSON Database Management ---

def load_json_db(file_path: Path) -> Dict[str, Any]:
    """Load JSON database file."""
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_json_db(file_path: Path, data: Dict[str, Any]):
    """Save JSON database file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fuzzy_match_oem(manufacturer_name: str, oems_db: Dict) -> Optional[Dict]:
    """
    Fuzzy match manufacturer name against known OEMs.
    Returns matched OEM data if similarity >= threshold, else None.
    """
    manufacturer_lower = manufacturer_name.lower().strip()
    
    for oem in oems_db.get("manufacturers", []):
        # Check exact match with name
        if oem["name"].lower() == manufacturer_lower:
            return oem
        
        # Check aliases
        for alias in oem.get("aliases", []):
            if alias.lower() == manufacturer_lower:
                return oem
        
        # Fuzzy match with name and aliases
        all_names = [oem["name"]] + oem.get("aliases", [])
        for name in all_names:
            similarity = SequenceMatcher(None, manufacturer_lower, name.lower()).ratio()
            if similarity >= OEM_MATCH_THRESHOLD:
                return oem
    
    return None


def prompt_user_new_oem(manufacturer_name: str, llm: AzureOpenAI, tracker: TokenTracker) -> Optional[Dict]:
    """
    Prompt user to confirm adding a new OEM.
    Uses AI to suggest proper name format and aliases.
    """
    print(f"\n{'='*60}")
    print(f"🏭 NEW MANUFACTURER DETECTED: {manufacturer_name}")
    print(f"{'='*60}")
    
    # Get AI suggestion for proper formatting
    prompt = f"""Given the manufacturer name "{manufacturer_name}", provide:
1. The official/proper company name
2. Common aliases and variations
3. Brief description of what they manufacture

This is for an industrial manufacturing equipment database."""
    
    try:
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=NewOEMProposal,
            llm=llm,
            prompt_template_str=prompt,
            verbose=False
        )
        proposal = program()
        tracker.record("new_oem_proposal", prompt, str(proposal))
        
        print(f"\nAI Suggestion:")
        print(f"  Name: {proposal.name}")
        print(f"  Aliases: {', '.join(proposal.aliases)}")
        print(f"  Description: {proposal.description}")
        
    except Exception as e:
        print(f"  (AI suggestion failed: {e})")
        proposal = NewOEMProposal(
            name=manufacturer_name,
            aliases=[],
            description="Industrial equipment manufacturer"
        )
    
    print(f"\nAdd this manufacturer to the database?")
    response = input("  [y]es / [n]o / [e]dit: ").strip().lower()
    
    if response == 'n':
        print("  ❌ Skipped")
        return None
    
    if response == 'e':
        name = input(f"  Official name [{proposal.name}]: ").strip() or proposal.name
        aliases_str = input(f"  Aliases (comma-separated) [{', '.join(proposal.aliases)}]: ").strip()
        aliases = [a.strip() for a in aliases_str.split(',')] if aliases_str else proposal.aliases
        description = input(f"  Description [{proposal.description}]: ").strip() or proposal.description
    else:
        name = proposal.name
        aliases = proposal.aliases
        description = proposal.description
    
    # Generate ID
    oem_id = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    
    new_oem = {
        "id": oem_id,
        "name": name,
        "aliases": aliases,
        "description": description
    }
    
    print(f"  ✅ Added: {name}")
    return new_oem


def match_factory_module(module_suggestion: str, modules_db: Dict, llm: AzureOpenAI, tracker: TokenTracker) -> Optional[str]:
    """
    Match suggested module to existing taxonomy.
    Prompts user if no match and AI suggests new module.
    Returns module ID or None.
    """
    module_lower = module_suggestion.lower().strip()
    
    # Try exact ID match
    for module in modules_db.get("modules", []):
        if module["id"] == module_lower:
            return module["id"]
    
    # Try fuzzy match on name and keywords
    best_match = None
    best_similarity = 0.7  # Lower threshold for modules
    
    for module in modules_db.get("modules", []):
        # Check name similarity
        similarity = SequenceMatcher(None, module_lower, module["name"].lower()).ratio()
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = module["id"]
        
        # Check if suggestion matches any keywords
        for keyword in module.get("keywords", []):
            if keyword.lower() in module_lower or module_lower in keyword.lower():
                return module["id"]
    
    if best_match:
        return best_match
    
    # No match - potentially new module
    print(f"\n{'='*60}")
    print(f"🔧 NEW FACTORY MODULE SUGGESTED: {module_suggestion}")
    print(f"{'='*60}")
    print("Existing modules:")
    for i, module in enumerate(modules_db.get("modules", [])[:5], 1):
        print(f"  {i}. {module['name']} ({module['id']})")
    print(f"  ... ({len(modules_db.get('modules', []))} total)")
    
    response = input(f"\nIs this truly a new module category? [y]es / [n]o (map to existing): ").strip().lower()
    
    if response != 'y':
        print("\nEnter existing module ID to use:")
        for module in modules_db.get("modules", []):
            print(f"  - {module['id']}: {module['name']}")
        module_id = input("Module ID: ").strip()
        if module_id:
            return module_id
        return None
    
    # Create new module
    module_id = re.sub(r'[^a-z0-9]+', '_', module_suggestion.lower()).strip('_')
    name = input(f"Display name [{module_suggestion}]: ").strip() or module_suggestion
    description = input("Description: ").strip()
    keywords_str = input("Keywords (comma-separated): ").strip()
    keywords = [k.strip() for k in keywords_str.split(',')] if keywords_str else []
    
    new_module = {
        "id": module_id,
        "name": name,
        "description": description,
        "keywords": keywords
    }
    
    # Add to database
    if "modules" not in modules_db:
        modules_db["modules"] = []
    modules_db["modules"].append(new_module)
    modules_db["metadata"]["total_modules"] = len(modules_db["modules"])
    modules_db["metadata"]["last_updated"] = datetime.now().isoformat()
    save_json_db(MODULES_JSON, modules_db)
    
    print(f"  ✅ Added new module: {name}")
    return module_id


def add_or_update_machine(machine_data: Dict, manual_path: Path, modules_db: Dict, llm: AzureOpenAI, tracker: TokenTracker):
    """Add or update machine entry in machines.json."""
    machines_db = load_json_db(MACHINES_JSON)
    
    if "machines" not in machines_db:
        machines_db["machines"] = []
    
    # Generate machine ID
    machine_id = f"{machine_data['oem_id']}_{re.sub(r'[^a-z0-9]+', '_', machine_data['model'].lower()).strip('_')}"
    
    # Check if machine exists
    existing_machine = None
    for machine in machines_db["machines"]:
        if machine.get("id") == machine_id:
            existing_machine = machine
            break
    
    if existing_machine:
        # Update existing machine - add manual if not already there
        manual_info = {
            "path": str(manual_path),
            "added_date": datetime.now().isoformat(),
            "pages": machine_data.get("pages", 0),
            "file_size_mb": machine_data.get("file_size_mb", 0)
        }
        
        if "manuals" not in existing_machine:
            existing_machine["manuals"] = []
        
        # Check if this manual already exists
        manual_exists = any(m.get("path") == str(manual_path) for m in existing_machine["manuals"])
        if not manual_exists:
            existing_machine["manuals"].append(manual_info)
            print(f"  📄 Added manual to existing machine: {existing_machine['name']}")
    else:
        # Create new machine entry
        new_machine = {
            "id": machine_id,
            "name": f"{machine_data['manufacturer']} {machine_data['model']}",
            "manufacturer": machine_data['manufacturer'],
            "oem_id": machine_data['oem_id'],
            "model": machine_data['model'],
            "series": machine_data.get('series'),
            "description": machine_data.get('description', ''),
            "machine_type": machine_data.get('machine_type', ''),
            "factory_module": machine_data.get('factory_module'),
            "manuals": [{
                "path": str(manual_path),
                "added_date": datetime.now().isoformat(),
                "pages": machine_data.get("pages", 0),
                "file_size_mb": machine_data.get("file_size_mb", 0)
            }],
            "created_date": datetime.now().isoformat()
        }
        machines_db["machines"].append(new_machine)
        print(f"  ✅ Created new machine entry: {new_machine['name']}")
    
    # Update metadata
    machines_db["metadata"]["total_machines"] = len(machines_db["machines"])
    machines_db["metadata"]["last_updated"] = datetime.now().isoformat()
    
    save_json_db(MACHINES_JSON, machines_db)


# --- PDF Classification Functions ---

def classify_pdf_digital_scanned(pdf_path: Path) -> Tuple[str, dict]:
    """
    Classify PDF as 'digital' or 'scanned' based on text extractability.
    """
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        pages_to_check = min(total_pages, MAX_PAGES_TO_SAMPLE)
        pages_with_text = 0
        total_chars = 0
        
        for i in range(pages_to_check):
            try:
                text = reader.pages[i].extract_text()
                char_count = len(text.strip())
                total_chars += char_count
                
                if char_count >= MIN_CHARS_PER_PAGE:
                    pages_with_text += 1
            except:
                continue
        
        text_ratio = pages_with_text / pages_to_check if pages_to_check > 0 else 0
        classification = "digital" if text_ratio >= MIN_TEXT_RATIO else "scanned"
        
        metadata = {
            "total_pages": total_pages,
            "pages_checked": pages_to_check,
            "text_ratio": round(text_ratio, 2),
            "classification": classification,
            "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2)
        }
        
        return classification, metadata
        
    except Exception as e:
        return "scanned", {"error": str(e), "classification": "scanned"}


def extract_pdf_text_sample(pdf_path: Path, max_chars: int = METADATA_TEXT_LENGTH) -> str:
    """Extract text sample from PDF for AI analysis."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page_num in range(min(5, len(reader.pages))):
            page_text = reader.pages[page_num].extract_text()
            text += page_text + "\n"
            
            if len(text) >= max_chars:
                break
        
        return text[:max_chars]
    except Exception as e:
        return f"[Error extracting text: {e}]"


def classify_pdf_single_shot(pdf_path: Path, llm: AzureOpenAI, tracker: TokenTracker, oems_db: Dict, modules_db: Dict) -> ClassificationResult:
    """Single-shot classification: relevance, OEM, machine, and module in one AI call."""
    text_sample = extract_pdf_text_sample(pdf_path)
    
    # Build known OEM list
    known_oems = [oem["name"] for oem in oems_db.get("manufacturers", [])]
    oem_list_str = ", ".join(known_oems[:20]) if known_oems else "(none)"
    
    # Build known module list
    known_modules = [m["id"] for m in modules_db.get("modules", [])]
    module_list_str = ", ".join(known_modules) if known_modules else "(none)"
    
    prompt = f"""Classify this PDF for an industrial manufacturing equipment database.

1. Is it manufacturing-related? (robots, CNCs, PLCs, drives, conveyors, presses, sensors, etc.)
2. If YES: Extract manufacturer (OEM) and machine model. Check if OEM is in known list: [{oem_list_str}]. Set is_new_oem=True if not found.
3. Extract machine type and match to module: [{module_list_str}].
4. If NOT manufacturing: set is_manufacturing=False, leave fields empty.

PDF excerpt:
{text_sample}

Classify concisely."""
    
    try:
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=ClassificationResult,
            llm=llm,
            prompt_template_str=prompt,
            verbose=False
        )
        result = program()
        tracker.record("classify_single_shot", prompt, str(result))
        return result
    except Exception as e:
        print(f"Classification failed: {e}")
        return ClassificationResult(
            is_manufacturing=False,
            reasoning=f"Error: {e}"
        )


# --- File Organization Functions ---

def is_fully_sorted(pdf_path: Path, manuals_dir: Path) -> bool:
    """
    Check if PDF is fully sorted (in OEM/Machine folder structure).
    """
    # Check absolute path parts for sorted_manuals structure (outside manuals folder)
    parts = pdf_path.parts
    for i, part in enumerate(parts):
        if part == "sorted_manuals":
            # Expect: sorted_manuals/digital/manufacturing/{OEM}/{Machine}/file.pdf
            if i + 3 < len(parts):
                if parts[i+1] == "digital" and parts[i+2] == "manufacturing":
                    return True
            break
    
    return False


def find_all_pdfs(root_dir: Path, manuals_dir: Path, skip_fully_sorted: bool = True) -> List[Path]:
    """Find all PDF files, optionally skipping fully sorted ones."""
    all_pdfs = list(root_dir.rglob("*.pdf")) + list(root_dir.rglob("*.PDF"))
    
    if skip_fully_sorted:
        pdfs = [pdf for pdf in all_pdfs if not is_fully_sorted(pdf, manuals_dir)]
        skipped = len(all_pdfs) - len(pdfs)
        if skipped > 0:
            print(f"Skipping {skipped} fully-sorted PDFs")
        return pdfs
    
    return all_pdfs


def find_pending_sorted_manuals(manuals_dir: Path) -> List[Path]:
    """Find PDFs in sorted_manuals/digital that are not fully processed yet.
    Pending means: files under sorted_manuals/digital/ that are NOT in
    - sorted_manuals/digital/other/
    - sorted_manuals/digital/manufacturing/{OEM}/{Machine}/
    """
    sorted_base = manuals_dir.parent / "sorted_manuals" / "digital"
    if not sorted_base.exists():
        return []
    candidates = list(sorted_base.rglob("*.pdf")) + list(sorted_base.rglob("*.PDF"))
    pending: List[Path] = []
    for pdf in candidates:
        parts = pdf.parts
        try:
            idx = parts.index("sorted_manuals")
        except ValueError:
            # Not under sorted_manuals; skip
            continue
        # Structure after idx: [sorted_manuals, digital, ...]
        if idx + 1 < len(parts) and parts[idx+1] == "digital":
            # If under other -> skip
            if idx + 2 < len(parts) and parts[idx+2] == "other":
                continue
            # If under manufacturing/{OEM}/{Machine} -> skip fully processed
            if idx + 2 < len(parts) and parts[idx+2] == "manufacturing":
                # Check if it has at least OEM/machine layers
                if idx + 4 < len(parts):
                    # Consider fully processed; skip
                    continue
                # Else it's under manufacturing but not in OEM/machine yet -> pending
                pending.append(pdf)
                continue
            # Files directly under digital or other subfolders -> pending
            pending.append(pdf)
    return pending


def organize_pdf(pdf_path: Path, manuals_dir: Path, stage: str, **kwargs) -> Path:
    """
    Move PDF to appropriate folder based on classification stage.
    
    Stages:
    - 'scanned': sorted_manuals/scanned/
    - 'digital': sorted_manuals/digital/
    - 'other': sorted_manuals/digital/other/
    - 'manufacturing': sorted_manuals/digital/manufacturing/
    - 'oem': sorted_manuals/digital/manufacturing/{OEM}/
    - 'machine': sorted_manuals/digital/manufacturing/{OEM}/{Machine}/
    """
    # Place sorted outputs outside the manuals folder for visibility
    sorted_base = manuals_dir.parent / "sorted_manuals"
    
    if stage == "scanned":
        dest_dir = sorted_base / "scanned"
    elif stage == "digital":
        dest_dir = sorted_base / "digital"
    elif stage == "other":
        dest_dir = sorted_base / "digital" / "other"
    elif stage == "manufacturing":
        dest_dir = sorted_base / "digital" / "manufacturing"
    elif stage == "oem":
        oem = kwargs.get("oem", "Unknown")
        dest_dir = sorted_base / "digital" / "manufacturing" / oem
    elif stage == "machine":
        oem = kwargs.get("oem", "Unknown")
        machine = kwargs.get("machine", "Unknown")
        dest_dir = sorted_base / "digital" / "manufacturing" / oem / machine
    else:
        dest_dir = sorted_base
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Preserve original filename
    dest_path = dest_dir / pdf_path.name
    
    # Handle duplicate filenames
    counter = 1
    while dest_path.exists() and dest_path != pdf_path:
        stem = pdf_path.stem
        dest_path = dest_dir / f"{stem}_{counter}{pdf_path.suffix}"
        counter += 1
    
    # Move file
    if pdf_path != dest_path:
        shutil.move(str(pdf_path), str(dest_path))
        # Log save location
        try:
            print(f"Saved: {pdf_path.name} -> {dest_path}")
        except Exception:
            pass
        # Clean up empty source directories inside manuals after move
        try:
            src_dir = pdf_path.parent
            manuals_root = manuals_dir
            # Only attempt cleanup if source is under manuals_dir
            if str(src_dir).startswith(str(manuals_root)):
                while src_dir != manuals_root and src_dir.exists():
                    # Stop if directory not empty
                    if any(src_dir.iterdir()):
                        break
                    src_dir.rmdir()
                    src_dir = src_dir.parent
        except Exception:
            # Ignore cleanup errors silently
            pass
    
    return dest_path


# --- Main Pipeline ---

def _format_desc(name: str, width: int = 40) -> str:
    """Create a fixed-width description to avoid tqdm bar jumping."""
    safe = (name or "")
    if len(safe) > width:
        safe = safe[:width-1] + "…"
    return f"{safe:<{width}}"

def _format_postfix(cost: float, tokens: int, width: int = 28) -> str:
    """Create a compact fixed-width postfix string for tqdm."""
    txt = f"Cost: ${cost:.4f} | Tok: {tokens:,}"
    if len(txt) > width:
        txt = txt[:width-1] + "…"
    return f"{txt:<{width}}"

def process_pdf_pipeline(pdf_path: Path, manuals_dir: Path, llm: Optional[AzureOpenAI], tracker: TokenTracker, stats: Dict, pbar: Optional[tqdm] = None):
    """
    Simplified processing pipeline using single-shot AI classification.
    """
    try:
        if pbar:
            pbar.set_description(_format_desc(f"Processing: {pdf_path.name}"))
        
        # Stage 1: Digital vs Scanned
        classification, metadata = classify_pdf_digital_scanned(pdf_path)
        
        if classification == "scanned":
            organize_pdf(pdf_path, manuals_dir, "scanned")
            stats["scanned"] += 1
            return
        
        stats["digital"] += 1
        
        # If no LLM, just move to digital folder
        if llm is None:
            organize_pdf(pdf_path, manuals_dir, "digital")
            stats["digital_unprocessed"] += 1
            return
        
        # Load databases for single-shot classification
        oems_db = load_json_db(OEMS_JSON)
        modules_db = load_json_db(MODULES_JSON)
        
        # Stage 2: Single-shot AI classification (relevance + OEM + machine + module)
        result = classify_pdf_single_shot(pdf_path, llm, tracker, oems_db, modules_db)
        
        # Not manufacturing -> move to other
        if not result.is_manufacturing:
            organize_pdf(pdf_path, manuals_dir, "other")
            stats["other"] += 1
            return
        
        stats["manufacturing"] += 1
        
        # No manufacturer found -> generic manufacturing folder
        if not result.manufacturer:
            organize_pdf(pdf_path, manuals_dir, "manufacturing")
            stats["manufacturing_no_oem"] += 1
            return
        
        # Update progress bar with OEM/Model
        if pbar:
            pbar.set_description(_format_desc(f"{result.manufacturer} {result.model or 'unknown'}"))
        
        # Stage 3: Handle OEM (new or existing)
        if result.is_new_oem:
            # Auto-add new OEM without prompting
            oem_id = re.sub(r'[^a-z0-9]+', '_', result.manufacturer.lower()).strip('_')
            new_oem = {
                "id": oem_id,
                "name": result.manufacturer,
                "aliases": [],
                "description": f"Auto-added: {result.machine_type or 'manufacturing equipment'}"
            }
            if "manufacturers" not in oems_db:
                oems_db["manufacturers"] = []
            oems_db["manufacturers"].append(new_oem)
            oems_db["metadata"]["total_manufacturers"] = len(oems_db["manufacturers"])
            oems_db["metadata"]["last_updated"] = datetime.now().isoformat()
            save_json_db(OEMS_JSON, oems_db)
            stats["new_oems"] += 1
        else:
            # Match existing OEM
            matched_oem = fuzzy_match_oem(result.manufacturer, oems_db)
            if matched_oem:
                oem_id = matched_oem["id"]
                result.manufacturer = matched_oem["name"]  # Use canonical name
            else:
                # Fallback: treat as new
                oem_id = re.sub(r'[^a-z0-9]+', '_', result.manufacturer.lower()).strip('_')
        
        # Stage 4: Organize into OEM/Machine folder
        machine_folder_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', (result.model or "unknown").strip()).strip('_')
        new_path = organize_pdf(pdf_path, manuals_dir, "machine", oem=result.manufacturer, machine=machine_folder_name)
        
        # Stage 5: Update machines.json
        machine_data = {
            "manufacturer": result.manufacturer,
            "oem_id": oem_id,
            "model": result.model or "unknown",
            "series": None,
            "description": result.reasoning,
            "machine_type": result.machine_type or "unknown",
            "factory_module": result.factory_module,
            "pages": metadata.get("total_pages", 0),
            "file_size_mb": metadata.get("file_size_mb", 0)
        }
        
        add_or_update_machine(machine_data, new_path, modules_db, llm, tracker)
        stats["fully_sorted"] += 1
        
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        stats["errors"] += 1


def main():
    parser = argparse.ArgumentParser(
        description="Sort and classify PDFs into organized manufacturing database"
    )
    parser.add_argument(
        "--manuals-dir",
        type=Path,
        default=Path(__file__).parent.parent / "manuals",
        help="Directory containing PDF files (default: ../manuals)"
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=100,
        help="Maximum number of PDFs to process (default: 100, use -1 for all)"
    )
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="Skip AI classification (only do digital/scanned sort)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without moving files"
    )
    
    args = parser.parse_args()
    
    if not args.manuals_dir.exists():
        print(f"Error: Directory not found: {args.manuals_dir}")
        return 1
    
    # Initialize
    # Ensure output base exists outside manuals folder
    (args.manuals_dir.parent / "sorted_manuals").mkdir(parents=True, exist_ok=True)
    tracker = TokenTracker()
    llm = None
    
    if not args.skip_ai:
        try:
            llm = AzureOpenAI(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=0.2
            )
            print("✅ Azure OpenAI initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize Azure OpenAI: {e}")
            print("Continuing with basic digital/scanned classification only")
    
    # Find PDFs
    print(f"\n{'='*60}")
    print(f"PDF Classification Pipeline")
    print(f"{'='*60}")
    print(f"Source: {args.manuals_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"AI Classification: {'ENABLED' if llm else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    # Prioritize pending files in sorted_manuals/digital before fresh ones
    pending_first = find_pending_sorted_manuals(args.manuals_dir)
    pdf_files_fresh = find_all_pdfs(args.manuals_dir, args.manuals_dir, skip_fully_sorted=True)
    # Avoid duplicates if paths overlap
    pending_set = set(str(p) for p in pending_first)
    combined = pending_first + [p for p in pdf_files_fresh if str(p) not in pending_set]
    pdf_files = combined
    
    # Limit number of PDFs to process
    if args.max_pdfs > 0:
        pdf_files = pdf_files[:args.max_pdfs]
    
    print(f"Found {len(pdf_files)} PDFs to process")
    
    if len(pdf_files) == 0:
        print("No PDFs to process.")
        return 0
    
    # Process PDFs
    stats = {
        "total": len(pdf_files),
        "scanned": 0,
        "digital": 0,
        "digital_unprocessed": 0,
        "manufacturing": 0,
        "other": 0,
        "manufacturing_no_oem": 0,
        "fully_sorted": 0,
        "new_oems": 0,
        "errors": 0
    }
    
    start_time = datetime.now()
    
    # Progress bar with cost tracking
    pbar = tqdm(
        pdf_files,
        desc=_format_desc("Processing PDFs"),
        bar_format="{l_bar}{bar}{r_bar} {desc}",
        ncols=140
    )
    for pdf_path in pbar:
        if args.dry_run:
            # Update description only; do not print per-file lines
            pbar.set_description(_format_desc(f"Dry-run: {pdf_path.name}"))
            continue
        
        process_pdf_pipeline(pdf_path, args.manuals_dir, llm, tracker, stats, pbar)
        
        # Update progress bar with compact fixed-width cost info
        if llm and tracker.input_tokens > 0:
            costs = current_cost_estimate(tracker)
            tokens_total = tracker.input_tokens + tracker.output_tokens
            pbar.set_postfix_str(_format_postfix(costs['total_cost'], tokens_total))
    
    end_time = datetime.now()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing Complete")
    print(f"{'='*60}")
    print(f"Total PDFs: {stats['total']}")
    print(f"  Scanned (needs OCR): {stats['scanned']}")
    print(f"  Digital: {stats['digital']}")
    if llm:
        print(f"    └─ Manufacturing: {stats['manufacturing']}")
        print(f"       └─ Fully Sorted (OEM/Machine): {stats['fully_sorted']}")
        print(f"       └─ No OEM detected: {stats['manufacturing_no_oem']}")
        print(f"    └─ Other (not manufacturing): {stats['other']}")
        print(f"\n  New OEMs added: {stats['new_oems']}")
    else:
        print(f"    └─ Unprocessed (no AI): {stats['digital_unprocessed']}")
    print(f"  Errors: {stats['errors']}")
    print(f"\nTime: {end_time - start_time}")
    
    if llm and tracker.input_tokens > 0:
        costs = current_cost_estimate(tracker)
        print(f"\nAPI Usage:")
        print(f"  Tokens: {tracker.input_tokens:,} in / {tracker.output_tokens:,} out")
        print(f"  Cost: ${costs['total_cost']:.4f}")
    
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
