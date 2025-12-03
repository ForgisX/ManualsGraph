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
  sorted_pdfs/
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
METADATA_TEXT_LENGTH = 3000  # Characters to send to AI for classification

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

class ManufacturingRelevance(BaseModel):
    """Assessment of PDF relevance to industrial manufacturing."""
    is_manufacturing: bool = Field(description="Whether this PDF is related to industrial manufacturing equipment")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of the decision")
    document_type: str = Field(description="Type of document (e.g., service manual, parts catalog, advertisement, user guide)")


class MachineMetadata(BaseModel):
    """Extracted machine information from PDF."""
    manufacturer: str = Field(description="Equipment manufacturer/OEM (e.g., 'ABB', 'Siemens', 'Fanuc')")
    model: str = Field(description="Specific model number or name (e.g., 'IRB 6700', 'S7-1500')")
    series: Optional[str] = Field(description="Product series if applicable (e.g., 'IRB', 'S7')")
    description: str = Field(description="Brief description of the equipment (1-2 sentences)")
    factory_module_suggestion: str = Field(description="Suggested factory module category (e.g., 'robotics', 'plc_control', 'drives_motors')")
    machine_type: str = Field(description="Type of machine (e.g., 'industrial robot', 'PLC', 'servo drive')")


class NewOEMProposal(BaseModel):
    """Proposal for a new OEM to add to the database."""
    name: str = Field(description="Official company name")
    aliases: List[str] = Field(description="Common variations/abbreviations of the name")
    description: str = Field(description="Brief description of what this company makes")


class NewModuleProposal(BaseModel):
    """Proposal for a new factory module to add to the taxonomy."""
    id: str = Field(description="Unique ID for the module (lowercase, underscores)")
    name: str = Field(description="Display name for the module")
    description: str = Field(description="What this module category represents")
    keywords: List[str] = Field(description="Keywords associated with this module type")


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


def assess_manufacturing_relevance(pdf_path: Path, llm: AzureOpenAI, tracker: TokenTracker) -> ManufacturingRelevance:
    """Use AI to determine if PDF is relevant to industrial manufacturing."""
    text_sample = extract_pdf_text_sample(pdf_path)
    
    prompt = f"""Analyze this PDF excerpt and determine if it relates to industrial manufacturing equipment.

Industrial manufacturing includes: robots, CNCs, PLCs, conveyors, assembly machines, presses, 
packaging equipment, inspection systems, drives, motors, controllers, sensors, etc.

NOT manufacturing: consumer products, cars (unless factory equipment), personal electronics, 
software, business documents, catalogs without technical specs, advertisements.

PDF Text Sample:
{text_sample}

Classify this document."""
    
    try:
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=ManufacturingRelevance,
            llm=llm,
            prompt_template_str=prompt,
            verbose=False
        )
        result = program()
        tracker.record("manufacturing_relevance", prompt, str(result))
        return result
    except Exception as e:
        print(f"    AI assessment failed: {e}")
        # Default to not manufacturing if AI fails
        return ManufacturingRelevance(
            is_manufacturing=False,
            confidence=0.0,
            reasoning=f"AI error: {e}",
            document_type="unknown"
        )


def extract_machine_metadata(pdf_path: Path, llm: AzureOpenAI, tracker: TokenTracker) -> Optional[MachineMetadata]:
    """Extract machine metadata using AI."""
    text_sample = extract_pdf_text_sample(pdf_path)
    
    prompt = f"""Extract machine/equipment information from this PDF.

Focus on:
- Manufacturer/OEM (e.g., ABB, Siemens, Fanuc)
- Model number (e.g., IRB 6700, S7-1500)
- Equipment type and purpose
- Which factory module it belongs to

PDF Text Sample:
{text_sample}

Extract structured metadata."""
    
    try:
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=MachineMetadata,
            llm=llm,
            prompt_template_str=prompt,
            verbose=False
        )
        result = program()
        tracker.record("machine_metadata", prompt, str(result))
        return result
    except Exception as e:
        print(f"    Metadata extraction failed: {e}")
        return None


# --- File Organization Functions ---

def is_fully_sorted(pdf_path: Path, manuals_dir: Path) -> bool:
    """
    Check if PDF is fully sorted (in OEM/Machine folder structure).
    """
    try:
        rel_path = pdf_path.relative_to(manuals_dir)
        parts = rel_path.parts
        
        # Fully sorted path: sorted_pdfs/digital/manufacturing/{OEM}/{Machine}/file.pdf
        if len(parts) >= 5 and parts[0] == "sorted_pdfs" and parts[1] == "digital" and parts[2] == "manufacturing":
            return True
    except:
        pass
    
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


def organize_pdf(pdf_path: Path, manuals_dir: Path, stage: str, **kwargs) -> Path:
    """
    Move PDF to appropriate folder based on classification stage.
    
    Stages:
    - 'scanned': sorted_pdfs/scanned/
    - 'digital': sorted_pdfs/digital/
    - 'other': sorted_pdfs/digital/other/
    - 'manufacturing': sorted_pdfs/digital/manufacturing/
    - 'oem': sorted_pdfs/digital/manufacturing/{OEM}/
    - 'machine': sorted_pdfs/digital/manufacturing/{OEM}/{Machine}/
    """
    sorted_base = manuals_dir / "sorted_pdfs"
    
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
    
    return dest_path


# --- Main Pipeline ---

def process_pdf_pipeline(pdf_path: Path, manuals_dir: Path, llm: Optional[AzureOpenAI], tracker: TokenTracker, stats: Dict):
    """
    Complete processing pipeline for a single PDF.
    """
    try:
        print(f"\n📄 {pdf_path.name}")
        
        # Stage 1: Digital vs Scanned
        classification, metadata = classify_pdf_digital_scanned(pdf_path)
        print(f"  └─ Classification: {classification} (text ratio: {metadata.get('text_ratio', 0):.0%})")
        
        if classification == "scanned":
            new_path = organize_pdf(pdf_path, manuals_dir, "scanned")
            stats["scanned"] += 1
            return
        
        stats["digital"] += 1
        
        # If no LLM, just move to digital folder
        if llm is None:
            organize_pdf(pdf_path, manuals_dir, "digital")
            stats["digital_unprocessed"] += 1
            return
        
        # Stage 2: Manufacturing Relevance
        relevance = assess_manufacturing_relevance(pdf_path, llm, tracker)
        print(f"  └─ Manufacturing: {'✅ Yes' if relevance.is_manufacturing else '❌ No'} ({relevance.confidence:.0%} confidence)")
        print(f"     Type: {relevance.document_type}")
        
        if not relevance.is_manufacturing:
            new_path = organize_pdf(pdf_path, manuals_dir, "other")
            stats["other"] += 1
            return
        
        stats["manufacturing"] += 1
        
        # Stage 3: Extract Machine Metadata
        machine_meta = extract_machine_metadata(pdf_path, llm, tracker)
        
        if not machine_meta or not machine_meta.manufacturer:
            print(f"  └─ ⚠️ Could not extract manufacturer, moving to manufacturing/ folder")
            organize_pdf(pdf_path, manuals_dir, "manufacturing")
            stats["manufacturing_no_oem"] += 1
            return
        
        print(f"  └─ Machine: {machine_meta.manufacturer} {machine_meta.model}")
        print(f"     Type: {machine_meta.machine_type}")
        
        # Stage 4: Match/Add OEM
        oems_db = load_json_db(OEMS_JSON)
        matched_oem = fuzzy_match_oem(machine_meta.manufacturer, oems_db)
        
        if matched_oem:
            oem_id = matched_oem["id"]
            oem_name = matched_oem["name"]
            print(f"  └─ Matched OEM: {oem_name}")
        else:
            # Prompt for new OEM
            new_oem = prompt_user_new_oem(machine_meta.manufacturer, llm, tracker)
            if new_oem:
                if "manufacturers" not in oems_db:
                    oems_db["manufacturers"] = []
                oems_db["manufacturers"].append(new_oem)
                oems_db["metadata"]["total_manufacturers"] = len(oems_db["manufacturers"])
                oems_db["metadata"]["last_updated"] = datetime.now().isoformat()
                save_json_db(OEMS_JSON, oems_db)
                
                oem_id = new_oem["id"]
                oem_name = new_oem["name"]
                stats["new_oems"] += 1
            else:
                print(f"  └─ ⚠️ OEM not added, moving to manufacturing/ folder")
                organize_pdf(pdf_path, manuals_dir, "manufacturing")
                stats["manufacturing_no_oem"] += 1
                return
        
        # Stage 5: Match Factory Module
        modules_db = load_json_db(MODULES_JSON)
        factory_module = match_factory_module(machine_meta.factory_module_suggestion, modules_db, llm, tracker)
        
        if factory_module:
            print(f"  └─ Factory Module: {factory_module}")
        
        # Stage 6: Organize into OEM/Machine folder
        machine_folder_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', machine_meta.model).strip('_')
        new_path = organize_pdf(pdf_path, manuals_dir, "machine", oem=oem_name, machine=machine_folder_name)
        
        # Stage 7: Update machines.json
        machine_data = {
            "manufacturer": oem_name,
            "oem_id": oem_id,
            "model": machine_meta.model,
            "series": machine_meta.series,
            "description": machine_meta.description,
            "machine_type": machine_meta.machine_type,
            "factory_module": factory_module,
            "pages": metadata.get("total_pages", 0),
            "file_size_mb": metadata.get("file_size_mb", 0)
        }
        
        add_or_update_machine(machine_data, new_path, modules_db, llm, tracker)
        stats["fully_sorted"] += 1
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
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
    
    pdf_files = find_all_pdfs(args.manuals_dir, args.manuals_dir, skip_fully_sorted=True)
    print(f"Found {len(pdf_files)} PDFs to process\n")
    
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
    pbar = tqdm(pdf_files, desc="Processing PDFs")
    for pdf_path in pbar:
        if args.dry_run:
            print(f"\n[DRY RUN] Would process: {pdf_path}")
            continue
        
        process_pdf_pipeline(pdf_path, args.manuals_dir, llm, tracker, stats)
        
        # Update progress bar with cost info
        if llm and tracker.input_tokens > 0:
            costs = current_cost_estimate(tracker)
            pbar.set_postfix({
                'Cost': f"${costs['total_cost']:.4f}",
                'Tokens': f"{tracker.input_tokens + tracker.output_tokens:,}"
            })
    
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
