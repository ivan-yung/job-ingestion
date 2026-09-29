"""Job requirement classification and enhancement."""
from __future__ import annotations

import re
from src.models.job import JobRecord, ClassifiedJobRecord


def extract_seniority(text: str | None) -> str | None:
    """Extract seniority from text keywords or years of experience."""
    if not text:
        return None
        
    text_lower = text.lower()
    
    # 1. Explicit Keywords
    if "director" in text_lower: return "director"
    if "lead" in text_lower: return "lead"
    if "senior" in text_lower or "sr." in text_lower or "sr " in text_lower: return "senior"
    if "mid" in text_lower or "intermediate" in text_lower: return "mid"
    if "junior" in text_lower or "jr." in text_lower or "jr " in text_lower or "entry-level" in text_lower or "entry level" in text_lower: return "junior"
    if "new grad" in text_lower: return "new_grad"

    # 2. Years of Experience (captures X or X-Y)
    match = re.search(r'(\d+)\s*(?:-\s*\d+)?\s*\+?\s*years', text_lower)
    if match:
        years = int(match.group(1))
        if years >= 12: return "director"
        if years >= 7: return "senior"
        if years >= 3: return "mid"
        if years >= 1: return "junior"
        return "new_grad"
        
    return None


def classify_requirement(text: str) -> str:
    """Classify a single requirement string as hard, preferred, or unknown."""
    text_lower = text.lower()
    
    # Check for negations first
    if "not required" in text_lower or "no prior experience required" in text_lower:
        return "preferred"
        
    if any(word in text_lower for word in ["must", "required", "mandatory", "minimum", "essential"]):
        return "hard"
    if any(word in text_lower for word in ["preferred", "plus", "nice to have", "ideal", "bonus", "desirable"]):
        return "preferred"
    return "unknown"


def classify_job_record(job: JobRecord) -> ClassifiedJobRecord:
    """Classify requirements and backfill missing critical fields."""
    job_dict = job.model_dump()
    
    # Backfill Seniority
    seniority = job.seniority
    if not seniority:
        seniority = extract_seniority(job.title)
    if not seniority:
        seniority = extract_seniority(job.description_raw)
        
    job_dict["seniority"] = seniority
    
    # Reset tracking lists to ensure correct schema format
    job_dict["required_qualifications_with_evidence"] = []
    job_dict["preferred_qualifications_with_evidence"] = []
        
    # Re-populate qualification evidence into the correct target list
    for req in job.required_qualifications:
        cls = classify_requirement(req)
        item = {"text": req, "classification": cls, "evidence_sentence": req}
        if cls == "preferred":
            job_dict["preferred_qualifications_with_evidence"].append(item)
        else:
            job_dict["required_qualifications_with_evidence"].append(item)
            
    for req in job.preferred_qualifications:
        cls = classify_requirement(req)
        item = {"text": req, "classification": cls, "evidence_sentence": req}
        if cls == "hard":
            job_dict["required_qualifications_with_evidence"].append(item)
        else:
            job_dict["preferred_qualifications_with_evidence"].append(item)
            
    return ClassifiedJobRecord(**job_dict)