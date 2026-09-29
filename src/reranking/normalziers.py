"""Normalization for skills and job titles."""

import re

SKILL_ALIASES = {
    "js": "javascript",
    "postgres": "postgresql",
    "node": "node.js",
    "nodejs": "node.js",
    "golang": "go",
    "ts": "typescript",
    "reactjs": "react",
    "vuejs": "vue",
    "k8s": "kubernetes",
    "aws": "amazon web services",
    "gcp": "google cloud",
}

ROLE_FAMILIES = {
    "frontend": ["frontend", "front end", "front-end", "ui ", "user interface"],
    "backend": ["backend", "back end", "back-end"],
    "fullstack": ["full stack", "fullstack", "full-stack"],
    "embedded": ["embedded", "firmware"],
    "data": ["data engineer", "data pipeline"],
    "ml_ai": ["machine learning", "ml", "ai ", "artificial intelligence"],
    "swe": ["software engineer", "software developer", "sde", "swe", "programmer"]
}


def normalize_skill(skill: str) -> str:
    """Normalize a skill, applying aliases while preserving distinct technologies."""
    cleaned = skill.lower().strip()
    return SKILL_ALIASES.get(cleaned, cleaned)


def normalize_title(title: str) -> str:
    """Map a job title to a broad role family."""
    if not title:
        return "unknown"
        
    title_lower = title.lower()
    for family, keywords in ROLE_FAMILIES.items():
        if any(kw in title_lower for kw in keywords):
            return family
            
    return "other"