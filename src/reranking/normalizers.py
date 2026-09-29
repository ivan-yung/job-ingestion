"""Conservative normalization for skills and job titles."""

import re

SKILL_ALIASES = {
    "js": "javascript",
    "javascript es6": "javascript",
    "postgres": "postgresql",
    "postgres db": "postgresql",
    "node": "node.js",
    "nodejs": "node.js",
    "golang": "go",
    "ts": "typescript",
    "reactjs": "react",
    "vuejs": "vue",
    "k8s": "kubernetes",
    "amazon web services": "aws",
    "google cloud platform": "gcp",
}

ROLE_FAMILIES = {
    "fullstack": ("full stack", "fullstack", "full-stack"),
    "embedded": ("embedded", "firmware"),
    "ml_ai": ("machine learning", "artificial intelligence", "ml engineer", "ai engineer"),
    "data": ("data engineer", "data pipeline", "analytics engineer"),
    "frontend": ("frontend", "front end", "front-end", "user interface", "ui developer"),
    "backend": ("backend", "back end", "back-end"),
    "software_engineer": ("software engineer", "software developer", "sde", "swe", "programmer"),
}

_SENIORITY_WORDS = re.compile(
    r"\b(junior|jr\.?|mid(?:-level)?|senior|sr\.?|lead|principal|staff|director|manager|head|vice president|vp)\b",
    re.IGNORECASE,
)


def normalize_skill(skill: str) -> str:
    """Normalize spelling and known aliases without merging distinct technologies."""
    cleaned = re.sub(r"\s+", " ", str(skill).strip().lower())
    cleaned = cleaned.replace("®", "").strip()
    return SKILL_ALIASES.get(cleaned, cleaned)


def normalize_title(title: str) -> str:
    """Map a title to its most specific role family."""
    if not title:
        return "unknown"

    title_lower = re.sub(r"\s+", " ", title.lower()).strip()
    for family, keywords in ROLE_FAMILIES.items():
        if any(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", title_lower) for keyword in keywords):
            return family
    return "other"


def title_tokens(title: str) -> set[str]:
    """Return role tokens after removing seniority and generic title words."""
    cleaned = _SENIORITY_WORDS.sub(" ", str(title).lower())
    tokens = set(re.findall(r"[a-z]+(?:-[a-z]+)?", cleaned))
    return tokens - {"engineer", "developer", "role", "position", "of", "the"}


def title_similarity(candidate_titles: list[str], job_title: str) -> float:
    """Compare explicit target roles to a job title using role families and tokens."""
    if not candidate_titles or not job_title:
        return 0.0

    job_family = normalize_title(job_title)
    job_tokens = title_tokens(job_title)
    best = 0.0
    for candidate_title in candidate_titles:
        candidate_family = normalize_title(candidate_title)
        candidate_tokens = title_tokens(candidate_title)
        if candidate_family != "other" and candidate_family == job_family:
            score = 1.0
        elif candidate_tokens and job_tokens:
            overlap = len(candidate_tokens & job_tokens)
            score = overlap / len(candidate_tokens | job_tokens)
        else:
            score = 0.0
        best = max(best, score)
    return min(1.0, best)
