"""
lexical.py - signal 3 of 3: exact keyword matching (the recruiter's checklist).

The recruiter lists MUST-HAVE and NICE-TO-HAVE skills. We check which of them
appear in the resume, after mapping common synonyms to one name
("postgres" -> postgresql, "k8s" -> kubernetes, "nodejs" -> node.js ...).

    must_coverage = must-have skills found  / total must-have skills
    nice_coverage = nice-to-have skills found / total nice-to-have skills

    S_lex = 0.7 * must_coverage + 0.3 * nice_coverage

If only one list is given, S_lex is that list's coverage. If the recruiter leaves
both lists empty, skills are auto-extracted from the JD and treated as must-have.

Example (JD must = Python, SQL, Docker; nice = AWS, Git):
    resume has Python, SQL, Git  ->  0.7 * 2/3 + 0.3 * 1/2 = 0.62
"""

from __future__ import annotations

import re
from functools import lru_cache

from . import config

# canonical skill name  ->  ways people write it in a resume
SKILL_ALIASES: dict[str, list[str]] = {
    "python": ["python", "python3"],
    "java": ["java"],
    "javascript": ["javascript", "js", "es6"],
    "typescript": ["typescript"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "golang": ["golang"],
    "sql": ["sql"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "docker": ["docker", "dockerfile", "docker compose", "docker-compose"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services", "ec2"],
    "azure": ["azure"],
    "gcp": ["gcp", "google cloud"],
    "git": ["git", "github", "gitlab"],
    "linux": ["linux", "ubuntu", "bash"],
    "rest api": ["rest api", "rest apis", "restful", "restful api", "restful apis"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
    "node.js": ["node.js", "nodejs", "express.js", "expressjs"],
    "react": ["react", "react.js", "reactjs"],
    "next.js": ["next.js", "nextjs"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "tailwind", "tailwindcss"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow", "keras"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "llm": ["llm", "llms", "large language model", "large language models"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "langchain": ["langchain"],
    "excel": ["excel", "ms excel"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "ci/cd": ["ci/cd", "cicd", "github actions", "jenkins"],
    "microservices": ["microservices", "microservice"],
}

# reverse lookup: any alias -> canonical name
_ALIAS_TO_SKILL = {alias: skill for skill, aliases in SKILL_ALIASES.items() for alias in aliases}


@lru_cache(maxsize=None)
def _pattern(alias: str) -> re.Pattern:
    """
    Regex that finds `alias` as a whole word, also for names with symbols
    ('c++', 'node.js', 'ci/cd'). 'ml' must not match inside 'html', and
    'sql' must not match inside 'mysql'.
    """
    body = r"\s+".join(re.escape(part) for part in alias.split())
    return re.compile(rf"(?<![a-z0-9+#./]){body}(?![a-z0-9+#/])")


def canonical(skill: str) -> str:
    """'Postgres' -> 'postgresql'. Unknown skills are kept as typed (lower-case)."""
    cleaned = " ".join(skill.lower().split())
    return _ALIAS_TO_SKILL.get(cleaned, cleaned)


def parse_skill_list(raw: str) -> list[str]:
    """'Python, SQL; docker' -> ['python', 'sql', 'docker'] (no duplicates, order kept)."""
    skills: list[str] = []
    for part in re.split(r"[,;\n]", raw or ""):
        if part.strip():
            skill = canonical(part)
            if skill not in skills:
                skills.append(skill)
    return skills


def has_skill(text_lower: str, skill: str) -> bool:
    """True if any spelling of `skill` appears in the (lower-case) text."""
    return any(_pattern(alias).search(text_lower) for alias in SKILL_ALIASES.get(skill, [skill]))


def extract_skills(text: str) -> list[str]:
    """All known skills mentioned in a text (used when the recruiter gives no skill list)."""
    lower = text.lower()
    return [skill for skill in SKILL_ALIASES if has_skill(lower, skill)]


def score(resume_text: str, must_have: list[str], nice_to_have: list[str]) -> dict:
    """Lexical score of one resume plus the matched / missing skills (shown in the UI)."""
    lower = resume_text.lower()
    must_found = [s for s in must_have if has_skill(lower, s)]
    nice_found = [s for s in nice_to_have if has_skill(lower, s)]

    must_cov = len(must_found) / len(must_have) if must_have else None
    nice_cov = len(nice_found) / len(nice_to_have) if nice_to_have else None

    if must_cov is not None and nice_cov is not None:
        value = config.LEX_MUST_WEIGHT * must_cov + config.LEX_NICE_WEIGHT * nice_cov
    elif must_cov is not None:
        value = must_cov
    elif nice_cov is not None:
        value = nice_cov
    else:
        value = 0.0

    return {
        "score": round(value, 4),
        "must_matched": must_found,
        "must_missing": [s for s in must_have if s not in must_found],
        "nice_matched": nice_found,
        "nice_missing": [s for s in nice_to_have if s not in nice_found],
    }
