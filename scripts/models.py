"""Data classes for the Britannica OCR article parser."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paragraph:
    index: int
    char_start: int
    char_end: int
    text: str
    preview: str  # first 300 chars


@dataclass
class Classification:
    index: int
    type: str  # article_start, subsection_start, running_header, cross_reference,
               # front_matter, back_matter, author_attribution, footnote_sep, body_text
    title: Optional[str] = None
    keywords: Optional[list[str]] = None
    target: Optional[str] = None  # for cross_reference


@dataclass
class Subsection:
    title: str
    paragraph_start: int  # index within article's paragraphs
    paragraph_end: int


@dataclass
class HeadingCandidate:
    """A candidate article heading extracted by regex, before LIS filtering."""
    headword: str
    sort_key: str
    char_start: int
    char_end: int
    pattern: str  # 'article', 'treatise', 'crossref', 'titlecase'
    crossref_target: Optional[str] = None
    confidence: float = 1.0


@dataclass
class Article:
    article_id: str
    title: str
    edition: str
    edition_year: int
    volume: int
    source_file: str
    type: str  # article, cross_reference, front_matter, back_matter
    char_start: int
    char_end: int
    text: str
    word_count: int
    paragraph_count: int
    keywords: Optional[list[str]] = None
    author_attribution: Optional[str] = None
    target: Optional[str] = None  # for cross_reference
    subsections: list[dict] = field(default_factory=list)
    lis_confidence: float = 1.0
    heading_pattern: str = 'article'
