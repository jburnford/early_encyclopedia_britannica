"""
Edition-specific configuration and parsing rules for Encyclopedia Britannica.

Each edition has unique characteristics that affect parsing:
- 1771 (1st): Arts & Sciences only, no biography
- 1778 (2nd): Added Biography & History, Vol 10 Appendix
- 1797 (3rd): Expert authors, supplement (1801)
- 1810 (4th): Reprint of 3rd with updates
- 1815 (5th): Largest article count
- 1823 (6th): Cleanest OCR (no long s)
- 1842 (7th): Has General Index (Vol 22)
- 1860 (8th): Final edition in corpus

Usage:
    from encyclopedia_parser.editions import get_edition_config, EditionRegistry

    # Get config for a specific edition
    config = get_edition_config(1771)
    print(config.has_biography)  # False

    # Check if article type is valid for edition
    from encyclopedia_parser.models import ArticleType
    if ArticleType.BIOGRAPHICAL in config.allowed_article_types:
        print("This edition has biographical entries")
"""

from typing import Optional
from ..models import EditionConfig, EDITION_CONFIGS, ArticleType


class EditionRegistry:
    """
    Registry for edition-specific configurations and rules.

    Provides a central point for accessing edition-specific behavior
    and validating articles against edition constraints.
    """

    _instance: Optional['EditionRegistry'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._configs = EDITION_CONFIGS.copy()
        self._initialized = True

    def get_config(self, year: int) -> EditionConfig:
        """Get configuration for a specific edition year."""
        if year in self._configs:
            return self._configs[year]
        # Return generic config for unknown editions
        return EditionConfig(
            year=year,
            name=f"Edition {year}",
            volumes=1
        )

    def is_valid_article_type(self, year: int, article_type: ArticleType) -> bool:
        """Check if an article type is valid for a given edition."""
        config = self.get_config(year)
        return article_type in config.allowed_article_types

    def has_biography(self, year: int) -> bool:
        """Check if edition includes biographical entries."""
        return self.get_config(year).has_biography

    def has_geography(self, year: int) -> bool:
        """Check if edition includes geographical entries."""
        return self.get_config(year).has_geography

    def get_major_treatises(self, year: int) -> set[str]:
        """Get the set of major treatise headwords for an edition."""
        return set(self.get_config(year).major_treatises)

    def has_index(self, year: int) -> bool:
        """Check if edition has an index volume."""
        return self.get_config(year).index_volume is not None

    def get_index_volume(self, year: int) -> Optional[int]:
        """Get the index volume number for an edition."""
        return self.get_config(year).index_volume

    def list_editions(self) -> list[int]:
        """Get list of all configured edition years."""
        return sorted(self._configs.keys())


# Module-level singleton instance
_registry = EditionRegistry()


def get_edition_config(year: int) -> EditionConfig:
    """Get configuration for a specific edition year."""
    return _registry.get_config(year)


def is_valid_article_type(year: int, article_type: ArticleType) -> bool:
    """Check if an article type is valid for a given edition."""
    return _registry.is_valid_article_type(year, article_type)


def get_major_treatises(year: int) -> set[str]:
    """Get the set of major treatise headwords for an edition."""
    return _registry.get_major_treatises(year)


__all__ = [
    'EditionRegistry',
    'get_edition_config',
    'is_valid_article_type',
    'get_major_treatises',
]
