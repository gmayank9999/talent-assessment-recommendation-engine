"""
Catalog loader and lookup structures.

Loads the SHL product catalog from JSON, computes test_type codes from the
raw 'keys' field, and builds fast-lookup sets for URL and name validation.
This module is the single source of truth for all product data.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# mapping from catalog 'keys' values to short test_type codes
# derived from the sample conversation traces (C1-C10)
KEYS_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
}


def _compute_test_type(keys: list[str]) -> str:
    """
    Convert a list of catalog key strings into a comma-separated
    test_type code string. Order follows the keys array order.

    Example: ["Competencies", "Knowledge & Skills"] -> "C,K"
    """
    codes = []
    for key in keys:
        code = KEYS_TO_CODE.get(key)
        if code:
            codes.append(code)
    return ",".join(codes)


def _normalize_url(link: str) -> str:
    """Ensure URL ends with a trailing slash for consistency."""
    if link and not link.endswith("/"):
        return link + "/"
    return link


class CatalogEntry:
    """Single assessment product from the SHL catalog."""

    __slots__ = (
        "entity_id", "name", "url", "description", "job_levels",
        "languages", "duration", "keys", "test_type", "remote",
        "adaptive", "status",
    )

    def __init__(self, raw: dict):
        self.entity_id: str = raw.get("entity_id", "")
        self.name: str = raw.get("name", "").strip()
        self.url: str = _normalize_url(raw.get("link", ""))
        self.description: str = raw.get("description", "").strip()
        self.job_levels: list[str] = raw.get("job_levels", [])
        self.languages: list[str] = raw.get("languages", [])
        self.duration: str = raw.get("duration", "").strip()
        self.keys: list[str] = raw.get("keys", [])
        self.test_type: str = _compute_test_type(self.keys)
        self.remote: str = raw.get("remote", "no")
        self.adaptive: str = raw.get("adaptive", "no")
        self.status: str = raw.get("status", "ok")

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "job_levels": self.job_levels,
            "languages": self.languages,
            "duration": self.duration,
            "keys": self.keys,
            "test_type": self.test_type,
            "remote": self.remote,
            "adaptive": self.adaptive,
        }


class CatalogStore:
    """
    Loads the full SHL catalog and provides fast lookups.

    Built once at startup. All retrieval and validation layers
    reference this store rather than re-reading the JSON.
    """

    def __init__(self, catalog_path: Optional[str] = None):
        if catalog_path is None:
            # default: look for shl_product_catalog.json at project root
            root = Path(__file__).resolve().parent.parent
            catalog_path = str(root / "shl_product_catalog.json")

        self.entries: list[CatalogEntry] = []
        self.url_set: set[str] = set()
        self.name_set: set[str] = set()
        self.url_to_entry: dict[str, CatalogEntry] = {}
        self.name_to_entry: dict[str, CatalogEntry] = {}

        self._load(catalog_path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            raw_catalog = json.load(f)

        seen_urls: set[str] = set()
        skipped = 0

        for raw in raw_catalog:
            entry = CatalogEntry(raw)

            # basic quality gate: must have name and valid SHL URL
            if not entry.name or not entry.url.startswith("https://www.shl.com/"):
                skipped += 1
                continue

            # dedupe by URL (pagination can produce repeats)
            if entry.url in seen_urls:
                skipped += 1
                continue
            seen_urls.add(entry.url)

            self.entries.append(entry)
            self.url_set.add(entry.url)
            self.name_set.add(entry.name)
            self.url_to_entry[entry.url] = entry
            self.name_to_entry[entry.name] = entry

        logger.info(
            "Catalog loaded: %d entries, %d skipped", len(self.entries), skipped
        )

    def get_by_name(self, name: str) -> Optional[CatalogEntry]:
        return self.name_to_entry.get(name)

    def get_by_url(self, url: str) -> Optional[CatalogEntry]:
        normalized = _normalize_url(url)
        return self.url_to_entry.get(normalized)

    def search_by_name_substring(self, query: str) -> list[CatalogEntry]:
        """Case-insensitive substring search on product names."""
        q = query.lower()
        return [e for e in self.entries if q in e.name.lower()]
