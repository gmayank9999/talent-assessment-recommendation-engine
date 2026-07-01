"""
Recommendation post-processor.

Runs between LLM output and the validator. Handles normalization and
cleaning -- tasks that are conceptually distinct from correctness checks.

Responsibilities:
    - Normalize URLs (trailing slash consistency)
    - Map names to canonical catalog names
    - Derive correct test_type from catalog if the LLM got it wrong
    - Deduplicate by URL
    - Trim to max recommendation count
"""

import logging
from typing import Optional

from app.catalog import CatalogStore
from app.config import MAX_RECOMMENDATIONS

logger = logging.getLogger(__name__)


class RecommendationPostProcessor:
    """
    Cleans and normalizes raw LLM recommendation output
    before validation checks.
    """

    def __init__(self, catalog: CatalogStore):
        self._url_to_entry = catalog.url_to_entry
        self._name_to_entry = catalog.name_to_entry

    def process(self, raw_recs: Optional[list[dict]]) -> Optional[list[dict]]:
        """
        Normalize a list of raw recommendation dicts from the LLM.
        Returns None if the input is None or all items are invalid.
        """
        if raw_recs is None:
            return None

        processed = []
        seen_urls = set()

        for rec in raw_recs:
            name = rec.get("name", "").strip()
            url = rec.get("url", "").strip()
            test_type = rec.get("test_type", "").strip()

            if not url:
                continue

            # normalize trailing slash -- SHL URLs are consistent
            # but the LLM sometimes adds or drops the trailing slash
            url_with_slash = url if url.endswith("/") else url + "/"
            url_without_slash = url.rstrip("/")

            # try to match against catalog (both URL forms)
            entry = self._url_to_entry.get(url_with_slash)
            if entry is None:
                entry = self._url_to_entry.get(url_without_slash)

            if entry is not None:
                # use canonical values from the catalog
                canonical_name = entry.name
                canonical_url = entry.url
                canonical_type = entry.test_type
            elif name and name in self._name_to_entry:
                # URL didn't match but name did -- use the catalog URL
                entry = self._name_to_entry[name]
                canonical_name = entry.name
                canonical_url = entry.url
                canonical_type = entry.test_type
            else:
                # no catalog match -- pass through for the validator to catch
                canonical_name = name
                canonical_url = url_with_slash
                canonical_type = test_type

            # deduplicate by URL
            if canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)

            processed.append({
                "name": canonical_name,
                "url": canonical_url,
                "test_type": canonical_type,
            })

        # enforce max count
        processed = processed[:MAX_RECOMMENDATIONS]

        return processed if processed else None
