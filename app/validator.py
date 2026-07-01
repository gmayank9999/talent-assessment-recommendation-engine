"""
Recommendation validator -- the last-defense layer.

Runs after the post-processor. This is a hard gate: any recommendation
whose URL or name is not in the catalog gets stripped. This guarantees
the "items from catalog only" evaluation criterion is met even if the
LLM hallucinates an entry.

Checks applied (in order):
    1. URL must exist in the catalog URL set
    2. Name must exist in the catalog name set
    3. No duplicate URLs
    4. Result count must be 1-10
    5. If everything was stripped, return null instead of empty array
"""

import logging
from typing import Optional

from app.catalog import CatalogStore

logger = logging.getLogger(__name__)


class RecommendationValidator:
    """
    Hard gate that verifies every recommendation against the catalog
    before the response leaves the server.
    """

    def __init__(self, catalog: CatalogStore):
        self._url_set = catalog.url_set
        self._name_set = catalog.name_set

    def validate(
        self, recommendations: Optional[list[dict]]
    ) -> tuple[Optional[list[dict]], int]:
        """
        Validate a list of recommendation dicts.

        Returns:
            (validated_list, failure_count) where failure_count is the
            number of items that were stripped.
        """
        if recommendations is None:
            return None, 0

        original_count = len(recommendations)
        valid = []

        for rec in recommendations:
            url = rec.get("url", "")
            name = rec.get("name", "")

            # rule 1: URL must exist in catalog
            if url not in self._url_set:
                logger.warning("Stripped hallucinated URL: %s", url)
                continue

            # rule 2: name must exist in catalog
            if name not in self._name_set:
                logger.warning("Stripped hallucinated name: %s", name)
                continue

            valid.append(rec)

        # rule 3: remove duplicates by URL
        seen_urls = set()
        deduped = []
        for rec in valid:
            if rec["url"] not in seen_urls:
                deduped.append(rec)
                seen_urls.add(rec["url"])

        # rule 4: enforce 1-10 constraint
        if len(deduped) > 10:
            deduped = deduped[:10]

        # rule 5: if all items were hallucinated, return null
        failures = original_count - len(deduped)
        if len(deduped) == 0 and original_count > 0:
            logger.critical("All recommendations failed validation")
            return None, failures

        return deduped if deduped else None, failures
