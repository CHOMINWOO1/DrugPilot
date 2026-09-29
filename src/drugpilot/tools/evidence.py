from __future__ import annotations

from hashlib import sha256
from threading import Lock
from typing import Any, Protocol, Sequence

import httpx

from drugpilot.agents.ma1 import EvidenceRetriever, TargetSeed
from drugpilot.domain.models import EvidenceRef, ResearchBrief
from drugpilot.tools.http import get_scientific_http_client


class PublicEvidenceRetriever(EvidenceRetriever):
    """Live, read-only retrieval from Europe PMC and Open Targets.

    The adapter returns compact provenance records rather than full documents,
    keeping downstream LLM requests within the configured context cap.
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        *,
        client: "HTTPClient | None" = None,
        include_open_targets: bool = True,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.client = client or get_scientific_http_client()
        self.include_open_targets = include_open_targets
        self._cache: dict[tuple[str, str], tuple[EvidenceRef, ...]] = {}
        self._cache_lock = Lock()

    def retrieve(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
    ) -> Sequence[EvidenceRef]:
        del iteration
        cache_key = (brief.disease.casefold().strip(), seed.symbol.casefold())
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return tuple(
                item.model_copy(
                    update={
                        "metadata": {
                            **item.metadata,
                            "provenance_class": "external_cache",
                            "cache_hit": True,
                        }
                    },
                    deep=True,
                )
                for item in cached
            )

        records: list[EvidenceRef] = []
        failures: list[str] = []
        if "Europe PMC" in brief.data_policy.allowed_sources:
            try:
                records.extend(self._europe_pmc(brief, seed))
            except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                failures.append(f"Europe PMC: {type(exc).__name__}")
        if self.include_open_targets and "Open Targets" in brief.data_policy.allowed_sources:
            try:
                records.extend(self._open_targets(seed))
            except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                failures.append(f"Open Targets: {type(exc).__name__}")
        if not records:
            raise RuntimeError(
                f"No public evidence returned for {seed.symbol}"
                + (f" ({'; '.join(failures)})" if failures else "")
            )
        frozen = tuple(item.model_copy(deep=True) for item in records)
        with self._cache_lock:
            self._cache[cache_key] = frozen
        return tuple(item.model_copy(deep=True) for item in frozen)

    def _europe_pmc(self, brief: ResearchBrief, seed: TargetSeed) -> list[EvidenceRef]:
        query = f'"{seed.symbol}" AND "{brief.disease}"'
        response = self.client.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "pageSize": 4, "resultType": "core"},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result", [])
        evidence: list[EvidenceRef] = []
        for item in results:
            pmid = item.get("pmid")
            pmcid = item.get("pmcid")
            external_id = pmid or pmcid or item.get("id")
            if not external_id:
                continue
            title = item.get("title") or f"Europe PMC record {external_id}"
            article_path = (
                f"MED/{pmid}" if pmid else f"PMC/{pmcid}" if pmcid else str(external_id)
            )
            evidence.append(
                EvidenceRef(
                    evidence_id=f"ev_epmc_{sha256(str(external_id).encode()).hexdigest()[:12]}",
                    source="Europe PMC",
                    external_id=str(external_id),
                    title=title,
                    url=f"https://europepmc.org/article/{article_path}",
                    supports="support",
                    excerpt_hash=sha256(title.encode("utf-8")).hexdigest(),
                    metadata={
                        "publication_year": item.get("pubYear"),
                        "journal": item.get("journalTitle"),
                        "cited_by_count": item.get("citedByCount"),
                        "live": True,
                        "mock": False,
                        "provenance_class": "external_live",
                        "query": query,
                        "evidence_scope": "target-disease co-mention; not causal proof",
                    },
                )
            )
        return evidence

    def _open_targets(self, seed: TargetSeed) -> list[EvidenceRef]:
        query = {
            "query": (
                "query Search($q:String!){search(queryString:$q,page:{index:0,size:3})"
                "{hits{id name entity description}}}"
            ),
            "variables": {"q": seed.symbol},
        }
        response = self.client.post(
            "https://api.platform.opentargets.org/api/v4/graphql",
            json=query,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        if response.status_code >= 400:
            return []
        try:
            hits = response.json().get("data", {}).get("search", {}).get("hits", [])
        except (ValueError, AttributeError):
            return []
        evidence: list[EvidenceRef] = []
        for hit in hits:
            if hit.get("entity") != "target":
                continue
            external_id = hit.get("id")
            if not external_id:
                continue
            title = f"Open Targets target record: {hit.get('name') or seed.symbol}"
            evidence.append(
                EvidenceRef(
                    evidence_id=f"ev_ot_{sha256(str(external_id).encode()).hexdigest()[:12]}",
                    source="Open Targets",
                    external_id=str(external_id),
                    title=title,
                    url=f"https://platform.opentargets.org/target/{external_id}",
                    supports="context",
                    excerpt_hash=sha256(title.encode("utf-8")).hexdigest(),
                    metadata={
                        "description": hit.get("description"),
                        "live": True,
                        "mock": False,
                        "provenance_class": "external_live",
                        "query": seed.symbol,
                        "evidence_scope": "target identity context",
                    },
                )
            )
        return evidence


class HTTPClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...


class ResilientEvidenceRetriever(EvidenceRetriever):
    """Combine keyless public evidence with explicit fixture safeguards.

    Public records provide real, traceable retrieval results.  The deterministic
    counter record is retained only as an audit-workflow placeholder because a
    keyword search result cannot honestly be relabelled as counter-evidence.
    If the public services are unavailable, the complete fixture pair keeps the
    local workflow testable and records the fallback in provenance metadata.
    """

    def __init__(
        self,
        public: EvidenceRetriever | None = None,
        fallback: EvidenceRetriever | None = None,
        *,
        fixture_fallback_enabled: bool = True,
    ) -> None:
        from drugpilot.agents.ma1 import MockEvidenceRetriever

        self.public = public or PublicEvidenceRetriever()
        self.fallback = fallback or MockEvidenceRetriever()
        self.fixture_fallback_enabled = bool(fixture_fallback_enabled)

    def retrieve(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
    ) -> Sequence[EvidenceRef]:
        fixtures = (
            list(self.fallback.retrieve(brief, seed, iteration))
            if self.fixture_fallback_enabled
            else []
        )
        try:
            public_records = list(self.public.retrieve(brief, seed, iteration))
        except Exception as exc:  # public-network errors must not break a local test run
            if not self.fixture_fallback_enabled:
                raise
            reason = type(exc).__name__
            return tuple(
                self._mark_fixture(item, fallback_reason=reason)
                for item in fixtures
            )

        if not public_records:
            if not self.fixture_fallback_enabled:
                return ()
            return tuple(
                self._mark_fixture(item, fallback_reason="empty_public_result")
                for item in fixtures
            )

        # Keep one explicitly labelled fixture counter.  It exercises the G0
        # counter-evidence path without pretending that a co-mention search is a
        # scientific refutation.  A fixture support record is used only if the
        # live adapter returned context records but no support record.
        if not self.fixture_fallback_enabled:
            return tuple(public_records)

        supplements = [
            self._mark_fixture(item, fallback_reason=None)
            for item in fixtures
            if item.supports == "counter"
        ]
        if not any(item.supports == "support" for item in public_records):
            supplements.extend(
                self._mark_fixture(item, fallback_reason="no_public_support_record")
                for item in fixtures
                if item.supports == "support"
            )
        return tuple(public_records + supplements)

    @staticmethod
    def _mark_fixture(
        item: EvidenceRef,
        *,
        fallback_reason: str | None,
    ) -> EvidenceRef:
        role = (
            "counter_evidence_placeholder"
            if item.supports == "counter"
            else "support_fallback"
        )
        metadata = {
            **item.metadata,
            "live": False,
            "mock": True,
            "provenance_class": "fixture_fallback",
            "fallback_role": role,
        }
        if fallback_reason:
            metadata["fallback_reason"] = fallback_reason
        return item.model_copy(update={"metadata": metadata}, deep=True)


__all__ = ["PublicEvidenceRetriever", "ResilientEvidenceRetriever"]
