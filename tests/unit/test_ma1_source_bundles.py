"""Raw-source persistence tests in memory only, without paid calls or real DBs."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from drugpilot.agents.evidence_semantics import abstract_context
from drugpilot.domain.models import EvidenceRef, RunStage
from drugpilot.orchestration.ma1_source_bundles import persist_source_bundle, load_source_bundles


class Harness:
    def __init__(self):
        self.snapshot = SimpleNamespace(run_id="unit", stage=RunStage.MA1, events=[])
        self.items = {}
        self.store = self
    async def get_snapshot(self, run_id):
        assert run_id == "unit"
        return self.snapshot
    async def commit_artifact_event(self, snapshot, identifier, kind, payload, event):
        self.items[identifier] = deepcopy(payload)
        self.snapshot.events.append(event)
        return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
    async def read_artifact(self, run_id, identifier, kind):
        return deepcopy(self.items.get(identifier))
    async def _trace_event(self, run_id, event_type, message, *, node, metadata):
        self.snapshot.events.append(SimpleNamespace(event_type=event_type, metadata=deepcopy(metadata)))


def _source(text="FLT3 was studied in AML cells. More source context remains unchanged."):
    excerpt = text.split(".")[0] + "."
    return EvidenceRef(source="PubMed", external_id="test-only", title="Synthetic source",
        supports="context", excerpt_hash=sha256(excerpt.encode()).hexdigest(),
        metadata={"excerpt": excerpt, "provenance_class": "external_live",
                  "content_hash_scope": "normalized_full_abstract",
                  "content_hash": sha256(text.encode()).hexdigest(),
                  "source_abstract_context": abstract_context(text)})


@pytest.mark.asyncio
async def test_raw_roundtrip_preserves_full_context_and_never_adds_interpretations(brief):
    from drugpilot.agents.evidence_semantics import ABSTRACT_CONTEXT_KEY
    evidence = _source()
    # Use the authoritative key rather than depending on a guessed label.
    evidence.metadata[ABSTRACT_CONTEXT_KEY] = evidence.metadata.pop("source_abstract_context")
    original = evidence.model_dump_json()
    harness = Harness()
    returned = await persist_source_bundle(harness, "unit", brief, SimpleNamespace(symbol="FLT3"), 1, [evidence])
    records, lineage = await load_source_bundles(harness, "unit", brief)
    assert evidence.model_dump_json() == original
    assert returned[0].model_dump_json() == original
    assert records["flt3"][0].model_dump_json() == original
    assert "semantic_interpretation" not in records["flt3"][0].metadata
    assert lineage[0]["new_model_calls"] == 0
    assert next(iter(harness.items.values()))["semantic_models"] == []


@pytest.mark.asyncio
async def test_changed_request_does_not_reuse_old_sources(brief):
    harness = Harness()
    await persist_source_bundle(harness, "unit", brief, SimpleNamespace(symbol="FLT3"), 1, [_source()])
    changed = brief.model_copy(update={"disease": "other disease"}, deep=True)
    assert await load_source_bundles(harness, "unit", changed) == ({}, [])


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["payload", "event_hash", "target"])
async def test_raw_bundle_or_event_tampering_fails_closed(brief, change):
    harness = Harness()
    await persist_source_bundle(harness, "unit", brief, SimpleNamespace(symbol="FLT3"), 1, [_source()])
    if change == "payload":
        next(iter(harness.items.values()))["evidence"][0]["title"] = "tampered"
    elif change == "event_hash":
        harness.snapshot.events[-1].metadata["evidence_bundle_sha256"] = "0" * 64
    else:
        harness.snapshot.events[-1].metadata["target"] = "other"
    with pytest.raises(ValueError, match="BUNDLE_INVALID"):
        await load_source_bundles(harness, "unit", brief)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["hash", "fixture"])
async def test_rehashed_invalid_source_cannot_become_reusable(brief, change):
    evidence = _source()
    if change == "hash":
        evidence.excerpt_hash = "0" * 64
    else:
        evidence.metadata["fixture"] = True
    harness = Harness()
    await persist_source_bundle(harness, "unit", brief, SimpleNamespace(symbol="FLT3"), 1, [evidence])
    with pytest.raises(ValueError, match="SOURCE_INVALID"):
        await load_source_bundles(harness, "unit", brief)
