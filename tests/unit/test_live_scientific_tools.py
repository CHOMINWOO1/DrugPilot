from __future__ import annotations

from hashlib import sha256
from importlib.util import find_spec
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest

from drugpilot.agents.ma3 import (
    MA3AssuranceAgent,
    _paper_v6_source_inventory_valid,
)
from drugpilot.domain.models import ActivityMeasurement, TargetCandidate
from drugpilot.tools.admet_ai_adapter import ADMETAIAdapter
from drugpilot.tools.brics_worker import generate_brics
from drugpilot.tools.executable import (
    ExecutableRunner,
    ScientificDataUnavailable,
    ScientificToolTimeout,
    ScientificToolUnavailable,
    bounded_excerpt,
    sanitize_argument,
    sanitize_arguments,
)
from drugpilot.tools.ligand_prep_worker import prepare_ligand
from drugpilot.tools.live_ma2_gateway import (
    LiveScientificGateway,
    NoMeasuredLigandCandidates,
    _assay_engagement_assessment,
    _canonical_sha256,
    _normalize_activity,
    _primary_parenthetical_binding,
)
from drugpilot.tools.live_repurposing_gateway import (
    PAIRED_V4_REQUIRED_POLICY,
    PublicRepurposingGateway,
)
from drugpilot.tools.primary_abstract_paper_native_search_v6 import (
    PAPER_NATIVE_V6_POLICY_TOKEN,
)
from drugpilot.tools.pose_rmsd import calculate_redocking_rmsd
from drugpilot.tools.vina_adapter import DockingProfileRegistry, VinaDockingAdapter


class _Response:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        if "page_meta" not in payload:
            for collection_key in ("activities", "mechanisms"):
                collection = payload.get(collection_key)
                if isinstance(collection, list):
                    payload["page_meta"] = {
                        "total_count": len(collection),
                    }
                    break
        self._payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload, sort_keys=True).encode()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, object]:
        return self._payload


class _ChEMBLClient:
    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("esummary.fcgi"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            requested = [
                item for item in str(params["id"]).split(",") if item
            ]
            if hasattr(self, "pubmed_calls"):
                self.pubmed_calls += 1
            if hasattr(self, "pubmed_requested_pmids"):
                self.pubmed_requested_pmids.append(requested)
            if getattr(self, "pubmed_unavailable", False):
                raise httpx.ReadTimeout("PubMed type lookup timed out")
            missing = set(getattr(self, "pubmed_missing_pmids", []))
            retained = [pmid for pmid in requested if pmid not in missing]
            result_uids = getattr(self, "pubmed_result_uids", retained)
            publication_types = getattr(
                self,
                "pubmed_publication_types",
                ["Journal Article"],
            )
            return _Response(
                {
                    "result": {
                        "uids": result_uids,
                        **{
                            pmid: (
                                "invalid-record"
                                if getattr(
                                    self, "pubmed_record_nonmapping", False
                                )
                                else {
                                    **(
                                        {}
                                        if getattr(
                                            self,
                                            "pubmed_omit_record_uid",
                                            False,
                                        )
                                        else {"uid": pmid}
                                    ),
                                    "pubtype": (
                                        "invalid-types"
                                        if getattr(
                                            self,
                                            "pubmed_pubtype_nonlist",
                                            False,
                                        )
                                        else list(publication_types)
                                    ),
                                }
                            )
                            for pmid in retained
                        },
                    }
                }
            )
        if url.endswith("status.json"):
            return _Response({"chembl_db_version": "ChEMBL_TEST"})
        if url.endswith("target.json"):
            return _Response(
                {
                    "targets": [
                        {
                            "target_chembl_id": "CHEMBL_TARGET",
                            "pref_name": "Example target",
                            "target_type": "SINGLE PROTEIN",
                            "organism": "Homo sapiens",
                            "target_components": [{"accession": "P12345"}],
                        }
                    ]
                }
            )
        if url.endswith("mechanism.json"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            assert params == {
                "target_chembl_id": "CHEMBL_TARGET",
                "action_type": "INHIBITOR",
                "max_phase__gte": 1,
                "direct_interaction": 1,
                "molecular_mechanism": 1,
                "order_by": "molecule_chembl_id",
                "limit": 80,
            }
            return _Response(
                {
                    "mechanisms": [
                        {
                            "mec_id": index,
                            "record_id": 1000 + index,
                            "action_type": "INHIBITOR",
                            "direct_interaction": 1,
                            "max_phase": phase,
                            "mechanism_of_action": "Example target inhibitor",
                            "mechanism_refs": [
                                {
                                    "ref_type": "PubMed",
                                    "ref_id": str(20000000 + index),
                                }
                            ],
                            "molecular_mechanism": 1,
                            "molecule_chembl_id": chembl_id,
                            "parent_molecule_chembl_id": chembl_id,
                            "target_chembl_id": "CHEMBL_TARGET",
                            "variant_sequence": None,
                        }
                        for index, (chembl_id, phase) in enumerate(
                            (("CHEMBL1", 4), ("CHEMBL2", 2)),
                            start=1,
                        )
                    ]
                }
            )
        if url.endswith("activity.json"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            requested_parent = str(
                params.get("parent_molecule_chembl_id") or ""
            )
            activities = [
                        {
                            "activity_id": 11,
                            "data_validity_comment": None,
                            "molecule_chembl_id": "CHEMBL1",
                            "parent_molecule_chembl_id": "CHEMBL1",
                            "target_chembl_id": "CHEMBL_TARGET",
                            "assay_type": "B",
                            "assay_chembl_id": "CHEMBL_ASSAY_1",
                            "standard_type": "IC50",
                            "standard_relation": "=",
                            "standard_value": "4.2",
                            "standard_units": "nM",
                            "pchembl_value": "8.3",
                            "bao_label": "single protein format",
                        },
                        {
                            "activity_id": 12,
                            "data_validity_comment": None,
                            "molecule_chembl_id": "CHEMBL2",
                            "parent_molecule_chembl_id": "CHEMBL2",
                            "target_chembl_id": "CHEMBL_TARGET",
                            "assay_type": "B",
                            "assay_chembl_id": "CHEMBL_ASSAY_2",
                            "standard_type": "IC50",
                            "standard_relation": "<",
                            "standard_value": "9.1",
                            "standard_units": "nM",
                            "pchembl_value": "8.0",
                            "bao_label": "single protein format",
                        },
                    ]
            if requested_parent:
                activities = [
                    row
                    for row in activities
                    if row["parent_molecule_chembl_id"] == requested_parent
                ]
            return _Response(
                {
                    "activities": activities,
                }
            )
        if url.endswith("assay.json"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            requested = [
                item
                for item in str(params["assay_chembl_id__in"]).split(",")
                if item
            ]
            assert 1 <= int(params["limit"]) <= 80
            assert int(params["limit"]) == len(requested)
            return _Response(
                {
                    "assays": [
                        {
                            "assay_chembl_id": assay_id,
                            "target_chembl_id": "CHEMBL_TARGET",
                            "document_chembl_id": f"DOC_{assay_id}",
                            "assay_type": "B",
                            "assay_category": "confirmatory",
                            "assay_organism": "Homo sapiens",
                            "bao_format": "BAO_0000357",
                            "bao_label": "single protein format",
                            "confidence_score": 9,
                            "description": "Exact biochemical activity assay.",
                            "variant_sequence": None,
                            "assay_parameters": [],
                        }
                        for assay_id in requested
                    ]
                }
            )
        if url.endswith("molecule.json"):
            return _Response(
                {
                    "molecules": [
                        {
                            "molecule_chembl_id": "CHEMBL1",
                            "pref_name": "Drug one",
                            "max_phase": 4,
                            "molecule_type": "Small molecule",
                            "molecule_structures": {
                                "canonical_smiles": "CCOc1ccccc1",
                                "standard_inchi_key": "TEST1",
                            },
                        },
                        {
                            "molecule_chembl_id": "CHEMBL2",
                            "pref_name": "Drug two",
                            "max_phase": 2,
                            "molecule_type": "Small molecule",
                            "molecule_structures": {
                                "canonical_smiles": "CCN(C)Cc1ccccc1",
                                "standard_inchi_key": "TEST2",
                            },
                        },
                    ]
                }
            )
        if url.endswith("drug_warning.json"):
            return _Response(
                {
                    "drug_warnings": [
                        {
                            "molecule_chembl_id": "CHEMBL2",
                            "warning_type": "Withdrawn",
                            "warning_class": "safety",
                            "warning_country": "Test jurisdiction",
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected URL: {url}")


class _NoExactTargetClient(_ChEMBLClient):
    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("target.json"):
            return _Response(
                {
                    "targets": [
                        {
                            "target_chembl_id": "CHEMBL_OTHER",
                            "pref_name": "Different target",
                            "target_type": "SINGLE PROTEIN",
                            "organism": "Homo sapiens",
                            "target_components": [{"accession": "P99999"}],
                        }
                    ]
                }
            )
        return super().get(url, **kwargs)


class _TargetAccessionFallbackClient(_ChEMBLClient):
    def __init__(
        self,
        *,
        xref_ids: tuple[str, ...] = ("CHEMBL_TARGET",),
        returned_accession: str = "P12345",
        component_accessions: tuple[str, ...] = ("P12345",),
        return_target: bool = True,
    ) -> None:
        self.xref_ids = xref_ids
        self.returned_accession = returned_accession
        self.component_accessions = component_accessions
        self.return_target = return_target
        self.accession_query_calls = 0
        self.uniprot_calls = 0
        self.target_id_calls = 0

    @staticmethod
    def _http_500(url: str) -> httpx.HTTPStatusError:
        request = httpx.Request("GET", url)
        response = httpx.Response(500, request=request)
        return httpx.HTTPStatusError(
            "exact accession target lookup unavailable",
            request=request,
            response=response,
        )

    def get(self, url: str, **kwargs: object) -> _Response:
        params = kwargs.get("params")
        assert isinstance(params, dict)
        if url.endswith("target.json") and "target_components__accession" in params:
            self.accession_query_calls += 1
            assert params == {
                "target_components__accession": "P12345",
                "limit": 30,
            }
            raise self._http_500(url)
        if url == "https://rest.uniprot.org/uniprotkb/P12345.json":
            self.uniprot_calls += 1
            assert params == {}
            return _Response(
                {
                    "primaryAccession": self.returned_accession,
                    "uniProtKBCrossReferences": [
                        {"database": "ChEMBL", "id": item}
                        for item in self.xref_ids
                    ],
                }
            )
        if url.endswith("target.json") and "target_chembl_id" in params:
            self.target_id_calls += 1
            assert params == {
                "target_chembl_id": "CHEMBL_TARGET",
                "limit": 1,
            }
            targets = []
            if self.return_target:
                targets.append(
                    {
                        "target_chembl_id": "CHEMBL_TARGET",
                        "pref_name": "Example target",
                        "target_type": "SINGLE PROTEIN",
                        "organism": "Homo sapiens",
                        "target_components": [
                            {"accession": accession}
                            for accession in self.component_accessions
                        ],
                    }
                )
            return _Response({"targets": targets})
        return super().get(url, **kwargs)


class _CellBasedChEMBLClient(_ChEMBLClient):
    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("activity.json"):
            return _Response(
                {
                    "activities": [
                        {
                            "activity_id": 11,
                            "data_validity_comment": None,
                            "molecule_chembl_id": "CHEMBL1",
                            "parent_molecule_chembl_id": "CHEMBL1",
                            "target_chembl_id": "CHEMBL_TARGET",
                            "assay_type": "B",
                            "assay_chembl_id": "CHEMBL_ASSAY_1",
                            "standard_type": "IC50",
                            "standard_relation": "=",
                            "standard_value": "4.2",
                            "standard_units": "nM",
                            "pchembl_value": "8.3",
                            "bao_label": "cell-based format",
                        }
                    ]
                }
            )
        return super().get(url, **kwargs)


class _UnspecifiedAssayChEMBLClient(_ChEMBLClient):
    def get(self, url: str, **kwargs: object) -> _Response:
        response = super().get(url, **kwargs)
        if not url.endswith("activity.json"):
            return response
        payload = response.json()
        activities = payload["activities"]
        assert isinstance(activities, list)
        for row in activities:
            assert isinstance(row, dict)
            row["bao_label"] = "unknown assay format"
        return _Response(payload)


class _DataValidityChEMBLClient(_ChEMBLClient):
    def __init__(
        self,
        *,
        value: object,
        include_key: bool = True,
    ) -> None:
        self.value = value
        self.include_key = include_key

    def get(self, url: str, **kwargs: object) -> _Response:
        response = super().get(url, **kwargs)
        if not url.endswith("activity.json"):
            return response
        payload = response.json()
        activities = payload["activities"]
        assert isinstance(activities, list)
        for activity in activities:
            assert isinstance(activity, dict)
            if self.include_key:
                activity["data_validity_comment"] = self.value
            else:
                activity.pop("data_validity_comment", None)
        return _Response(payload)


class _PchemblOrderingClient(_ChEMBLClient):
    def __init__(self, *, first: float, second: float) -> None:
        self.values = (first, second)

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("activity.json"):
            return _Response(
                {
                    "activities": [
                        {
                            "activity_id": activity_id,
                            "data_validity_comment": None,
                            "molecule_chembl_id": f"CHEMBL{index}",
                            "parent_molecule_chembl_id": f"CHEMBL{index}",
                            "target_chembl_id": "CHEMBL_TARGET",
                            "assay_type": "B",
                            "assay_chembl_id": f"CHEMBL_ASSAY_{index}",
                            "standard_type": "IC50",
                            "standard_relation": "=",
                            "standard_value": str(index),
                            "standard_units": "nM",
                            "pchembl_value": str(pchembl),
                            "bao_label": "single protein format",
                        }
                        for index, (activity_id, pchembl) in enumerate(
                            ((11, self.values[0]), (12, self.values[1])),
                            start=1,
                        )
                    ]
                }
            )
        if url.endswith("drug_warning.json"):
            return _Response({"drug_warnings": []})
        return super().get(url, **kwargs)


class _PrimaryDocumentChEMBLClient(_ChEMBLClient):
    abstract = (
        "Background context does not identify the tested pair. "
        "Drug one inhibited GENE with an IC50 of 4.2 nM in the reported assay. "
        "Additional experiments evaluated unrelated pharmacology."
    )

    def __init__(
        self,
        *,
        abstract: str | None = abstract,
        document_unavailable: bool = False,
        doc_type: str = "PUBLICATION",
        title: str = "Primary activity study",
        publication_types: list[str] | None = None,
        pubmed_publication_types: list[str] | None = None,
        pubmed_id: int | None = 12345678,
        pubmed_missing: bool = False,
        pubmed_unavailable: bool = False,
        pubmed_result_uids: list[str] | None = None,
        pubmed_omit_record_uid: bool = False,
        pubmed_record_nonmapping: bool = False,
        pubmed_pubtype_nonlist: bool = False,
        doi: str | None = "10.1000/example.primary",
        molecule_synonyms: list[str] | None = None,
    ) -> None:
        self.abstract_value = abstract
        self.document_unavailable = document_unavailable
        self.doc_type = doc_type
        self.title = title
        self.publication_types = publication_types
        self.pubmed_publication_types = (
            ["Journal Article"]
            if pubmed_publication_types is None
            else pubmed_publication_types
        )
        self.pubmed_id = pubmed_id
        self.pubmed_missing_pmids = (
            [str(pubmed_id)] if pubmed_missing and pubmed_id is not None else []
        )
        self.pubmed_unavailable = pubmed_unavailable
        if pubmed_result_uids is not None:
            self.pubmed_result_uids = list(pubmed_result_uids)
        self.pubmed_omit_record_uid = pubmed_omit_record_uid
        self.pubmed_record_nonmapping = pubmed_record_nonmapping
        self.pubmed_pubtype_nonlist = pubmed_pubtype_nonlist
        self.doi = doi
        self.pubmed_calls = 0
        self.pubmed_requested_pmids: list[list[str]] = []
        self.molecule_synonyms = molecule_synonyms
        self.document_calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("activity.json"):
            return _Response(
                {
                    "activities": [
                        {
                            "activity_id": 11,
                            "data_validity_comment": None,
                            "molecule_chembl_id": "CHEMBL1",
                            "parent_molecule_chembl_id": "CHEMBL1",
                            "target_chembl_id": "CHEMBL_TARGET",
                            "assay_type": "B",
                            "assay_chembl_id": "CHEMBL_ASSAY_1",
                            "document_chembl_id": "CHEMBL900001",
                            "standard_type": "IC50",
                            "standard_relation": "=",
                            "standard_value": "4.2",
                            "standard_units": "nM",
                            "pchembl_value": "8.3",
                            "bao_label": "single protein format",
                        }
                    ]
                }
            )
        if url.endswith("document.json"):
            self.document_calls += 1
            if self.document_unavailable:
                raise httpx.ReadTimeout("document lookup timed out")
            params = kwargs.get("params")
            assert isinstance(params, dict)
            assert params["document_chembl_id__in"] == "CHEMBL900001"
            return _Response(
                {
                    "documents": [
                        {
                            "document_chembl_id": "CHEMBL900001",
                            "doc_type": self.doc_type,
                            "pubmed_id": self.pubmed_id,
                            "doi": self.doi,
                            "title": self.title,
                            "abstract": self.abstract_value,
                            "publication_types": self.publication_types,
                        }
                    ]
                }
            )
        if url.endswith("molecule.json") and self.molecule_synonyms is not None:
            payload = super().get(url, **kwargs).json()
            molecules = payload["molecules"]
            assert isinstance(molecules, list)
            molecules[0]["molecule_synonyms"] = [
                {"molecule_synonym": synonym}
                for synonym in self.molecule_synonyms
            ]
            return _Response(payload)
        return super().get(url, **kwargs)


class _MalformedSuccessfulSourceRowClient(_PrimaryDocumentChEMBLClient):
    def __init__(self, source: str) -> None:
        super().__init__()
        self.malformed_source = source

    def get(self, url: str, **kwargs: object) -> _Response:
        if self.malformed_source == "warning" and url.endswith(
            "drug_warning.json"
        ):
            return _Response({"drug_warnings": [None]})
        if self.malformed_source == "assay" and url.endswith("assay.json"):
            return _Response({"assays": [None]})
        if self.malformed_source == "document" and url.endswith(
            "document.json"
        ):
            return _Response({"documents": [None]})
        return super().get(url, **kwargs)


class _TopKPrimaryDocumentChEMBLClient(_ChEMBLClient):
    def __init__(self, *, row_count: int, valid_rank: int) -> None:
        self.row_count = row_count
        self.valid_rank = valid_rank
        self.requested_document_ids: list[str] = []
        self.pubmed_calls = 0
        self.pubmed_requested_pmids: list[list[str]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("activity.json"):
            return _Response(
                {
                    "activities": [
                        {
                            "activity_id": 10 + rank,
                            "data_validity_comment": None,
                            "molecule_chembl_id": "CHEMBL1",
                            "parent_molecule_chembl_id": "CHEMBL1",
                            "target_chembl_id": "CHEMBL_TARGET",
                            "assay_type": "B",
                            "assay_chembl_id": f"CHEMBL_ASSAY_{rank}",
                            "document_chembl_id": f"CHEMBL{900000 + rank}",
                            "standard_type": "IC50",
                            "standard_relation": "=",
                            "standard_value": "4.2",
                            "standard_units": "nM",
                            "pchembl_value": str(10.0 - rank),
                            "bao_label": "single protein format",
                        }
                        for rank in range(1, self.row_count + 1)
                    ]
                }
            )
        if url.endswith("document.json"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            self.requested_document_ids = str(
                params["document_chembl_id__in"]
            ).split(",")
            return _Response(
                {
                    "documents": [
                        {
                            "document_chembl_id": document_id,
                            "doc_type": (
                                "PUBLICATION"
                                if int(document_id.removeprefix("CHEMBL"))
                                - 900000
                                == self.valid_rank
                                else "PATENT"
                            ),
                            "pubmed_id": 12345000
                            + (
                                int(document_id.removeprefix("CHEMBL"))
                                - 900000
                            ),
                            "doi": None,
                            "title": "Primary activity study",
                            "abstract": (
                                "Drug one inhibited GENE with an IC50 of 4.2 nM."
                            ),
                        }
                        for document_id in self.requested_document_ids
                    ]
                }
            )
        return super().get(url, **kwargs)


class _AssayTimeoutChEMBLClient(_PrimaryDocumentChEMBLClient):
    def __init__(self) -> None:
        super().__init__()
        self.assay_calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("assay.json"):
            self.assay_calls += 1
            raise httpx.ReadTimeout("assay lookup timed out")
        return super().get(url, **kwargs)


class _AssayBatchClient(_ChEMBLClient):
    def __init__(self) -> None:
        self.assay_batches: list[list[str]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("assay.json"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            self.assay_batches.append(
                str(params["assay_chembl_id__in"]).split(",")
            )
        return super().get(url, **kwargs)


class _AliasConflictPrimaryDocumentChEMBLClient(_ChEMBLClient):
    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("activity.json"):
            return _Response(
                {
                    "activities": [
                        {
                            "activity_id": 11,
                            "data_validity_comment": None,
                            "molecule_chembl_id": "CHEMBL1",
                            "parent_molecule_chembl_id": "CHEMBL1",
                            "target_chembl_id": "CHEMBL_TARGET",
                            "assay_type": "B",
                            "document_chembl_id": "CHEMBL900001",
                            "standard_type": "IC50",
                            "standard_relation": "=",
                            "standard_value": "4.2",
                            "standard_units": "nM",
                            "pchembl_value": "8.3",
                            "bao_label": "single protein format",
                        },
                        {
                            "activity_id": 12,
                            "data_validity_comment": None,
                            "molecule_chembl_id": "CHEMBL2",
                            "parent_molecule_chembl_id": "CHEMBL2",
                            "target_chembl_id": "CHEMBL_TARGET",
                            "assay_type": "B",
                            "document_chembl_id": "CHEMBL900002",
                            "standard_type": "IC50",
                            "standard_relation": "=",
                            "standard_value": "5.1",
                            "standard_units": "nM",
                            "pchembl_value": "8.2",
                            "bao_label": "single protein format",
                        },
                    ]
                }
            )
        if url.endswith("molecule.json"):
            payload = super().get(url, **kwargs).json()
            molecules = payload["molecules"]
            assert isinstance(molecules, list)
            for molecule in molecules:
                molecule["molecule_synonyms"] = [
                    {"molecule_synonym": "Shared research code"}
                ]
            return _Response(payload)
        if url.endswith("document.json"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            requested = set(
                str(params["document_chembl_id__in"]).split(",")
            )
            documents = [
                {
                    "document_chembl_id": "CHEMBL900001",
                    "doc_type": "PUBLICATION",
                    "pubmed_id": 12345671,
                    "title": "Candidate one activity",
                    "abstract": (
                        "Drug one inhibited GENE with an IC50 of 4.2 nM."
                    ),
                },
                {
                    "document_chembl_id": "CHEMBL900002",
                    "doc_type": "PUBLICATION",
                    "pubmed_id": 12345672,
                    "title": "Candidate two activity",
                    "abstract": (
                        "Drug two inhibited GENE with an IC50 of 5.1 nM."
                    ),
                },
            ]
            return _Response(
                {
                    "documents": [
                        item
                        for item in documents
                        if item["document_chembl_id"] in requested
                    ]
                }
            )
        return super().get(url, **kwargs)


class _CompetingPrimaryDocumentChEMBLClient(
    _AliasConflictPrimaryDocumentChEMBLClient
):
    def __init__(self, *, abstract: str) -> None:
        self.abstract = abstract

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("molecule.json"):
            return _ChEMBLClient.get(self, url, **kwargs)
        if url.endswith("document.json"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            requested = set(
                str(params["document_chembl_id__in"]).split(",")
            )
            documents = [
                        {
                            "document_chembl_id": "CHEMBL900001",
                            "doc_type": "PUBLICATION",
                            "pubmed_id": 12345671,
                            "title": "Candidate one activity",
                            "abstract": self.abstract,
                        },
                        {
                            "document_chembl_id": "CHEMBL900002",
                            "doc_type": "PUBLICATION",
                            "pubmed_id": 12345672,
                            "title": "Candidate two activity",
                            "abstract": (
                                "Drug two inhibited GENE with an IC50 of 5.1 nM."
                            ),
                        },
            ]
            return _Response(
                {
                    "documents": [
                        item
                        for item in documents
                        if item["document_chembl_id"] in requested
                    ]
                }
            )
        return super().get(url, **kwargs)


class _MissingPoolMoleculePrimaryClient(
    _CompetingPrimaryDocumentChEMBLClient
):
    def __init__(self) -> None:
        super().__init__(
            abstract="Drug one inhibited GENE with an IC50 of 4.2 nM."
        )

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("molecule.json"):
            payload = _ChEMBLClient.get(self, url, **kwargs).json()
            molecules = payload["molecules"]
            assert isinstance(molecules, list)
            return _Response({"molecules": molecules[:1]})
        return super().get(url, **kwargs)


class _ReadTimeoutClient:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls += 1
        raise httpx.ReadTimeout("upstream read timed out")


class _TimeoutThenSuccessClient(_ChEMBLClient):
    def __init__(self) -> None:
        self.status_calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("status.json"):
            self.status_calls += 1
            if self.status_calls == 1:
                raise httpx.ReadTimeout("one transient upstream timeout")
        return super().get(url, **kwargs)


class _StatusEndpoint500Client(_ChEMBLClient):
    def __init__(self) -> None:
        self.status_calls = 0
        self.target_calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("status.json"):
            self.status_calls += 1
            request = httpx.Request("GET", url)
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError(
                "metadata endpoint unavailable",
                request=request,
                response=response,
            )
        if url.endswith("target.json"):
            self.target_calls += 1
        return super().get(url, **kwargs)


class _StatusEndpointReadTimeoutClient(_ChEMBLClient):
    def __init__(self) -> None:
        self.status_calls = 0
        self.target_calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("status.json"):
            self.status_calls += 1
            raise httpx.ReadTimeout("release metadata endpoint timed out")
        if url.endswith("target.json"):
            self.target_calls += 1
        return super().get(url, **kwargs)


class _MalformedStatusVersionClient(_ChEMBLClient):
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("status.json"):
            return _Response({"chembl_db_version": self.value})
        return super().get(url, **kwargs)


class _OrderedActivity500Client(_ChEMBLClient):
    def __init__(self) -> None:
        self.ordered_calls = 0
        self.unordered_calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("activity.json"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            if "order_by" in params:
                self.ordered_calls += 1
                request = httpx.Request("GET", url)
                response = httpx.Response(500, request=request)
                raise httpx.HTTPStatusError(
                    "server-side activity ordering unavailable",
                    request=request,
                    response=response,
                )
            self.unordered_calls += 1
        return super().get(url, **kwargs)


class _BoundedActivityPaginationClient(_ChEMBLClient):
    def __init__(
        self,
        *,
        fail_second_page: bool = False,
        fail_all_pages: bool = False,
    ) -> None:
        self.fail_second_page = fail_second_page
        self.fail_all_pages = fail_all_pages
        self.ordered_calls = 0
        self.page_params: list[dict[str, object]] = []

    @staticmethod
    def _http_500(url: str) -> httpx.HTTPStatusError:
        request = httpx.Request("GET", url)
        response = httpx.Response(500, request=request)
        return httpx.HTTPStatusError(
            "large activity request unavailable",
            request=request,
            response=response,
        )

    def get(self, url: str, **kwargs: object) -> _Response:
        if not url.endswith("activity.json"):
            return super().get(url, **kwargs)
        params = kwargs.get("params")
        assert isinstance(params, dict)
        if "order_by" in params:
            self.ordered_calls += 1
            raise self._http_500(url)
        copied = dict(params)
        self.page_params.append(copied)
        offset = int(copied["offset"])
        if self.fail_all_pages or (self.fail_second_page and offset == 20):
            raise httpx.ReadTimeout("bounded activity page timed out")
        if offset == 0:
            activities = []
            for index in range(20):
                activities.append(
                    {
                        "activity_id": 100 + index,
                        "data_validity_comment": None,
                        "molecule_chembl_id": (
                            "CHEMBL1" if index % 2 == 0 else "CHEMBL2"
                        ),
                        "parent_molecule_chembl_id": (
                            "CHEMBL1" if index % 2 == 0 else "CHEMBL2"
                        ),
                        "target_chembl_id": "CHEMBL_TARGET",
                        "assay_type": "B",
                        "standard_type": "IC50" if index % 2 == 0 else "Ki",
                        "standard_relation": "=",
                        "standard_value": str(5 + index),
                        "standard_units": "nM",
                        "pchembl_value": str(8.5 - index / 100),
                        "bao_label": "single protein format",
                    }
                )
            return _Response(
                {
                    "activities": activities,
                    "page_meta": {"total_count": 21},
                }
            )
        return _Response(
            {
                "activities": [
                    {
                        "activity_id": 121,
                        "data_validity_comment": None,
                        "molecule_chembl_id": "CHEMBL1",
                        "parent_molecule_chembl_id": "CHEMBL1",
                        "target_chembl_id": "CHEMBL_TARGET",
                        "assay_type": "B",
                        "standard_type": "IC50",
                        "standard_relation": "=",
                        "standard_value": "4.2",
                        "standard_units": "nM",
                        "pchembl_value": "8.6",
                        "bao_label": "single protein format",
                    }
                ],
                "page_meta": {"total_count": 21},
            }
        )


class _CountingChEMBLClient(_ChEMBLClient):
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls[url] = self.calls.get(url, 0) + 1
        return super().get(url, **kwargs)


class _ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _BudgetConsumingTimeoutClient:
    def __init__(self, clock: _ManualClock) -> None:
        self.clock = clock
        self.calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls += 1
        self.clock.advance(0.6)
        raise httpx.ReadTimeout("slow upstream read")


class _WarningIsolationClient(_ChEMBLClient):
    def __init__(self, *, unresolved_id: str | None = None) -> None:
        self.unresolved_id = unresolved_id
        self.batch_warning_calls = 0
        self.individual_warning_calls: list[str] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        if not url.endswith("drug_warning.json"):
            return super().get(url, **kwargs)
        params = kwargs.get("params")
        assert isinstance(params, dict)
        requested = str(params["molecule_chembl_id__in"])
        if "," in requested:
            self.batch_warning_calls += 1
            raise httpx.ReadTimeout("batch warning lookup timed out")
        self.individual_warning_calls.append(requested)
        if requested == self.unresolved_id:
            raise httpx.ReadTimeout("individual warning lookup timed out")
        warnings = []
        if requested == "CHEMBL2":
            warnings.append(
                {
                    "molecule_chembl_id": "CHEMBL2",
                    "warning_type": "Withdrawn",
                    "warning_class": "safety",
                    "warning_country": "Test jurisdiction",
                }
            )
        return _Response({"drug_warnings": warnings})


class _LargeBoundedRepurposingClient(_ChEMBLClient):
    def __init__(
        self,
        *,
        raw_count: int = 79,
        reverse_pchembl: bool = False,
        fail_warning_batch: bool = False,
        unresolved_warning_ids: set[str] | None = None,
    ) -> None:
        self.raw_count = raw_count
        self.reverse_pchembl = reverse_pchembl
        self.fail_warning_batch = fail_warning_batch
        self.unresolved_warning_ids = set(unresolved_warning_ids or set())
        self.activity_limits: list[int] = []
        self.activity_parents: list[str] = []
        self.molecule_batches: list[list[str]] = []
        self.warning_batches: list[list[str]] = []
        self.individual_warning_calls: list[str] = []
        self.document_batches: list[list[str]] = []
        self.assay_batches: list[list[str]] = []
        self.pubmed_calls = 0
        self.pubmed_requested_pmids: list[list[str]] = []

    @staticmethod
    def molecule_id(index: int) -> str:
        return f"CHEMBL{100000 + index}"

    @staticmethod
    def document_id(chembl_id: str) -> str:
        return f"CHEMBL{500000 + int(chembl_id.removeprefix('CHEMBL'))}"

    @staticmethod
    def max_phase(index: int) -> int:
        if index % 7 == 0:
            return 4
        if index % 5 == 0:
            return 3
        if index % 3 == 0:
            return 2
        return 1

    def expected_prewarning_ids(self, limit: int) -> list[str]:
        rows = [
            (self.max_phase(index), self.molecule_id(index))
            for index in range(1, min(self.raw_count, 80) + 1)
        ]
        rows.sort(key=lambda item: (-item[0], item[1]))
        return [chembl_id for _phase, chembl_id in rows[:limit]]

    @staticmethod
    def _http_500(url: str) -> httpx.HTTPStatusError:
        request = httpx.Request("GET", url)
        response = httpx.Response(500, request=request)
        return httpx.HTTPStatusError(
            "bounded warning batch unavailable",
            request=request,
            response=response,
        )

    def get(self, url: str, **kwargs: object) -> _Response:
        params = kwargs.get("params")
        assert isinstance(params, dict)
        if url.endswith("mechanism.json"):
            return _Response(
                {
                    "mechanisms": [
                        {
                            "mec_id": 300000 + index,
                            "record_id": 400000 + index,
                            "action_type": "INHIBITOR",
                            "direct_interaction": 1,
                            "max_phase": self.max_phase(index),
                            "mechanism_of_action": "Example target inhibitor",
                            "mechanism_refs": [
                                {
                                    "ref_type": "PubMed",
                                    "ref_id": str(60000000 + index),
                                }
                            ],
                            "molecular_mechanism": 1,
                            "molecule_chembl_id": self.molecule_id(index),
                            "parent_molecule_chembl_id": self.molecule_id(index),
                            "target_chembl_id": "CHEMBL_TARGET",
                            "variant_sequence": None,
                        }
                        for index in range(1, self.raw_count + 1)
                    ]
                }
            )
        if url.endswith("activity.json"):
            self.activity_limits.append(int(params["limit"]))
            chembl_id = str(params["parent_molecule_chembl_id"])
            self.activity_parents.append(chembl_id)
            index = int(chembl_id.removeprefix("CHEMBL")) - 100000
            pchembl = (
                5.0 + index / 1000
                if self.reverse_pchembl
                else 13.0 - index / 1000
            )
            activities = [
                {
                    "activity_id": 500000 + index,
                    "data_validity_comment": None,
                    "molecule_chembl_id": chembl_id,
                    "parent_molecule_chembl_id": chembl_id,
                    "target_chembl_id": "CHEMBL_TARGET",
                    "assay_type": "B",
                    "assay_chembl_id": f"ASSAY_{chembl_id}",
                    "document_chembl_id": self.document_id(chembl_id),
                    "standard_type": "IC50",
                    "standard_relation": "=",
                    "standard_value": str(index),
                    "standard_units": "nM",
                    "pchembl_value": str(pchembl),
                    "bao_label": "single protein format",
                }
            ]
            return _Response({"activities": activities})
        if url.endswith("molecule.json"):
            requested = [
                item
                for item in str(params["molecule_chembl_id__in"]).split(",")
                if item
            ]
            self.molecule_batches.append(requested)
            return _Response(
                {
                    "molecules": [
                        {
                            "molecule_chembl_id": chembl_id,
                            "pref_name": f"Drug {chembl_id}",
                            "max_phase": self.max_phase(
                                int(chembl_id.removeprefix("CHEMBL")) - 100000
                            ),
                            "molecule_type": "Small molecule",
                            "molecule_structures": {
                                "canonical_smiles": "CCOc1ccccc1",
                                "standard_inchi_key": f"KEY-{chembl_id}",
                            },
                        }
                        for chembl_id in requested
                    ]
                }
            )
        if url.endswith("drug_warning.json"):
            requested = [
                item
                for item in str(params["molecule_chembl_id__in"]).split(",")
                if item
            ]
            if len(requested) > 1:
                self.warning_batches.append(requested)
                if self.fail_warning_batch:
                    raise self._http_500(url)
                return _Response({"drug_warnings": []})
            chembl_id = requested[0]
            if self.fail_warning_batch and not self.warning_batches:
                self.warning_batches.append(requested)
                raise self._http_500(url)
            self.individual_warning_calls.append(chembl_id)
            if chembl_id in self.unresolved_warning_ids:
                raise httpx.ReadTimeout("individual warning lookup timed out")
            return _Response({"drug_warnings": []})
        if url.endswith("document.json"):
            requested = [
                item
                for item in str(params["document_chembl_id__in"]).split(",")
                if item
            ]
            self.document_batches.append(requested)
            return _Response(
                {
                    "documents": [
                        {
                            "document_chembl_id": document_id,
                            "doc_type": "PUBLICATION",
                            "pubmed_id": 70000000 + index,
                            "title": "Bounded primary activity study",
                            "abstract": (
                                "Drug CHEMBL"
                                f"{int(document_id.removeprefix('CHEMBL')) - 500000} inhibited "
                                f"GENE with an IC50 of {index} nM."
                            ),
                        }
                        for index, document_id in enumerate(requested, start=1)
                    ]
                }
            )
        if url.endswith("assay.json"):
            requested = [
                item
                for item in str(params["assay_chembl_id__in"]).split(",")
                if item
            ]
            self.assay_batches.append(requested)
        return super().get(url, **kwargs)


class _MissingMechanismFieldClient(_ChEMBLClient):
    def __init__(self) -> None:
        self.molecule_calls = 0
        self.activity_calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("mechanism.json"):
            payload = super().get(url, **kwargs).json()
            mechanisms = payload["mechanisms"]
            assert isinstance(mechanisms, list)
            for mechanism in mechanisms:
                assert isinstance(mechanism, dict)
                mechanism.pop("mechanism_refs")
            return _Response(payload)
        if url.endswith("molecule.json"):
            self.molecule_calls += 1
        if url.endswith("activity.json"):
            self.activity_calls += 1
        return super().get(url, **kwargs)


class _MalformedMechanismReferenceClient(_ChEMBLClient):
    def __init__(self) -> None:
        self.molecule_calls = 0
        self.activity_calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("mechanism.json"):
            payload = super().get(url, **kwargs).json()
            mechanisms = payload["mechanisms"]
            assert isinstance(mechanisms, list)
            for mechanism in mechanisms:
                assert isinstance(mechanism, dict)
                mechanism["mechanism_refs"] = [{}]
            return _Response(payload)
        if url.endswith("molecule.json"):
            self.molecule_calls += 1
        if url.endswith("activity.json"):
            self.activity_calls += 1
        return super().get(url, **kwargs)


class _MalformedMoleculePayloadClient(_ChEMBLClient):
    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("molecule.json"):
            return _Response({"molecules": None})
        return super().get(url, **kwargs)


class _FormParentConflictClient(_ChEMBLClient):
    def __init__(self) -> None:
        self.molecule_calls = 0
        self.activity_calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("mechanism.json"):
            payload = super().get(url, **kwargs).json()
            mechanisms = payload["mechanisms"]
            assert isinstance(mechanisms, list)
            for mechanism in mechanisms:
                assert isinstance(mechanism, dict)
                mechanism["molecule_chembl_id"] = "CHEMBL_FORM"
            return _Response(payload)
        if url.endswith("molecule.json"):
            self.molecule_calls += 1
        if url.endswith("activity.json"):
            self.activity_calls += 1
        return super().get(url, **kwargs)


class _PerParentActivityAdversarialClient(_ChEMBLClient):
    def __init__(self) -> None:
        self.activity_params: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        if not url.endswith("activity.json"):
            return super().get(url, **kwargs)
        params = kwargs.get("params")
        assert isinstance(params, dict)
        self.activity_params.append(dict(params))
        parent_id = str(params["parent_molecule_chembl_id"])
        if parent_id == "CHEMBL2":
            raise httpx.ReadTimeout("one exact parent activity lookup failed")
        return _Response(
            {
                "activities": [
                    {
                        "activity_id": 8000 + index,
                        "data_validity_comment": None,
                        "molecule_chembl_id": "CHEMBL1",
                        "parent_molecule_chembl_id": "CHEMBL1",
                        "target_chembl_id": "CHEMBL_TARGET",
                        "assay_type": "B",
                        "standard_type": "IC50",
                        "standard_relation": "=",
                        "standard_value": str(index + 1),
                        "standard_units": "nM",
                        "pchembl_value": str(8.0 - index / 100),
                        "bao_label": "single protein format",
                    }
                    for index in range(25)
                ]
            }
        )


class _ExactActivityFilterAdversarialClient(_ChEMBLClient):
    def get(self, url: str, **kwargs: object) -> _Response:
        if not url.endswith("activity.json"):
            return super().get(url, **kwargs)
        params = kwargs.get("params")
        assert isinstance(params, dict)
        if params["parent_molecule_chembl_id"] != "CHEMBL1":
            return super().get(url, **kwargs)
        exact = {
            "data_validity_comment": None,
            "molecule_chembl_id": "CHEMBL1",
            "parent_molecule_chembl_id": "CHEMBL1",
            "target_chembl_id": "CHEMBL_TARGET",
            "assay_type": "B",
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_value": "4.2",
            "standard_units": "nM",
            "pchembl_value": "8.3",
            "bao_label": "single protein format",
        }
        return _Response(
            {
                "activities": [
                    {**exact, "activity_id": 9101, "target_chembl_id": None},
                    {
                        **exact,
                        "activity_id": 9102,
                        "target_chembl_id": "CHEMBL_OTHER",
                    },
                    {
                        **exact,
                        "activity_id": 9103,
                        "parent_molecule_chembl_id": "CHEMBL_OTHER",
                    },
                    {**exact, "activity_id": 9104, "standard_type": "Ki"},
                    {**exact, "activity_id": 9105, "assay_type": "F"},
                    {**exact, "activity_id": 9106, "pchembl_value": "4.9"},
                    {**exact, "activity_id": 9107},
                ]
            }
        )


class _EmptyDocumentLargeClient(_LargeBoundedRepurposingClient):
    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("document.json"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            requested = [
                item
                for item in str(params["document_chembl_id__in"]).split(",")
                if item
            ]
            self.document_batches.append(requested)
            return _Response({"documents": []})
        return super().get(url, **kwargs)


class _Docking:
    source_version = "dummy-docking"
    last_result = None

    def dock(self, *, smiles: str, target: TargetCandidate) -> float:
        raise AssertionError("docking was not expected in this test")

    def readiness(self, target: TargetCandidate | None = None) -> dict[str, object]:
        return {"status": "completed", "fixture": False, "reason": None}


class _ADMET:
    source_version = "dummy-admet"
    last_result = None

    def predict(self, *, smiles: str) -> dict[str, float]:
        raise AssertionError("ADMET was not expected in this test")

    def readiness(self) -> dict[str, object]:
        return {"status": "completed", "fixture": False, "reason": None}


def test_executable_provenance_sanitizes_paths_secrets_and_output(tmp_path: Path) -> None:
    assert sanitize_argument(str(tmp_path / "private" / "file.txt")).startswith("<path:")
    assert sanitize_argument("--api-key=abc") == "--api-key=<redacted>"
    assert sanitize_arguments(("--api-key", "abc", "--limit", "2")) == (
        "<redacted>",
        "<redacted>",
        "--limit",
        "2",
    )
    excerpt = bounded_excerpt(f"saved C:\\Users\\public-user\\secret\\result.txt\nready")
    assert "alice" not in excerpt
    assert "<redacted" in excerpt

    runner = ExecutableRunner("python", executable=sys.executable)
    result = runner.run(
        ("-c", "print('scientific-tool-ok')"),
        timeout_seconds=10,
    )
    assert result.stdout.strip() == "scientific-tool-ok"
    assert result.provenance.exit_code == 0
    assert result.provenance.stdout_sha256


def test_executable_timeout_is_fail_closed() -> None:
    runner = ExecutableRunner("python", executable=sys.executable)
    with pytest.raises(ScientificToolTimeout):
        runner.run(
            ("-c", "import time; time.sleep(0.2)"),
            timeout_seconds=0.01,
        )


def test_docking_profile_requires_exact_explicit_selection(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("ATOM      1  C   ALA A   1       0.0 0.0 0.0\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "profiles": [
                    {
                        "profile_id": "IDH1-mutant-smoke-v1",
                        "profile_version": "1",
                        "target_symbol": "IDH1",
                        "uniprot_id": "O75874",
                        "receptor_pdbqt": "receptor.pdbqt",
                        "validated": True,
                        "explicit_selection_required": True,
                        "validation_method": "co-crystal ligand box",
                        "box": {"center": [1, 2, 3], "size": [20, 20, 20]},
                        "receptor_source": {"structure_id": "TEST"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = DockingProfileRegistry.from_json(profile)
    with pytest.raises(ScientificToolUnavailable, match="explicit-only"):
        registry.require(TargetCandidate(symbol="IDH1", uniprot_id="O75874"))
    selected = registry.require(
        TargetCandidate(
            symbol="IDH1",
            uniprot_id="O75874",
            metadata={"docking_profile_id": "IDH1-mutant-smoke-v1"},
        )
    )
    assert selected.profile_id == "IDH1-mutant-smoke-v1"


def test_experimental_docking_profile_fails_closed_without_test_override(
    tmp_path: Path,
) -> None:
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("ATOM      1  C   ALA A   1       0.0 0.0 0.0\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "profiles": [
                    {
                        "profile_id": "IDH1-experimental-v1",
                        "profile_version": "1",
                        "target_symbol": "IDH1",
                        "uniprot_id": "O75874",
                        "receptor_pdbqt": "receptor.pdbqt",
                        "validated": True,
                        "explicit_selection_required": True,
                        "validation_method": "execution smoke only",
                        "profile_status": "experimental_smoke_only",
                        "production_ready": False,
                        "box": {"center": [1, 2, 3], "size": [20, 20, 20]},
                        "receptor_source": {"structure_id": "TEST"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = DockingProfileRegistry.from_json(profile)
    target = TargetCandidate(
        symbol="IDH1",
        uniprot_id="O75874",
        metadata={"docking_profile_id": "IDH1-experimental-v1"},
    )
    production = VinaDockingAdapter(
        profiles=registry,
        vina_executable=sys.executable,
        python_executable=sys.executable,
    )
    with pytest.raises(ScientificToolUnavailable, match="not production_ready"):
        production._require_profile(target)

    test_only = VinaDockingAdapter(
        profiles=registry,
        vina_executable=sys.executable,
        python_executable=sys.executable,
        allow_experimental_profiles=True,
        runtime_variant="scientific-test",
    )
    assert test_only._require_profile(target).profile_id == "IDH1-experimental-v1"


def test_admet_exact_endpoint_mapping_and_missing_endpoint_fail_closed() -> None:
    raw = {
        "DILI": 0.8,
        "hERG": 0.2,
        "AMES": 0.1,
        "CYP3A4_Veith": 0.3,
        "Bioavailability_Ma": 0.7,
        "Clearance_Hepatocyte_AZ": 25.0,
        "Clearance_Hepatocyte_AZ_drugbank_approved_percentile": 73.0,
    }
    scores, selected, mapping = ADMETAIAdapter._map_panel(raw)
    assert scores == {
        "hepatotoxicity": 0.8,
        "herg_risk": 0.2,
        "ames_risk": 0.1,
        "cyp3a4_inhibition": 0.3,
        "oral_bioavailability": 0.7,
        "clearance_risk": 0.73,
    }
    assert selected["Clearance_Hepatocyte_AZ"] == 25.0
    assert "not a toxicity probability" in mapping["clearance_risk"]["semantic"]
    with pytest.raises(ScientificDataUnavailable, match="AMES"):
        ADMETAIAdapter._map_panel({key: value for key, value in raw.items() if key != "AMES"})


def test_live_chembl_gateway_returns_measured_non_fixture_candidates_and_warning() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_ChEMBLClient(),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )
    seeds = list(gateway.retrieve_repurposing(target, direction="modulate", limit=2))
    assert [seed.name for seed in seeds] == ["Drug one", "Drug two"]
    assert all(seed.metadata["fixture"] is False for seed in seeds)
    assert all(seed.activities for seed in seeds)
    assert seeds[0].activities[0].value == 4.2
    assert seeds[0].metadata["development_stage"] == "chembl_max_phase_4"
    assert seeds[0].metadata["regulatory_approval_claimed"] is False
    assert "not an independent regulatory approval" in seeds[0].metadata[
        "development_stage_basis"
    ]
    assert seeds[1].metadata["regulatory_status"] == "withdrawn"
    assert seeds[1].metadata["drug_warnings"][0]["warning_type"] == "Withdrawn"
    assert seeds[0].metadata["measured_target_uniprot_id"] == "P12345"
    assert seeds[0].metadata["measured_target_gene"] == "GENE"
    activity_evidence = seeds[0].evidence[0]
    assert activity_evidence.metadata["target_accession"] == "P12345"
    assert activity_evidence.metadata["target_gene"] == "GENE"
    assert activity_evidence.metadata["target_chembl_id"] == "CHEMBL_TARGET"
    assert "ChEMBL-curated inhibitor mechanism" in seeds[0].mechanism
    assert gateway.source_version.startswith("chembl=ChEMBL_TEST")


@pytest.mark.parametrize(
    "deposited_value",
    [None, "", "Manually validated"],
)
def test_live_chembl_preserves_exact_deposited_data_validity_comment(
    deposited_value: object,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_DataValidityChEMBLClient(value=deposited_value),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="modulate", limit=1)
    )[0]

    metadata = seed.evidence[0].metadata
    assert "data_validity_comment" in metadata
    assert metadata["data_validity_comment"] == deposited_value


@pytest.mark.parametrize(
    ("include_key", "deposited_value"),
    [
        pytest.param(False, None, id="missing-key"),
        pytest.param(True, "Outside typical range", id="nonallowed-value"),
        pytest.param(True, " Manually validated ", id="no-stripping"),
    ],
)
def test_live_chembl_data_validity_comment_fails_closed(
    include_key: bool,
    deposited_value: object,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_DataValidityChEMBLClient(
            value=deposited_value,
            include_key=include_key,
        ),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(NoMeasuredLigandCandidates) as exc_info:
        gateway.retrieve_repurposing(target, direction="modulate", limit=1)

    assert exc_info.value.eligible_count == 0
    assert (
        exc_info.value.reason_code
        == "INSUFFICIENT_FIXED_POOL_EXACT_PARENT_ACTIVITY"
    )


def test_activity_normalization_requires_exact_deposited_standard_fields() -> None:
    row = {
        "activity_id": 11,
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "4.2",
        "standard_units": "nM",
        "bao_label": "single protein format",
    }

    measurement = _normalize_activity(row)

    assert measurement is not None
    assert (
        measurement.endpoint,
        measurement.relation,
        measurement.value,
        measurement.unit,
    ) == ("IC50", "=", 4.2, "nM")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("standard_type", None),
        ("standard_relation", None),
        ("standard_value", None),
        ("standard_units", None),
        ("standard_type", "ic50"),
        ("standard_type", " IC50"),
        ("standard_relation", " ="),
        ("standard_units", "nm"),
        ("standard_units", "µM"),
        ("standard_units", "μM"),
        ("standard_units", " nM"),
    ],
)
def test_activity_normalization_never_defaults_or_canonicalizes_raw_fields(
    field: str,
    replacement: object,
) -> None:
    row: dict[str, object] = {
        "activity_id": 11,
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "4.2",
        "standard_units": "nM",
    }
    if replacement is None:
        row.pop(field)
    else:
        row[field] = replacement

    assert _normalize_activity(row) is None


def test_live_chembl_candidate_order_is_invariant_to_pchembl_swap() -> None:
    def ordered_ids(first: float, second: float) -> list[str]:
        gateway = LiveScientificGateway(
            docking=_Docking(),
            admet=_ADMET(),
            client=_PchemblOrderingClient(first=first, second=second),
            python_executable=sys.executable,
        )
        target = gateway.normalize_target(
            TargetCandidate(symbol="GENE", uniprot_id="P12345")
        )
        return [
            str(seed.metadata["chembl_id"])
            for seed in gateway.retrieve_repurposing(
                target,
                direction="modulate",
                limit=2,
            )
        ]

    assert ordered_ids(5.0, 12.0) == ordered_ids(12.0, 5.0) == [
        "CHEMBL1",
        "CHEMBL2",
    ]


def test_chembl_activity_preserves_bounded_verified_primary_excerpt() -> None:
    client = _PrimaryDocumentChEMBLClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]

    evidence = seed.evidence[0]
    primary = evidence.metadata["primary_source"]
    excerpt = primary["excerpt"]
    assert primary["schema_version"] == "candidate_primary_evidence_v2"
    assert primary["verification_status"] == "verified"
    audited_candidate = SimpleNamespace(
        name=seed.name,
        target=target,
        metadata=seed.metadata,
        evidence=seed.evidence,
        activities=seed.activities,
    )
    assert MA3AssuranceAgent._is_verified_candidate_primary_evidence(
        audited_candidate,
        audited_candidate.activities[0],
        evidence,
        pool_candidates=[audited_candidate],
    )
    assert primary["retrieval_status"] == "completed"
    assert primary["is_primary_support"] is True
    assert primary["pmid"] == "12345678"
    assert primary["doi"] == "10.1000/example.primary"
    assert primary["publication_types"] == ["Journal Article"]
    assert primary["full_text_stored"] is False
    assert primary["activity_binding"] == {
        "activity_id": "11",
        "molecule_chembl_id": "CHEMBL1",
        "target_chembl_id": "CHEMBL_TARGET",
        "assay_chembl_id": "CHEMBL_ASSAY_1",
        "document_chembl_id": "CHEMBL900001",
        "endpoint": "IC50",
            "relation": "=",
            "value": 4.2,
            "unit": "nM",
            "assay_type": "single protein format",
            "assay_context": "direct_single_protein_context",
            "direct_single_protein_engagement": True,
            "direction_evidence_rule": "measured_ic50_endpoint",
            "mechanism_identity_sha256": _canonical_sha256(
                evidence.metadata["mechanism_identity"]
            ),
            "mechanism_claim": (
                "ChEMBL-curated inhibitor mechanism and measured direct "
                "IC50 activity against GENE"
            ),
            "mechanism_claim_sha256": _canonical_sha256(
                "ChEMBL-curated inhibitor mechanism and measured direct "
                "IC50 activity against GENE"
            ),
            "deposited_activity": {
            "schema_version": "chembl_deposited_activity_exact_v1",
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_value": "4.2",
            "standard_units": "nM",
            "typed_exact_match": True,
            "missing_values_imputed": False,
            "normalization_applied": False,
            "unit_conversion_applied": False,
        },
    }
    assert primary["excerpt_location"]["field"] == "abstract"
    assert set(primary["relevance"]["matched_groups"]) == {
        "candidate",
        "target",
        "activity",
    }
    assert primary["relevance"]["matched_exact_value_unit"] == ["4.2 nM"]
    assert primary["relevance"]["policy"] == (
        "exact_candidate_target_endpoint_value_same_clause_v3"
    )
    assert primary["relevance"]["binding_pattern"] == (
        "candidate_target_endpoint_value"
    )
    assert primary["excerpt_location"]["clause_index"] == 0
    assert primary["candidate_exact_aliases"] == ["Drug one"]
    assert seed.metadata["chembl_exact_aliases"] == ["Drug one"]
    competitor = primary["competitor_alias_provenance"]
    assert competitor["competitor_exact_aliases"] == ["Drug two"]
    assert competitor["truncated"] is False
    pubmed_type = primary["pubmed_type_verification"]
    assert pubmed_type["verification_status"] == "verified"
    assert pubmed_type["type_policy"] == (
        "journal_article_without_excluded_publication_type_v1"
    )
    assert pubmed_type["query"]["source"] == "PubMed"
    assert pubmed_type["query"]["status_code"] == 200
    assert pubmed_type["batch"]["requested_pmids"] == ["12345678"]
    assert client.pubmed_calls == 1
    selection = primary["activity_document_selection"]
    assert selection["policy"] == "bounded_top_k_activity_document_rows_v1"
    assert selection["inspected_row_count"] == 1
    assert selection["selected_activity_id"] == "11"
    assay_context = evidence.metadata["chembl_assay_context"]
    assert assay_context["retrieval_status"] == "completed"
    assert assay_context["binding_status"] == "exact"
    assert assay_context["record"]["assay_chembl_id"] == "CHEMBL_ASSAY_1"
    assert assay_context["record"]["target_chembl_id"] == "CHEMBL_TARGET"
    assert assay_context["condition_coverage"] == {
        "construct": "unknown_no_dedicated_field",
        "mutation_state": "unknown_not_assumed_wild_type",
        "atp": "unknown",
        "substrate": "unknown",
        "detection_technology": "unknown",
    }
    assert assay_context["missing_values_imputed"] is False
    assert assay_context["inference_used"] is False
    assert evidence.excerpt_hash == sha256(excerpt.encode("utf-8")).hexdigest()
    assert primary["excerpt_hash"] == evidence.excerpt_hash
    assert client.abstract not in json.dumps(
        primary,
        ensure_ascii=False,
        sort_keys=True,
    )
    assert "abstract" not in primary
    assert all(
        endpoint != gateway.document_endpoint
        for endpoint, _parameter_hash in gateway._json_cache
    ), "full document responses must not remain in the gateway response cache"


def test_chembl_assay_timeout_retains_measurement_as_partial_unknown() -> None:
    client = _AssayTimeoutChEMBLClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        max_http_attempts=1,
        retry_backoff_seconds=0.0,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]

    assert seed.activities[0].value == 4.2
    assert seed.activities[0].unit == "nM"
    context = seed.evidence[0].metadata["chembl_assay_context"]
    assert context["retrieval_status"] == "partial"
    assert context["binding_status"] == "unresolved"
    assert context["reason_code"] == "ASSAY_LOOKUP_UNAVAILABLE"
    assert context["record"] is None
    assert context["condition_coverage"] == {
        "construct": "unknown",
        "mutation_state": "unknown_not_assumed_wild_type",
        "atp": "unknown",
        "substrate": "unknown",
        "detection_technology": "unknown",
    }
    assert context["missing_values_imputed"] is False
    assert context["inference_used"] is False
    assert client.assay_calls == 1


@pytest.mark.parametrize("stage_name", ["warning", "assay", "document"])
def test_http_200_malformed_source_rows_are_not_silently_skipped(
    stage_name: str,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_MalformedSuccessfulSourceRowClient(stage_name),
        python_executable=sys.executable,
        max_http_attempts=1,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    bounds = seed.metadata["candidate_discovery_bounds"]
    stage = bounds["source_call_inventory"]["stages"][stage_name]

    assert stage["successful_call_count"] == 1
    assert stage["failure_count"] >= 1
    assert bounds["source_call_inventory"]["failure_count"] >= 1


def test_chembl_assay_lookup_is_bounded_to_batches_of_eighty() -> None:
    client = _AssayBatchClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
    )
    assay_ids = [f"CHEMBL_ASSAY_{index:03d}" for index in range(161)]

    contexts = gateway._fetch_assay_contexts(assay_ids)

    assert [len(batch) for batch in client.assay_batches] == [80, 80, 1]
    assert set(contexts) == set(assay_ids)
    assert all(
        context["retrieval_status"] == "completed"
        for context in contexts.values()
    )


def test_chembl_exact_research_code_alias_can_support_primary_excerpt() -> None:
    client = _PrimaryDocumentChEMBLClient(
        abstract="RC-1 inhibited GENE with an IC50 of 4.2 nM.",
        molecule_synonyms=["RC-1"],
    )
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]

    assert primary["verification_status"] == "verified"
    assert primary["candidate_exact_aliases"] == ["Drug one", "RC-1"]
    assert primary["relevance"]["matched_terms"]["candidate"] == ["rc 1"]


@pytest.mark.parametrize(
    ("abstract", "expected_pattern"),
    [
        (
            "The IC50 of Drug one against GENE was 4.2 nM.",
            "endpoint_of_candidate_against_target_value",
        ),
        (
            "Drug one had an IC50 of 4.2 nM against GENE.",
            "candidate_endpoint_value_against_target",
        ),
    ],
)
def test_chembl_primary_clause_accepts_explicit_target_binding_grammar(
    abstract: str,
    expected_pattern: str,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_PrimaryDocumentChEMBLClient(abstract=abstract),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]

    assert primary["verification_status"] == "verified"
    assert primary["relevance"]["binding_pattern"] == expected_pattern


@pytest.mark.parametrize(
    ("abstract", "expected_pattern"),
    [
        (
            "Reference inhibitor Drug one (GENE IC50 = 4.2 nM) was included, "
            "whereas compounds A and B measured 1 and 2 nM, respectively.",
            "candidate_parenthetical_target_endpoint_value",
        ),
        (
            "Drug one is a potent JAK2 (IC50 = 7 and 9 nM for wild type and "
            "mutant JAK2, respectively) and GENE (IC50 = 4.2 nM) inhibitor.",
            "candidate_subject_target_parenthetical_endpoint_value",
        ),
        (
            "GENE was confirmed in a kinase inhibition assay, where values "
            "were 2.1 nM for A and 3.0 nM for B, which were comparable to "
            "that of Drug one (IC50 = 4.2 nM).",
            "single_target_context_candidate_parenthetical_endpoint_value",
        ),
    ],
)
def test_chembl_primary_excerpt_accepts_atomic_parenthetical_without_list_inference(
    abstract: str,
    expected_pattern: str,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_PrimaryDocumentChEMBLClient(abstract=abstract),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]

    assert primary["verification_status"] == "verified"
    assert primary["is_primary_support"] is True
    assert primary["relevance"]["binding_pattern"] == expected_pattern
    assert primary["relevance"]["matched_exact_value_unit"] == ["= 4.2 nM"]
    assert primary["excerpt"] in abstract
    assert primary["excerpt"] == abstract.rstrip(".")
    assert primary["excerpt_location"]["selection_scope"] == (
        "atomic_parenthetical_with_sentence_context"
    )


@pytest.mark.parametrize(
    "abstract",
    [
        "Drug one (GENE IC50 = 4.2 nM and 8.0 nM) was tested.",
        "Drug one (GENE and EGFR IC50 = 4.2 nM) was tested.",
        "GENE was tested and Drug one (IC50 = 4.2 nM) was mentioned.",
        "Predicted control Drug one (GENE IC50 = 4.2 nM) was evaluated.",
    ],
)
def test_chembl_primary_parenthetical_rejects_ambiguous_or_inferred_ownership(
    abstract: str,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_PrimaryDocumentChEMBLClient(abstract=abstract),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]

    assert primary["verification_status"] == "unverified"
    assert primary["is_primary_support"] is False
    assert primary["reason_code"] == "PRIMARY_ABSTRACT_RELEVANCE_UNRESOLVED"


@pytest.mark.parametrize(
    ("sentence", "candidate_aliases", "value", "expected_pattern"),
    [
        (
            "SB1518 is an innovative pyrimidine-based macrocycle that shows "
            "a unique kinase profile with selective inhibition of Janus "
            "Kinase-2 (JAK2; IC50=23 and 19 nM for JAK2(WT) and "
            "JAK2(V617F), respectively) within the JAK family (IC50=1280, "
            "520 and 50 nM for JAK1, JK3 and TYK2, respectively) and "
            "fms-like tyrosine kinase-3 (FLT3; IC50=22 nM).",
            ["PACRITINIB", "SB1518", "SB-1518"],
            22.0,
            "candidate_subject_parenthetical_target_endpoint_value",
        ),
        (
            "Among these, compound RSH-7 inhibited BTK and FLT3 most "
            "effectively, with IC50 values of 47 and 12 nM, respectively, "
            "which were superior to spebrutinib (BTK IC50 = 54 nM) and "
            "sorafenib (FLT3 IC50 = 33 nM).",
            ["SORAFENIB"],
            33.0,
            "candidate_parenthetical_target_endpoint_value",
        ),
        (
            "FLT3-ITD was confirmed as the degradation target through a "
            "kinase inhibition assay, where the IC50 values were 2.12 nM "
            "(Compound A1), 1.29 nM (Compound B2), and 3.06 nM (Compound "
            "C2), which were comparable to that of Gilteritinib "
            "(IC50 = 0.43 nM).",
            ["GILTERITINIB"],
            0.43,
            "single_target_context_candidate_parenthetical_endpoint_value",
        ),
    ],
)
def test_primary_parenthetical_binding_accepts_three_observed_flt3_sources(
    sentence: str,
    candidate_aliases: list[str],
    value: float,
    expected_pattern: str,
) -> None:
    """Lock the exact grammatical patterns observed in official ChEMBL docs."""

    binding = _primary_parenthetical_binding(
        sentence,
        candidate_aliases=candidate_aliases,
        competitor_aliases=[],
        target=TargetCandidate(
            symbol="FLT3",
            uniprot_id="P36888",
            target_name="Fms related receptor tyrosine kinase 3",
        ),
        measurement=ActivityMeasurement(
            endpoint="IC50",
            relation="=",
            value=value,
            unit="nM",
            assay_type="single protein format",
        ),
    )

    assert binding is not None
    assert binding["binding_pattern"] == expected_pattern
    assert binding["matched_exact_value_unit"]


@pytest.mark.parametrize(
    "abstract",
    [
        "Drug one inhibited GENE; Drug two had an IC50 of 4.2 nM.",
        (
            "Drug one and Drug two inhibited GENE with IC50 values of "
            "9.9 and 4.2 nM, respectively."
        ),
        "Drug one and Drug two inhibited GENE with an IC50 of 4.2 nM.",
    ],
)
def test_chembl_primary_clause_rejects_cross_clause_or_competing_candidate_value(
    abstract: str,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_CompetingPrimaryDocumentChEMBLClient(abstract=abstract),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=2)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]

    assert primary["verification_status"] == "unverified"
    assert primary["reason_code"] == "PRIMARY_ABSTRACT_RELEVANCE_UNRESOLVED"
    competitor = primary["competitor_alias_provenance"]
    assert competitor["competitor_exact_aliases"] == ["Drug two"]


def test_chembl_primary_clause_rejects_repeated_exact_value_as_ambiguous() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_PrimaryDocumentChEMBLClient(
            abstract=(
                "Drug one inhibited GENE with IC50 values of "
                "4.2 nM and 4.2 nM."
            )
        ),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]

    assert primary["verification_status"] == "unverified"
    assert primary["reason_code"] == "PRIMARY_ABSTRACT_RELEVANCE_UNRESOLVED"


def test_chembl_missing_molecule_is_excluded_before_selected_alias_inventory() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_MissingPoolMoleculePrimaryClient(),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]
    inventory = primary["competitor_alias_provenance"]["inventory"]
    bounds = seed.metadata["candidate_discovery_bounds"]

    assert primary["verification_status"] == "verified"
    assert inventory["source_candidate_ids"] == ["CHEMBL1"]
    assert inventory["missing_molecule_ids"] == []
    assert inventory["complete"] is True
    assert bounds["molecule_resolution_unresolved_ids"] == ["CHEMBL2"]
    missing = next(
        item
        for item in bounds["identity_resolution"]
        if item["molecule_chembl_id"] == "CHEMBL2"
    )
    assert missing["status"] == "excluded"
    assert missing["reason_codes"] == [
        "MOLECULE_METADATA_NOT_EXACTLY_ONE"
    ]


def test_chembl_bounded_top_k_can_select_lower_ranked_verified_document() -> None:
    client = _TopKPrimaryDocumentChEMBLClient(row_count=2, valid_rank=2)
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    evidence = seed.evidence[0]
    primary = evidence.metadata["primary_source"]
    selection = primary["activity_document_selection"]

    assert evidence.external_id == "12"
    assert primary["verification_status"] == "verified"
    assert selection["eligible_row_count"] == 2
    assert selection["inspected_row_count"] == 2
    assert selection["selected_rank"] == 2
    assert [row["reason_code"] for row in selection["rows"]] == [
        "DOCUMENT_NOT_PRIMARY_PUBLICATION",
        None,
    ]
    assert client.pubmed_calls == 1
    assert client.pubmed_requested_pmids == [
        ["12345001", "12345002"]
    ]


def test_chembl_activity_document_lookup_never_exceeds_top_k_per_molecule() -> None:
    client = _TopKPrimaryDocumentChEMBLClient(row_count=4, valid_rank=4)
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]
    selection = primary["activity_document_selection"]

    assert client.requested_document_ids == [
        "CHEMBL900001",
        "CHEMBL900002",
        "CHEMBL900003",
    ]
    assert primary["verification_status"] == "unverified"
    assert selection["eligible_row_count"] == 4
    assert selection["inspected_row_count"] == 3
    assert selection["truncated"] is True
    assert len(selection["rows"]) == 3
    assert client.pubmed_calls == 1
    assert client.pubmed_requested_pmids == [
        ["12345001", "12345002", "12345003"]
    ]


def test_chembl_exact_alias_conflict_fails_closed_for_both_candidates() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_AliasConflictPrimaryDocumentChEMBLClient(),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seeds = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=2)
    )

    assert [seed.name for seed in seeds] == ["Drug one", "Drug two"]
    for seed in seeds:
        primary = seed.evidence[0].metadata["primary_source"]
        assert primary["verification_status"] == "unverified"
        assert primary["reason_code"] == "CANDIDATE_ALIAS_CONFLICT"
        assert primary["candidate_alias_conflicts"] == [
            "Shared research code"
        ]
        assert seed.metadata["chembl_alias_conflicts"] == [
            "Shared research code"
        ]


@pytest.mark.parametrize(
    "abstract",
    [
        "GENE had an IC50 of 4.2 nM in the reported assay.",
        "Drug inhibited GENE with an IC50 of 4.2 nM.",
        "Drug one caused GENE activation at 4.2 nM.",
        "Drug one inhibited GENE with a low nanomolar IC50.",
        "Drug one inhibited GENE with an IC50 of 0.0042 uM.",
        (
            "Drug one inhibited EGFR with an IC50 of 4.2 nM, and GENE "
            "was separately measured."
        ),
        "Drug one inhibited EGFR and GENE with an IC50 of 4.2 nM.",
        (
            "Drug one and GENE were evaluated "
            + ("background context " * 30)
            + "with an IC50 of 4.2 nM."
        ),
    ],
)
def test_chembl_primary_excerpt_fails_closed_without_all_exact_bound_terms(
    abstract: str,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_PrimaryDocumentChEMBLClient(abstract=abstract),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]

    assert primary["verification_status"] == "unverified"
    assert primary["is_primary_support"] is False
    assert primary["reason_code"] == "PRIMARY_ABSTRACT_RELEVANCE_UNRESOLVED"
    assert primary["excerpt"] is None


@pytest.mark.parametrize(
    "client",
    [
        _PrimaryDocumentChEMBLClient(doc_type="PATENT"),
        _PrimaryDocumentChEMBLClient(
            title="A systematic review of primary activity studies"
        ),
        _PrimaryDocumentChEMBLClient(publication_types=["Review"]),
    ],
)
def test_chembl_patent_or_review_is_never_primary_support(
    client: _PrimaryDocumentChEMBLClient,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]

    assert primary["verification_status"] == "unverified"
    assert primary["is_primary_support"] is False
    assert primary["reason_code"] in {
        "DOCUMENT_NOT_PRIMARY_PUBLICATION",
        "DOCUMENT_REVIEW_NOT_PRIMARY_ACTIVITY_REPORT",
    }


@pytest.mark.parametrize(
    "excluded_type",
    [
        "Review",
        "Systematic Review",
        "Meta-Analysis",
        "Guideline",
        "Editorial",
        "Letter",
        "Comment",
        "Expression of Concern",
        "Retraction Notice",
        "Retraction of Publication",
        "Protocol",
        "Clinical Trial Protocol",
    ],
)
def test_pubmed_excluded_type_overrides_journal_article(
    excluded_type: str,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_PrimaryDocumentChEMBLClient(
            pubmed_publication_types=["Journal Article", excluded_type]
        ),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]

    assert primary["verification_status"] == "unverified"
    assert primary["is_primary_support"] is False
    assert primary["pubmed_type_verification"]["verification_status"] == (
        "unverified"
    )


@pytest.mark.parametrize(
    ("client", "reason_code"),
    [
        (
            _PrimaryDocumentChEMBLClient(
                pubmed_publication_types=["Clinical Trial"]
            ),
            "PUBMED_JOURNAL_ARTICLE_TYPE_MISSING",
        ),
        (
            _PrimaryDocumentChEMBLClient(pubmed_missing=True),
            "PUBMED_EXACT_PMID_RESULT_MISSING",
        ),
        (
            _PrimaryDocumentChEMBLClient(
                pubmed_result_uids=["12345678", "12345678"]
            ),
            "PUBMED_RESULT_UID_SET_INVALID",
        ),
        (
            _PrimaryDocumentChEMBLClient(
                pubmed_result_uids=["12345678", "99999999"]
            ),
            "PUBMED_RESULT_UID_SET_INVALID",
        ),
        (
            _PrimaryDocumentChEMBLClient(pubmed_omit_record_uid=True),
            "PUBMED_RESULT_PMID_MISMATCH",
        ),
        (
            _PrimaryDocumentChEMBLClient(pubmed_unavailable=True),
            "PUBMED_TYPE_LOOKUP_UNAVAILABLE",
        ),
        (
            _PrimaryDocumentChEMBLClient(pubmed_id=None),
            "PUBMED_ID_REQUIRED_FOR_PRIMARY_TYPE",
        ),
    ],
)
def test_pubmed_type_missing_failure_or_doi_only_is_unverified(
    client: _PrimaryDocumentChEMBLClient,
    reason_code: str,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        max_http_attempts=1,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]

    assert primary["verification_status"] == "unverified"
    assert primary["reason_code"] == reason_code


@pytest.mark.parametrize(
    ("client", "reason_code"),
    [
        (
            _PrimaryDocumentChEMBLClient(
                pubmed_result_uids=["12345678", "12345678"]
            ),
            "PUBMED_RESULT_UID_SET_INVALID",
        ),
        (
            _PrimaryDocumentChEMBLClient(pubmed_omit_record_uid=True),
            "PUBMED_RESULT_PMID_MISMATCH",
        ),
        (
            _PrimaryDocumentChEMBLClient(pubmed_record_nonmapping=True),
            "PUBMED_RESULT_RECORD_SCHEMA_INVALID",
        ),
        (
            _PrimaryDocumentChEMBLClient(pubmed_pubtype_nonlist=True),
            "PUBMED_PUBLICATION_TYPE_SCHEMA_INVALID",
        ),
        (
            _PrimaryDocumentChEMBLClient(
                pubmed_publication_types=[
                    "Journal Article",
                    {"malformed": "publication type row"},  # type: ignore[list-item]
                ]
            ),
            "PUBMED_PUBLICATION_TYPE_SCHEMA_INVALID",
        ),
    ],
)
def test_pubmed_declared_record_schema_errors_are_partial_inventory_failures(
    client: _PrimaryDocumentChEMBLClient,
    reason_code: str,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        max_http_attempts=1,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]
    primary = seed.evidence[0].metadata["primary_source"]
    pubmed_type = primary["pubmed_type_verification"]
    inventory = seed.metadata["candidate_discovery_bounds"][
        "source_call_inventory"
    ]

    assert primary["reason_code"] == reason_code
    assert pubmed_type["retrieval_status"] == "partial"
    assert inventory["stages"]["pubmed_type"]["failure_count"] == 1
    assert inventory["failure_count"] >= 1


def test_chembl_title_only_document_is_not_promoted_to_primary_support() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_PrimaryDocumentChEMBLClient(abstract=None),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]

    evidence = seed.evidence[0]
    primary = evidence.metadata["primary_source"]
    assert primary["title"] == "Primary activity study"
    assert primary["verification_status"] == "metadata_only"
    assert primary["is_primary_support"] is False
    assert primary["reason_code"] == "PRIMARY_ABSTRACT_NOT_AVAILABLE"
    assert primary["excerpt"] is None
    assert evidence.excerpt_hash is None


def test_chembl_document_timeout_keeps_candidate_partial_and_unverified() -> None:
    client = _PrimaryDocumentChEMBLClient(document_unavailable=True)
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]

    evidence = seed.evidence[0]
    primary = evidence.metadata["primary_source"]
    assert primary["verification_status"] == "unverified"
    assert primary["retrieval_status"] == "partial"
    assert primary["is_primary_support"] is False
    assert primary["reason_code"] == "PRIMARY_DOCUMENT_LOOKUP_UNAVAILABLE"
    assert primary["excerpt"] is None
    assert evidence.excerpt_hash is None
    assert client.document_calls == 2


def test_transient_batch_warning_timeout_falls_back_to_candidate_queries() -> None:
    client = _WarningIsolationClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seeds = list(gateway.retrieve_repurposing(target, direction="modulate", limit=2))

    assert [seed.name for seed in seeds] == ["Drug one", "Drug two"]
    assert client.batch_warning_calls == 2
    assert client.individual_warning_calls.count("CHEMBL1") == 1
    assert client.individual_warning_calls.count("CHEMBL2") == 1
    summary = seeds[0].metadata["candidate_pool_warning_screen"]
    assert summary["batch_fallback_used"] is True
    assert summary["unresolved_candidate_count"] == 0
    assert seeds[1].metadata["regulatory_status"] == "withdrawn"


def test_unresolved_candidate_warning_timeout_drops_only_that_candidate() -> None:
    client = _WarningIsolationClient(unresolved_id="CHEMBL2")
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seeds = list(gateway.retrieve_repurposing(target, direction="modulate", limit=2))

    assert [seed.name for seed in seeds] == ["Drug one"]
    summary = seeds[0].metadata["candidate_pool_warning_screen"]
    assert summary["unresolved_candidate_count"] == 1
    assert summary["unresolved_candidate_ids"] == ["CHEMBL2"]
    assert summary["missing_values_imputed"] is False
    assert client.individual_warning_calls.count("CHEMBL2") == 2


def test_repurposing_narrows_79_identities_to_12_before_downstream_calls() -> None:
    client = _LargeBoundedRepurposingClient(raw_count=79)
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        max_http_attempts=1,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )
    expected_ids = client.expected_prewarning_ids(12)

    seeds = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=12)
    )

    assert client.activity_limits == [20] * 12
    assert client.activity_parents == expected_ids
    assert len(client.molecule_batches) == 1
    assert len(client.molecule_batches[0]) == 79
    assert client.warning_batches == [expected_ids]
    assert set(client.document_batches[0]) == {
        client.document_id(chembl_id) for chembl_id in expected_ids
    }
    assert set(client.assay_batches[0]) == {
        f"ASSAY_{chembl_id}" for chembl_id in expected_ids
    }
    assert len(client.pubmed_requested_pmids[0]) == 12
    assert [str(seed.metadata["chembl_id"]) for seed in seeds] == expected_ids

    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    assert _paper_v6_source_inventory_valid(bounds)
    source_inventory = bounds["source_call_inventory"]
    assert source_inventory["stages"]["molecule"]["scope_ids"] == list(
        bounds["valid_mechanism_identity_ids"]
    )
    assert source_inventory["stages"]["molecule"]["successful_calls"] == [
        bounds["molecule_query"]
    ]
    assert source_inventory["stages"]["pubmed_type"][
        "successful_calls"
    ][0]["source"] == "PubMed"
    assert bounds["schema_version"] == (
        "bounded_mechanism_first_candidate_identity_pool_v1"
    )
    assert bounds["raw_mechanism_row_limit"] == 80
    assert bounds["raw_mechanism_response_count"] == 79
    assert bounds["mechanism_total_count"] == 79
    assert bounds["mechanism_complete_within_bound"] is True
    assert bounds["chembl_version"] == "ChEMBL_TEST"
    assert len(bounds["raw_mechanism_rows_sha256"]) == 64
    assert len(bounds["mechanism_identity_sha256"]) == 64
    assert bounds["discovered_identity_count"] == 79
    assert bounds["valid_mechanism_identity_count"] == 79
    assert bounds["valid_mechanism_identity_ids"] == sorted(
        client.molecule_batches[0]
    )
    assert bounds["invalid_mechanism_identity_ids"] == []
    assert bounds["molecule_resolution_batch_count"] == 1
    assert len(bounds["molecule_response_rows_sha256"]) == 64
    assert bounds["molecule_resolution_requested_count"] == 79
    assert bounds["molecule_resolution_count"] == 79
    assert bounds["prewarning_pool_limit"] == 12
    assert bounds["prewarning_pool_ids"] == expected_ids
    assert bounds["preactivity_pool_ids"] == expected_ids
    assert [
        item["parent_molecule_chembl_id"]
        for item in bounds["preactivity_pool"]
    ] == expected_ids
    assert len(bounds["preactivity_excluded_identity_ids"]) == 67
    assert set(bounds["preactivity_exclusion_reason_codes_by_id"]) == set(
        bounds["preactivity_excluded_identity_ids"]
    )
    assert bounds["warning_screen_identity_count"] == 12
    assert bounds["warning_screen_identity_ids"] == expected_ids
    assert bounds["warning_screen_within_prewarning_limit"] is True
    assert bounds["document_lookup_candidate_identity_count"] == 12
    assert bounds["pubmed_type_lookup_candidate_identity_count"] == 12
    assert bounds["assay_lookup_candidate_identity_count"] == 12
    assert bounds["downstream_candidate_identity_limit_satisfied"] is True
    assert bounds["pchembl_used_for_prewarning_selection"] is False
    assert bounds["primary_document_used_for_prewarning_selection"] is False
    assert bounds["warning_used_for_prewarning_selection"] is False
    assert bounds["activity_used_for_prewarning_selection"] is False
    assert set(bounds["mechanism_provenance_by_id"]) == set(
        bounds["valid_mechanism_identity_ids"]
    )
    assert all(
        len(item["rows_sha256"]) == 64
        for item in bounds["mechanism_provenance_by_id"].values()
    )
    assert all(
        len(item["response_sha256"]) == 64
        and len(item["response_rows_sha256"]) == 64
        and len(item["inspected_rows_sha256"]) == 64
        for item in bounds["activity_counts_by_id"].values()
    )
    assert bounds["no_activity_first_fallback"] is True
    assert bounds["no_backfill_beyond_prewarning_pool"] is True


def test_mechanism_first_rejects_upstream_identity_total_over_80() -> None:
    client = _LargeBoundedRepurposingClient(raw_count=95)
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        max_http_attempts=1,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(ScientificDataUnavailable) as captured:
        gateway.retrieve_repurposing(
            target,
            direction="inhibit",
            limit=1,
        )

    provenance = captured.value.provenance
    assert provenance["reason_code"] == (
        "MECHANISM_IDENTITY_SET_INCOMPLETE_OR_OVER_CAP"
    )
    assert provenance["raw_mechanism_row_limit"] == 80
    assert provenance["raw_mechanism_response_count"] == 95
    assert provenance["mechanism_total_count"] == 95
    assert provenance["no_activity_first_fallback"] is True
    assert client.molecule_batches == []
    assert client.activity_limits == []
    assert client.warning_batches == []


def test_mechanism_first_missing_required_fields_fails_before_molecule_or_activity() -> None:
    client = _MissingMechanismFieldClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(NoMeasuredLigandCandidates) as captured:
        gateway.retrieve_repurposing(
            target,
            direction="inhibit",
            limit=2,
        )

    assert captured.value.reason_code == "NO_VALID_BOUNDED_MECHANISM_IDENTITIES"
    malformed = captured.value.provenance["malformed_mechanism_rows"]
    assert len(malformed) == 2
    assert all(
        "MECHANISM_REQUIRED_FIELDS_MISSING" in row["reason_codes"]
        and row["missing_fields"] == ["mechanism_refs"]
        for row in malformed
    )
    assert captured.value.provenance["no_activity_first_fallback"] is True
    assert client.molecule_calls == 0
    assert client.activity_calls == 0


def test_mechanism_first_rejects_nonempty_but_unusable_reference_records() -> None:
    client = _MalformedMechanismReferenceClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(NoMeasuredLigandCandidates) as captured:
        gateway.retrieve_repurposing(
            target,
            direction="inhibit",
            limit=2,
        )

    assert captured.value.reason_code == "NO_VALID_BOUNDED_MECHANISM_IDENTITIES"
    malformed = captured.value.provenance["malformed_mechanism_rows"]
    assert len(malformed) == 2
    assert all(
        "MECHANISM_REFERENCE_MALFORMED" in row["reason_codes"]
        for row in malformed
    )
    assert client.molecule_calls == 0
    assert client.activity_calls == 0


def test_mechanism_first_molecule_schema_failure_preserves_bounded_provenance() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_MalformedMoleculePayloadClient(),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(ScientificDataUnavailable) as captured:
        gateway.retrieve_repurposing(
            target,
            direction="inhibit",
            limit=2,
        )

    provenance = captured.value.provenance
    assert provenance["reason_code"] == "MOLECULE_RESPONSE_LIST_MISSING"
    assert provenance["chembl_version"] == "ChEMBL_TEST"
    assert provenance["raw_mechanism_response_count"] == 2
    assert provenance["mechanism_total_count"] == 2
    assert len(provenance["raw_mechanism_rows_sha256"]) == 64
    assert provenance["discovered_identity_ids"] == ["CHEMBL1", "CHEMBL2"]
    assert provenance["valid_mechanism_identity_ids"] == [
        "CHEMBL1",
        "CHEMBL2",
    ]
    assert len(provenance["molecule_query"]["response_sha256"]) == 64
    assert provenance["no_activity_first_fallback"] is True


def test_mechanism_first_form_parent_conflict_fails_closed_without_backfill() -> None:
    client = _FormParentConflictClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(NoMeasuredLigandCandidates) as captured:
        gateway.retrieve_repurposing(
            target,
            direction="inhibit",
            limit=2,
        )

    assert captured.value.reason_code == "NO_VALID_BOUNDED_MECHANISM_IDENTITIES"
    assert captured.value.provenance["form_parent_conflicts"] == {
        "CHEMBL_FORM": ["CHEMBL1", "CHEMBL2"]
    }
    assert captured.value.provenance["no_activity_first_fallback"] is True
    assert client.molecule_calls == 0
    assert client.activity_calls == 0


def test_mechanism_first_requires_exact_human_single_protein_target() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_ChEMBLClient(),
        python_executable=sys.executable,
    )
    normalized = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )
    target = normalized.model_copy(
        update={
            "metadata": {
                **normalized.metadata,
                "chembl_target_type": "PROTEIN COMPLEX",
            }
        }
    )
    gateway._normalized_targets[target.symbol.casefold()] = target

    with pytest.raises(ScientificDataUnavailable) as captured:
        gateway.retrieve_repurposing(
            target,
            direction="inhibit",
            limit=2,
        )

    assert captured.value.provenance["reason_code"] == (
        "TARGET_NOT_EXACT_HUMAN_SINGLE_PROTEIN"
    )
    assert captured.value.provenance["target_type"] == "PROTEIN COMPLEX"
    assert captured.value.provenance["target_organism"] == "Homo sapiens"


def test_fixed_paper_policy_rejects_candidate_identity_limit_over_12() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_ChEMBLClient(),
        python_executable=sys.executable,
        repurposing_evidence_policy=PAIRED_V4_REQUIRED_POLICY,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(ValueError, match="must not exceed 12"):
        gateway.retrieve_repurposing(
            target,
            direction="inhibit",
            limit=13,
        )


def test_per_parent_activity_cap_and_failure_do_not_backfill_fixed_pool() -> None:
    client = _PerParentActivityAdversarialClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        max_http_attempts=1,
        retry_backoff_seconds=0,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seeds = gateway.retrieve_repurposing(
        target,
        direction="inhibit",
        limit=2,
    )

    assert [seed.metadata["chembl_id"] for seed in seeds] == ["CHEMBL1"]
    assert [params["parent_molecule_chembl_id"] for params in client.activity_params] == [
        "CHEMBL1",
        "CHEMBL2",
    ]
    assert all(params["limit"] == 20 for params in client.activity_params)
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    assert bounds["preactivity_pool_ids"] == ["CHEMBL1", "CHEMBL2"]
    assert bounds["activity_counts_by_id"]["CHEMBL1"] == {
        **bounds["activity_counts_by_id"]["CHEMBL1"],
        "response_count": 25,
        "inspected_count": 20,
        "inspected_cap": 20,
        "truncated": True,
        "eligible_directional_count": 20,
        "retained_document_rows": 3,
    }
    assert sorted(bounds["activity_failures_by_id"]) == ["CHEMBL2"]
    assert bounds["fixed_pool_with_activity_ids"] == ["CHEMBL1"]
    assert bounds["no_activity_first_fallback"] is True
    assert bounds["no_backfill_beyond_prewarning_pool"] is True


def test_mechanism_first_revalidates_exact_activity_filters_locally() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_ExactActivityFilterAdversarialClient(),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seeds = gateway.retrieve_repurposing(
        target,
        direction="modulate",
        limit=1,
    )

    assert len(seeds) == 2
    assert seeds[0].evidence[0].external_id == "9107"
    counts = seeds[0].metadata["candidate_discovery_bounds"][
        "activity_counts_by_id"
    ]["CHEMBL1"]
    assert counts["response_count"] == 7
    assert counts["eligible_directional_count"] == 1
    assert counts["target_identity_mismatch_count"] == 2
    assert counts["parent_identity_mismatch_count"] == 1
    assert counts["identity_mismatch_count"] == 3
    assert counts["endpoint_mismatch_count"] == 1
    assert counts["assay_type_mismatch_count"] == 1
    assert counts["pchembl_threshold_mismatch_count"] == 1


def test_warning_batch_fallback_is_bounded_to_selected_pool() -> None:
    client = _LargeBoundedRepurposingClient(
        raw_count=79,
        fail_warning_batch=True,
    )
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        max_http_attempts=1,
        retry_backoff_seconds=0,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )
    expected_ids = client.expected_prewarning_ids(12)

    seeds = gateway.retrieve_repurposing(
        target,
        direction="inhibit",
        limit=12,
    )

    assert client.warning_batches == [expected_ids]
    assert client.individual_warning_calls == expected_ids
    summary = seeds[0].metadata["candidate_pool_warning_screen"]
    assert summary["batch_fallback_used"] is True
    assert summary["requested_candidate_count"] == 12
    assert summary["screened_candidate_count"] == 12
    assert summary["unresolved_candidate_count"] == 0
    assert summary["logical_call_count"] == 13
    assert summary["logical_call_limit"] == 13
    assert summary["logical_call_limit_satisfied"] is True
    assert summary["batch_failure"]["error_type"] == "upstream_http_500"


def test_prewarning_phase_id_selection_is_invariant_to_pchembl_swap() -> None:
    def selected_ids(
        reverse_pchembl: bool,
        *,
        empty_documents: bool = False,
    ) -> list[str]:
        client_type = (
            _EmptyDocumentLargeClient
            if empty_documents
            else _LargeBoundedRepurposingClient
        )
        client = client_type(
            raw_count=30,
            reverse_pchembl=reverse_pchembl,
        )
        gateway = LiveScientificGateway(
            docking=_Docking(),
            admet=_ADMET(),
            client=client,
            python_executable=sys.executable,
            max_http_attempts=1,
        )
        target = gateway.normalize_target(
            TargetCandidate(symbol="GENE", uniprot_id="P12345")
        )
        seeds = gateway.retrieve_repurposing(
            target,
            direction="inhibit",
            limit=5,
        )
        return seeds[0].metadata["candidate_discovery_bounds"][
            "prewarning_pool_ids"
        ]

    forward = selected_ids(False)
    reverse = selected_ids(True)
    without_documents = selected_ids(True, empty_documents=True)

    assert forward == reverse == without_documents
    assert forward == _LargeBoundedRepurposingClient(
        raw_count=30
    ).expected_prewarning_ids(12)


def test_partial_warning_failure_never_backfills_beyond_selected_pool() -> None:
    client = _LargeBoundedRepurposingClient(
        raw_count=20,
        fail_warning_batch=True,
    )
    selected_ids = client.expected_prewarning_ids(12)
    client.unresolved_warning_ids = {selected_ids[0]}
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        max_http_attempts=1,
        retry_backoff_seconds=0,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seeds = gateway.retrieve_repurposing(
        target,
        direction="inhibit",
        limit=3,
    )

    returned_ids = [str(seed.metadata["chembl_id"]) for seed in seeds]
    assert returned_ids == selected_ids[1:]
    assert client.individual_warning_calls.count(selected_ids[0]) == 1
    assert set(returned_ids).issubset(selected_ids)
    outside_fixed_pool = client.expected_prewarning_ids(13)[12]
    assert client.activity_parents == selected_ids
    assert outside_fixed_pool not in client.activity_parents
    assert outside_fixed_pool not in returned_ids
    summary = seeds[0].metadata["candidate_pool_warning_screen"]
    assert summary["unresolved_candidate_ids"] == [selected_ids[0]]
    assert summary["missing_values_imputed"] is False
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    assert bounds["no_backfill_beyond_prewarning_pool"] is True
    assert bounds["requested_eligible_count"] == 3
    assert bounds["gateway_return_cap_applied"] is False
    assert bounds["gateway_return_candidate_ids"] == selected_ids[1:]
    failed_resolution = next(
        item
        for item in bounds["identity_resolution"]
        if item["molecule_chembl_id"] == selected_ids[0]
    )
    assert failed_resolution["warning_screen_status"] == "unresolved"
    assert failed_resolution["downstream_status"] == "excluded"
    assert "REGULATORY_WARNING_STATUS_UNRESOLVED" in failed_resolution[
        "reason_codes"
    ]
    assert client.document_id(selected_ids[0]) not in client.document_batches[0]
    assert f"ASSAY_{selected_ids[0]}" not in client.assay_batches[0]


def test_persistent_selected_pool_warning_failure_is_fail_closed_and_audited() -> None:
    client = _LargeBoundedRepurposingClient(
        raw_count=20,
        fail_warning_batch=True,
    )
    selected_ids = client.expected_prewarning_ids(12)
    client.unresolved_warning_ids = set(selected_ids)
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        max_http_attempts=1,
        retry_backoff_seconds=0,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(ScientificDataUnavailable) as captured:
        gateway.retrieve_repurposing(
            target,
            direction="inhibit",
            limit=2,
        )

    provenance = captured.value.provenance
    assert provenance["operation"] == "candidate_regulatory_warning_screen"
    assert sorted(provenance["candidate_failures"]) == sorted(selected_ids)
    bounds = provenance["candidate_discovery_bounds"]
    assert bounds["prewarning_pool_ids"] == selected_ids
    assert bounds["warning_screen_identity_count"] == 12
    assert bounds["warning_screen"]["logical_call_limit"] == 13
    assert bounds["warning_screen"]["logical_call_limit_satisfied"] is True
    assert bounds["warning_screen"]["batch_failure"]["status_code"] == 500
    last = gateway.last_operation_provenance
    assert last["operation"] == "retrieve_repurposing"
    assert last["status"] == "error"
    assert last["result_returned"] is False
    assert last["failure"]["candidate_discovery_bounds"][
        "prewarning_pool_ids"
    ] == selected_ids
    assert "FINITE_DISCOVERY_POOL_EXHAUSTED" not in json.dumps(last)


def test_live_chembl_inhibitory_route_excludes_binding_only_measurement() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_ChEMBLClient(),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seeds = list(gateway.retrieve_repurposing(target, direction="inhibit", limit=2))

    assert [seed.name for seed in seeds] == ["Drug one", "Drug two"]
    assert seeds[0].activities[0].endpoint == "IC50"
    assert seeds[0].metadata["direction_evidence_rules"] == [
        "measured_ic50_endpoint"
    ]
    assert "ChEMBL-curated inhibitor mechanism" in seeds[0].mechanism


def test_cell_based_ic50_preserves_context_without_claiming_direct_engagement() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_CellBasedChEMBLClient(),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]

    assert seed.activities[0].assay_type == "cell-based format"
    assert seed.metadata["activity_assay_context"] == (
        "cellular_or_system_context"
    )
    assert seed.metadata["direct_single_protein_engagement"] is False
    assert "direct single-protein engagement not established" in seed.mechanism
    evidence = seed.evidence[0]
    assert evidence.metadata["assay_type"] == "cell-based format"
    assert evidence.metadata["assay_context"] == "cellular_or_system_context"
    assert evidence.metadata["direct_single_protein_engagement"] is False


def test_unspecified_assay_context_is_not_described_as_cellular() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_UnspecifiedAssayChEMBLClient(),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seed = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=1)
    )[0]

    expected = (
        "ChEMBL-curated inhibitor mechanism with measured IC50 activity for "
        "GENE; assay context unspecified; direct engagement not established"
    )
    assert seed.activities[0].assay_type == "unknown assay format"
    assert seed.metadata["activity_assay_context"] == (
        "assay_context_unspecified"
    )
    assert seed.metadata["direct_single_protein_engagement"] is False
    assert seed.mechanism == expected
    assert "cellular" not in seed.mechanism.casefold()
    evidence = seed.evidence[0]
    assert evidence.metadata["assay_context"] == "assay_context_unspecified"
    binding = evidence.metadata["primary_source"]["activity_binding"]
    assert binding["assay_type"] == "unknown assay format"
    assert binding["assay_context"] == "assay_context_unspecified"
    assert binding["direct_single_protein_engagement"] is False
    assert binding["direction_evidence_rule"] == "measured_ic50_endpoint"
    assert binding["mechanism_claim"] == expected
    assert binding["mechanism_claim_sha256"] == _canonical_sha256(expected)
    assert binding["mechanism_identity_sha256"] == _canonical_sha256(
        evidence.metadata["mechanism_identity"]
    )


def test_exact_recombinant_biochemical_record_recovers_generic_bao_context() -> None:
    description = (
        "Inhibition of recombinant N-terminal GST-tagged FLT3 expressed in "
        "baculovirus using poly(Glu,Tyr) as substrate in presence of ATP"
    )
    context = {
        "schema_version": "chembl_assay_context_v1",
        "retrieval_status": "completed",
        "binding_status": "exact",
        "assay_chembl_id": "CHEMBL_ASSAY",
        "missing_values_imputed": False,
        "record": {
            "assay_chembl_id": "CHEMBL_ASSAY",
            "target_chembl_id": "CHEMBL_TARGET",
            "assay_type": "B",
            "assay_organism": "Homo sapiens",
            "relationship_type": "D",
            "confidence_score": 9,
            "assay_cell_type": None,
            "assay_tissue": None,
            "assay_strain": None,
            "cell_chembl_id": None,
            "tissue_chembl_id": None,
            "description": {
                "excerpt": description,
                "excerpt_truncated": False,
                "content_length": len(description),
                "content_sha256": sha256(description.encode()).hexdigest(),
            },
        },
    }

    assessment = _assay_engagement_assessment(
        assay_type="assay format",
        assay_context=context,
        assay_chembl_id="CHEMBL_ASSAY",
        target_chembl_id="CHEMBL_TARGET",
        target_symbol="FLT3",
    )

    assert assessment["direct_single_protein_engagement"] is True
    assert assessment["assay_context"] == "direct_single_protein_context"
    assert assessment["basis"] == (
        "exact_bound_recombinant_biochemical_assay_record"
    )
    assert assessment["description_similarity_used"] is False


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("wrong_target", "EXACT_ASSAY_TARGET_BINDING_MISMATCH"),
        ("low_confidence", "DIRECT_HUMAN_TARGET_ASSIGNMENT_UNATTESTED"),
        ("cellular", "CELLULAR_OR_TISSUE_CONTEXT_PRESENT"),
        ("truncated", "EXACT_ASSAY_DESCRIPTION_UNATTESTED"),
        ("missing_atp", "EXPLICIT_RECOMBINANT_ASSAY_BINDING_UNATTESTED"),
    ],
)
def test_generic_bao_direct_engagement_fallback_fails_closed(
    mutation: str,
    reason_code: str,
) -> None:
    description = (
        "Inhibition of recombinant FLT3 kinase using peptide substrate and ATP"
    )
    record = {
        "assay_chembl_id": "CHEMBL_ASSAY",
        "target_chembl_id": "CHEMBL_TARGET",
        "assay_type": "B",
        "assay_organism": "Homo sapiens",
        "relationship_type": "D",
        "confidence_score": 9,
        "assay_cell_type": None,
        "assay_tissue": None,
        "assay_strain": None,
        "cell_chembl_id": None,
        "tissue_chembl_id": None,
        "description": {
            "excerpt": description,
            "excerpt_truncated": False,
            "content_length": len(description),
            "content_sha256": sha256(description.encode()).hexdigest(),
        },
    }
    if mutation == "wrong_target":
        record["target_chembl_id"] = "CHEMBL_OTHER"
    elif mutation == "low_confidence":
        record["confidence_score"] = 8
    elif mutation == "cellular":
        record["assay_cell_type"] = "HEK293"
    elif mutation == "truncated":
        record["description"]["excerpt_truncated"] = True
    elif mutation == "missing_atp":
        changed = description.replace(" and ATP", "")
        record["description"] = {
            "excerpt": changed,
            "excerpt_truncated": False,
            "content_length": len(changed),
            "content_sha256": sha256(changed.encode()).hexdigest(),
        }
    context = {
        "schema_version": "chembl_assay_context_v1",
        "retrieval_status": "completed",
        "binding_status": "exact",
        "assay_chembl_id": "CHEMBL_ASSAY",
        "missing_values_imputed": False,
        "record": record,
    }

    assessment = _assay_engagement_assessment(
        assay_type="assay format",
        assay_context=context,
        assay_chembl_id="CHEMBL_ASSAY",
        target_chembl_id="CHEMBL_TARGET",
        target_symbol="FLT3",
    )

    assert assessment["direct_single_protein_engagement"] is False
    assert reason_code in assessment["reason_codes"]


def test_live_chembl_activation_without_explicit_action_is_unsearched_capability_gap() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_ChEMBLClient(),
        python_executable=sys.executable,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(ScientificToolUnavailable) as captured:
        gateway.retrieve_repurposing(target, direction="activate", limit=2)

    assert captured.value.conclusive is False
    assert captured.value.reason_code == "UNSUPPORTED_MECHANISM_ACTION_DIRECTION"
    assert captured.value.provenance["search_completed"] is False
    assert captured.value.provenance["query_completed"] is False
    assert captured.value.provenance["outcome"] == "unsupported_capability_not_searched"
    assert "eligible_count" not in captured.value.provenance


def test_no_exact_chembl_target_is_typed_conclusive_negative_result() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_NoExactTargetClient(),
        python_executable=sys.executable,
    )

    with pytest.raises(NoMeasuredLigandCandidates) as captured:
        gateway.normalize_target(
            TargetCandidate(symbol="GENE", uniprot_id="P12345")
        )

    assert captured.value.reason_code == "NO_EXACT_CHEMBL_TARGET"
    assert captured.value.target_uniprot_id == "P12345"
    assert captured.value.conclusive is True
    assert captured.value.provenance["matching_policy"] == (
        "exact target-component UniProt accession"
    )


def test_target_accession_500_resolves_through_one_exact_uniprot_xref() -> None:
    client = _TargetAccessionFallbackClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0,
    )

    normalized = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    assert normalized.metadata["chembl_target_id"] == "CHEMBL_TARGET"
    assert normalized.metadata["chembl_target_type"] == "SINGLE PROTEIN"
    assert normalized.metadata["chembl_organism"] == "Homo sapiens"
    fallback = normalized.metadata["chembl_target_resolution_fallback"]
    assert fallback["used"] is True
    assert fallback["fixture"] is False
    assert fallback["uniprot_primary_accession"] == "P12345"
    assert fallback["uniprot_chembl_xref"] == "CHEMBL_TARGET"
    assert fallback["uniprot_record_query"]["source"] == "UniProt"
    assert fallback["chembl_target_query"]["source"] == "ChEMBL"
    assert len(fallback["uniprot_record_query"]["response_sha256"]) == 64
    assert len(fallback["chembl_target_query"]["response_sha256"]) == 64
    assert fallback["independent_validation"] == {
        "target_chembl_id_exact": True,
        "target_component_accessions": ["P12345"],
        "target_type": "SINGLE PROTEIN",
        "organism": "Homo sapiens",
    }
    assert fallback["target_accession_query_failure"]["error_type"] == (
        "upstream_http_500"
    )
    assert client.accession_query_calls == 2
    assert client.uniprot_calls == 1
    assert client.target_id_calls == 1


def test_target_xref_fallback_rejects_conflicting_component_accession() -> None:
    client = _TargetAccessionFallbackClient(
        component_accessions=("P99999",)
    )
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0,
    )

    with pytest.raises(ScientificDataUnavailable) as captured:
        gateway.normalize_target(
            TargetCandidate(symbol="GENE", uniprot_id="P12345")
        )

    assert captured.value.provenance["reason_code"] == (
        "CHEMBL_XREF_TARGET_ACCESSION_CONFLICT"
    )
    assert captured.value.provenance["source"] == "ChEMBL"
    assert captured.value.provenance["detail"][
        "returned_component_accessions"
    ] == ["P99999"]
    assert captured.value.provenance["fixture"] is False


def test_target_xref_fallback_rejects_multiple_uniprot_chembl_xrefs() -> None:
    client = _TargetAccessionFallbackClient(
        xref_ids=("CHEMBL_TARGET", "CHEMBL_OTHER")
    )
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0,
    )

    with pytest.raises(ScientificDataUnavailable) as captured:
        gateway.normalize_target(
            TargetCandidate(symbol="GENE", uniprot_id="P12345")
        )

    assert captured.value.provenance["reason_code"] == (
        "UNIPROT_CHEMBL_XREF_AMBIGUOUS"
    )
    assert captured.value.provenance["source"] == "UniProt"
    assert captured.value.provenance["detail"]["chembl_xref_count"] == 2
    assert client.target_id_calls == 0


def test_target_xref_fallback_rejects_missing_uniprot_chembl_xref() -> None:
    client = _TargetAccessionFallbackClient(xref_ids=())
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0,
    )

    with pytest.raises(ScientificDataUnavailable) as captured:
        gateway.normalize_target(
            TargetCandidate(symbol="GENE", uniprot_id="P12345")
        )

    assert captured.value.provenance["reason_code"] == (
        "UNIPROT_CHEMBL_XREF_MISSING"
    )
    assert captured.value.provenance["source"] == "UniProt"
    assert client.target_id_calls == 0


def test_target_xref_fallback_rejects_zero_chembl_target_response() -> None:
    client = _TargetAccessionFallbackClient(return_target=False)
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0,
    )

    with pytest.raises(ScientificDataUnavailable) as captured:
        gateway.normalize_target(
            TargetCandidate(symbol="GENE", uniprot_id="P12345")
        )

    assert captured.value.provenance["reason_code"] == (
        "CHEMBL_XREF_TARGET_NOT_FOUND"
    )
    assert captured.value.provenance["source"] == "ChEMBL"
    assert captured.value.provenance["uniprot_record_query"]["source"] == (
        "UniProt"
    )
    assert captured.value.provenance["chembl_target_query"]["source"] == (
        "ChEMBL"
    )


def test_chembl_read_timeout_is_typed_scientific_data_unavailable() -> None:
    client = _ReadTimeoutClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0,
    )

    with pytest.raises(ScientificDataUnavailable) as captured:
        gateway.generate_de_novo(
            TargetCandidate(symbol="GENE", uniprot_id="P12345"),
            direction="inhibit",
            limit=2,
        )

    assert "read_timeout" in str(captured.value)
    assert captured.value.provenance["error_type"] == "read_timeout"
    assert captured.value.provenance["attempt_count"] == 2
    assert captured.value.provenance["max_attempts"] == 2
    assert captured.value.provenance["exhausted"] is True
    assert captured.value.provenance["cache_hit"] is False
    assert captured.value.provenance["fixture"] is False
    # The optional release-metadata timeout is isolated. The exact ChEMBL
    # accession request then triggers the bounded official-UniProt fallback,
    # which independently fails closed on the same transport outage.
    assert captured.value.provenance["source"] == "UniProt"
    assert captured.value.provenance["endpoint"].endswith("P12345.json")
    assert captured.value.provenance["reason_code"] == (
        "UNIPROT_RECORD_UNAVAILABLE"
    )
    assert captured.value.provenance["target_accession_query_failure"][
        "endpoint"
    ].endswith("target.json")
    assert client.calls == 6


def test_chembl_transient_read_timeout_retries_once_then_uses_live_response() -> None:
    client = _TimeoutThenSuccessClient()
    sleeps: list[float] = []
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0.25,
        sleeper=sleeps.append,
    )

    normalized = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    assert normalized.metadata["chembl_target_id"] == "CHEMBL_TARGET"
    assert client.status_calls == 2
    assert sleeps == [0.25]
    status_call = next(
        item for item in gateway._http_calls if item.endpoint.endswith("status.json")
    )
    assert status_call.attempt_count == 2
    assert status_call.exhausted is False
    assert status_call.cache_hit is False
    assert [item["error_type"] for item in status_call.attempts[:1]] == [
        "read_timeout"
    ]


def test_chembl_status_metadata_500_does_not_block_exact_live_activity_data() -> None:
    client = _StatusEndpoint500Client()
    sleeps: list[float] = []
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0.25,
        sleeper=sleeps.append,
    )
    target = TargetCandidate(symbol="GENE", uniprot_id="P12345")

    normalized = gateway.normalize_target(target)
    seeds = gateway.retrieve_repurposing(
        normalized,
        direction="inhibit",
        limit=2,
    )

    assert normalized.metadata["chembl_target_id"] == "CHEMBL_TARGET"
    assert normalized.metadata["chembl_version"] is None
    assert normalized.metadata["chembl_version_resolution"]["status"] == "unavailable"
    assert (
        normalized.metadata["chembl_version_resolution"]["reason_code"]
        == "CHEMBL_VERSION_METADATA_UNAVAILABLE"
    )
    assert [seed.name for seed in seeds] == ["Drug one", "Drug two"]
    assert client.status_calls == 2
    assert client.target_calls == 1
    assert sleeps == [0.25]
    assert gateway.last_operation_provenance["chembl_version"] is None
    assert (
        gateway.last_operation_provenance["chembl_version_resolution"]["status"]
        == "unavailable"
    )


def test_chembl_status_metadata_timeout_does_not_block_exact_live_activity_data() -> None:
    client = _StatusEndpointReadTimeoutClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        retry_backoff_seconds=0,
    )

    normalized = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )
    seeds = gateway.retrieve_repurposing(
        normalized,
        direction="inhibit",
        limit=2,
    )

    assert normalized.metadata["chembl_target_id"] == "CHEMBL_TARGET"
    assert normalized.metadata["chembl_version"] is None
    resolution = normalized.metadata["chembl_version_resolution"]
    assert resolution["status"] == "unavailable"
    assert resolution["reason_code"] == "CHEMBL_VERSION_METADATA_UNAVAILABLE"
    assert resolution["provenance"]["error_type"] == "read_timeout"
    assert resolution["provenance"]["retryable"] is True
    assert [seed.name for seed in seeds] == ["Drug one", "Drug two"]
    assert client.status_calls == 2
    assert client.target_calls == 1


@pytest.mark.parametrize(
    "malformed_version",
    [True, {"release": "34"}, ["34"], float("nan"), float("inf")],
)
def test_chembl_status_malformed_version_is_not_string_coerced(
    malformed_version: object,
) -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_MalformedStatusVersionClient(malformed_version),
        python_executable=sys.executable,
        retry_backoff_seconds=0,
    )

    normalized = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    assert normalized.metadata["chembl_version"] is None
    resolution = normalized.metadata["chembl_version_resolution"]
    assert resolution["status"] == "unavailable"
    assert resolution["reason_code"] == "CHEMBL_VERSION_METADATA_INVALID"
    assert gateway._chembl_version is None


def test_mechanism_first_never_falls_back_to_unordered_activity() -> None:
    client = _OrderedActivity500Client()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        max_http_attempts=1,
        retry_backoff_seconds=0,
    )
    normalized = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(NoMeasuredLigandCandidates) as captured:
        gateway.retrieve_repurposing(
            normalized,
            direction="inhibit",
            limit=2,
        )

    bounds = captured.value.provenance["candidate_discovery_bounds"]
    assert bounds["preactivity_pool_ids"] == ["CHEMBL1", "CHEMBL2"]
    assert bounds["activity_failure_count"] == 2
    assert sorted(bounds["activity_failures_by_id"]) == [
        "CHEMBL1",
        "CHEMBL2",
    ]
    assert bounds["no_activity_first_fallback"] is True
    assert bounds["no_backfill_beyond_prewarning_pool"] is True
    assert client.ordered_calls == 2
    assert client.unordered_calls == 0


@pytest.mark.parametrize(
    ("fail_second_page", "fail_all_pages"),
    [
        (False, False),
        (True, False),
        (False, True),
    ],
)
def test_mechanism_first_never_invokes_activity_pagination_fallback(
    fail_second_page: bool,
    fail_all_pages: bool,
) -> None:
    client = _BoundedActivityPaginationClient(
        fail_second_page=fail_second_page,
        fail_all_pages=fail_all_pages,
    )
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        max_http_attempts=1,
        retry_backoff_seconds=0,
    )
    normalized = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(NoMeasuredLigandCandidates) as captured:
        gateway.retrieve_repurposing(
            normalized,
            direction="modulate",
            limit=2,
        )

    bounds = captured.value.provenance["candidate_discovery_bounds"]
    assert bounds["activity_rows_per_candidate_limit"] == 20
    assert bounds["activity_logical_call_count"] == 0
    assert bounds["activity_failure_count"] == 2
    assert bounds["fixed_pool_with_activity_ids"] == []
    assert bounds["no_activity_first_fallback"] is True
    assert client.ordered_calls == 2
    assert client.page_params == []


def test_chembl_exact_success_cache_is_run_local_and_provenance_marked() -> None:
    client = _CountingChEMBLClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
    )
    target = TargetCandidate(symbol="GENE", uniprot_id="P12345")

    first = gateway.normalize_target(target)
    second = gateway.normalize_target(target)

    assert first.metadata["chembl_target_id"] == second.metadata["chembl_target_id"]
    assert client.calls[gateway.target_endpoint] == 1
    assert second.metadata["chembl_target_resolution"]["cache_hit"] is True
    assert second.metadata["chembl_target_resolution"]["attempt_count"] == 0
    assert second.metadata["chembl_target_resolution"]["response_sha256"]
    assert gateway._json_cache


def test_chembl_operation_deadline_stops_retry_before_budget_overrun() -> None:
    clock = _ManualClock()
    client = _BudgetConsumingTimeoutClient(clock)
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=client,
        python_executable=sys.executable,
        timeout_seconds=1,
        operation_timeout_seconds=1,
        retry_backoff_seconds=0.5,
        sleeper=clock.advance,
        clock=clock,
    )

    with pytest.raises(ScientificDataUnavailable) as captured:
        gateway.normalize_target(
            TargetCandidate(symbol="GENE", uniprot_id="P12345")
        )

    assert client.calls == 1
    assert captured.value.provenance["error_type"] == "operation_timeout"
    assert captured.value.provenance["operation"] == "normalize_target"
    assert captured.value.provenance["exhausted"] is True
    assert captured.value.provenance["attempt_count"] == 1


@pytest.mark.skipif(find_spec("rdkit") is None, reason="RDKit optional dependency is absent")
def test_live_brics_readiness_executes_real_rdkit_probe() -> None:
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=_ChEMBLClient(),
        python_executable=sys.executable,
    )

    readiness = gateway._brics_readiness()

    assert readiness["status"] == "completed"
    assert readiness["fixture"] is False
    assert readiness["version"]
    execution = readiness["provenance"]["readiness_execution"]
    assert execution["exit_code"] == 0
    assert execution["stdout_sha256"]


@pytest.mark.skipif(find_spec("rdkit") is None, reason="RDKit optional dependency is absent")
def test_brics_generation_requires_measured_seed_pool() -> None:
    with pytest.raises(ValueError, match="at least two"):
        generate_brics(["CCOc1ccccc1"], limit=1, max_builds=100)
    result = generate_brics(
        [
            "CCOc1ccc(NC(=O)C)cc1",
            "CCN(CC)C(=O)c1ccc(OC)cc1",
            "COc1ccc(CCNC(=O)C)cc1",
        ],
        limit=2,
        max_builds=5_000,
    )
    assert len(result["molecules"]) == 2
    assert result["generator"] == "RDKit BRICS"


@pytest.mark.skipif(
    find_spec("rdkit") is None or find_spec("meeko") is None,
    reason="RDKit/Meeko optional dependencies are absent",
)
def test_meeko_ligand_preparation_is_real_and_versioned(tmp_path: Path) -> None:
    output = tmp_path / "ligand.pdbqt"
    result = prepare_ligand("CCOc1ccccc1", output, seed=20260819)
    assert output.is_file()
    assert "ROOT" in output.read_text(encoding="utf-8")
    assert result["rdkit_version"]
    assert result["meeko_version"]


@pytest.mark.skipif(
    find_spec("rdkit") is None or find_spec("meeko") is None,
    reason="RDKit/Meeko optional dependencies are absent",
)
def test_redocking_rmsd_fails_closed_when_reference_mapping_is_not_exact(
    tmp_path: Path,
) -> None:
    docked = tmp_path / "docked.pdbqt"
    reference = tmp_path / "reference.sdf"
    crystal = tmp_path / "crystal.pdb"
    docked.write_text("MODEL 1\nENDMDL\n", encoding="utf-8")
    reference.write_text("invalid", encoding="utf-8")
    crystal.write_text("END\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference component SDF"):
        calculate_redocking_rmsd(
            docked_pdbqt=docked,
            reference_component_sdf=reference,
            crystal_ligand_pdb=crystal,
        )


class _PrimarySearchResponse:
    def __init__(
        self,
        value: dict[str, object] | str,
        *,
        status_code: int = 200,
    ) -> None:
        self.status_code = status_code
        if isinstance(value, str):
            self.text = value
            self.content = value.encode("utf-8")
            self._payload: dict[str, object] | None = None
        else:
            self._payload = value
            self.content = json.dumps(value, sort_keys=True).encode("utf-8")

    def json(self) -> dict[str, object]:
        if self._payload is None:
            raise ValueError("not a JSON response")
        return self._payload


class _PairedPubMedClient:
    def __init__(
        self,
        records: dict[str, str],
        *,
        max_attempts: int = 1,
        failed_stage: str | None = None,
    ) -> None:
        self.records = dict(records)
        self.max_attempts = max_attempts
        self.failed_stage = failed_stage
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _PrimarySearchResponse:
        params = kwargs.get("params")
        assert isinstance(params, dict)
        if url.endswith("esearch.fcgi"):
            stage = "esearch"
        elif url.endswith("esummary.fcgi"):
            stage = "esummary"
        elif url.endswith("efetch.fcgi"):
            stage = "efetch"
        else:
            raise AssertionError(f"unexpected PubMed URL: {url}")
        self.calls.append({"stage": stage, "url": url, "params": dict(params)})
        if stage == self.failed_stage:
            return _PrimarySearchResponse({}, status_code=503)
        if stage == "esearch":
            return _PrimarySearchResponse(
                {"esearchresult": {"idlist": list(self.records)}}
            )
        if stage == "esummary":
            requested = [item for item in str(params["id"]).split(",") if item]
            return _PrimarySearchResponse(
                {
                    "result": {
                        "uids": requested,
                        **{
                            document_id: {
                                "uid": document_id,
                                "pubtype": ["Journal Article"],
                            }
                            for document_id in requested
                        },
                    }
                }
            )
        requested = [item for item in str(params["id"]).split(",") if item]
        articles = "".join(
            (
                "<PubmedArticle><MedlineCitation>"
                f"<PMID>{document_id}</PMID><Article><Abstract>"
                f'<AbstractText NlmCategory="RESULTS">{self.records[document_id]}'
                "</AbstractText></Abstract></Article></MedlineCitation>"
                "</PubmedArticle>"
            )
            for document_id in requested
        )
        return _PrimarySearchResponse(f"<PubmedArticleSet>{articles}</PubmedArticleSet>")


class _PaperNativeV5PubMedClient:
    def __init__(
        self,
        records: dict[str, str],
        *,
        max_attempts: int = 1,
        failed_stage: str | None = None,
        malformed_esearch: bool = False,
    ) -> None:
        self.records = dict(records)
        self.max_attempts = max_attempts
        self.failed_stage = failed_stage
        self.malformed_esearch = malformed_esearch
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _PrimarySearchResponse:
        params = kwargs.get("params")
        assert isinstance(params, dict)
        if url.endswith("esearch.fcgi"):
            stage = "esearch"
        elif url.endswith("esummary.fcgi"):
            stage = "esummary"
        elif url.endswith("efetch.fcgi"):
            stage = "efetch"
        else:
            raise AssertionError(f"unexpected PubMed URL: {url}")
        self.calls.append({"stage": stage, "url": url, "params": dict(params)})
        if stage == self.failed_stage:
            return _PrimarySearchResponse({}, status_code=503)
        if stage == "esearch":
            assert params["sort"] == "pub date"
            assert params["retmax"] == 8
            if self.malformed_esearch:
                return _PrimarySearchResponse({"esearchresult": {"idlist": []}})
            return _PrimarySearchResponse(
                {
                    "esearchresult": {
                        "count": str(len(self.records)),
                        "idlist": list(self.records),
                        "querytranslation": "bounded translated query",
                    }
                }
            )
        requested = [item for item in str(params["id"]).split(",") if item]
        if stage == "esummary":
            return _PrimarySearchResponse(
                {
                    "result": {
                        "uids": requested,
                        **{
                            document_id: {
                                "uid": document_id,
                                "pubtype": [
                                    "Journal Article",
                                    "Comparative Study",
                                ],
                            }
                            for document_id in requested
                        },
                    }
                }
            )
        articles = "".join(
            (
                "<PubmedArticle><MedlineCitation>"
                f"<PMID>{document_id}</PMID><Article>"
                "<ArticleTitle>Experimental kinase inhibitor comparison.</ArticleTitle>"
                "<PublicationTypeList>"
                "<PublicationType>Journal Article</PublicationType>"
                "<PublicationType>Comparative Study</PublicationType>"
                "</PublicationTypeList><Abstract>"
                '<AbstractText NlmCategory="METHODS">Inhibitory potency was measured using biochemical assays.</AbstractText>'
                f'<AbstractText NlmCategory="RESULTS">{self.records[document_id]}'
                "</AbstractText></Abstract></Article></MedlineCitation>"
                "</PubmedArticle>"
            )
            for document_id in requested
        )
        return _PrimarySearchResponse(f"<PubmedArticleSet>{articles}</PubmedArticleSet>")


class _PairedV4ChEMBLClient(_ChEMBLClient):
    ids = ("CHEMBL101", "CHEMBL102", "CHEMBL103", "CHEMBL104")
    names = {
        "CHEMBL101": "Alphatinib",
        "CHEMBL102": "Betanib",
        "CHEMBL103": "Gammanib",
        "CHEMBL104": "Controlib",
    }
    values = {
        "CHEMBL101": "17",
        "CHEMBL102": "29",
        "CHEMBL103": "39",
        "CHEMBL104": "2.9",
    }

    def __init__(
        self,
        *,
        split_assays: bool = False,
        condition_mode: str = "exact",
        alias_collision: bool = False,
        identity_count: int = 4,
        control_first: bool = False,
    ) -> None:
        assert condition_mode in {
            "exact",
            "missing",
            "ambiguous",
            "conflict",
            "activity_empty",
            "parameter_empty",
            "activity_missing_field",
            "activity_non_list",
            "activity_non_mapping",
            "parameter_missing_field",
            "parameter_non_list",
            "parameter_non_mapping",
        }
        assert 4 <= identity_count <= 13
        self.split_assays = split_assays
        self.condition_mode = condition_mode
        self.alias_collision = alias_collision
        self.ids = tuple(
            f"CHEMBL{100 + index}"
            for index in range(1, identity_count + 1)
        )
        self.names = {
            **type(self).names,
            **{
                chembl_id: f"Pooldrug{index}"
                for index, chembl_id in enumerate(self.ids, start=1)
                if chembl_id not in type(self).names
            },
        }
        self.values = {
            **type(self).values,
            **{
                chembl_id: str(40 + index)
                for index, chembl_id in enumerate(self.ids, start=1)
                if chembl_id not in type(self).values
            },
        }
        if control_first:
            self.names.update(
                {
                    "CHEMBL101": "Controlib",
                    "CHEMBL102": "Alphatinib",
                    "CHEMBL103": "Betanib",
                    "CHEMBL104": "Gammanib",
                }
            )
            self.values.update(
                {
                    "CHEMBL101": "2.9",
                    "CHEMBL102": "17",
                    "CHEMBL103": "29",
                    "CHEMBL104": "39",
                }
            )
        self.activity_parents: list[str] = []

    def assay_id(self, chembl_id: str) -> str:
        return f"ASSAY_{chembl_id}" if self.split_assays else "ASSAY_SHARED"

    def get(self, url: str, **kwargs: object) -> _Response:
        params = kwargs.get("params")
        assert isinstance(params, dict)
        if url.endswith("mechanism.json"):
            return _Response(
                {
                    "mechanisms": [
                        {
                            "mec_id": 8100 + index,
                            "record_id": 9100 + index,
                            "action_type": "INHIBITOR",
                            "direct_interaction": 1,
                            "max_phase": 4,
                            "mechanism_of_action": "Example target inhibitor",
                            "mechanism_refs": [
                                {"ref_type": "PubMed", "ref_id": str(81000000 + index)}
                            ],
                            "molecular_mechanism": 1,
                            "molecule_chembl_id": chembl_id,
                            "parent_molecule_chembl_id": chembl_id,
                            "target_chembl_id": "CHEMBL_TARGET",
                            "variant_sequence": None,
                        }
                        for index, chembl_id in enumerate(self.ids, start=1)
                    ]
                }
            )
        if url.endswith("molecule.json"):
            requested = [
                item
                for item in str(params["molecule_chembl_id__in"]).split(",")
                if item
            ]
            return _Response(
                {
                    "molecules": [
                        {
                            "molecule_chembl_id": chembl_id,
                            "pref_name": (
                                "ALPHATINIB"
                                if self.alias_collision and chembl_id == "CHEMBL102"
                                else self.names[chembl_id]
                            ),
                            "max_phase": 4,
                            "molecule_type": "Small molecule",
                            "molecule_structures": {
                                "canonical_smiles": f"CC{'C' * index}O",
                                "standard_inchi_key": f"PAIR-{index}",
                            },
                        }
                        for index, chembl_id in enumerate(requested, start=1)
                    ]
                }
            )
        if url.endswith("activity.json"):
            chembl_id = str(params["parent_molecule_chembl_id"])
            self.activity_parents.append(chembl_id)
            properties: object = (
                []
                if self.condition_mode in {"missing", "activity_empty"}
                else [
                    {
                        "type": "ATP",
                        "value": "1",
                        "units": "mM",
                    },
                    *(
                        [
                            {
                                "type": "ATP",
                                "value": "2",
                                "units": "mM",
                            }
                        ]
                        if self.condition_mode == "ambiguous"
                        else []
                    ),
                ]
            )
            if self.condition_mode == "activity_non_list":
                properties = {"type": "ATP", "value": "1", "units": "mM"}
            elif self.condition_mode == "activity_non_mapping":
                properties = ["malformed-condition-row"]
            activity = {
                "activity_id": 8200 + self.ids.index(chembl_id),
                "data_validity_comment": None,
                "molecule_chembl_id": chembl_id,
                "parent_molecule_chembl_id": chembl_id,
                "target_chembl_id": "CHEMBL_TARGET",
                "assay_type": "B",
                "assay_chembl_id": self.assay_id(chembl_id),
                "standard_type": "IC50",
                "standard_relation": "=",
                "standard_value": self.values[chembl_id],
                "standard_units": "nM",
                "pchembl_value": "8.0",
                "bao_label": "single protein format",
                "activity_properties": properties,
            }
            if self.condition_mode == "activity_missing_field":
                activity.pop("activity_properties")
            return _Response(
                {
                    "activities": [activity]
                }
            )
        if url.endswith("assay.json"):
            requested = [
                item
                for item in str(params["assay_chembl_id__in"]).split(",")
                if item
            ]
            assays: list[dict[str, object]] = []
            for assay_id in requested:
                parameters: object = (
                    []
                    if self.condition_mode in {"missing", "parameter_empty"}
                    else [
                        {
                            "standard_type": "ATP",
                            "standard_value": (
                                "2"
                                if self.condition_mode == "conflict"
                                else "1"
                            ),
                            "standard_units": "mM",
                        }
                    ]
                )
                if self.condition_mode == "parameter_non_list":
                    parameters = {
                        "standard_type": "ATP",
                        "standard_value": "1",
                        "standard_units": "mM",
                    }
                elif self.condition_mode == "parameter_non_mapping":
                    parameters = ["malformed-condition-row"]
                assay: dict[str, object] = {
                    "assay_chembl_id": assay_id,
                    "target_chembl_id": "CHEMBL_TARGET",
                    "document_chembl_id": "DOC_SHARED",
                    "assay_type": "B",
                    "assay_category": "confirmatory",
                    "assay_organism": "Homo sapiens",
                    "bao_format": "BAO_0000357",
                    "bao_label": "single protein format",
                    "confidence_score": 9,
                    "description": "One exact shared biochemical assay.",
                    "variant_sequence": None,
                    "assay_parameters": parameters,
                }
                if self.condition_mode == "parameter_missing_field":
                    assay.pop("assay_parameters")
                assays.append(assay)
            return _Response({"assays": assays})
        if url.endswith("drug_warning.json"):
            return _Response({"drug_warnings": []})
        return super().get(url, **kwargs)


_GENERATED_PAPER_NATIVE_V5_DOCUMENT = object()


class _PaperNativeV5ChEMBLClient(_PairedV4ChEMBLClient):
    def __init__(
        self,
        *,
        identity_count: int = 5,
        support_failure_id: str | None = None,
        malformed_alias_source_id: str | None = None,
        alias_source_mode: str | None = None,
        include_activity_document: bool = True,
        activity_document_value: object = (
            _GENERATED_PAPER_NATIVE_V5_DOCUMENT
        ),
        mechanism_max_phase: object = 4,
        molecule_max_phase: object = 4,
        activity_total_count_by_id: dict[str, int] | None = None,
        activity_failure_offset_by_id: dict[str, int] | None = None,
    ) -> None:
        super().__init__(split_assays=True, identity_count=identity_count)
        self.support_failure_id = support_failure_id
        self.malformed_alias_source_id = malformed_alias_source_id
        self.alias_source_mode = alias_source_mode
        self.include_activity_document = include_activity_document
        self.activity_document_value = activity_document_value
        self.mechanism_max_phase = mechanism_max_phase
        self.molecule_max_phase = molecule_max_phase
        self.activity_total_count_by_id = dict(
            activity_total_count_by_id or {}
        )
        self.activity_failure_offset_by_id = dict(
            activity_failure_offset_by_id or {}
        )
        self.document_calls = 0
        self.legacy_pubmed_calls = 0
        self.molecule_calls = 0
        self.last_mechanism_payload: dict[str, object] | None = None
        self.last_molecule_payload: dict[str, object] | None = None

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("document.json"):
            self.document_calls += 1
            params = kwargs.get("params")
            assert isinstance(params, dict)
            requested = str(params["document_chembl_id__in"]).split(",")
            return _Response(
                {
                    "documents": [
                        {
                            "document_chembl_id": document_id,
                            "doc_type": "PUBLICATION",
                            "pubmed_id": 92000000 + index,
                            "title": "Legacy document that v5 must not fetch",
                            "abstract": "Legacy abstract lookup is out of contract.",
                        }
                        for index, document_id in enumerate(requested, start=1)
                    ]
                }
            )
        if url.endswith("esummary.fcgi"):
            self.legacy_pubmed_calls += 1
            return super().get(url, **kwargs)
        if url.endswith("activity.json"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            parent_id = str(params["parent_molecule_chembl_id"])
            offset = int(params.get("offset", 0))
            if self.activity_failure_offset_by_id.get(parent_id) == offset:
                raise httpx.ReadTimeout("synthetic later activity page failure")
        response = super().get(url, **kwargs)
        if url.endswith("mechanism.json"):
            payload = response.json()
            mechanisms = payload.get("mechanisms")
            assert isinstance(mechanisms, list)
            for mechanism in mechanisms:
                assert isinstance(mechanism, dict)
                mechanism["max_phase"] = self.mechanism_max_phase
            self.last_mechanism_payload = payload
            return _Response(payload)
        if url.endswith("activity.json"):
            payload = response.json()
            activities = payload.get("activities")
            assert isinstance(activities, list)
            params = kwargs.get("params")
            assert isinstance(params, dict)
            parent_id = str(params["parent_molecule_chembl_id"])
            total_count = self.activity_total_count_by_id.get(parent_id, 1)
            offset = int(params.get("offset", 0))
            limit = int(params.get("limit", 20))
            assert activities and isinstance(activities[0], dict)
            template = dict(activities[0])
            generated: list[dict[str, object]] = []
            for absolute_index in range(
                offset, min(total_count, offset + limit)
            ):
                activity = dict(template)
                activity["activity_id"] = (
                    9_300_000
                    + self.ids.index(parent_id) * 10_000
                    + absolute_index
                    + 1
                )
                activity["pchembl_value"] = (
                    "9.0"
                    if total_count > 20 and absolute_index == total_count - 1
                    else "8.0"
                    if absolute_index == 0
                    else "5.0"
                )
                if self.include_activity_document:
                    activity["document_chembl_id"] = (
                        f"CHEMBL{930000 + self.ids.index(parent_id)}"
                        if self.activity_document_value
                        is _GENERATED_PAPER_NATIVE_V5_DOCUMENT
                        else self.activity_document_value
                    )
                generated.append(activity)
            payload["activities"] = generated
            page_meta = payload.setdefault("page_meta", {})
            assert isinstance(page_meta, dict)
            page_meta["total_count"] = total_count
            return _Response(payload)
        if url.endswith("molecule.json"):
            self.molecule_calls += 1
            payload = response.json()
            molecules = payload.get("molecules")
            assert isinstance(molecules, list)
            for molecule in molecules:
                assert isinstance(molecule, dict)
                molecule["max_phase"] = self.molecule_max_phase
                chembl_id = str(molecule["molecule_chembl_id"])
                if chembl_id == self.malformed_alias_source_id:
                    mode = self.alias_source_mode or "non_list"
                    if mode == "missing":
                        molecule.pop("molecule_synonyms", None)
                    elif mode == "null":
                        molecule["molecule_synonyms"] = None
                    elif mode == "non_list":
                        molecule["molecule_synonyms"] = {
                            "molecule_synonym": "malformed"
                        }
                    elif mode == "nonmapping_row":
                        molecule["molecule_synonyms"] = [
                            "malformed synonym row"
                        ]
                    elif mode == "empty":
                        molecule["molecule_synonyms"] = []
                    elif mode == "target_punctuation":
                        molecule["molecule_synonyms"] = [
                            {
                                "molecule_synonym": "GENE-1 inhibitor IV",
                                "syn_type": "RESEARCH_CODE",
                            }
                        ]
                    elif mode == "clean_research_code":
                        molecule["molecule_synonyms"] = [
                            {
                                "molecule_synonym": "ZX-9001",
                                "syn_type": "RESEARCH_CODE",
                            }
                        ]
                    else:
                        raise AssertionError(f"unknown alias source mode: {mode}")
                else:
                    molecule["molecule_synonyms"] = [
                        {
                            "molecule_synonym": f"{molecule['pref_name']} research",
                            "syn_type": "RESEARCH_CODE",
                        },
                        {
                            "molecule_synonym": chembl_id,
                            "syn_type": "OTHER",
                        },
                    ]
            self.last_molecule_payload = payload
            return _Response(payload)
        if url.endswith("assay.json") and self.support_failure_id is not None:
            payload = response.json()
            assays = payload.get("assays")
            assert isinstance(assays, list)
            failed_assay_id = self.assay_id(self.support_failure_id)
            for assay in assays:
                assert isinstance(assay, dict)
                if assay.get("assay_chembl_id") == failed_assay_id:
                    assay["assay_organism"] = "Mus musculus"
            return _Response(payload)
        return response


class _PaperNativeV6ChEMBLClient(_PaperNativeV5ChEMBLClient):
    """Serve one distinct, exact primary abstract for every fixed-pool row."""

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("document.json"):
            self.document_calls += 1
            params = kwargs.get("params")
            assert isinstance(params, dict)
            requested = str(params["document_chembl_id__in"]).split(",")
            documents: list[dict[str, object]] = []
            for document_id in requested:
                document_index = int(document_id.removeprefix("CHEMBL")) - 930000
                chembl_id = self.ids[document_index]
                name = self.names[chembl_id]
                value = self.values[chembl_id]
                documents.append(
                    {
                        "document_chembl_id": document_id,
                        "doc_type": "PUBLICATION",
                        "pubmed_id": 93000000 + document_index,
                        "title": f"Independent biochemical study of {name}",
                        "abstract": (
                            f"{name} inhibits GENE with an experimentally "
                            f"measured IC50 of {value} nM."
                        ),
                    }
                )
            return _Response({"documents": documents})
        return super().get(url, **kwargs)


class _PaperNativeV6LatePrimaryChEMBLClient(_PaperNativeV5ChEMBLClient):
    """Expose exact primary support only on a late, fully paginated row."""

    def __init__(self, *, late_rank: int = 81) -> None:
        super().__init__(
            identity_count=12,
            activity_total_count_by_id={"CHEMBL101": late_rank},
        )
        self.late_rank = late_rank
        self.document_bindings: dict[
            str, tuple[str, str, int, bool]
        ] = {}

    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("document.json"):
            self.document_calls += 1
            params = kwargs.get("params")
            assert isinstance(params, dict)
            requested = str(params["document_chembl_id__in"]).split(",")
            documents: list[dict[str, object]] = []
            for document_id in requested:
                chembl_id, value, rank, verified = self.document_bindings[
                    document_id
                ]
                name = self.names[chembl_id]
                abstract = (
                    f"{name} inhibits GENE with an experimentally measured "
                    f"IC50 of {value} nM."
                    if verified
                    else (
                        "An unrelated molecule was evaluated in a separate "
                        "biochemical experiment."
                    )
                )
                documents.append(
                    {
                        "document_chembl_id": document_id,
                        "doc_type": "PUBLICATION",
                        "pubmed_id": 95000000 + int(document_id.removeprefix("CHEMBL")),
                        "title": f"Primary record {rank} for {name}",
                        "abstract": abstract,
                    }
                )
            return _Response({"documents": documents})
        response = super().get(url, **kwargs)
        if url.endswith("activity.json"):
            params = kwargs.get("params")
            assert isinstance(params, dict)
            chembl_id = str(params["parent_molecule_chembl_id"])
            payload = response.json()
            activities = payload.get("activities")
            assert isinstance(activities, list)
            offset = int(params.get("offset", 0))
            for local_index, activity in enumerate(activities, start=1):
                assert isinstance(activity, dict)
                rank = offset + local_index
                activity["pchembl_value"] = "8.0"
                document_id = f"CHEMBL{100_000_000 + int(activity['activity_id'])}"
                activity["document_chembl_id"] = document_id
                self.document_bindings[document_id] = (
                    chembl_id,
                    str(activity["standard_value"]),
                    rank,
                    chembl_id != "CHEMBL101" or rank == self.late_rank,
                )
            return _Response(payload)
        return response


class _V6PublicationTypeClient:
    max_attempts = 1

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, "kwargs": dict(kwargs)})
        if not url.endswith("esummary.fcgi"):
            raise AssertionError(
                "paper-native V6 must not execute paired-v4 search/fetch"
            )
        params = kwargs.get("params")
        assert isinstance(params, dict)
        requested = [item for item in str(params["id"]).split(",") if item]
        return _Response(
            {
                "result": {
                    "uids": requested,
                    **{
                        pmid: {"uid": pmid, "pubtype": ["Journal Article"]}
                        for pmid in requested
                    },
                }
            }
        )


class _V6InvalidAssaySchemaClient(_PaperNativeV6ChEMBLClient):
    def get(self, url: str, **kwargs: object) -> _Response:
        if url.endswith("assay.json"):
            return _Response({"assays": {"not": "a list"}})
        return super().get(url, **kwargs)


class _V6InvalidPubMedSchemaClient(_V6PublicationTypeClient):
    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, "kwargs": dict(kwargs)})
        return _Response({"result": {"uids": "not-a-list"}})


def test_pubmed_type_batches_report_global_multi_batch_semantics_exactly(
) -> None:
    chembl = _PaperNativeV6ChEMBLClient(identity_count=12)
    primary = _V6PublicationTypeClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=chembl,
        primary_search_client=primary,
        max_http_attempts=2,
        retry_backoff_seconds=0,
    )
    documents = {
        f"CHEMBL{800_000 + index}": {"pubmed_id": str(10_000_000 + index)}
        for index in range(301)
    }

    result = gateway._fetch_pubmed_publication_types(documents)

    assert len(primary.calls) == 2
    first = result["10000000"]["batch"]
    last = result["10000300"]["batch"]
    assert first["batch_index"] == 1
    assert first["batch_count"] == 2
    assert first["global_eligible_pmid_count"] == 301
    assert first["single_batch"] is False
    assert first["requested_count"] == 300
    assert last["batch_index"] == 2
    assert last["batch_count"] == 2
    assert last["global_eligible_pmid_count"] == 301
    assert last["single_batch"] is False
    assert last["requested_count"] == 1


@pytest.mark.parametrize(
    ("invalid_assay", "failed_stage", "reason_code"),
    [
        (True, "assay", "ASSAY_RESPONSE_INVALID"),
        (False, "pubmed_type", "PUBMED_RESULT_UID_SET_INVALID"),
    ],
)
def test_v6_source_inventory_promotes_http_200_schema_errors_to_failures(
    invalid_assay: bool,
    failed_stage: str,
    reason_code: str,
) -> None:
    chembl = (
        _V6InvalidAssaySchemaClient(identity_count=12)
        if invalid_assay
        else _PaperNativeV6ChEMBLClient(identity_count=12)
    )
    primary = (
        _V6PublicationTypeClient()
        if invalid_assay
        else _V6InvalidPubMedSchemaClient()
    )
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=chembl,
        primary_search_client=primary,
        max_http_attempts=2,
        retry_backoff_seconds=0,
        repurposing_evidence_policy=PAPER_NATIVE_V6_POLICY_TOKEN,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seeds = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=3)
    )
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    stage = bounds["source_call_inventory"]["stages"][failed_stage]

    assert stage["successful_call_count"] >= 1
    assert stage["failure_count"] >= 1
    assert all(
        item["failure"]["retrieval_status"] == "partial"
        and item["failure"]["reason_code"] == reason_code
        for item in stage["failures"]
    )
    assert bounds["source_call_inventory"]["failure_count"] >= 1


_PAIRED_THREE = (
    "Alphatinib was the most potent GENE inhibitor "
    "(IC50 at physiological [1mM] ATP, 17nM), followed by "
    "Betanib (29nM), and Gammanib (39nM), and the most selective."
)
_PAIRED_FOUR = (
    "Alphatinib was the most potent GENE inhibitor "
    "(IC50 at physiological [1mM] ATP, 17nM), followed by "
    "Betanib (29nM), Gammanib (39nM), and Controlib (2.9nM), "
    "and the most selective."
)
_PAIRED_RESERVED_THREE = (
    "Alphatinib was the most potent GENE inhibitor "
    "(IC50 at physiological [1mM] ATP, 17nM), followed by "
    "Betanib (29nM), and Controlib (2.9nM), and the most selective."
)


def _run_paired_v4(
    primary_client: _PairedPubMedClient,
    *,
    split_assays: bool = False,
    condition_mode: str = "exact",
    alias_collision: bool = False,
    identity_count: int = 4,
    limit: int = 4,
    control_first: bool = False,
) -> tuple[list[object], _PairedV4ChEMBLClient, LiveScientificGateway]:
    chembl = _PairedV4ChEMBLClient(
        split_assays=split_assays,
        condition_mode=condition_mode,
        alias_collision=alias_collision,
        identity_count=identity_count,
        control_first=control_first,
    )
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=chembl,
        primary_search_client=primary_client,
        max_http_attempts=1,
        retry_backoff_seconds=0,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )
    seeds = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=limit)
    )
    return seeds, chembl, gateway


def test_paired_v4_live_gateway_owns_distinct_single_attempt_pubmed_pool() -> None:
    gateway = LiveScientificGateway(docking=_Docking(), admet=_ADMET())
    try:
        assert gateway.primary_search_client is not gateway.client
        assert gateway.primary_search_client.max_attempts == 1
        assert gateway.client.max_attempts == 2
    finally:
        gateway.close()


def test_paper_native_v6_uses_independent_candidate_primary_rows_and_never_paired_v4(
) -> None:
    chembl = _PaperNativeV6ChEMBLClient(identity_count=12)
    primary = _V6PublicationTypeClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=chembl,
        primary_search_client=primary,
        max_http_attempts=2,
        retry_backoff_seconds=0,
        repurposing_evidence_policy=PAPER_NATIVE_V6_POLICY_TOKEN,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seeds = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=3)
    )

    assert primary.calls
    assert all(
        str(call["url"]).endswith("esummary.fcgi")
        for call in primary.calls
    )
    assert chembl.legacy_pubmed_calls == 0
    pubmed_provenance = [
        call for call in gateway._http_calls if call.source == "PubMed"
    ]
    assert len(pubmed_provenance) == len(primary.calls) == 1
    assert pubmed_provenance[0].max_attempts == 1
    assert pubmed_provenance[0].attempt_count == 1
    assert any(
        call["source"] == "PubMed"
        for call in gateway.last_operation_provenance["http_calls"]
    )
    assert [seed.metadata["chembl_id"] for seed in seeds] == list(chembl.ids)
    assert len(seeds) == 12
    seen_primary_ids: set[tuple[str, str]] = set()
    for index, seed in enumerate(seeds):
        evidence_metadata = seed.evidence[0].metadata
        primary = evidence_metadata["primary_source"]
        selection = primary["activity_document_selection"]
        expected_document_id = f"CHEMBL{930000 + index}"
        expected_pmid = str(93000000 + index)
        assert primary["verification_status"] == "verified"
        assert primary["is_primary_support"] is True
        assert primary["document_chembl_id"] == expected_document_id
        assert primary["pmid"] == expected_pmid
        binding = primary["activity_binding"]
        expected_mechanism = (
            "ChEMBL-curated inhibitor mechanism and measured direct "
            "IC50 activity against GENE"
        )
        assert binding["assay_type"] == seed.activities[0].assay_type
        assert binding["assay_context"] == (
            "direct_single_protein_context"
        )
        assert binding["direct_single_protein_engagement"] is True
        assert binding["direction_evidence_rule"] == (
            "measured_ic50_endpoint"
        )
        assert binding["mechanism_identity_sha256"] == _canonical_sha256(
            seed.metadata["mechanism_identity"]
        )
        assert binding["mechanism_claim"] == expected_mechanism
        assert binding["mechanism_claim_sha256"] == _canonical_sha256(
            expected_mechanism
        )
        assert seed.mechanism == expected_mechanism
        assert chembl.names[chembl.ids[index]] in primary["excerpt"]
        assert "primary_abstract_paired_v4" not in evidence_metadata
        document_batch = primary["document_query_batch"]
        assert document_batch["schema_version"] == (
            "chembl_document_query_batch_v1"
        )
        assert document_batch["document_count"] == len(seeds)
        assert document_batch["document_chembl_ids"] == sorted(
            f"CHEMBL{930000 + item}" for item in range(len(seeds))
        )
        assert document_batch["selected_document_chembl_id"] == (
            expected_document_id
        )
        assert document_batch["selected_document_in_batch"] is True
        assert document_batch["query_parameters"] == {
            "document_chembl_id__in": ",".join(
                document_batch["document_chembl_ids"]
            ),
            "limit": len(seeds),
        }
        assert document_batch["query_parameter_hash"] == primary[
            "document_query"
        ]["parameter_hash"]
        assert document_batch["truncated"] is False
        assert selection["selected_by_candidate_primary_record_v6"] is True
        assert selection["paper_native_v6_paired_search_or_selection_used"] is False
        assert selection["selected_by_pre_pubmed_paired_v4_cohort"] is False
        assert selection["paired_v4_selection_precedes_pubmed_search"] is False
        assert selection["policy"] == (
            "complete_exact_parent_activity_document_rows_v1"
        )
        assert selection["top_k"] is None
        assert selection["result_row_cap"] is None
        assert selection["all_eligible_rows_inspected"] is True
        assert selection["truncated"] is False
        count = seed.metadata["candidate_discovery_bounds"][
            "activity_counts_by_id"
        ][seed.metadata["chembl_id"]]
        selection_activity_ids = [
            row["activity_id"] for row in selection["rows"]
        ]
        assert selection_activity_ids == count[
            "eligible_directional_activity_ids"
        ]
        assert selection_activity_ids == count[
            "retained_document_activity_ids"
        ]
        assert count["eligible_directional_count"] == len(
            selection_activity_ids
        )
        assert count["retained_document_rows"] == len(
            selection_activity_ids
        )
        assert count["eligible_directional_activity_ids_sha256"] == (
            _canonical_sha256(selection_activity_ids)
        )
        assert count["retained_document_activity_ids_sha256"] == (
            _canonical_sha256(selection_activity_ids)
        )
        seen_primary_ids.add((primary["document_chembl_id"], primary["pmid"]))
    assert len(seen_primary_ids) == len(seeds)
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    molecule_projection = bounds["fixed_pool_molecule_projection"]
    assert molecule_projection["schema_version"] == (
        "chembl_fixed_pool_molecule_projection_v1"
    )
    assert molecule_projection["complete"] is True
    assert molecule_projection["fixed_pool_candidate_ids"] == list(
        chembl.ids
    )
    assert molecule_projection["molecule_query"] == bounds["molecule_query"]
    assert molecule_projection["row_count"] == len(seeds)
    for index, (seed, row) in enumerate(
        zip(seeds, molecule_projection["rows"], strict=True)
    ):
        assert row["pool_index"] == index
        assert row["molecule_chembl_id"] == seed.metadata["chembl_id"]
        assert row["canonical_parent_molecule_chembl_id"] == (
            seed.metadata["chembl_id"]
        )
        assert row["canonical_smiles"] == seed.smiles
        assert row["canonical_smiles"] == seed.metadata["canonical_smiles"]
        assert row["max_phase"] == seed.metadata["max_phase"]
        assert row["exact_aliases"] == seed.metadata["chembl_exact_aliases"]
        assert row["exact_aliases"] == seed.evidence[0].metadata[
            "primary_source"
        ]["candidate_exact_aliases"]
        assert row["molecule_query_parameter_sha256"] == bounds[
            "molecule_query"
        ]["parameter_hash"]
        assert row["molecule_response_sha256"] == bounds["molecule_query"][
            "response_sha256"
        ]
        assert row["projection_complete"] is True
        assert row["fixture"] is False
    delegate = bounds["paper_native_v6_delegate_primary_selection"]
    assert delegate == {
        "schema_version": "paper_native_v6_delegate_primary_selection_v1",
        "policy": PAPER_NATIVE_V6_POLICY_TOKEN,
        "selection_scope": "candidate_specific_verified_primary_record",
        "paired_v4_search_executed": False,
        "paired_v4_promotion_used": False,
        "paired_v4_retained_index_used": False,
        "candidate_ordering_changed": False,
        "fixture": False,
    }
    seed_projection = bounds["delegate_seed_source_binding_projection"]
    assert seed_projection == gateway.last_operation_provenance[
        "delegate_seed_source_binding_projection"
    ]
    assert seed_projection["fixed_pool_candidate_ids"] == list(chembl.ids)
    assert seed_projection["delegate_return_candidate_ids"] == list(
        chembl.ids
    )
    assert seed_projection["row_count"] == len(seeds)
    assert seed_projection["complete"] is True
    for index, (seed, row) in enumerate(
        zip(seeds, seed_projection["rows"], strict=True)
    ):
        primary = seed.evidence[0].metadata["primary_source"]
        assert row["pool_index"] == index
        assert row["fixed_pool_rank"] == index + 1
        assert row["molecule_chembl_id"] == seed.metadata["chembl_id"]
        assert row["delegate_returned"] is True
        assert row["activity_id"] == str(seed.evidence[0].external_id)
        assert row["document_chembl_id"] == primary["document_chembl_id"]
        assert row["pmid"] == primary["pmid"]
        assert row["primary_attestation_sha256"] == _canonical_sha256(
            primary
        )


def test_paper_native_v6_audits_all_activity_rows_and_selects_rank_81_primary(
) -> None:
    chembl = _PaperNativeV6LatePrimaryChEMBLClient(late_rank=81)
    primary_client = _V6PublicationTypeClient()
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=chembl,
        primary_search_client=primary_client,
        max_http_attempts=2,
        retry_backoff_seconds=0,
        repurposing_evidence_policy=PAPER_NATIVE_V6_POLICY_TOKEN,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    seeds = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=3)
    )

    first = seeds[0]
    evidence = first.evidence[0]
    primary = evidence.metadata["primary_source"]
    selection = primary["activity_document_selection"]
    bounds = first.metadata["candidate_discovery_bounds"]
    assert _paper_v6_source_inventory_valid(bounds)
    count = bounds["activity_counts_by_id"]["CHEMBL101"]
    assert primary["verification_status"] == "verified"
    audited_candidates = [
        SimpleNamespace(
            name=seed.name,
            target=target,
            metadata=seed.metadata,
            evidence=seed.evidence,
            activities=seed.activities,
        )
        for seed in seeds
    ]
    assert MA3AssuranceAgent._is_verified_candidate_primary_evidence(
        audited_candidates[0],
        audited_candidates[0].activities[0],
        evidence,
        pool_candidates=audited_candidates,
    )
    assert evidence.external_id == str(9_300_000 + 81)
    assert selection["policy"] == (
        "complete_exact_parent_activity_document_rows_v1"
    )
    assert selection["eligible_row_count"] == 81
    assert selection["inspected_row_count"] == 81
    assert selection["selected_rank"] == 81
    assert selection["top_k"] is None
    assert selection["result_row_cap"] is None
    assert selection["truncated"] is False
    assert selection["all_eligible_rows_inspected"] is True
    assert [row["verification_status"] for row in selection["rows"][:80]] == [
        "unverified"
    ] * 80
    assert selection["rows"][80]["verification_status"] == "verified"
    assert count["total_count"] == 81
    assert count["inspected_count"] == 81
    assert count["retained_document_rows"] == 81
    selection_activity_ids = [
        row["activity_id"] for row in selection["rows"]
    ]
    assert count["eligible_directional_activity_ids"] == (
        selection_activity_ids
    )
    assert count["retained_document_activity_ids"] == (
        selection_activity_ids
    )
    assert count["eligible_directional_activity_ids_sha256"] == (
        _canonical_sha256(selection_activity_ids)
    )
    assert count["retained_document_activity_ids_sha256"] == (
        _canonical_sha256(selection_activity_ids)
    )
    assert count["primary_document_audit_result_cap"] is None
    assert count["all_eligible_rows_retained_for_primary_audit"] is True
    assert count["truncated"] is False
    assert count["complete_result_set"] is True
    assert [page["offset"] for page in count["page_audits"]] == [
        0, 20, 40, 60, 80
    ]
    # 81 exact documents for CHEMBL101 plus one for each other pool member
    # cross the 80-ID document wire batch without truncation.
    assert chembl.document_calls == 2
    assert len(primary["document_query_batch"]["document_chembl_ids"]) <= 80
    assert primary["document_query_batch"]["selected_document_in_batch"] is True
    assert all(
        str(call["url"]).endswith("esummary.fcgi")
        for call in primary_client.calls
    )
    assert chembl.legacy_pubmed_calls == 0
    batch = primary["document_query_batch"]
    batch["selected_document_chembl_id"] = next(
        item
        for item in batch["document_chembl_ids"]
        if item != primary["document_chembl_id"]
    )
    body = dict(batch)
    body.pop("attestation_sha256")
    batch["attestation_sha256"] = sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert not MA3AssuranceAgent._is_verified_candidate_primary_evidence(
        audited_candidates[0],
        audited_candidates[0].activities[0],
        evidence,
        pool_candidates=audited_candidates,
    )


def test_public_wrapper_accepts_and_preserves_actual_paired_v4_delegate_contract(
    tmp_path: Path,
) -> None:
    primary = _PairedPubMedClient({"90000001": _PAIRED_THREE})
    chembl = _PairedV4ChEMBLClient()
    delegate = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=chembl,
        primary_search_client=primary,
        max_http_attempts=1,
        retry_backoff_seconds=0,
    )
    wrapper = PublicRepurposingGateway(
        drugcentral_dataset_path=tmp_path / "intentionally-unused.tsv",
        delegate=delegate,
        repurposing_evidence_policy=PAIRED_V4_REQUIRED_POLICY,
    )
    try:
        seeds = wrapper.retrieve_repurposing(
            TargetCandidate(symbol="GENE", uniprot_id="P12345"),
            direction="inhibit",
            limit=3,
        )

        assert [seed.metadata["chembl_id"] for seed in seeds] == list(
            chembl.ids
        )
        assert wrapper.last_operation_provenance[
            "paired_v4_envelope_candidate_ids"
        ] == list(chembl.ids[:3])
        assert wrapper.last_operation_provenance[
            "delegate_pool_preserved_exactly"
        ] is True
    finally:
        delegate.close()


def test_paired_v4_promotes_three_exact_pairs_without_changing_chembl_activity() -> None:
    primary = _PairedPubMedClient({"90000001": _PAIRED_THREE})
    seeds, chembl, _gateway = _run_paired_v4(primary)

    assert [seed.metadata["chembl_id"] for seed in seeds] == list(chembl.ids)
    assert chembl.activity_parents == list(chembl.ids)
    promoted = seeds[:3]
    assert len(primary.calls) == 3
    signatures = set()
    for seed in promoted:
        evidence = seed.evidence[0]
        envelope = evidence.metadata["primary_abstract_paired_v4"]
        assert evidence.source == "ChEMBL"
        assert seed.activities[0].source_evidence_id == evidence.evidence_id
        assert evidence.metadata["primary_source"]["schema_version"] == (
            "candidate_primary_evidence_v2"
        )
        assert envelope["schema_version"] == "candidate_primary_evidence_paired_v4"
        assert envelope["verification_status"] == "verified"
        assert envelope["pmid"] == "90000001"
        assert envelope["selected_measurement"]["exact_decimal_value_match"] is True
        assert envelope["selected_measurement"]["exact_unit_match"] is True
        assert envelope["selected_measurement"]["unit_conversion_applied"] is False
        assert envelope["pubmed_or_result_used_for_cohort_selection"] is False
        condition = envelope["selected_assay_condition_binding"]
        assert condition["status"] == "exact"
        assert (condition["subject"], condition["value_raw"], condition["unit_raw"]) == (
            "ATP",
            "1",
            "mM",
        )
        assert condition == envelope["selected_assay_signature"][
            "assay_condition_binding"
        ]
        assert condition["source_inventory_count"] == 2
        assert condition["source_inventory_complete"] is True
        assert len(condition["source_inventory_sha256"]) == 64
        assert [
            item["source"] for item in condition["source_inventory"]
        ] == [
            "activity_properties",
            "chembl_assay_context.record.assay_parameters",
        ]
        for item in condition["source_inventory"]:
            assert item["field_present"] is True
            assert item["list_typed"] is True
            assert item["row_count"] == 1
            assert item["mapping_row_count"] == 1
            assert item["non_mapping_row_indices"] == []
            assert item["complete"] is True
            assert len(item["rows_sha256"]) == 64
            inventory_body = dict(item)
            inventory_sha256 = inventory_body.pop("inventory_sha256")
            assert sha256(
                json.dumps(
                    inventory_body,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest() == inventory_sha256
        assert envelope["selected_measurement"]["pair_assay_condition"][
            "subject"
        ] == "ATP"
        signatures.add(envelope["selected_assay_signature"]["signature_sha256"])
        body = dict(envelope)
        attestation = body.pop("attestation_sha256")
        assert sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest() == attestation
    assert len(signatures) == 1
    assert "primary_abstract_paired_v4" not in seeds[3].evidence[0].metadata
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    audit = bounds["primary_abstract_search"]
    assert audit["transport"] == {
        "dedicated_from_chembl_client": True,
        "max_attempts": 1,
        "required_max_attempts": 1,
        "logical_call_limit": 3,
    }
    assert len(audit["search_outcome"]["calls"]) == 3
    inventory = audit["alias_inventory"]
    assert inventory["records_scope"] == "pre_pubmed_frozen_cohort_query_subset"
    assert inventory["fixed_pool_alias_record_count"] == 4
    assert inventory["pre_pubmed_frozen_cohort"][
        "pubmed_or_result_used_for_cohort_selection"
    ] is False
    assert inventory["pre_pubmed_frozen_cohort"]["selected_candidate_count"] == 4


def test_paired_v4_limit_three_freezes_top_twelve_before_result_and_keeps_control(
) -> None:
    positive = _PairedPubMedClient({"90000001": _PAIRED_RESERVED_THREE})
    mismatch = _PairedPubMedClient(
        {
            "90000001": _PAIRED_RESERVED_THREE.replace(
                "Controlib (2.9nM)", "Controlib (3.0nM)"
            )
        }
    )
    positive_seeds, positive_chembl, positive_gateway = _run_paired_v4(
        positive,
        identity_count=13,
        limit=3,
        control_first=True,
    )
    mismatch_seeds, mismatch_chembl, mismatch_gateway = _run_paired_v4(
        mismatch,
        identity_count=13,
        limit=3,
        control_first=True,
    )

    fixed_top_twelve = list(positive_chembl.ids[:12])
    outside_fixed_pool = positive_chembl.ids[12]
    assert positive_chembl.activity_parents == fixed_top_twelve
    assert mismatch_chembl.activity_parents == fixed_top_twelve
    assert outside_fixed_pool not in positive_chembl.activity_parents
    assert outside_fixed_pool not in mismatch_chembl.activity_parents
    positive_returned = [
        seed.metadata["chembl_id"] for seed in positive_seeds
    ]
    mismatch_returned = [
        seed.metadata["chembl_id"] for seed in mismatch_seeds
    ]
    assert positive_returned == mismatch_returned == fixed_top_twelve
    for seeds, gateway in (
        (positive_seeds, positive_gateway),
        (mismatch_seeds, mismatch_gateway),
    ):
        bounds = seeds[0].metadata["candidate_discovery_bounds"]
        assert bounds["prewarning_pool_limit"] == 12
        assert bounds["prewarning_pool_ids"] == fixed_top_twelve
        assert bounds["preactivity_pool_ids"] == fixed_top_twelve
        assert bounds["returned_candidate_ordering_policy"] == (
            "surviving_fixed_prewarning_clinical_phase_desc_then_"
            "stable_parent_molecule_chembl_id_subsequence_v1"
        )
        assert bounds["activity_query_attempted_ids"] == fixed_top_twelve
        assert bounds["fixed_pool_with_activity_ids"] == fixed_top_twelve
        assert bounds["requested_eligible_count"] == 3
        assert bounds["gateway_return_candidate_ids"] == fixed_top_twelve
        assert bounds["gateway_return_candidate_count"] == 12
        assert bounds["gateway_return_cap_applied"] is False
        assert bounds["eligible_quota_evaluation_owner"] == "MA2"
        assert bounds["gateway_returned_from_fixed_prewarning_pool"] is True
        assert bounds[
            "gateway_return_order_matches_fixed_prewarning_subsequence"
        ] is True
        assert bounds["no_backfill_beyond_prewarning_pool"] is True
        assert outside_fixed_pool in bounds["preactivity_excluded_identity_ids"]
        assert gateway.last_operation_provenance["requested_limit"] == 3
        assert gateway.last_operation_provenance["requested_eligible_count"] == 3
        assert gateway.last_operation_provenance["candidate_count"] == 12
        assert gateway.last_operation_provenance[
            "gateway_return_cap_applied"
        ] is False
        assert gateway.last_operation_provenance[
            "gateway_return_order_matches_fixed_prewarning_subsequence"
        ] is True
    envelope = positive_seeds[0].evidence[0].metadata[
        "primary_abstract_paired_v4"
    ]
    assert set(envelope["group"]["candidate_ids"]) == {
        "CHEMBL101",
        "CHEMBL102",
        "CHEMBL103",
    }
    assert envelope["same_assay_cohort_candidate_ids"] == [
        "CHEMBL101",
        "CHEMBL102",
        "CHEMBL103",
    ]
    assert envelope["alias_inventory"]["fixed_pool_alias_record_count"] == 12
    assert envelope["alias_inventory"]["pre_pubmed_frozen_cohort"][
        "selected_candidate_ids"
    ] == fixed_top_twelve
    assert positive_seeds[0].name == "Controlib"
    assert positive_seeds[0].metadata["chembl_id"] == "CHEMBL101"
    assert positive.calls and mismatch.calls


@pytest.mark.parametrize(
    ("condition_mode", "empty_source"),
    [
        ("activity_empty", "activity_properties"),
        (
            "parameter_empty",
            "chembl_assay_context.record.assay_parameters",
        ),
    ],
)
def test_paired_v4_explicit_empty_condition_source_is_complete_when_other_is_exact(
    condition_mode: str,
    empty_source: str,
) -> None:
    primary = _PairedPubMedClient({"90000001": _PAIRED_THREE})
    seeds, _chembl, _gateway = _run_paired_v4(
        primary,
        condition_mode=condition_mode,
    )

    assert len(primary.calls) == 3
    condition = seeds[0].evidence[0].metadata[
        "primary_abstract_paired_v4"
    ]["selected_assay_condition_binding"]
    assert condition["status"] == "exact"
    assert condition["source_inventory_complete"] is True
    inventory = {
        item["source"]: item for item in condition["source_inventory"]
    }
    empty = inventory[empty_source]
    assert empty["field_present"] is True
    assert empty["list_typed"] is True
    assert empty["row_count"] == 0
    assert empty["mapping_row_count"] == 0
    assert empty["non_mapping_row_indices"] == []
    assert empty["complete"] is True
    assert empty["rows_sha256"] == sha256(b"[]").hexdigest()


@pytest.mark.parametrize(
    ("condition_mode", "source", "reason_code"),
    [
        (
            "activity_missing_field",
            "activity_properties",
            "CONDITION_SOURCE_FIELD_MISSING",
        ),
        (
            "activity_non_list",
            "activity_properties",
            "CONDITION_SOURCE_FIELD_NOT_LIST",
        ),
        (
            "activity_non_mapping",
            "activity_properties",
            "CONDITION_SOURCE_ROW_NOT_OBJECT",
        ),
        (
            "parameter_missing_field",
            "chembl_assay_context.record.assay_parameters",
            "CONDITION_SOURCE_FIELD_MISSING",
        ),
        (
            "parameter_non_list",
            "chembl_assay_context.record.assay_parameters",
            "CONDITION_SOURCE_FIELD_NOT_LIST",
        ),
        (
            "parameter_non_mapping",
            "chembl_assay_context.record.assay_parameters",
            "CONDITION_SOURCE_ROW_NOT_OBJECT",
        ),
    ],
)
def test_paired_v4_malformed_condition_source_inventory_fails_before_search(
    condition_mode: str,
    source: str,
    reason_code: str,
) -> None:
    primary = _PairedPubMedClient({"90000001": _PAIRED_THREE})
    seeds, _chembl, _gateway = _run_paired_v4(
        primary,
        condition_mode=condition_mode,
    )

    assert primary.calls == []
    assert all(
        "primary_abstract_paired_v4" not in seed.evidence[0].metadata
        for seed in seeds
    )
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    frozen = audit["alias_inventory"]["pre_pubmed_frozen_cohort"]
    binding = frozen["retained_top_k_rows"][0][
        "assay_condition_binding"
    ]
    assert binding["status"] == "unverified"
    assert binding["reason_code"] == (
        "ASSAY_CONDITION_RECORD_INCOMPLETE_OR_CONFLICTING"
    )
    assert binding["source_inventory_count"] == 2
    assert binding["source_inventory_complete"] is False
    assert len(binding["source_inventory_sha256"]) == 64
    source_record = next(
        item for item in binding["source_inventory"]
        if item["source"] == source
    )
    assert source_record["complete"] is False
    observed_reasons = {
        code
        for record in binding["invalid_records"]
        if record["source"] == source
        for code in record["reason_codes"]
    }
    assert reason_code in observed_reasons
    if reason_code == "CONDITION_SOURCE_FIELD_MISSING":
        assert source_record["field_present"] is False
        assert source_record["row_count"] is None
    elif reason_code == "CONDITION_SOURCE_FIELD_NOT_LIST":
        assert source_record["field_present"] is True
        assert source_record["list_typed"] is False
        assert source_record["row_count"] is None
        assert len(source_record["observed_value_sha256"]) == 64
    else:
        assert source_record["field_present"] is True
        assert source_record["list_typed"] is True
        assert source_record["row_count"] == 1
        assert source_record["mapping_row_count"] == 0
        assert source_record["non_mapping_row_indices"] == [0]
        assert len(source_record["rows_sha256"]) == 64
    assert audit["search_outcome"]["calls"] == []


def test_paired_v4_under_three_pairs_never_promotes_or_changes_pool() -> None:
    sentence = (
        "Alphatinib was the most potent GENE inhibitor "
        "(IC50 at physiological [1mM] ATP, 17nM), followed by "
        "Betanib (29nM), and the most selective."
    )
    primary = _PairedPubMedClient({"90000001": sentence})
    seeds, chembl, _gateway = _run_paired_v4(primary)

    assert [seed.metadata["chembl_id"] for seed in seeds] == list(chembl.ids)
    assert all(
        "primary_abstract_paired_v4" not in seed.evidence[0].metadata
        for seed in seeds
    )
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["search_outcome"]["status"] == "not_verified"
    assert audit["promotion"]["promoted_candidate_ids"] == []


def test_paired_v4_partial_http_ledger_fails_closed_without_promotion() -> None:
    primary = _PairedPubMedClient(
        {"90000001": _PAIRED_THREE},
        failed_stage="efetch",
    )
    seeds, _chembl, _gateway = _run_paired_v4(primary)

    assert len(primary.calls) == 3
    assert all(
        "primary_abstract_paired_v4" not in seed.evidence[0].metadata
        for seed in seeds
    )
    outcome = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]["search_outcome"]
    assert outcome["status"] == "failed"
    assert outcome["calls"][-1]["stage"] == "efetch"
    assert outcome["calls"][-1]["status"] == "failed"


def test_paired_v4_value_mismatch_cannot_select_an_alternate_activity() -> None:
    mismatch = _PAIRED_THREE.replace("Betanib (29nM)", "Betanib (30nM)")
    primary = _PairedPubMedClient({"90000001": mismatch})
    seeds, _chembl, _gateway = _run_paired_v4(primary)

    assert all(
        "primary_abstract_paired_v4" not in seed.evidence[0].metadata
        for seed in seeds
    )
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["search_outcome"]["status"] == "verified"
    assert audit["promotion"]["reason_code"] == (
        "NO_THREE_CANDIDATE_EXACT_ASSAY_MATCH"
    )
    frozen = audit["alias_inventory"]["pre_pubmed_frozen_cohort"]
    assert frozen["selected_candidate_rows"][1]["retained_rank"] == 1


def test_paired_v4_cross_group_value_conflict_rejects_all_groups() -> None:
    conflict = _PAIRED_THREE.replace("17nM", "18nM", 1)
    primary = _PairedPubMedClient(
        {"90000001": _PAIRED_THREE, "90000002": conflict}
    )
    seeds, _chembl, _gateway = _run_paired_v4(primary)

    assert all(
        "primary_abstract_paired_v4" not in seed.evidence[0].metadata
        for seed in seeds
    )
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["search_outcome"]["status"] == "verified"
    assert audit["promotion"]["reason_code"] == (
        "CROSS_GROUP_VALUE_OR_UNIT_MISMATCH"
    )
    assert audit["promotion"]["cross_group_conflict_candidate_ids"] == [
        "CHEMBL101"
    ]


def test_paired_v4_reserved_named_member_stays_in_fixed_pool_and_group() -> None:
    primary = _PairedPubMedClient({"90000001": _PAIRED_FOUR})
    seeds, chembl, _gateway = _run_paired_v4(primary)

    assert [seed.metadata["chembl_id"] for seed in seeds] == list(chembl.ids)
    control = seeds[-1]
    envelope = control.evidence[0].metadata["primary_abstract_paired_v4"]
    assert envelope["group"]["candidate_ids"] == list(chembl.ids)
    assert envelope["same_assay_cohort_candidate_ids"] == list(chembl.ids)
    assert envelope["selected_measurement"]["molecule_chembl_id"] == "CHEMBL104"


def test_paired_v4_no_prepubmed_three_candidate_assay_cohort_makes_no_http_call() -> None:
    primary = _PairedPubMedClient({"90000001": _PAIRED_THREE})
    seeds, chembl, _gateway = _run_paired_v4(primary, split_assays=True)

    assert primary.calls == []
    assert [seed.metadata["chembl_id"] for seed in seeds] == list(chembl.ids)
    assert all(
        "primary_abstract_paired_v4" not in seed.evidence[0].metadata
        for seed in seeds
    )
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["search_outcome"]["reason_code"] == (
        "NO_PRE_PUBMED_THREE_CANDIDATE_EXACT_ASSAY_COHORT"
    )
    assert audit["search_outcome"]["calls"] == []


def test_paired_v4_non_single_attempt_client_is_rejected_before_http() -> None:
    primary = _PairedPubMedClient(
        {"90000001": _PAIRED_THREE},
        max_attempts=2,
    )
    seeds, _chembl, _gateway = _run_paired_v4(primary)

    assert primary.calls == []
    assert all(
        "primary_abstract_paired_v4" not in seed.evidence[0].metadata
        for seed in seeds
    )
    outcome = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]["search_outcome"]
    assert outcome["status"] == "failed"
    assert outcome["reason_code"] == "client_retry_policy_not_single_attempt"
    assert outcome["calls"] == []


def test_paired_v4_accepts_source_alias_target_and_endpoint_case_variation() -> None:
    mixed_case = (
        _PAIRED_THREE.replace("Alphatinib", "alphatinib")
        .replace("GENE", "gene")
        .replace("IC50", "ic50")
        .replace("Betanib", "BETANIB")
        .replace("Gammanib", "gammanib")
    )
    primary = _PairedPubMedClient({"90000001": mixed_case})
    seeds, _chembl, _gateway = _run_paired_v4(primary)

    assert len(primary.calls) == 3
    for seed in seeds[:3]:
        envelope = seed.evidence[0].metadata["primary_abstract_paired_v4"]
        assert envelope["verification_status"] == "verified"
        assert envelope["target_binding"]["target_alias"] == "gene"
        assert envelope["selected_assay_condition_binding"]["subject"] == "ATP"


def test_paired_v4_nfkc_casefold_alias_owner_collision_fails_before_http() -> None:
    primary = _PairedPubMedClient({"90000001": _PAIRED_THREE})
    seeds, _chembl, _gateway = _run_paired_v4(
        primary,
        alias_collision=True,
    )

    assert primary.calls == []
    assert all(
        "primary_abstract_paired_v4" not in seed.evidence[0].metadata
        for seed in seeds
    )
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["alias_inventory"]["candidate_alias_conflict_ids"] == [
        "CHEMBL101",
        "CHEMBL102",
    ]
    assert audit["alias_inventory"]["complete"] is False
    assert audit["search_outcome"]["reason_code"] == (
        "PRE_PUBMED_ALIAS_INVENTORY_INCOMPLETE"
    )


def test_paired_v4_paper_condition_must_exactly_match_frozen_chembl_condition() -> None:
    two_mm_atp = _PAIRED_THREE.replace("[1mM] ATP", "[2mM] ATP")
    primary = _PairedPubMedClient({"90000001": two_mm_atp})
    seeds, _chembl, _gateway = _run_paired_v4(primary)

    assert len(primary.calls) == 3
    assert all(
        "primary_abstract_paired_v4" not in seed.evidence[0].metadata
        for seed in seeds
    )
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["search_outcome"]["status"] == "verified"
    assert audit["promotion"]["reason_code"] == (
        "PRIMARY_ABSTRACT_ASSAY_CONDITION_MISMATCH"
    )
    assert audit["promotion"]["assay_condition_mismatch_candidate_ids"] == [
        "CHEMBL101",
        "CHEMBL102",
        "CHEMBL103",
    ]
    binding = audit["alias_inventory"]["pre_pubmed_frozen_cohort"][
        "selected_signature"
    ]["assay_condition_binding"]
    assert (binding["subject"], binding["value_raw"], binding["unit_raw"]) == (
        "ATP",
        "1",
        "mM",
    )


@pytest.mark.parametrize(
    ("condition_mode", "reason_code"),
    [
        ("missing", "ASSAY_CONDITION_MISSING"),
        ("ambiguous", "ASSAY_CONDITION_AMBIGUOUS_OR_CONFLICTING"),
        ("conflict", "ASSAY_CONDITION_AMBIGUOUS_OR_CONFLICTING"),
    ],
)
def test_paired_v4_unresolved_chembl_condition_prevents_search_and_promotion(
    condition_mode: str,
    reason_code: str,
) -> None:
    primary = _PairedPubMedClient({"90000001": _PAIRED_THREE})
    seeds, _chembl, _gateway = _run_paired_v4(
        primary,
        condition_mode=condition_mode,
    )

    assert primary.calls == []
    assert all(
        "primary_abstract_paired_v4" not in seed.evidence[0].metadata
        for seed in seeds
    )
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    frozen = audit["alias_inventory"]["pre_pubmed_frozen_cohort"]
    assert frozen["selection_status"] == "not_selected"
    assert frozen["reason_code"] == (
        "NO_PRE_PUBMED_THREE_CANDIDATE_EXACT_ASSAY_COHORT"
    )
    assert frozen["retained_top_k_rows"][0]["assay_condition_binding"][
        "reason_code"
    ] == reason_code
    assert audit["search_outcome"]["calls"] == []


_PAPER_NATIVE_V5_POLICY = (
    "dynamic_pubmed_primary_abstract_paper_native_v5_required"
)
_PAPER_NATIVE_V5_GROUP_FOUR = (
    "Alphatinib was the most potent GENE inhibitor "
    "(half-maximal inhibitory concentration [IC50] at physiological "
    "[1mM] ATP, 501nM), followed by Betanib (702nM), "
    "Gammanib (903nM), and Controlib (1104nM), and the most selective."
)


def _run_paper_native_v5(
    primary_client: _PaperNativeV5PubMedClient,
    *,
    support_failure_id: str | None = None,
    malformed_alias_source_id: str | None = None,
    alias_source_mode: str | None = None,
    include_activity_document: bool = True,
    activity_document_value: object = _GENERATED_PAPER_NATIVE_V5_DOCUMENT,
    target_symbol: str = "GENE",
    mechanism_max_phase: object = 4,
    molecule_max_phase: object = 4,
    activity_total_count_by_id: dict[str, int] | None = None,
    activity_failure_offset_by_id: dict[str, int] | None = None,
) -> tuple[list[object], _PaperNativeV5ChEMBLClient, LiveScientificGateway]:
    chembl = _PaperNativeV5ChEMBLClient(
        support_failure_id=support_failure_id,
        malformed_alias_source_id=malformed_alias_source_id,
        alias_source_mode=alias_source_mode,
        include_activity_document=include_activity_document,
        activity_document_value=activity_document_value,
        mechanism_max_phase=mechanism_max_phase,
        molecule_max_phase=molecule_max_phase,
        activity_total_count_by_id=activity_total_count_by_id,
        activity_failure_offset_by_id=activity_failure_offset_by_id,
    )
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=chembl,
        primary_search_client=primary_client,
        max_http_attempts=2,
        retry_backoff_seconds=0,
        repurposing_evidence_policy=_PAPER_NATIVE_V5_POLICY,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol=target_symbol, uniprot_id="P12345")
    )
    seeds = list(
        gateway.retrieve_repurposing(target, direction="inhibit", limit=3)
    )
    return seeds, chembl, gateway


def test_paper_native_v5_promotes_one_indivisible_group_from_fixed_top_five(
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    seeds, chembl, _gateway = _run_paper_native_v5(primary)

    assert [seed.metadata["chembl_id"] for seed in seeds] == list(chembl.ids)
    assert all(
        seed.metadata["regulatory_status"]
        == "unknown_no_matching_chembl_warning_record"
        for seed in seeds
    )
    assert all(seed.metadata["drug_warnings"] == [] for seed in seeds)
    assert chembl.activity_parents == list(chembl.ids)
    assert [call["stage"] for call in primary.calls] == [
        "esearch",
        "esummary",
        "efetch",
    ]
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    assert bounds["prewarning_pool_ids"] == list(chembl.ids)
    assert bounds["paper_native_v5_direct_support_candidate_ids"] == list(
        chembl.ids
    )
    snapshots = bounds["paper_native_v5_direct_support_snapshots"]
    assert [item["molecule_chembl_id"] for item in snapshots] == list(
        chembl.ids
    )
    assert len(bounds["paper_native_v5_direct_support_snapshots_sha256"]) == 64
    audit = bounds["primary_abstract_search"]
    assert audit["schema_version"] == "primary_abstract_paper_native_gateway_v5"
    assert audit["policy"] == _PAPER_NATIVE_V5_POLICY
    assert audit["search_outcome"]["search_sort"] == "pub date"
    assert audit["search_outcome"]["status"] == "verified"
    assert audit["promotion"]["verified_member_candidate_ids"] == list(
        chembl.ids[:4]
    )
    assert audit["no_paired_v4_search_or_fallback"] is True
    support_inventory = audit["direct_support_inventory"]
    assert len(support_inventory["fixed_pool_identity_bindings"]) == len(
        chembl.ids
    )
    assert len(support_inventory["fixed_pool_identity_bindings_sha256"]) == 64
    for record in support_inventory["records"]:
        binding = record["candidate_identity_binding"]
        assert binding["binding_status"] == "exact"
        assert binding["reason_codes"] == []
        assert binding["target_chembl_id"] == "CHEMBL_TARGET"
        assert record["candidate_identity_binding_sha256"] == binding[
            "binding_sha256"
        ]
        for row_audit in record["row_audits"]:
            assert row_audit["assay_context"]["record"]["bao_format"] == (
                "BAO_0000357"
            )
            assert len(row_audit["assay_context_sha256"]) == 64
            assert row_audit["chembl_activity_query"]["source"] == "ChEMBL"
            assert row_audit["chembl_activity_query_parameters"][
                "parent_molecule_chembl_id"
            ] == record["molecule_chembl_id"]
            assert row_audit["chembl_activity_query_parameters_sha256"] == (
                row_audit["chembl_activity_query"]["parameter_hash"]
            )
            assert len(row_audit["chembl_activity_query_sha256"]) == 64
            assert len(
                row_audit[
                    "chembl_activity_query_candidate_binding_sha256"
                ]
            ) == 64

    for seed in seeds[:4]:
        envelope = seed.evidence[0].metadata[
            "primary_abstract_paper_native_v5"
        ]
        assert envelope == seed.metadata["primary_abstract_paper_native_v5"]
        assert envelope["schema_version"] == (
            "candidate_primary_evidence_paper_native_v5"
        )
        assert envelope["source_contract"] == (
            "primary_abstract_paper_native_v5"
        )
        assert envelope["verification_status"] == "verified"
        assert envelope["molecule_chembl_id"] == seed.metadata["chembl_id"]
        assert envelope["group_member_candidate_ids"] == list(chembl.ids[:4])
        assert envelope["selected_group_sha256"] == envelope["group"][
            "group_sha256"
        ]
        assert envelope["selected_pair_evidence_sha256"] == envelope[
            "selected_pair"
        ]["evidence_sha256"]
        assert envelope["cross_source_numeric_equality_required"] is False
        assert envelope["cross_source_numeric_comparison_performed"] is False
        assert envelope["pubmed_value_used_for_activity_selection"] is False
        assert envelope["pubmed_value_used_for_candidate_ordering"] is False
        support = envelope["pre_pubmed_chembl_support"]
        matching_snapshot = next(
            item
            for item in snapshots
            if item["molecule_chembl_id"] == seed.metadata["chembl_id"]
        )
        assert support == matching_snapshot
        assert support["typed_measurement_tuple"]["value"] == (
            seed.activities[0].value
        )
        assert support["pubmed_value_used_for_selection"] is False
        body = dict(envelope)
        attestation = body.pop("attestation_sha256")
        assert sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest() == attestation
        assert "primary_abstract_paired_v4" not in seed.evidence[0].metadata

    nonmember = seeds[4].metadata["primary_abstract_paper_native_v5"]
    assert nonmember["verification_status"] == "not_verified"
    assert nonmember["reason_code"] == "NOT_IN_SELECTED_PAPER_NATIVE_V5_GROUP"
    assert nonmember["pre_pubmed_chembl_support"] is not None
    assert nonmember["group"] is None


@pytest.mark.parametrize("wire_phase", [4.0, "4.0"])
def test_paper_native_v5_canonicalizes_integral_decimal_wire_phase_and_promotes_group(
    wire_phase: object,
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    seeds, chembl, _gateway = _run_paper_native_v5(
        primary,
        mechanism_max_phase=wire_phase,
        molecule_max_phase=wire_phase,
    )

    assert [seed.metadata["chembl_id"] for seed in seeds] == list(chembl.ids)
    assert chembl.activity_parents == list(chembl.ids)
    assert [call["stage"] for call in primary.calls] == [
        "esearch",
        "esummary",
        "efetch",
    ]
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    assert bounds["paper_native_v5_direct_support_candidate_ids"] == list(
        chembl.ids
    )
    assert len(bounds["raw_mechanism_rows_sha256"]) == 64
    assert len(bounds["molecule_response_rows_sha256"]) == 64
    assert chembl.last_mechanism_payload is not None
    assert chembl.last_molecule_payload is not None
    raw_mechanism_rows = chembl.last_mechanism_payload["mechanisms"]
    raw_molecule_rows = chembl.last_molecule_payload["molecules"]
    assert isinstance(raw_mechanism_rows, list)
    assert isinstance(raw_molecule_rows, list)
    assert all(row["max_phase"] == wire_phase for row in raw_molecule_rows)
    assert bounds["raw_mechanism_rows_sha256"] == sha256(
        json.dumps(
            raw_mechanism_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert bounds["molecule_response_rows_sha256"] == sha256(
        json.dumps(
            raw_molecule_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected_mechanism_response_sha256 = sha256(
        json.dumps(
            chembl.last_mechanism_payload,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    audit = bounds["primary_abstract_search"]
    assert audit["search_outcome"]["status"] == "verified"
    assert audit["promotion"]["verified_member_candidate_ids"] == list(
        chembl.ids[:4]
    )
    for binding in audit["direct_support_inventory"][
        "fixed_pool_identity_bindings"
    ]:
        assert binding["binding_status"] == "exact"
        assert binding["identity_resolution"]["max_phase"] == 4
        assert type(binding["identity_resolution"]["max_phase"]) is int
        assert binding["molecule_metadata"] == {
            "molecule_chembl_id": binding["molecule_chembl_id"],
            "max_phase": 4,
            "molecule_type": "Small molecule",
        }
        assert all(
            row["max_phase"] == wire_phase
            for row in binding["mechanism_provenance"]["rows"]
        )
        assert len(
            binding["mechanism_provenance"]["query"]["response_sha256"]
        ) == 64
        assert binding["mechanism_provenance"]["query"][
            "response_sha256"
        ] == expected_mechanism_response_sha256


@pytest.mark.parametrize(
    ("molecule_max_phase", "expected_reason"),
    [
        ("4.5", "CLINICAL_PHASE_UNRESOLVED"),
        ("not-a-phase", "CLINICAL_PHASE_UNRESOLVED"),
        ("Infinity", "CLINICAL_PHASE_UNRESOLVED"),
        (True, "CLINICAL_PHASE_UNRESOLVED"),
        ("0.0", "CLINICAL_PHASE_UNRESOLVED"),
        (-1, "CLINICAL_PHASE_UNRESOLVED"),
        ("5.0", "CLINICAL_PHASE_UNRESOLVED"),
        (99, "CLINICAL_PHASE_UNRESOLVED"),
        ("3.0", "MECHANISM_MOLECULE_PHASE_CONFLICT"),
    ],
)
def test_paper_native_v5_invalid_or_conflicting_wire_phase_fails_before_activity_and_pubmed(
    molecule_max_phase: object,
    expected_reason: str,
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    chembl = _PaperNativeV5ChEMBLClient(
        mechanism_max_phase="4.0",
        molecule_max_phase=molecule_max_phase,
    )
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=chembl,
        primary_search_client=primary,
        max_http_attempts=2,
        retry_backoff_seconds=0,
        repurposing_evidence_policy=_PAPER_NATIVE_V5_POLICY,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(NoMeasuredLigandCandidates) as captured:
        gateway.retrieve_repurposing(target, direction="inhibit", limit=3)

    assert captured.value.reason_code == (
        "INSUFFICIENT_FIXED_POOL_EXACT_PARENT_ACTIVITY"
    )
    bounds = captured.value.provenance["candidate_discovery_bounds"]
    assert bounds["prewarning_pool_ids"] == []
    assert bounds["activity_logical_call_count"] == 0
    assert bounds["activity_query_attempted_count"] == 0
    assert chembl.activity_parents == []
    assert primary.calls == []
    assert set(bounds["preactivity_exclusion_reason_codes_by_id"]) == set(
        chembl.ids
    )
    assert all(
        expected_reason in reason_codes
        for reason_codes in bounds[
            "preactivity_exclusion_reason_codes_by_id"
        ].values()
    )


@pytest.mark.parametrize(
    "mechanism_max_phase",
    ["4.5", "not-a-phase", "Infinity", True, "0.0", -1, "5.0", 99],
)
def test_paper_native_v5_invalid_mechanism_phase_fails_before_molecule_activity_and_pubmed(
    mechanism_max_phase: object,
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    chembl = _PaperNativeV5ChEMBLClient(
        mechanism_max_phase=mechanism_max_phase,
        molecule_max_phase="4.0",
    )
    gateway = LiveScientificGateway(
        docking=_Docking(),
        admet=_ADMET(),
        client=chembl,
        primary_search_client=primary,
        max_http_attempts=2,
        retry_backoff_seconds=0,
        repurposing_evidence_policy=_PAPER_NATIVE_V5_POLICY,
    )
    target = gateway.normalize_target(
        TargetCandidate(symbol="GENE", uniprot_id="P12345")
    )

    with pytest.raises(NoMeasuredLigandCandidates) as captured:
        gateway.retrieve_repurposing(target, direction="inhibit", limit=3)

    assert captured.value.reason_code == "NO_VALID_BOUNDED_MECHANISM_IDENTITIES"
    assert all(
        "MECHANISM_CLINICAL_PHASE_INVALID" in row["reason_codes"]
        for row in captured.value.provenance["malformed_mechanism_rows"]
    )
    assert chembl.molecule_calls == 0
    assert chembl.activity_parents == []
    assert primary.calls == []


def test_paper_native_v5_real_activity_documents_never_trigger_legacy_lookups(
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    seeds, chembl, _gateway = _run_paper_native_v5(
        primary,
        include_activity_document=True,
    )

    assert [call["stage"] for call in primary.calls] == [
        "esearch",
        "esummary",
        "efetch",
    ]
    assert chembl.document_calls == 0
    assert chembl.legacy_pubmed_calls == 0
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    provenance = bounds["primary_abstract_search"]
    expected = {
        "legacy_primary_document_lookup_executed": False,
        "legacy_primary_document_lookup_candidate_count": 0,
        "legacy_primary_document_lookup_candidate_ids": [],
        "legacy_primary_document_lookup_document_count": 0,
        "legacy_primary_document_lookup_document_ids": [],
        "legacy_pubmed_type_lookup_executed": False,
        "legacy_pubmed_type_lookup_candidate_count": 0,
        "legacy_pubmed_type_lookup_candidate_ids": [],
        "legacy_pubmed_type_lookup_pmid_count": 0,
        "legacy_pubmed_type_lookup_pmid_ids": [],
    }
    for key, value in expected.items():
        assert bounds[key] == value
        assert provenance[key] == value
    assert bounds["document_lookup_candidate_identity_count"] == 0
    assert bounds["document_lookup_candidate_identity_ids"] == []
    assert bounds["document_lookup_id_count"] == 0
    assert bounds["pubmed_type_lookup_candidate_identity_count"] == 0
    assert bounds["pubmed_type_lookup_candidate_identity_ids"] == []
    assert bounds["pubmed_type_lookup_pmid_count"] == 0
    snapshots = bounds["paper_native_v5_direct_support_snapshots"]
    for snapshot in snapshots:
        expected_document_id = (
            f"CHEMBL{930000 + chembl.ids.index(snapshot['molecule_chembl_id'])}"
        )
        assert snapshot["document_chembl_id"] == expected_document_id
        assert snapshot["raw_measurement_tuple"][
            "document_chembl_id"
        ] == expected_document_id
    for seed in seeds:
        evidence_metadata = seed.evidence[0].metadata
        assert "primary_source" not in evidence_metadata
        separation = evidence_metadata["primary_source_contract"]
        assert separation["schema_version"] == (
            "paper_native_v5_independent_source_separation_v1"
        )
        assert separation["legacy_candidate_primary_evidence_v2_applicable"] is False
        assert separation["legacy_chembl_document_metadata_lookup_executed"] is False
        assert separation["same_document_binding_required"] is False
        assert separation["independent_measurements_not_equated"] is True
        assert separation["chembl_document_chembl_id"].startswith("CHEMBL")
        assert len(separation["attestation_sha256"]) == 64


@pytest.mark.parametrize(
    ("include_activity_document", "activity_document_value"),
    [
        pytest.param(
            False,
            _GENERATED_PAPER_NATIVE_V5_DOCUMENT,
            id="missing",
        ),
        pytest.param(True, None, id="null"),
        pytest.param(True, "", id="empty"),
        pytest.param(True, " CHEMBL930000 ", id="whitespace-repaired"),
        pytest.param(True, 930000, id="numeric"),
        pytest.param(True, "DOC_CHEMBL101", id="malformed"),
    ],
)
def test_paper_native_v5_invalid_activity_document_fails_support_before_pubmed(
    include_activity_document: bool,
    activity_document_value: object,
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    seeds, chembl, _gateway = _run_paper_native_v5(
        primary,
        include_activity_document=include_activity_document,
        activity_document_value=activity_document_value,
    )

    assert [seed.metadata["chembl_id"] for seed in seeds] == list(chembl.ids)
    assert primary.calls == []
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    assert bounds["paper_native_v5_direct_support_candidate_ids"] == []
    assert bounds["paper_native_v5_direct_support_snapshots"] == []
    audit = bounds["primary_abstract_search"]
    assert audit["search_outcome"]["calls"] == []
    assert audit["search_outcome"]["reason_code"] == (
        "FEWER_THAN_THREE_PRE_PUBMED_DIRECT_SUPPORT_CANDIDATES"
    )
    for record in audit["direct_support_inventory"]["records"]:
        assert record["selection_status"] == "not_selected"
        assert "DOCUMENT_PROVENANCE_INCOMPLETE" in record[
            "observed_row_reason_codes"
        ]
        assert record["selected_support"] is None
        assert record["row_audits"]
        assert all(
            row_audit["document_chembl_id"] is None
            and "DOCUMENT_PROVENANCE_INCOMPLETE"
            in row_audit["reason_codes"]
            for row_audit in record["row_audits"]
        )


def test_paper_native_v5_member_support_failure_rejects_whole_group_without_partial(
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    seeds, chembl, _gateway = _run_paper_native_v5(
        primary,
        support_failure_id="CHEMBL104",
    )

    assert [seed.metadata["chembl_id"] for seed in seeds] == list(chembl.ids)
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    assert bounds["paper_native_v5_direct_support_candidate_ids"] == [
        "CHEMBL101",
        "CHEMBL102",
        "CHEMBL103",
        "CHEMBL105",
    ]
    assert bounds["primary_abstract_search"]["search_outcome"]["groups"] == []
    assert bounds["primary_abstract_search"]["promotion"][
        "verified_member_candidate_ids"
    ] == []
    assert all(
        seed.metadata["primary_abstract_paper_native_v5"][
            "verification_status"
        ]
        == "not_verified"
        for seed in seeds
    )
    assert all(
        seed.metadata["primary_abstract_paper_native_v5"]["group"] is None
        for seed in seeds
    )


def test_paper_native_v5_activity_total_above_page_size_reads_every_row(
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    seeds, chembl, _gateway = _run_paper_native_v5(
        primary,
        activity_total_count_by_id={"CHEMBL101": 21},
    )

    assert [seed.metadata["chembl_id"] for seed in seeds] == list(chembl.ids)
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    count = bounds["activity_counts_by_id"]["CHEMBL101"]
    assert count["total_count"] == 21
    assert count["inspected_count"] == 21
    assert count["inspected_cap"] is None
    assert count["truncated"] is False
    assert count["complete_result_set"] is True
    assert count["page_size"] == 20
    assert count["page_count"] == 2
    assert [page["offset"] for page in count["page_audits"]] == [0, 20]
    assert [page["response_count"] for page in count["page_audits"]] == [
        20,
        1,
    ]
    assert count["eligible_directional_count"] == 21
    assert count["retained_document_rows"] == 3
    resolution = next(
        item
        for item in bounds["identity_resolution"]
        if item["parent_molecule_chembl_id"] == "CHEMBL101"
    )
    assert resolution["status"] == "selected"
    assert "ACTIVITY_TOTAL_COUNT_EXCEEDS_BOUND" not in resolution["reason_codes"]
    assert "CHEMBL101" in bounds[
        "paper_native_v5_direct_support_candidate_ids"
    ]
    assert bounds["activity_complete_pagination_required"] is True
    assert bounds["activity_rows_per_candidate_limit_semantics"] == (
        "wire_page_size_not_result_cap"
    )
    assert bounds["activity_result_row_limit"] is None


def test_paper_native_v5_later_activity_page_failure_discards_partial_rows(
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    seeds, chembl, _gateway = _run_paper_native_v5(
        primary,
        activity_total_count_by_id={"CHEMBL101": 42},
        activity_failure_offset_by_id={"CHEMBL101": 20},
    )

    assert [seed.metadata["chembl_id"] for seed in seeds] == list(
        chembl.ids[1:]
    )
    bounds = seeds[0].metadata["candidate_discovery_bounds"]
    failure = bounds["activity_failures_by_id"]["CHEMBL101"]
    assert failure["reason_code"] == (
        "ACTIVITY_COMPLETE_PAGINATION_UNAVAILABLE"
    )
    assert failure["completed_row_count"] == 20
    assert failure["partial_rows_discarded"] is True
    assert "CHEMBL101" not in bounds["fixed_pool_with_activity_ids"]
    assert "CHEMBL101" not in bounds[
        "paper_native_v5_direct_support_candidate_ids"
    ]


def test_paper_native_v5_pubmed_value_swap_cannot_change_activity_or_order(
) -> None:
    swapped = (
        _PAPER_NATIVE_V5_GROUP_FOUR.replace("501nM", "1104nM", 1)
        .replace("702nM", "903nM", 1)
        .replace("903nM", "702nM", 1)
        .replace("1104nM", "501nM", 1)
    )
    baseline, _chembl_a, _gateway_a = _run_paper_native_v5(
        _PaperNativeV5PubMedClient(
            {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
        )
    )
    altered, _chembl_b, _gateway_b = _run_paper_native_v5(
        _PaperNativeV5PubMedClient({"91000001": swapped})
    )

    projection = lambda seeds: [
        (
            seed.metadata["chembl_id"],
            seed.evidence[0].external_id,
            seed.activities[0].endpoint,
            seed.activities[0].relation,
            seed.activities[0].value,
            seed.activities[0].unit,
        )
        for seed in seeds
    ]
    assert projection(baseline) == projection(altered)
    assert all(
        seed.metadata["primary_abstract_paper_native_v5"][
            "pubmed_value_used_for_activity_selection"
        ]
        is False
        for seed in altered
    )


def test_paper_native_v5_http_failure_has_no_paired_v4_fallback() -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR},
        failed_stage="esearch",
    )
    seeds, _chembl, _gateway = _run_paper_native_v5(primary)

    assert [call["stage"] for call in primary.calls] == ["esearch"]
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["policy"] == _PAPER_NATIVE_V5_POLICY
    assert audit["no_paired_v4_search_or_fallback"] is True
    assert audit["search_outcome"]["status"] == "failed"
    assert all(
        "primary_abstract_paired_v4" not in seed.evidence[0].metadata
        for seed in seeds
    )
    assert all(
        seed.metadata["primary_abstract_paper_native_v5"][
            "verification_status"
        ]
        == "not_verified"
        for seed in seeds
    )


def test_default_policy_preserves_existing_paired_v4_serialization() -> None:
    seeds, _chembl, gateway = _run_paired_v4(
        _PairedPubMedClient({"90000001": _PAIRED_THREE})
    )

    assert gateway.repurposing_evidence_policy is None
    assert "primary_abstract_paired_v4" in seeds[0].evidence[0].metadata
    assert all(
        "primary_abstract_paper_native_v5" not in seed.evidence[0].metadata
        for seed in seeds
    )
    assert seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]["schema_version"] == "primary_abstract_search_gateway_v1"


def test_paper_native_v5_malformed_alias_source_fails_closed_before_http(
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    seeds, _chembl, _gateway = _run_paper_native_v5(
        primary,
        malformed_alias_source_id="CHEMBL103",
    )

    assert primary.calls == []
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["alias_inventory"]["complete"] is False
    malformed = next(
        item
        for item in audit["alias_inventory"]["records"]
        if item["molecule_chembl_id"] == "CHEMBL103"
    )
    assert "MOLECULE_SYNONYMS_NOT_LIST" in malformed[
        "malformed_source_codes"
    ]
    assert audit["search_outcome"]["calls"] == []
    assert all(
        seed.metadata["primary_abstract_paper_native_v5"][
            "verification_status"
        ]
        == "not_verified"
        for seed in seeds
    )


@pytest.mark.parametrize(
    ("source_mode", "reason_code", "row_count"),
    [
        ("missing", "MOLECULE_SYNONYMS_KEY_MISSING", None),
        ("null", "MOLECULE_SYNONYMS_NULL", None),
        ("non_list", "MOLECULE_SYNONYMS_NOT_LIST", None),
        ("nonmapping_row", "MOLECULE_SYNONYM_ROW_NOT_OBJECT", 1),
    ],
)
def test_paper_native_v5_incomplete_synonym_source_has_hashed_inventory_and_no_pubmed(
    source_mode: str,
    reason_code: str,
    row_count: int | None,
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    seeds, chembl, _gateway = _run_paper_native_v5(
        primary,
        malformed_alias_source_id="CHEMBL103",
        alias_source_mode=source_mode,
        include_activity_document=True,
    )

    assert primary.calls == []
    assert chembl.document_calls == 0
    assert chembl.legacy_pubmed_calls == 0
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["alias_inventory"]["complete"] is False
    record = next(
        item
        for item in audit["alias_inventory"]["records"]
        if item["molecule_chembl_id"] == "CHEMBL103"
    )
    source_inventory = record["molecule_synonyms_source_inventory"]
    assert source_inventory["source_row_count"] == row_count
    assert reason_code in source_inventory["reason_codes"]
    assert reason_code in record["molecule_synonyms_source_reason_codes"]
    assert reason_code in record["malformed_source_codes"]
    assert record["molecule_synonyms_source_sha256"] == source_inventory[
        "inventory_sha256"
    ]
    assert len(record["molecule_synonyms_source_sha256"]) == 64
    assert audit["search_outcome"]["reason_code"] == (
        "PRE_PUBMED_ALIAS_INVENTORY_INCOMPLETE"
    )
    assert audit["search_outcome"]["calls"] == []


def test_paper_native_v5_explicit_empty_synonym_list_is_complete_source(
) -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR}
    )
    seeds, _chembl, _gateway = _run_paper_native_v5(
        primary,
        malformed_alias_source_id="CHEMBL103",
        alias_source_mode="empty",
    )

    assert [call["stage"] for call in primary.calls] == [
        "esearch",
        "esummary",
        "efetch",
    ]
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["alias_inventory"]["complete"] is True
    record = next(
        item
        for item in audit["alias_inventory"]["records"]
        if item["molecule_chembl_id"] == "CHEMBL103"
    )
    source_inventory = record["molecule_synonyms_source_inventory"]
    assert source_inventory["source_key_present"] is True
    assert source_inventory["source_value_type"] == "list"
    assert source_inventory["source_row_count"] == 0
    assert source_inventory["rows"] == []
    assert source_inventory["reason_codes"] == []
    assert record["source_attestation_complete"] is True
    assert record["query_eligible"] is True


def test_paper_native_v5_target_punctuation_equivalence_is_rejected_but_clean_code_is_retained(
) -> None:
    target_sentence = _PAPER_NATIVE_V5_GROUP_FOUR.replace("GENE", "GENE1")
    target_primary = _PaperNativeV5PubMedClient({"91000001": target_sentence})
    target_seeds, _target_chembl, _gateway = _run_paper_native_v5(
        target_primary,
        malformed_alias_source_id="CHEMBL103",
        alias_source_mode="target_punctuation",
        target_symbol="GENE1",
    )
    target_inventory = target_seeds[0].metadata[
        "candidate_discovery_bounds"
    ]["primary_abstract_search"]["alias_inventory"]
    target_record = next(
        item
        for item in target_inventory["records"]
        if item["molecule_chembl_id"] == "CHEMBL103"
    )
    assert {
        (item["alias"], item["reason_code"])
        for item in target_record["rejected_aliases"]
    } >= {("GENE-1 inhibitor IV", "TARGET_CONTAINING_ALIAS")}
    assert "GENE-1 inhibitor IV" not in target_record["accepted_aliases"]

    clean_primary = _PaperNativeV5PubMedClient({"91000001": target_sentence})
    clean_seeds, _clean_chembl, _gateway = _run_paper_native_v5(
        clean_primary,
        malformed_alias_source_id="CHEMBL103",
        alias_source_mode="clean_research_code",
        target_symbol="GENE1",
    )
    clean_inventory = clean_seeds[0].metadata[
        "candidate_discovery_bounds"
    ]["primary_abstract_search"]["alias_inventory"]
    clean_record = next(
        item
        for item in clean_inventory["records"]
        if item["molecule_chembl_id"] == "CHEMBL103"
    )
    assert clean_inventory["complete"] is True
    assert "ZX-9001" in clean_record["accepted_aliases"]
    assert [call["stage"] for call in clean_primary.calls] == [
        "esearch",
        "esummary",
        "efetch",
    ]


@pytest.mark.parametrize(
    "unsafe_sentence",
    [
        _PAPER_NATIVE_V5_GROUP_FOUR.replace(
            "most potent GENE", "most potent phase3 GENE"
        ),
        _PAPER_NATIVE_V5_GROUP_FOUR.replace(
            "Alphatinib was the most potent",
            "Alphatinib was predicted in silico the most potent",
        ),
    ],
)
def test_paper_native_v5_gateway_rejects_contextual_or_computational_sentences(
    unsafe_sentence: str,
) -> None:
    primary = _PaperNativeV5PubMedClient({"91000001": unsafe_sentence})
    seeds, _chembl, _gateway = _run_paper_native_v5(primary)

    assert [call["stage"] for call in primary.calls] == [
        "esearch",
        "esummary",
        "efetch",
    ]
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["search_outcome"]["groups"] == []
    assert audit["promotion"]["verified_member_candidate_ids"] == []
    assert all(
        seed.metadata["primary_abstract_paper_native_v5"][
            "verification_status"
        ]
        == "not_verified"
        for seed in seeds
    )


def test_paper_native_v5_malformed_pubmed_payload_fails_closed() -> None:
    primary = _PaperNativeV5PubMedClient(
        {"91000001": _PAPER_NATIVE_V5_GROUP_FOUR},
        malformed_esearch=True,
    )
    seeds, _chembl, _gateway = _run_paper_native_v5(primary)

    assert [call["stage"] for call in primary.calls] == ["esearch"]
    audit = seeds[0].metadata["candidate_discovery_bounds"][
        "primary_abstract_search"
    ]
    assert audit["search_outcome"]["status"] == "failed"
    assert audit["promotion"]["verified_member_candidate_ids"] == []
    assert all(
        seed.metadata["primary_abstract_paper_native_v5"]["group"] is None
        for seed in seeds
    )


def test_paper_native_v5_gateway_contains_no_synthetic_scenario_literals() -> None:
    source = (
        Path(__file__).parents[2]
        / "src"
        / "drugpilot"
        / "tools"
        / "live_ma2_gateway.py"
    ).read_text(encoding="utf-8")
    for scenario_literal in (
        "Alphatinib",
        "Betanib",
        "Gammanib",
        "Controlib",
        "91000001",
    ):
        assert scenario_literal not in source
