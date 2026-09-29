from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal


AccessKind = Literal[
    "public_api",
    "federated_api",
    "optional_sdk",
    "local_dataset",
    "credential_required",
    "cloud_dataset",
    "evaluation_dataset",
]

SourceStatusKind = Literal[
    "connected_adapter",
    "federated",
    "available_not_default",
    "catalog_only",
    "not_configured",
    "configured_no_adapter",
    "evaluation_only",
]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id: str
    name: str
    stages: tuple[str, ...]
    access: AccessKind
    endpoint: str
    requirement: str | None = None
    note: str = ""

    def public_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceConfiguration:
    """Non-secret source configuration used to build a truthful registry.

    The registry is intentionally a description of installed access paths, not
    a network health check.  API credentials are represented only by booleans.
    """

    public_evidence_enabled: bool = True
    public_candidate_enrichment_enabled: bool = False
    drugcentral_dataset_configured: bool = False
    drugcentral_dataset_available: bool = False
    archs4_dataset_configured: bool = False
    archs4_dataset_available: bool = False
    cellxgene_sidecar_configured: bool = False
    biogrid_credential_configured: bool = False
    google_patents_project_configured: bool = False
    drugbank_credential_configured: bool = False


@dataclass(frozen=True, slots=True)
class SourceStatus:
    source_id: str
    name: str
    stages: tuple[str, ...]
    access: AccessKind
    endpoint: str
    status: SourceStatusKind
    adapter: str | None
    default_queried: bool
    configured: bool
    reachability: Literal["not_checked"] = "not_checked"
    configuration_state: str = "ready"
    requirement: str | None = None
    note: str = ""

    def public_dict(self) -> dict[str, object]:
        return asdict(self)


# Source names follow the inventory stated in DrugPilot_final.pdf. ``stages``
# records the component that directly issues the query, not every downstream
# consumer: MA3 audits persisted evidence and does not re-query these sources.
# Stage adapters query on demand and the API reports access requirements rather
# than presenting gated sources as connected.
RUNTIME_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("pubmed", "PubMed", ("MA1",), "public_api", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"),
    SourceSpec("europe_pmc", "Europe PMC", ("MA1",), "public_api", "https://www.ebi.ac.uk/europepmc/webservices/rest/"),
    SourceSpec("biorxiv", "bioRxiv", ("MA1",), "federated_api", "https://www.ebi.ac.uk/europepmc/webservices/rest/", note="Retrieved through Europe PMC preprint metadata."),
    SourceSpec("medrxiv", "medRxiv", ("MA1",), "federated_api", "https://www.ebi.ac.uk/europepmc/webservices/rest/", note="Retrieved through Europe PMC preprint metadata."),
    SourceSpec("uniprot", "UniProt", ("MA1",), "public_api", "https://rest.uniprot.org/"),
    SourceSpec("chembl", "ChEMBL", ("MA2",), "public_api", "https://www.ebi.ac.uk/chembl/api/data/"),
    SourceSpec("bindingdb", "BindingDB", ("MA2",), "public_api", "https://bindingdb.org/rest/"),
    SourceSpec("dgidb", "DGIdb", ("MA1",), "public_api", "https://dgidb.org/api/graphql"),
    SourceSpec("drugcentral", "DrugCentral", ("MA2",), "local_dataset", "https://drugcentral.org/download", requirement="DrugCentral dump or public PostgreSQL mirror"),
    SourceSpec("open_targets", "Open Targets", ("MA1",), "public_api", "https://api.platform.opentargets.org/api/v4/graphql"),
    SourceSpec("rcsb_pdb", "RCSB PDB", ("MA2",), "public_api", "https://search.rcsb.org/rcsbsearch/v2/query"),
    SourceSpec("alphafold_db", "AlphaFold DB", ("MA2",), "public_api", "https://alphafold.ebi.ac.uk/api/"),
    SourceSpec("pubchem", "PubChem", ("MA2",), "public_api", "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"),
    SourceSpec(
        "pharos",
        "Pharos",
        ("MA1",),
        "public_api",
        "https://pharos-api.ncats.io/graphql",
        note=(
            "Orchestrator queries a bounded set of ranked MA1 targets only when "
            "ResearchBrief.human_priorities explicitly requests Pharos, TDL, "
            "target development level, druggability, or the supported Korean "
            "equivalents. Excluded from the interactive default."
        ),
    ),
    SourceSpec("cellxgene", "CELLxGENE Census", ("MA1",), "optional_sdk", "https://chanzuckerberg.github.io/cellxgene-census/", requirement="Operator-supplied Linux/macOS CELLxGENE Census sidecar; native Windows SDK unsupported"),
    SourceSpec("human_cell_atlas", "Human Cell Atlas", ("MA1",), "public_api", "https://service.azul.data.humancellatlas.org/index/projects", note="Official keyless Azul project-metadata API; disease/cell-type context only."),
    SourceSpec("gtex", "GTEx", ("MA1",), "public_api", "https://gtexportal.org/api/v2/"),
    SourceSpec("archs4", "ARCHS4", ("MA1",), "public_api", "https://maayanlab.cloud/sigpy/data/correlation", note="Interactive gene-context lookup uses the public SigPy endpoint; bulk HDF5 remains optional."),
    SourceSpec("string", "STRING", ("MA1",), "public_api", "https://string-db.org/api/"),
    SourceSpec("biogrid", "BioGRID", ("MA1",), "credential_required", "https://webservice.thebiogrid.org/", requirement="BIOGRID_ACCESS_KEY"),
    SourceSpec("reactome", "Reactome", ("MA1",), "public_api", "https://reactome.org/ContentService/"),
    SourceSpec("kegg", "KEGG", ("MA1",), "public_api", "https://rest.kegg.jp/", note="Use subject to KEGG REST and licensing terms."),
    SourceSpec("quickgo", "QuickGO", ("MA1",), "public_api", "https://www.ebi.ac.uk/QuickGO/services/"),
    SourceSpec("clinical_trials", "ClinicalTrials.gov", ("MA1",), "public_api", "https://clinicaltrials.gov/api/v2/"),
    SourceSpec("google_patents", "Google Patents", ("MA2",), "cloud_dataset", "https://patents.google.com/", requirement="Google Patents Public Datasets/BigQuery project, ADC and optional google-cloud-bigquery dependency"),
    SourceSpec("drugbank", "DrugBank", ("MA2",), "credential_required", "https://go.drugbank.com/", requirement="DrugBank licence and API/data credentials"),
)


EVALUATION_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("repodb", "repoDB", ("EVALUATION",), "evaluation_dataset", "https://unmtid-shinyapps.net/shiny/repodb/"),
    SourceSpec("chembl_eval", "ChEMBL", ("EVALUATION",), "evaluation_dataset", "https://www.ebi.ac.uk/chembl/"),
    SourceSpec("tdc", "Therapeutics Data Commons", ("EVALUATION",), "evaluation_dataset", "https://tdcommons.ai/"),
)


# These mappings describe code paths that exist in this repository.  They do
# not imply that a remote service answered successfully; each run records that
# separately in evidence provenance.
_DEFAULT_BIOLOGY_ADAPTERS = {
    "pubmed": "PublicBiologyRetriever",
    "europe_pmc": "PublicBiologyRetriever",
    "uniprot": "PublicBiologyRetriever",
    "dgidb": "PublicBiologyRetriever",
    "open_targets": "PublicBiologyRetriever",
}
_ON_DEMAND_BIOLOGY_ADAPTERS = {
    "pharos": "PublicBiologyRetriever",
}
_FEDERATED_SOURCES = {
    "biorxiv": "MA1PublicContextRetriever (Europe PMC federation)",
    "medrxiv": "MA1PublicContextRetriever (Europe PMC federation)",
}
_CONTEXT_ADAPTERS = {
    "gtex": "MA1PublicContextRetriever",
    "string": "MA1PublicContextRetriever",
    "reactome": "MA1PublicContextRetriever",
    "kegg": "MA1PublicContextRetriever",
    "quickgo": "MA1PublicContextRetriever",
    "clinical_trials": "MA1PublicContextRetriever",
}
_EXTENDED_CONTEXT_ADAPTERS = {
    "human_cell_atlas": "PublicCellDataRetriever",
    "archs4": "PublicExpressionInteractionRetriever",
}
_OPTIONAL_CHEMISTRY_ADAPTERS = {
    "chembl": "ChEMBLAdapter",
    "bindingdb": "BindingDBAdapter",
    "pubchem": "PubChemAdapter",
    "rcsb_pdb": "RCSBPDBAdapter",
    "alphafold_db": "AlphaFoldDBAdapter",
}


def _status_for_runtime_source(
    source: SourceSpec,
    config: SourceConfiguration,
) -> SourceStatus:
    if source.source_id in _ON_DEMAND_BIOLOGY_ADAPTERS:
        return SourceStatus(
            **source.public_dict(),
            status="available_not_default",
            adapter=_ON_DEMAND_BIOLOGY_ADAPTERS[source.source_id],
            default_queried=False,
            configured=True,
            configuration_state=(
                "explicit_human_priority_trigger"
                if config.public_evidence_enabled
                else "disabled_by_policy"
            ),
        )

    if source.source_id in _DEFAULT_BIOLOGY_ADAPTERS:
        enabled = config.public_evidence_enabled
        return SourceStatus(
            **source.public_dict(),
            status="connected_adapter" if enabled else "available_not_default",
            adapter=_DEFAULT_BIOLOGY_ADAPTERS[source.source_id],
            default_queried=enabled,
            configured=True,
            configuration_state="enabled" if enabled else "disabled_by_policy",
        )

    if source.source_id in _FEDERATED_SOURCES:
        return SourceStatus(
            **source.public_dict(),
            status="federated",
            adapter=_FEDERATED_SOURCES[source.source_id],
            # The current MA1 search can receive preprints through Europe PMC,
            # but it does not issue a dedicated bioRxiv/medRxiv query.
            default_queried=False,
            configured=config.public_evidence_enabled,
            configuration_state=(
                "enabled" if config.public_evidence_enabled else "disabled_by_policy"
            ),
        )

    if source.source_id in _OPTIONAL_CHEMISTRY_ADAPTERS:
        enabled = config.public_candidate_enrichment_enabled
        return SourceStatus(
            **source.public_dict(),
            status="connected_adapter" if enabled else "available_not_default",
            adapter=_OPTIONAL_CHEMISTRY_ADAPTERS[source.source_id],
            default_queried=enabled,
            configured=True,
            configuration_state=("enabled" if enabled else "adapter_available"),
        )

    if source.source_id in _CONTEXT_ADAPTERS:
        return SourceStatus(
            **source.public_dict(),
            status="available_not_default",
            adapter=_CONTEXT_ADAPTERS[source.source_id],
            default_queried=False,
            configured=True,
            configuration_state="on_demand_context_adapter",
        )

    if source.source_id in _EXTENDED_CONTEXT_ADAPTERS:
        return SourceStatus(
            **source.public_dict(),
            status="available_not_default",
            adapter=_EXTENDED_CONTEXT_ADAPTERS[source.source_id],
            default_queried=False,
            configured=True,
            configuration_state=(
                "on_demand_context_adapter"
                if config.public_evidence_enabled
                else "disabled_by_policy"
            ),
        )

    if source.source_id == "drugcentral":
        if config.drugcentral_dataset_configured:
            ready = config.drugcentral_dataset_available
            enabled = ready and config.public_candidate_enrichment_enabled
            return SourceStatus(
                **source.public_dict(),
                status=(
                    "connected_adapter"
                    if enabled
                    else "available_not_default"
                    if ready
                    else "not_configured"
                ),
                adapter="DrugCentralLocalAdapter",
                default_queried=enabled,
                configured=ready,
                configuration_state=(
                    "enabled"
                    if enabled
                    else "dataset_ready"
                    if ready
                    else "dataset_missing"
                ),
            )
        return SourceStatus(
            **source.public_dict(),
            status="not_configured",
            adapter="DrugCentralLocalAdapter",
            default_queried=False,
            configured=False,
            configuration_state="local_dataset_required",
        )

    conditional_configuration = {
        "cellxgene": (
            config.cellxgene_sidecar_configured,
            "sidecar_configured",
            "sidecar_required",
            "PublicCellDataRetriever",
        ),
        "biogrid": (
            config.biogrid_credential_configured,
            "credential_configured",
            "credential_required",
            "PublicExpressionInteractionRetriever",
        ),
        "google_patents": (
            config.google_patents_project_configured,
            "cloud_project_configured",
            "cloud_project_required",
            "GooglePatentsBigQueryAdapter",
        ),
        "drugbank": (
            config.drugbank_credential_configured,
            "credential_configured",
            "credential_required",
            "DrugBankLicensedAdapter",
        ),
    }
    if source.source_id in conditional_configuration:
        configured, configured_state, missing_state, adapter = conditional_configuration[
            source.source_id
        ]
        return SourceStatus(
            **source.public_dict(),
            status="available_not_default" if configured else "not_configured",
            adapter=adapter,
            default_queried=False,
            configured=configured,
            configuration_state=configured_state if configured else missing_state,
        )

    # Public availability alone is not an implementation claim.  Sources in
    # this category occur in the proposal and policy but currently have no
    # query adapter connected to a workflow stage.
    return SourceStatus(
        **source.public_dict(),
        status="catalog_only",
        adapter=None,
        default_queried=False,
        configured=False,
        configuration_state="adapter_not_implemented",
    )


def source_status_inventory(
    config: SourceConfiguration | None = None,
) -> dict[str, object]:
    """Return all proposal sources with implementation/configuration status.

    ``reachability`` is always ``not_checked``.  Request-time evidence metadata
    is the authoritative record of live, cached, failed, or fixture retrieval.
    """

    config = config or SourceConfiguration()
    runtime = [_status_for_runtime_source(source, config) for source in RUNTIME_SOURCES]
    evaluation = [
        SourceStatus(
            **source.public_dict(),
            status="evaluation_only",
            adapter=None,
            default_queried=False,
            configured=False,
            configuration_state="evaluation_dataset",
        )
        for source in EVALUATION_SOURCES
    ]
    counts = Counter(item.status for item in runtime)
    return {
        "scope_note": (
            "Adapter/configuration registry only; network reachability is not checked. "
            "Stages identify the component that directly queries the source; MA3 audits "
            "the persisted upstream artifact rather than querying these sources itself. "
            "Per-run evidence provenance reports live, cache, error, or fixture use."
        ),
        "runtime": [item.public_dict() for item in runtime],
        "evaluation": [item.public_dict() for item in evaluation],
        "summary": {
            "runtime_total": len(runtime),
            "evaluation_total": len(evaluation),
            "default_queried_total": sum(item.default_queried for item in runtime),
            "status_counts": dict(sorted(counts.items())),
            "network_reachability_checked": False,
        },
    }


def report_source_inventory() -> dict[str, list[dict[str, object]]]:
    return {
        "runtime": [source.public_dict() for source in RUNTIME_SOURCES],
        "evaluation": [source.public_dict() for source in EVALUATION_SOURCES],
    }


__all__ = [
    "EVALUATION_SOURCES",
    "RUNTIME_SOURCES",
    "SourceConfiguration",
    "SourceSpec",
    "SourceStatus",
    "report_source_inventory",
    "source_status_inventory",
]
