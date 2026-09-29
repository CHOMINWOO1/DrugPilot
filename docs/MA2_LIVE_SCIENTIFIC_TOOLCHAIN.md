# MA2 live scientific toolchain handoff

This implementation exists only in the `drugpilot-llm-intake-test` validation
clone. It invokes real public data and local scientific software and never
substitutes fixture values, deterministic hashes, or synthetic scientific
scores. A missing measurement, model endpoint, executable, receptor, docking
box, or timeout is a fail-closed error.

## Runtime composition

```python
profiles = DockingProfileRegistry.from_json(
    "data/docking/IDH1/6U4J-PWV-A503/profile.json"
)
docking = VinaDockingAdapter(
    profiles=profiles,
    vina_executable="tools/bin/vina_1.2.7_win.exe",
    python_executable=sys.executable,
    ligand_prep_timeout_seconds=60,
    docking_timeout_seconds=180,
    exhaustiveness=8,
    num_modes=9,
    cpu=1,
    seed=20260819,
)
admet = ADMETAIAdapter(
    python_executable=sys.executable,
    startup_timeout_seconds=180,
    prediction_timeout_seconds=90,
)
gateway = LiveScientificGateway(
    docking=docking,
    admet=admet,
    client=None,
    timeout_seconds=12,
    min_pchembl_value=5.0,
    python_executable=sys.executable,
    brics_timeout_seconds=120,
    brics_max_builds=20_000,
)
```

`LiveScientificGateway.backend_kind == "live"` and
`LiveScientificGateway.fixture is False`. Its
`readiness(target: TargetCandidate | None = None)` method returns this mapping:

```text
{
  backend_kind: "live",
  fixture: false,
  status: "completed" | "unavailable",
  source_version: str,
  tools: [
    {
      tool, operation, status, backend, version, fixture,
      required, reason, provenance
    }, ...
  ],
  reason: str | null
}
```

The four tool entries are ChEMBL, RDKit BRICS, AutoDock Vina, and ADMET-AI.
Shared DTO code may convert this mapping to `GatewayReadiness` and
`ToolReadiness`; the adapter itself deliberately has no dependency on shared
DTOs. `last_operation_provenance` is refreshed after every gateway operation.
Live MA2 must use `max_refinement_iterations=0`: `refine()` is intentionally
unavailable because this adapter will not invent an improved molecule or score.

## Scientific operations

- Target resolution requires an exact UniProt accession in the official ChEMBL
  REST API. Measured IC50/Ki/Kd/EC50 values retain relation, unit, assay, source
  evidence ID, and ChEMBL database version.
- Repurposing candidates require measured target activity and clinical/approved
  status. ChEMBL drug-warning records are retrieved and retained.
- De novo candidates require at least two measured target ligands. RDKit BRICS
  runs in an isolated subprocess with a bounded fragment pool, build count, and
  timeout. Candidates are structural proposals with **unmeasured activity**.
  Eligibility uses real RDKit calculations: validity, MW 200–550, at least 15
  heavy atoms and one ring, HBD/HBA limits, PAINS/BRENK exclusion, and Morgan
  Tanimoto 0.30–0.85 to the nearest measured seed. QED is used only to order
  already eligible molecules, not as an activity claim.
- Vina runs only when `TargetCandidate.metadata["docking_profile_id"]` exactly
  selects a configured receptor/box profile. No target symbol or arbitrary
  pocket is used as a general fallback.
- ADMET-AI loads its packaged pretrained Chemprop ensembles in one persistent,
  timeout-controlled worker. Every required endpoint must be present.

## ADMET six-field mapping

| DrugPilot field | ADMET-AI endpoint | Meaning |
|---|---|---|
| `hepatotoxicity` | `DILI` | direct classification probability |
| `herg_risk` | `hERG` | direct classification probability |
| `ames_risk` | `AMES` | direct classification probability |
| `cyp3a4_inhibition` | `CYP3A4_Veith` | direct classification probability |
| `oral_bioavailability` | `Bioavailability_Ma` | direct classification probability |
| `clearance_risk` | `Clearance_Hepatocyte_AZ_drugbank_approved_percentile / 100` | high-clearance relative percentile; **not** a toxicity probability |

The raw `Clearance_Hepatocyte_AZ` regression value in
`uL/min/10^6 cells` is retained in provenance. The adapter fails closed when
any of the seven source values (five classification endpoints, raw clearance,
and its packaged reference percentile) is absent.

## 6U4J/PWV docking profile status

Profile `IDH1-R132H-6U4J-PWV-A503-v1` is an explicit-only, versioned smoke
profile for the human IDH1 R132H mutant homodimer and co-crystal ligand PWV
(FT-2102/olutasidenib). It is not a general IDH1 fallback and remains marked
`experimental_smoke_only`, `production_ready: false`.

- RCSB structure: 6U4J; protein chains A/B; PWV chain A residue 503.
- Box: co-crystal ligand bounds plus 6 Å padding; center
  `(10.910, -4.143, 23.671)` Å and size `(20.922, 20.194, 19.208)` Å.
- Fixed protocol: Vina 1.2.7, seed 20260819, exhaustiveness 8, 9 requested
  modes, one CPU.
- Actual top affinity: -12.000 kcal/mol. A docking score is not experimental
  potency, efficacy, or safety evidence.
- Actual symmetry-aware 25-heavy-atom top-pose RMSD: **0.248 Å**, passing the
  declared 2.0 Å threshold. No coordinate alignment was applied because both
  poses use the receptor/co-crystal coordinate frame. Mapping required an exact
  RCSB component-SDF/PDB element-sequence match and a complete chirality-aware
  RDKit graph isomorphism. The exact artifact and hashes are in
  `data/docking/IDH1/6U4J-PWV-A503/redocking/validation.json`.

This single-ligand redocking result verifies this fixed smoke protocol; it does
not establish production performance across targets or chemotypes.

## Reproducibility and footprint

- Python: 3.12.10, Windows x86-64.
- Direct pins: `requirements/ma2-scientific.txt`.
- Fully resolved environment: `requirements/ma2-scientific-lock.txt`.
- Official Vina binary: `tools/bin/vina_1.2.7_win.exe`, SHA-256
  `e0c4b2715e0c1a74f6e92d0f3be0328ac97542eafbc111e6b1efad897a73cce5`.
- Clone virtual environment: 1,274,024,533 bytes (1.187 GiB).
- 8766 baseline virtual environment: 154,943,227 bytes (0.144 GiB), read only.
- Clone minus baseline: 1,119,081,306 bytes (1.042 GiB).
- Vina binary: 1,233,920 bytes; IDH1 source/profile/redocking assets:
  20,435,678 bytes.

Install the clone itself after the locked external packages so the editable
package path does not point to another copy:

```powershell
python -m pip install -r requirements\ma2-scientific-lock.txt
python -m pip install -e . --no-deps
```

Primary project documentation: [ChEMBL API](https://www.ebi.ac.uk/chembl/api/data/docs),
[RDKit BRICS](https://www.rdkit.org/docs/source/rdkit.Chem.BRICS.html),
[AutoDock Vina](https://autodock-vina.readthedocs.io/en/latest/),
[Meeko](https://meeko.readthedocs.io/),
[ADMET-AI](https://github.com/swansonk14/admet_ai), and
[RCSB 6U4J](https://www.rcsb.org/structure/6U4J).
