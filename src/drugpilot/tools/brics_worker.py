from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys


def _worker_identity() -> dict[str, object]:
    return {
        "python_isolated": bool(sys.flags.isolated),
        "no_user_site": bool(sys.flags.no_user_site),
        "worker_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def generate_brics(
    parent_smiles: list[str],
    *,
    limit: int,
    max_builds: int,
) -> dict[str, object]:
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import BRICS, Descriptors, FilterCatalog, Lipinski, QED, rdFingerprintGenerator
    except (ImportError, OSError) as exc:
        raise RuntimeError("RDKit is required for BRICS generation") from exc
    # This worker already converts invalid generated molecules into bounded
    # structured skips. Avoid flooding the parent service with repeated RDKit
    # kekulization diagnostics for those same rejected intermediates.
    RDLogger.DisableLog("rdApp.error")
    if limit < 1 or max_builds < limit:
        raise ValueError("limit must be positive and max_builds must cover limit")

    parents: list[object] = []
    canonical_parents: set[str] = set()
    for smiles in parent_smiles:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if canonical in canonical_parents:
            continue
        canonical_parents.add(canonical)
        parents.append(molecule)
    if len(parents) < 2:
        raise ValueError("BRICS generation requires at least two measured ligand seeds")

    fragment_support: dict[str, int] = {}
    for molecule in parents:
        # Support counts keep the bounded fragment library anchored to motifs
        # observed across multiple measured ligands.  This also prevents a
        # large target series from making BRICSBuild expand an effectively
        # unbounded Cartesian product before the process timeout can help.
        for smiles in set(BRICS.BRICSDecompose(molecule)):
            fragment_support[smiles] = fragment_support.get(smiles, 0) + 1
    fragment_records: list[tuple[str, object, int, int]] = []
    for smiles, support in fragment_support.items():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        heavy_atoms = sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms())
        if heavy_atoms < 2 or heavy_atoms > 30:
            continue
        fragment_records.append((smiles, molecule, support, heavy_atoms))
    fragment_records.sort(
        key=lambda item: (
            -item[2],
            abs(item[3] - 8),
            item[0],
        )
    )
    fragment_pool_cap = min(48, max(16, len(parents) * 3))
    selected_fragment_records = fragment_records[:fragment_pool_cap]
    fragment_molecules = [item[1] for item in selected_fragment_records]
    if len(fragment_molecules) < 2:
        raise ValueError("Measured ligand seeds yielded an insufficient BRICS fragment pool")
    generation_depth = 3 if len(fragment_molecules) <= 16 else 2

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )
    parent_fingerprints = [
        fingerprint_generator.GetFingerprint(molecule) for molecule in parents
    ]
    catalog_parameters = FilterCatalog.FilterCatalogParams()
    catalog_parameters.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog_parameters.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    alert_catalog = FilterCatalog.FilterCatalog(catalog_parameters)

    eligible: list[dict[str, object]] = []
    seen = set(canonical_parents)
    attempted = 0
    builder = BRICS.BRICSBuild(
        fragment_molecules,
        onlyCompleteMols=True,
        uniquify=True,
        scrambleReagents=False,
        # A small fragment pool can safely explore a third join; target series
        # with larger pools are capped at two joins to avoid combinatorial
        # expansion in an interactive fail-closed workflow.
        maxDepth=generation_depth,
    )
    for molecule in builder:
        attempted += 1
        if attempted > max_builds:
            break
        try:
            Chem.SanitizeMol(molecule)
            canonical = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )
        except Exception:
            continue
        if canonical in seen or "*" in canonical:
            continue
        molecular_weight = float(Descriptors.MolWt(molecule))
        if not 200.0 <= molecular_weight <= 550.0:
            continue
        if molecule.GetNumHeavyAtoms() < 15 or Lipinski.RingCount(molecule) < 1:
            continue
        if Lipinski.NumHDonors(molecule) > 5 or Lipinski.NumHAcceptors(molecule) > 10:
            continue
        alerts = [match.GetDescription() for match in alert_catalog.GetMatches(molecule)]
        if alerts:
            continue
        fingerprint = fingerprint_generator.GetFingerprint(molecule)
        similarities = [
            float(DataStructs.TanimotoSimilarity(fingerprint, parent))
            for parent in parent_fingerprints
        ]
        nearest_parent_index = max(range(len(similarities)), key=similarities.__getitem__)
        nearest_similarity = similarities[nearest_parent_index]
        # Require target-chemotype continuity while avoiding exact/near-exact
        # copies of a measured seed. This is a structural eligibility rule,
        # not an invented activity prediction.
        if not 0.30 <= nearest_similarity <= 0.85:
            continue
        try:
            inchikey = Chem.MolToInchiKey(molecule) or None
        except Exception:
            inchikey = None
        seen.add(canonical)
        eligible.append(
            {
                "smiles": canonical,
                "inchikey": inchikey,
                "molecular_weight": molecular_weight,
                "hbd": Lipinski.NumHDonors(molecule),
                "hba": Lipinski.NumHAcceptors(molecule),
                "ring_count": Lipinski.RingCount(molecule),
                "qed": float(QED.qed(molecule)),
                "nearest_measured_seed_index": nearest_parent_index,
                "nearest_measured_seed_tanimoto": nearest_similarity,
                "structural_alerts": alerts,
            }
        )
        if len(eligible) >= max(8, limit * 4):
            break
    eligible.sort(
        key=lambda item: (
            -float(item["qed"]),
            abs(float(item["nearest_measured_seed_tanimoto"]) - 0.55),
            str(item["smiles"]),
        )
    )
    generated = eligible[:limit]
    if not generated:
        raise RuntimeError(
            "BRICS generated no eligible molecule from the measured-ligand seed pool"
        )
    return {
        "generator": "RDKit BRICS",
        **_worker_identity(),
        "rdkit_version": _package_version("rdkit"),
        "parent_count": len(parents),
        "fragment_count": len(fragment_records),
        "selected_fragment_count": len(fragment_molecules),
        "fragment_pool_cap": fragment_pool_cap,
        "generation_depth": generation_depth,
        "rdkit_error_log_policy": "suppressed_in_worker_structured_skips_retained",
        "fragment_selection_policy": (
            "BRICS fragments with 2-30 heavy atoms; prioritize support across "
            "measured ligands, then proximity to 8 heavy atoms; deterministic "
            "cap of min(48, max(16, 3 x measured-seed count))"
        ),
        "attempted_builds": attempted,
        "novelty_scope": "not an exact canonical-SMILES duplicate of measured seeds",
        "selection_policy": (
            "RDKit-valid; MW 200-550; >=15 heavy atoms; >=1 ring; HBD<=5; "
            "HBA<=10; no PAINS/BRENK alerts; nearest measured-seed Morgan "
            "Tanimoto 0.30-0.85; prioritize higher RDKit QED"
        ),
        "molecules": generated,
        "requested_limit": limit,
        "returned_count": len(generated),
        "partial_result": len(generated) < limit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate BRICS candidates from measured ligands")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-builds", type=int, default=20_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        try:
            from rdkit import rdBase

            print(
                json.dumps(
                    {
                        "status": "ready",
                        "generator": "RDKit BRICS",
                        "rdkit_version": str(rdBase.rdkitVersion),
                        **_worker_identity(),
                    },
                    sort_keys=True,
                )
            )
            return 0
        except (ImportError, OSError) as exc:
            print(
                json.dumps(
                    {"status": "unavailable", "detail": type(exc).__name__},
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
    if args.input is None or args.output is None or args.limit is None:
        print(
            json.dumps(
                {"error_type": "ValueError", "detail": "input, output and limit are required"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        result = generate_brics(
            list(payload["parent_smiles"]),
            limit=args.limit,
            max_builds=args.max_builds,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
    except Exception as exc:
        print(
            json.dumps(
                {"error_type": type(exc).__name__, "detail": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "generated_count": len(result["molecules"]),
                "rdkit_version": result["rdkit_version"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
