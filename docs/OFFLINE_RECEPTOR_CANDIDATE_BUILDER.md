# Offline receptor candidate builder

`scripts/build_receptor_candidate_profile.py` is an offline, fail-closed bridge
between an exact receptor qualification and a later human-reviewed docking
profile promotion. It is deliberately not called by MA2 and does not read or
write the active docking-profile registry.

The checked-in IDH1 manifest uses the real local 6U4J PDB and PWV component
SDF. It pins both files by SHA-256, explicitly requests human IDH1 O75874 with
the R132H mutation, classifies every known heterogen role, and selects
`PWV:A:503` as the co-crystal box ligand. The command parses the PDB again and
runs `ReceptorQualifier`; the manifest attestation alone is never trusted.

The command requires `DRUGPILOT_RECEPTOR_BUILDER_ENABLED=true` plus the exact
clone-local PDBFixer/OpenMM Python, Meeko receptor and ligand CLIs, P2Rank/Java,
and Vina paths in the selected env file. It executes the existing local backend
and emits a profile only when both PoseBusters and symmetry-aware redocking RMSD
pass. Fixture input, incomplete identity, hash drift, missing tools, failed
validation, and any pre-existing output directory stop the build.

Example from the 8765 clone root:

```powershell
.\.venv\Scripts\python.exe scripts\build_receptor_candidate_profile.py `
  --manifest data\docking\IDH1\qualified-6U4J-PWV-A503-build-manifest.json `
  --output-dir data\docking\IDH1\candidate-builds\6U4J-PWV-A503-run-001
```

Each output directory is single-use. A successful directory contains hashed
preparation and validation artifacts, a byte-identical manifest copy, and
`candidate_docking_profile.json` using
`drugpilot.candidate-docking-profile/2.0`. Its acceptance is always false and
its status is always `candidate_not_accepted`. There is intentionally no CLI
flag that can accept, promote, or register the result.
