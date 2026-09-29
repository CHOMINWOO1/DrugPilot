# Clone-local receptor toolchain

This directory records the receptor-preparation bootstrap performed on
2026-08-21 for the 8765 validation clone only.  It did not modify system
`PATH`, install a system Java/Python, edit `.env`, restart a service, touch
8766, or promote a docking profile.

## Installed locations

- Python 3.12.13 environment: `.venv-receptor-tools`
- Temurin JRE 21.0.12+8: `tools/java/jdk-21.0.12+8-jre`
- P2Rank 2.5.1: `tools/p2rank/p2rank_2.5.1`
- Existing AutoDock Vina 1.2.7: `tools/bin/vina_1.2.7_win.exe`

`manifest.json` records official URLs, archive/file SHA-256 values, measured
disk use and smoke outcomes.  The pip `--report` JSON files retain resolved
download URLs, hashes and installed metadata.  Complete Java legal notices
and the P2Rank MIT license remain in their extracted distributions.

Measured clone-local disk increase after all retained archives and smoke
artifacts: 1,270,467,236 bytes (1.183 GiB).  The temporary ASCII staging
directory used to isolate the Windows path issue was removed after verification.

## Windows non-ASCII path requirement

The workspace has a Korean path.  RDKit and OpenMM C++ loaders cannot reliably
consume their absolute data/plugin paths there.  The backend must compute paths
relative to each subprocess `cwd=output` and pass a merged environment to both
Python and Meeko workers.

The exact successful probe from
`tools/receptor-toolchain/smoke/pdbfixer-idh1-chain-a` used:

```text
RDBASE=..\..\..\..\.venv-receptor-tools\Lib\site-packages\rdkit
OPENMM_PLUGIN_DIR=..\..\..\..\.venv-receptor-tools\Lib\site-packages\OpenMM.libs\lib\plugins
```

These strings must not be hard-coded for another output directory.  Use
`os.path.relpath(runtime_directory, start=output_directory)`.  With the plugin
path, OpenMM exposes Reference, CPU and OpenCL and reports no plugin-load
failure.  With no plugin path, only Reference is visible; the actual no-MD,
no-hydrogen IDH1 PDBFixer repair still completed successfully, but that is a
bounded compatibility observation rather than a general performance claim.

Set `JAVA_HOME` only in the service/worker process to the clone-local JRE.
`ExecutableRunner(shell=False)` successfully executed the upstream
`prank.bat -v` and reported P2Rank 2.5.1.  No machine-wide Java setting is
required.

## Verified, but not accepted for production

- `pip check`: pass.
- Meeko receptor/ligand CLI: pass.
- P2Rank 2.5.1 `alphafold` prediction on its bundled 1FBL test structure:
  exit 0 and prediction artifacts emitted.  This is not an FLT3 result.
- Focused receptor unit tests: see the final task log.
- Actual IDH1 6U4J/PWV full backend: PDBFixer/OpenMM, Meeko, Vina 1.2.7,
  PoseBusters and symmetry-aware RMSD all executed.  PoseBusters passed 28/28;
  top-pose RMSD was 0.270882614808196 A at a 2.0 A threshold.

The IDH1 run is an experimental smoke only.  No `ProductionAcceptance` was
created and no registry/profile promotion occurred.  Reviewed target-specific
cofactor/metal/water policies, an external redocking benchmark and explicit
acceptance remain required.  FLT3 still needs real, exact-identity structure
retrieval and validation; the FLT3 fixture is not scientific evidence.

The main `.venv` remains broken because it points to a removed Python 3.12.10
installation.  This separate environment makes the receptor tools executable
without disturbing the currently running service, but it does not by itself
repair the normal 8765 restart environment or wire the toolchain into the
graph/API/UI.
