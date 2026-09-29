# Small public/test inputs

Only the following test inputs are included; internal service databases and live-run exports are excluded.

- `evaluation/demo_v1/`: the application's versioned demonstration benchmark and its checksum. This is not clinical validation.
- `docking/IDH1/source/6U4J.pdb`: public PDB structure 6U4J, used for exact-identity/qualification regression tests. Original structure entry: https://www.rcsb.org/structure/6U4J ; source download: https://files.rcsb.org/download/6U4J.pdb . Preserve the structure's author/publication attribution in its HEADER and REMARK records.
- The small IDH1 JSON manifests describe test qualification inputs. They do not constitute a deployable, qualified docking installation; workers and generated receptor artifacts are excluded.

The request in `docs/validation/` is retained only as a schema regression fixture. Publishing or testing its schema does not execute that research request.
