# Public snapshot validation

Date: 2026-09-29. Checks used the staged public source on Windows with available local dependency environments. This is not a claim of a fresh cross-platform dependency installation.

## Executed checks

- Test result: **See focused and broad-suite results below**.
- Command: `python -m pytest tests/unit/test_beta_access_guard.py tests/integration/test_source_api.py -q`.
- The public configuration was checked with `python -m drugpilot.cli doctor`. It selected mock mode with no provider credentials. No paid research run, external scientific worker, or internal service deployment was started.
- All retained Python source was parsed for syntax errors.
- The primary README's local links were checked.
- Gitleaks 8.30.1 and an additional identity/address scan were used before upload. Actual credentials are excluded; explicit test-only dummy strings are not provider credentials.

## Boundaries

The source snapshot omits large raw datasets, model weights, training checkpoints, service databases, and unrelated administration/report-generation archives. Historical research values remain historical values. Paths/identities in historical notes were generalized; old artifact manifests are not a checksum guarantee for this curated publication.

## DrugPilot test details

- Broad run: `tests/unit` plus `tests/integration/test_source_api.py`; **3266 passed and 19 failed** in the initial curated copy. All failures were missing selected fixtures or omitted preflight helpers.
- After restoring those small dependencies, all affected modules, access-control checks, source-redaction checks, and idempotency fixtures were rerun: **86 passed, 0 failed**.
- The full broad suite was not repeated after those fixture-only repairs. The first run and focused follow-up cover the retained unit/API scope; other integration and optional scientific-tool suites are not claimed as fully rerun.
- Test-only UUIDs were made visibly deterministic examples. No production logic was changed to make tests pass.

## Clean Git-tree verification

The files selected by Git were exported to a separate clean directory and the stated public test scope was run again from that export. This checks that ignored local packages or uncommitted helper files are not required by those tests. No private source history is included.
