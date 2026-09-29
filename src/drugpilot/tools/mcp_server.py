from __future__ import annotations

from drugpilot.tools.chemistry import ChemistryValidator


def build_mcp_server():
    """Build the optional local FastMCP boundary declared in the proposal."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "MCP support is optional. Install with: pip install -e .[mcp]"
        ) from exc

    mcp = FastMCP("DrugPilot Scientific Tools")
    chemistry = ChemistryValidator()

    @mcp.tool()
    def validate_smiles(smiles: str) -> dict:
        """Validate and standardize a SMILES string without mutating it."""

        report = chemistry.inspect(smiles)
        return {
            "valid": report.valid,
            "canonical_smiles": report.canonical_smiles,
            "inchikey": report.inchikey,
            "metrics": report.metrics.model_dump(),
            "backend": report.backend,
            "errors": list(report.errors),
        }

    return mcp


def main() -> None:  # pragma: no cover - standalone optional server
    build_mcp_server().run()


if __name__ == "__main__":
    main()

