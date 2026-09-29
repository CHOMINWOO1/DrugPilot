from __future__ import annotations

import argparse
import json

from drugpilot.config import get_settings
from drugpilot.domain.models import RunMode


def main() -> None:
    parser = argparse.ArgumentParser(prog="drugpilot")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="run the local web dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    serve.add_argument("--reload", action="store_true")

    subparsers.add_parser("doctor", help="show configuration without secrets")
    args = parser.parse_args()

    if args.command == "doctor":
        settings = get_settings()
        print(
            json.dumps(
                {
                    "mode": settings.effective_mode.value,
                    "configured_mode": settings.mode.value,
                    "effective_mode": settings.effective_mode.value,
                    "mode_auto_activated": settings.openai_auto_activated,
                    "llm_backend": (
                        "openai"
                        if settings.effective_mode is not RunMode.MOCK
                        and bool(settings.openai_api_key)
                        else "mock"
                    ),
                    "database": str(settings.database_path),
                    "openai_configured": bool(settings.openai_api_key),
                    "anthropic_configured": bool(settings.anthropic_api_key),
                    "candidates_per_route": settings.candidates_per_route,
                    "max_run_tokens": settings.max_run_tokens,
                    "hard_max_run_tokens": settings.hard_max_run_tokens,
                    "max_llm_calls": settings.max_llm_calls,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command in {None, "serve"}:
        import uvicorn

        uvicorn.run(
            "drugpilot.api.app:app",
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 8000),
            reload=getattr(args, "reload", False),
        )


if __name__ == "__main__":
    main()
