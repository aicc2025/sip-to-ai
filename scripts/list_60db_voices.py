"""List the voices available to your 60db account (GET /myvoices).

Use this to pick a SIXTYDB_VOICE_ID for the Deepgram-brain + 60db-voice setup.

Usage:
    uv run python scripts/list_60db_voices.py

Reads SIXTYDB_API_KEY from the environment / .env.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv


def _fetch_voices(api_key: str) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        "https://api.60db.ai/myvoices",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"message": body[:500]}
        return exc.code, payload


def main() -> int:
    load_dotenv()
    api_key = os.getenv("SIXTYDB_API_KEY", "")
    if not api_key:
        print("SIXTYDB_API_KEY is not set (env or .env).", file=sys.stderr)
        return 1

    status, payload = _fetch_voices(api_key)
    if status != 200 or not payload.get("success", True):
        print(f"Request failed (HTTP {status}): {payload}", file=sys.stderr)
        return 1

    voices = payload.get("data", [])
    if not voices:
        print("No voices found for this account.")
        return 0

    print(f"{'voice_id':38}  {'name':24}  {'category':12}  {'model':12}  labels")
    print("-" * 110)
    for v in voices:
        labels = v.get("labels", {}) or {}
        label_str = ", ".join(
            str(labels.get(k))
            for k in ("language_name", "gender", "accent")
            if labels.get(k)
        )
        print(
            f"{v.get('voice_id', ''):38}  "
            f"{(v.get('name') or ''):24.24}  "
            f"{(v.get('category') or ''):12.12}  "
            f"{(v.get('model') or ''):12.12}  "
            f"{label_str}"
        )

    print("\nSet one as your voice:  SIXTYDB_VOICE_ID=<voice_id>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
