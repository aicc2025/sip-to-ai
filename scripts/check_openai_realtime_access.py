"""Check whether the configured OpenAI key can access the configured realtime model."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv


def _request(path: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(f"https://api.openai.com/v1{path}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"error": {"message": body[:500]}}
        return exc.code, payload


def _error(payload: dict[str, Any]) -> dict[str, Any]:
    err = payload.get("error", payload)
    if not isinstance(err, dict):
        return {"message": str(err)}
    return {
        "type": err.get("type"),
        "code": err.get("code"),
        "message": err.get("message"),
        "param": err.get("param"),
    }


def _headers(api_key: str, *, project: str = "", organization: str = "") -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if project:
        headers["OpenAI-Project"] = project
    if organization:
        headers["OpenAI-Organization"] = organization
    return headers


def main() -> int:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-realtime")
    project = os.getenv("OPENAI_PROJECT", os.getenv("OPENAI_PROJECT_ID", ""))
    organization = os.getenv("OPENAI_ORGANIZATION", os.getenv("OPENAI_ORG_ID", ""))

    if not api_key:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    variants = [
        ("configured_scope", project, organization),
        ("no_project_header", "", organization),
        ("no_scope_headers", "", ""),
    ]

    for name, variant_project, variant_organization in variants:
        headers = _headers(api_key, project=variant_project, organization=variant_organization)
        status, payload = _request(f"/models/{model}", headers)
        result: dict[str, Any] = {
            "variant": name,
            "status": status,
            "model": model,
            "sent_project_header": bool(variant_project),
            "sent_organization_header": bool(variant_organization),
        }
        if status == 200:
            result["accessible"] = True
            result["id"] = payload.get("id")
        else:
            result["accessible"] = False
            result["error"] = _error(payload)
        print(json.dumps(result, ensure_ascii=False))

    status, payload = _request("/models", _headers(api_key))
    if status == 200:
        models = sorted(
            item.get("id", "")
            for item in payload.get("data", [])
            if "realtime" in item.get("id", "") or "audio" in item.get("id", "")
        )
        print(json.dumps({"variant": "visible_realtime_models", "status": status, "models": models}, ensure_ascii=False))
    else:
        print(json.dumps({"variant": "visible_realtime_models", "status": status, "error": _error(payload)}, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
