"""Tests for OpenAI realtime configuration defaults."""

import os
from unittest.mock import patch


def test_openai_ai_config_default_model_is_ga_realtime() -> None:
    """The in-code default should match the current GA Realtime model."""
    from app.config import AIConfig

    assert AIConfig().openai_model == "gpt-realtime-2.1"


def test_openai_config_load_default_model_is_ga_realtime() -> None:
    """Config.load should fall back to gpt-realtime-2.1 when OPENAI_MODEL is unset."""
    from app.config import Config

    with patch.dict(os.environ, {}, clear=True):
        cfg = Config.load()

    assert cfg.ai.openai_model == "gpt-realtime-2.1"


def test_openai_config_load_respects_model_override() -> None:
    """OPENAI_MODEL remains configurable for gated aliases and snapshots."""
    from app.config import Config

    with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-realtime-2"}, clear=True):
        cfg = Config.load()

    assert cfg.ai.openai_model == "gpt-realtime-2"


def test_openai_config_loads_project_and_organization_scope() -> None:
    """OpenAI project/org env vars should be available to request headers."""
    from app.config import Config

    with patch.dict(
        os.environ,
        {
            "OPENAI_PROJECT": "proj_123",
            "OPENAI_ORGANIZATION": "org_123",
        },
        clear=True,
    ):
        cfg = Config.load()

    assert cfg.ai.openai_project == "proj_123"
    assert cfg.ai.openai_organization == "org_123"


def test_openai_client_builds_scoped_headers() -> None:
    """Realtime WebSocket requests should carry optional project/org scope."""
    from app.ai.openai_realtime import OpenAIRealtimeClient

    client = OpenAIRealtimeClient(
        api_key="sk-test",
        model="gpt-realtime-2",
        ws_endpoint="wss://example.test/realtime/",
        project="proj_123",
        organization="org_123",
    )

    assert client._ws_url == "wss://example.test/realtime"
    assert client._build_headers() == {
        "Authorization": "Bearer sk-test",
        "OpenAI-Project": "proj_123",
        "OpenAI-Organization": "org_123",
    }


def test_openai_client_formats_mismatched_project_error() -> None:
    """Project mismatch should explain that headers cannot switch API-key projects."""
    from app.ai.openai_realtime import OpenAIRealtimeClient

    client = OpenAIRealtimeClient(api_key="sk-test", model="gpt-realtime-2")
    message = client._format_error_message(
        {
            "message": "OpenAI-Project header should match project for API key",
            "type": "invalid_request_error",
            "code": "mismatched_project",
        }
    )

    assert "OPENAI_PROJECT does not match" in message
    assert "Use an API key created in that project" in message


def test_openai_config_load_audio_defaults() -> None:
    """Without overrides the G.711 passthrough, near-field NR and marin voice stay in place."""
    from app.config import Config

    with patch.dict(os.environ, {}, clear=True):
        cfg = Config.load()

    assert cfg.ai.openai_audio_format == "pcmu"
    assert cfg.ai.openai_noise_reduction == "near_field"
    assert cfg.ai.openai_voice == "marin"


def test_openai_config_load_cascade_overrides() -> None:
    """pcm16 / none / empty voice are parsed for Realtime-compatible gateways."""
    from app.config import Config

    with patch.dict(
        os.environ,
        {
            "OPENAI_AUDIO_FORMAT": "PCM16",
            "OPENAI_NOISE_REDUCTION": "none",
            "OPENAI_VOICE": "",
        },
        clear=True,
    ):
        cfg = Config.load()

    assert cfg.ai.openai_audio_format == "pcm16"
    assert cfg.ai.openai_noise_reduction == "none"
    assert cfg.ai.openai_voice == ""


def test_openai_config_rejects_unknown_audio_format() -> None:
    """An unknown OPENAI_AUDIO_FORMAT fails fast instead of silently using pcmu."""
    import pytest

    from app.config import Config

    with patch.dict(os.environ, {"OPENAI_AUDIO_FORMAT": "opus"}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_AUDIO_FORMAT"):
            Config.load()


def test_openai_config_rejects_unknown_noise_reduction() -> None:
    """An unknown OPENAI_NOISE_REDUCTION fails fast."""
    import pytest

    from app.config import Config

    with patch.dict(os.environ, {"OPENAI_NOISE_REDUCTION": "off"}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_NOISE_REDUCTION"):
            Config.load()


def test_openai_client_rejects_unknown_audio_format() -> None:
    """The client validates audio_format itself as well."""
    import pytest

    from app.ai.openai_realtime import OpenAIRealtimeClient

    with pytest.raises(ValueError, match="audio_format"):
        OpenAIRealtimeClient(api_key="sk-test", audio_format="opus")
