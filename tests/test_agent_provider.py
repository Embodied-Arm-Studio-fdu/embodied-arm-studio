import json
import os
import unittest
import tempfile
import io
import urllib.error
import socket
import errno
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.provider import (
    OpenAICompatibleProvider,
    ProviderConfig,
    provider_config,
    provider_status,
    validate_runtime_provider_config,
)


ROOT = Path(__file__).resolve().parents[1]


class AgentProviderTest(unittest.TestCase):
    def test_deepseek_environment_configuration_is_available(self):
        environment = {
            "DEEPSEEK_API_KEY": "server-secret",
            "DEEPSEEK_MODEL": "deepseek-chat",
        }
        with patch.dict(os.environ, environment, clear=True):
            status = provider_status("deepseek")
            config = provider_config("deepseek")

        self.assertTrue(status["configured"])
        self.assertEqual(status["models"], ["deepseek-chat"])
        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "deepseek-chat")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.api_key, "")

    def test_deepseek_request_uses_openai_compatible_endpoint_and_bearer_key(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "configured reply"}}]
        }).encode("utf-8")
        config = ProviderConfig(
            provider="deepseek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key="runtime-secret",
        )

        with patch("agent.provider.urllib.request.urlopen", return_value=response) as urlopen:
            content = OpenAICompatibleProvider().complete(
                [{"role": "user", "content": "hello"}], config
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(content, "configured reply")
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer runtime-secret")
        self.assertEqual(payload["model"], "deepseek-chat")

    def test_empty_openai_base_url_uses_default_endpoint(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "configured reply"}}]
        }).encode("utf-8")
        environment = {
            "OPENAI_API_KEY": "server-secret",
            "OPENAI_MODEL": "test-model",
            "OPENAI_BASE_URL": "",
        }

        with patch.dict(os.environ, environment, clear=True):
            config = provider_config("openai-compatible")
            with patch(
                "agent.provider.urllib.request.urlopen", return_value=response
            ) as urlopen:
                OpenAICompatibleProvider().complete(
                    [{"role": "user", "content": "hello"}],
                    ProviderConfig(
                        provider="openai-compatible",
                        model="test-model",
                    ),
                )

        self.assertEqual(config.base_url, "https://api.openai.com/v1")
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://api.openai.com/v1/chat/completions",
        )

    def test_runtime_provider_url_rejects_private_networks(self):
        with patch(
            "agent.provider.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("127.0.0.1", 443))],
        ):
            with self.assertRaisesRegex(ValueError, "本机或内网"):
                validate_runtime_provider_config(
                    "openai-compatible", "test-model", "https://localhost.example"
                )

    def test_configured_is_not_a_verified_connection(self):
        runtime = {"deepseek": {"api_key": "test-key", "model": "test-model"}}
        self.assertEqual(provider_status("deepseek", runtime)["status"], "unknown")
        self.assertEqual(provider_status("deepseek", runtime, {"deepseek": {"status": "degraded"}})["status"], "degraded")

    def test_http_error_does_not_echo_provider_body_or_key(self):
        error = urllib.error.HTTPError("https://api.deepseek.com", 401, "denied", {}, io.BytesIO(b'private-key-and-content'))
        with patch("agent.provider.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as raised:
                OpenAICompatibleProvider().complete([], ProviderConfig(provider="deepseek", model="test", api_key="secret"))
        self.assertIn("HTTP 401", str(raised.exception))
        self.assertNotIn("private-key", str(raised.exception))
        self.assertNotIn("secret", str(raised.exception))

    def test_refused_outbound_proxy_is_explained(self):
        error = urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
        with patch("agent.provider.urllib.request.urlopen", side_effect=error), patch("agent.provider.urllib.request.getproxies", return_value={"https": "http://127.0.0.1:1"}):
            with self.assertRaisesRegex(RuntimeError, "代理连接被拒绝"):
                OpenAICompatibleProvider().complete([], ProviderConfig(provider="deepseek", model="test", api_key="secret"))

    def test_connection_probe_state_and_clear(self):
        from apps.api.main import create_app
        from tests.auth import StaticCurrentUserProvider
        from tests.http_client import ASGITestClient
        with tempfile.TemporaryDirectory() as temp:
            app = create_app(storage_root=Path(temp)/"runs", project_root=Path(temp)/"projects", run_repository_mode="memory", current_user_provider=StaticCurrentUserProvider("alice"))
            with ASGITestClient(app) as client, patch.dict(os.environ, {}, clear=True):
                app.state.agent_provider_configs["deepseek"] = {"api_key": "probe-secret", "model": "test"}
                self.assertEqual(client.post("/api/agent/providers/deepseek/check", headers={"Authorization": ""}).status_code, 401)
                with patch("agent.provider.OpenAICompatibleProvider.complete", side_effect=RuntimeError("network unavailable")):
                    result = client.post("/api/agent/providers/deepseek/check").json()
                    self.assertEqual(result["status"], "degraded")
                    self.assertNotIn("probe-secret", json.dumps(result))
                with patch("agent.provider.OpenAICompatibleProvider.complete", return_value="OK") as complete:
                    result = client.post("/api/agent/providers/deepseek/check").json()
                    self.assertEqual(result["status"], "healthy")
                    self.assertEqual(complete.call_args.args[0], [{"role": "user", "content": "Reply with OK."}])
                client.delete("/api/agent/providers/deepseek/configuration")
                self.assertNotIn("deepseek", app.state.agent_provider_checks)

    def test_provider_policy_registers_deepseek(self):
        document = json.loads(
            (ROOT / "agent" / "policy" / "provider_limits.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            document["providers"]["deepseek"],
            {"requires_server_secret": True, "timeout_seconds": 30},
        )


if __name__ == "__main__":
    unittest.main()
