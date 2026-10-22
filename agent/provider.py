from __future__ import annotations

import json
import os
import ipaddress
import socket
import ssl
import errno
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = "mock"
    model: str = "teaching-mock-v1"
    base_url: str = ""
    api_key: str = ""


PROVIDER_METADATA: Dict[str, Dict[str, Any]] = {
    "mock": {
        "name": "内置基础 Agent",
        "models": ["teaching-mock-v1"],
        "secret_environment": None,
        "model_environment": None,
        "capabilities": {"chat": True, "structured_output": False, "tool_calls": False},
        "data_scope": "仅使用当前会话中提供的教学上下文，不发送外部请求。",
    },
    "openai-compatible": {
        "name": "OpenAI 兼容 API",
        "models": [],
        "secret_environment": "OPENAI_API_KEY",
        "model_environment": "OPENAI_MODEL",
        "capabilities": {"chat": True, "structured_output": False, "tool_calls": False},
        "data_scope": "会将系统提示和当前会话消息发送到管理员配置的兼容端点。",
    },
    "deepseek": {
        "name": "DeepSeek",
        "models": [],
        "secret_environment": "DEEPSEEK_API_KEY",
        "model_environment": "DEEPSEEK_MODEL",
        "default_base_url": "https://api.deepseek.com",
        "capabilities": {"chat": True, "structured_output": False, "tool_calls": False},
        "data_scope": "会将系统提示和当前会话消息发送到管理员或当前会话配置的 DeepSeek API。",
    },
    "anthropic": {
        "name": "Anthropic API",
        "models": [],
        "secret_environment": "ANTHROPIC_API_KEY",
        "model_environment": "ANTHROPIC_MODEL",
        "capabilities": {"chat": True, "structured_output": False, "tool_calls": False},
        "data_scope": "会将系统提示和当前会话消息发送到 Anthropic API。",
    },
}


class LLMProvider(Protocol):
    def complete(self, messages: List[Dict[str, str]], config: ProviderConfig) -> str: ...


class MockLLMProvider:
    def complete(self, messages: List[Dict[str, str]], _config: ProviderConfig) -> str:
        learner_text = messages[-1]["content"] if messages else ""
        if any("Reply in English." in item["content"] for item in messages if item["role"] == "system"):
            return "Review the recorded distance curve and action_scale. If progress is slow, propose a small allowed adjustment and compare using the same seed. I do not directly edit files. Your question: " + learner_text[:160]
        return (
            "先看当前实验的 distance 曲线和 action_scale。"
            "如果 distance 下降太慢，可在允许范围内小幅提高 action_scale，再用相同 seed 重跑。"
            "我不会直接修改文件；需要修改时请明确要求生成 patch。当前问题：" + learner_text[:160]
        )


class OpenAICompatibleProvider:
    def complete(self, messages: List[Dict[str, str]], config: ProviderConfig) -> str:
        secret_environment = PROVIDER_METADATA[config.provider]["secret_environment"]
        api_key = config.api_key or os.environ.get(secret_environment or "")
        if not api_key:
            raise RuntimeError("服务端未配置当前模型供应商 API Key")
        base_url = (
            config.base_url
            or (os.environ.get("OPENAI_BASE_URL") if config.provider == "openai-compatible" else "")
            or PROVIDER_METADATA[config.provider].get(
                "default_base_url", "https://api.openai.com/v1"
            )
        )
        payload = {"model": config.model, "messages": messages, "temperature": 0.2}
        response = _post_json(base_url.rstrip("/") + "/chat/completions", payload, {"Authorization": "Bearer " + api_key})
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("模型供应商返回格式无效") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("模型供应商返回了空回复")
        return content


class AnthropicProvider:
    def complete(self, messages: List[Dict[str, str]], config: ProviderConfig) -> str:
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("服务端未配置 ANTHROPIC_API_KEY")
        system = "\n".join(item["content"] for item in messages if item["role"] == "system")
        user_messages = [item for item in messages if item["role"] != "system"]
        payload = {"model": config.model, "system": system, "messages": user_messages, "max_tokens": 800}
        response = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        try:
            content = response["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("模型供应商返回格式无效") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("模型供应商返回了空回复")
        return content


PROVIDERS: Dict[str, LLMProvider] = {
    "mock": MockLLMProvider(),
    "openai-compatible": OpenAICompatibleProvider(),
    "deepseek": OpenAICompatibleProvider(),
    "anthropic": AnthropicProvider(),
}


def provider_status(provider_id: str, runtime_configs: Optional[Dict[str, Dict[str, str]]] = None, checks: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    metadata = PROVIDER_METADATA[provider_id]
    secret_environment = metadata["secret_environment"]
    model_environment = metadata["model_environment"]
    runtime_config = (runtime_configs or {}).get(provider_id, {})
    configured = (
        secret_environment is None
        or bool(
            (runtime_config.get("api_key") or os.environ.get(secret_environment))
            and (runtime_config.get("model") or os.environ.get(model_environment))
        )
    )
    models = metadata["models"]
    if configured and model_environment:
        model = runtime_config.get("model") or os.environ.get(model_environment)
        models = [model] if model else []
    return {
        "id": provider_id,
        "name": metadata["name"],
        "status": ("healthy" if provider_id == "mock" else (checks or {}).get(provider_id, {}).get("status", "unknown")) if configured else "unavailable",
        "connection": (checks or {}).get(provider_id, {"status": "untested"}),
        "configured": configured,
        "models": models,
        "capabilities": metadata["capabilities"],
        "data_scope": metadata["data_scope"],
    }


def provider_config(provider_id: str, runtime_configs: Optional[Dict[str, Dict[str, str]]] = None) -> ProviderConfig:
    runtime_config = (runtime_configs or {}).get(provider_id, {})
    status = provider_status(provider_id, runtime_configs)
    if not status["configured"]:
        raise RuntimeError("模型供应商尚未在服务端完整配置")
    if provider_id == "openai-compatible":
        return ProviderConfig(
            provider=provider_id,
            model=runtime_config.get("model") or os.environ["OPENAI_MODEL"],
            base_url=runtime_config.get("base_url")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1",
            api_key=runtime_config.get("api_key", ""),
        )
    if provider_id == "deepseek":
        return ProviderConfig(
            provider=provider_id,
            model=runtime_config.get("model") or os.environ["DEEPSEEK_MODEL"],
            base_url=runtime_config.get("base_url") or "https://api.deepseek.com",
            api_key=runtime_config.get("api_key", ""),
        )
    if provider_id == "anthropic":
        return ProviderConfig(
            provider=provider_id,
            model=runtime_config.get("model") or os.environ["ANTHROPIC_MODEL"],
            api_key=runtime_config.get("api_key", ""),
        )
    return ProviderConfig()


def validate_runtime_provider_config(provider_id: str, model: str, base_url: str = "") -> str:
    if provider_id not in PROVIDER_METADATA or provider_id == "mock":
        raise ValueError("该 Provider 不支持网页端 API 配置")
    if not model.strip() or len(model) > 160:
        raise ValueError("模型名称不能为空且不能超过 160 个字符")
    if provider_id == "anthropic":
        return ""
    candidate = base_url.strip() or PROVIDER_METADATA[provider_id].get("default_base_url", "")
    parsed = urlparse(candidate)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path.endswith("/chat/completions") or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("兼容 API 地址必须是 HTTPS 根地址，不能包含 chat/completions")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError("无法解析兼容 API 地址") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("兼容 API 地址不能指向本机或内网地址")
    return candidate.rstrip("/")


def _post_json(url: str, payload: Dict, headers: Dict[str, str]) -> Dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Never echo provider bodies: they can contain credentials or request data.
        messages = {
            400: "模型供应商拒绝请求，请核对模型 ID 和 API 地址",
            401: "API Key 无效或已撤销，请重新配置",
            402: "模型账户余额不足，请检查供应商账户",
            403: "模型供应商拒绝访问，请检查密钥权限和地区限制",
            404: "模型或 API 路径不存在，请核对模型 ID 和根地址",
            422: "模型供应商不支持当前请求参数",
            429: "模型供应商请求限流，请稍后重试",
        }
        raise RuntimeError(messages.get(exc.code, "模型供应商暂不可用，请稍后重试") + " (HTTP %s)" % exc.code) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError("模型供应商请求超时，请检查地址和网络") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, OSError) and exc.reason.errno == errno.ECONNREFUSED and urllib.request.getproxies().get("https"):
            message = "API 服务器的出站代理连接被拒绝，请检查代理服务和端口"
        elif isinstance(exc.reason, ssl.SSLError):
            message = "模型供应商 TLS 证书验证失败，请检查 API 服务器证书配置"
        elif isinstance(exc.reason, socket.gaierror):
            message = "API 服务器无法解析模型供应商域名，请检查 DNS"
        elif isinstance(exc.reason, (TimeoutError, socket.timeout)):
            message = "模型供应商请求超时，请检查地址和网络"
        else:
            message = "API 服务器无法连接模型供应商，请检查出站网络和代理配置"
        raise RuntimeError(message) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("模型供应商返回了无法解析的响应") from exc

    if not isinstance(body, dict):
        raise RuntimeError("模型供应商返回格式无效")
    return body
