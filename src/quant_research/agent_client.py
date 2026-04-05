from __future__ import annotations

import configparser
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class AgentConfig:
    vendor: str
    base_url: str
    api_key: str
    model: str
    timeout_sec: int
    temperature: float
    system_prompt: str


def _expand_env(value: str) -> str:
    return os.path.expandvars(value).strip()


def load_agent_config(config_path: str | Path = "agent.config") -> AgentConfig:
    path = Path(config_path)
    parser = configparser.ConfigParser()
    if path.exists():
        parser.read(path, encoding="utf-8")

    section = parser["agent"] if "agent" in parser else {}
    vendor = _expand_env(str(section.get("vendor", "openai"))).lower()
    base_url = _expand_env(str(section.get("base_url", "https://api.openai.com/v1"))).rstrip("/")
    api_key = _expand_env(str(section.get("api_key", "${OPENAI_API_KEY}")))
    model = _expand_env(str(section.get("model", "gpt-4o-mini")))
    timeout_sec = int(section.get("timeout_sec", "45"))
    temperature = float(section.get("temperature", "0.2"))
    system_prompt = str(section.get("system_prompt", "你是 AlphaForge 的量化研究助手。")).strip()
    return AgentConfig(
        vendor=vendor,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_sec=timeout_sec,
        temperature=temperature,
        system_prompt=system_prompt,
    )


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {text}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response: {body[:300]}") from exc


def chat_completion(messages: list[dict[str, str]], cfg: AgentConfig) -> str:
    if cfg.vendor in {"openai", "openai_compatible"}:
        if not cfg.api_key or cfg.api_key.startswith("${"):
            raise RuntimeError("缺少 API Key，请在 agent.config 中配置 api_key 或设置环境变量 OPENAI_API_KEY。")
        url = f"{cfg.base_url}/chat/completions"
        payload = {
            "model": cfg.model,
            "messages": [{"role": "system", "content": cfg.system_prompt}, *messages],
            "temperature": cfg.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        }
        data = _post_json(url, payload, headers, cfg.timeout_sec)
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"OpenAI response parse error: {data}") from exc

    if cfg.vendor == "ollama":
        url = f"{cfg.base_url}/api/chat"
        payload = {
            "model": cfg.model,
            "messages": [{"role": "system", "content": cfg.system_prompt}, *messages],
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        data = _post_json(url, payload, headers, cfg.timeout_sec)
        msg = data.get("message", {})
        content = msg.get("content", "")
        if not content:
            raise RuntimeError(f"Ollama response parse error: {data}")
        return str(content).strip()

    raise RuntimeError(f"Unsupported vendor: {cfg.vendor}")
