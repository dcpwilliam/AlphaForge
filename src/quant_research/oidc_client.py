from __future__ import annotations

import configparser
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request


def _expand_env(value: str) -> str:
    return os.path.expandvars(value).strip()


@dataclass(frozen=True)
class OIDCConfig:
    client_id: str
    client_secret: str
    issuer: str
    auth_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    jwks_endpoint: str
    redirect_uri: str
    scope: str


def load_oidc_config(config_path: str | Path = "agent.config") -> OIDCConfig:
    path = Path(config_path)
    parser = configparser.ConfigParser()
    if path.exists():
        parser.read(path, encoding="utf-8")
    section = parser["oidc"] if "oidc" in parser else {}

    client_id = _expand_env(str(section.get("client_id", os.getenv("OIDC_CLIENT_ID", "69ba3eccdb057f1398c742c5"))))
    client_secret = _expand_env(str(section.get("client_secret", os.getenv("OIDC_CLIENT_SECRET", "${OIDC_CLIENT_SECRET}"))))
    issuer = _expand_env(str(section.get("issuer", "https://dcp-ultimate.authing.cn/oidc"))).rstrip("/")
    auth_endpoint = _expand_env(str(section.get("auth_endpoint", "https://dcp-ultimate.authing.cn/oidc/auth")))
    token_endpoint = _expand_env(str(section.get("token_endpoint", "https://dcp-ultimate.authing.cn/oidc/token")))
    userinfo_endpoint = _expand_env(str(section.get("userinfo_endpoint", "https://dcp-ultimate.authing.cn/oidc/me")))
    jwks_endpoint = _expand_env(str(section.get("jwks_endpoint", "https://dcp-ultimate.authing.cn/oidc/.well-known/jwks.json")))
    redirect_uri = _expand_env(str(section.get("redirect_uri", os.getenv("OIDC_REDIRECT_URI", "http://127.0.0.1:8501"))))
    scope = _expand_env(str(section.get("scope", "openid profile")))
    return OIDCConfig(
        client_id=client_id,
        client_secret=client_secret,
        issuer=issuer,
        auth_endpoint=auth_endpoint,
        token_endpoint=token_endpoint,
        userinfo_endpoint=userinfo_endpoint,
        jwks_endpoint=jwks_endpoint,
        redirect_uri=redirect_uri,
        scope=scope,
    )


def generate_state() -> str:
    return secrets.token_urlsafe(24)


def build_authorize_url(cfg: OIDCConfig, state: str) -> str:
    qs = parse.urlencode(
        {
            "client_id": cfg.client_id,
            "redirect_uri": cfg.redirect_uri,
            "response_type": "code",
            "scope": cfg.scope,
            "state": state,
        }
    )
    return f"{cfg.auth_endpoint}?{qs}"


def _http_post_form(url: str, form: dict[str, Any], timeout_sec: int = 20) -> dict[str, Any]:
    data = parse.urlencode(form).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
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


def _http_get_json(url: str, headers: dict[str, str], timeout_sec: int = 20) -> dict[str, Any]:
    req = request.Request(url=url, headers=headers, method="GET")
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


def exchange_code_for_token(cfg: OIDCConfig, code: str) -> dict[str, Any]:
    if not cfg.client_id:
        raise RuntimeError("OIDC client_id 不能为空。")
    if not cfg.client_secret or cfg.client_secret.startswith("${"):
        raise RuntimeError("OIDC client_secret 未配置。请在环境变量 OIDC_CLIENT_SECRET 或 agent.config [oidc] 中配置。")
    return _http_post_form(
        cfg.token_endpoint,
        {
            "grant_type": "authorization_code",
            "client_id": cfg.client_id,
            "client_secret": cfg.client_secret,
            "redirect_uri": cfg.redirect_uri,
            "code": code,
        },
    )


def fetch_me(cfg: OIDCConfig, access_token: str) -> dict[str, Any]:
    if not access_token:
        raise RuntimeError("access_token 为空。")
    return _http_get_json(
        cfg.userinfo_endpoint,
        headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
    )


def _extract_any(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        cur: Any = data
        ok = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def _normalize_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    items: list[Any]
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    out: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            for k in ("id", "asset_id", "assetId", "strategy_id", "strategyId", "factor_id", "factorId"):
                if k in item and item[k] is not None:
                    out.add(str(item[k]).strip())
        elif item is not None:
            out.add(str(item).strip())
    return {x for x in out if x}


def normalize_me_profile(me: dict[str, Any]) -> dict[str, Any]:
    user_id = str(
        _extract_any(
            me,
            ["sub", "user_id", "userId", "id", "profile.user_id", "profile.userId", "profile.id"],
            default="",
        )
        or ""
    ).strip()
    display_name = str(
        _extract_any(
            me,
            ["name", "nickname", "display_name", "displayName", "username", "profile.name", "profile.nickname"],
            default=user_id or "UnknownUser",
        )
        or (user_id or "UnknownUser")
    ).strip()

    raw_balance = _extract_any(me, ["balance", "funds", "capital", "profile.balance", "profile.funds"], default=0)
    try:
        balance = float(raw_balance)
    except (TypeError, ValueError):
        balance = 0.0

    strategy_ids = _normalize_ids(
        _extract_any(
            me,
            [
                "strategy_ids",
                "strategyIds",
                "allowed_strategy_ids",
                "allowedStrategyIds",
                "permissions.strategy_ids",
                "permissions.strategyIds",
                "permissions.strategies",
                "profile.strategy_ids",
                "profile.strategyIds",
            ],
            default=[],
        )
    )
    factor_ids = _normalize_ids(
        _extract_any(
            me,
            [
                "factor_ids",
                "factorIds",
                "allowed_factor_ids",
                "allowedFactorIds",
                "permissions.factor_ids",
                "permissions.factorIds",
                "permissions.factors",
                "profile.factor_ids",
                "profile.factorIds",
            ],
            default=[],
        )
    )

    return {
        "user_id": user_id or "unknown",
        "display_name": display_name or (user_id or "UnknownUser"),
        "balance": balance,
        "strategy_ids": strategy_ids,
        "factor_ids": factor_ids,
        "raw_me": me,
    }
