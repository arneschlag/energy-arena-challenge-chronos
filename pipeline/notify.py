"""SMTP-Versand fuer die kombinierte Tageszusammenfassung.

Die Konfiguration stammt ausschliesslich aus ``SMTP_*``/``ALERT_*``-Variablen.
Passwoerter und andere Konfigurationswerte werden nie geloggt oder in
Fehlermeldungen dieses Moduls eingebettet.
"""
from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Callable, Mapping


class NotificationConfigError(ValueError):
    """Unvollstaendige, aber nicht geheime SMTP-Konfiguration."""


def _first(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _boolean(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on", "ja"}


def _recipients(raw: str) -> tuple[str, ...]:
    addresses = tuple(
        address.strip() for part in raw.split(";") for address in part.split(",") if address.strip()
    )
    if not addresses:
        raise NotificationConfigError("Keine Empfängeradresse konfiguriert")
    return addresses


@dataclass(frozen=True)
class SMTPConfig:
    host: str
    port: int
    sender: str
    recipients: tuple[str, ...]
    username: str | None
    password: str | None
    use_ssl: bool
    starttls: bool
    timeout: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SMTPConfig":
        env = os.environ if env is None else env
        host = _first(
            env, "SMTP_HOST", "SMTP_SERVER", "ALERT_SMTP_HOST", "ALERT_SMTP_SERVER"
        )
        to = _first(
            env,
            "ALERT_EMAIL_TO",
            "ALERT_TO",
            "ALERT_EMAIL",
            "ALERT_RECIPIENT",
            "ALERT_RECIPIENTS",
            "SMTP_TO",
            "SMTP_RECIPIENTS",
        )
        username = _first(
            env, "SMTP_USERNAME", "SMTP_USER", "ALERT_SMTP_USERNAME", "ALERT_SMTP_USER"
        )
        password = _first(
            env,
            "SMTP_PASSWORD",
            "SMTP_PASS",
            "ALERT_SMTP_PASSWORD",
            "ALERT_SMTP_PASS",
        )
        sender = _first(
            env,
            "ALERT_EMAIL_FROM",
            "ALERT_FROM",
            "ALERT_SENDER",
            "SMTP_FROM",
            "SMTP_SENDER",
        ) or username
        missing = [
            name
            for name, value in (("SMTP_HOST", host), ("Empfänger", to), ("Absender", sender))
            if not value
        ]
        if missing:
            raise NotificationConfigError(
                "SMTP-Konfiguration unvollständig: " + ", ".join(missing)
            )
        try:
            port = int(_first(env, "SMTP_PORT", "ALERT_SMTP_PORT") or "587")
            timeout = float(_first(env, "SMTP_TIMEOUT", "ALERT_SMTP_TIMEOUT") or "30")
        except ValueError as exc:
            raise NotificationConfigError("SMTP_PORT/SMTP_TIMEOUT ist keine Zahl") from exc
        use_ssl = _boolean(_first(env, "SMTP_SSL", "ALERT_SMTP_SSL"), port == 465)
        starttls_default = not use_ssl and port in {25, 587}
        starttls = _boolean(
            _first(env, "SMTP_STARTTLS", "SMTP_TLS", "ALERT_SMTP_STARTTLS"),
            starttls_default,
        )
        return cls(
            host=host,
            port=port,
            sender=sender,
            recipients=_recipients(to),
            username=username,
            password=password,
            use_ssl=use_ssl,
            starttls=starttls,
            timeout=timeout,
        )


def build_message(config: SMTPConfig, subject: str, body: str) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config.sender
    message["To"] = ", ".join(config.recipients)
    message.set_content(body)
    return message


def send_mail(
    subject: str,
    body: str,
    *,
    config: SMTPConfig | None = None,
    env: Mapping[str, str] | None = None,
    smtp_factory: Callable[..., smtplib.SMTP] | None = None,
    smtp_ssl_factory: Callable[..., smtplib.SMTP_SSL] | None = None,
) -> None:
    """Mail senden; bei Erfolg oder Fehler werden keine Zugangsdaten ausgegeben."""
    config = config or SMTPConfig.from_env(env)
    smtp_factory = smtp_factory or smtplib.SMTP
    smtp_ssl_factory = smtp_ssl_factory or smtplib.SMTP_SSL
    factory = smtp_ssl_factory if config.use_ssl else smtp_factory
    with factory(config.host, config.port, timeout=config.timeout) as smtp:
        if config.starttls and not config.use_ssl:
            smtp.starttls()
        if config.username:
            smtp.login(config.username, config.password or "")
        smtp.send_message(build_message(config, subject, body))
