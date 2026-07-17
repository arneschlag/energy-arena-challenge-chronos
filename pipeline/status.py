"""CLI fuer die kombinierte G1_C1-/G3_C5-Tagesmail.

Beispiel::

    python -m pipeline.status --send

Ohne ``--send`` wird die Zusammenfassung nur auf stdout ausgegeben.  So kann
der Summary-Container vor Aktivierung des SMTP-Versands sicher geprueft werden.
"""
from __future__ import annotations

import argparse
import sys

from loaders import config as lc

from .notify import NotificationConfigError, send_mail
from .summary import G1, G3, load_daily_summary, state_path


def parser() -> argparse.ArgumentParser:
    lc.load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g1-state", default=str(state_path(G1)))
    p.add_argument("--g3-state", default=str(state_path(G3)))
    p.add_argument("--send", action="store_true", help="Zusammenfassung per SMTP senden")
    p.add_argument(
        "--print",
        dest="print_summary",
        action="store_true",
        help="Text auch bei --send auf stdout ausgeben",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    summary = load_daily_summary(g1_path=args.g1_state, g3_path=args.g3_state)
    if not args.send or args.print_summary:
        print(summary.subject)
        print(summary.body, end="")
    if args.send:
        try:
            send_mail(summary.subject, summary.body)
        except NotificationConfigError as exc:
            print(f"Statusmail nicht gesendet: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            # Keine SMTP-Parameter oder State-Inhalte in die Meldung aufnehmen.
            print(f"Statusmail nicht gesendet ({type(exc).__name__})", file=sys.stderr)
            return 1
        print("Kombinierte Statusmail gesendet.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
