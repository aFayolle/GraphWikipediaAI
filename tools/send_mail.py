"""Send the readable-graph report by email, reusing ENWISE credentials."""
from __future__ import annotations

import smtplib, ssl, sys
from email.message import EmailMessage
from pathlib import Path

CONFIG = Path.home() / ".config" / "enwise" / "mail.env"
HTML = Path("reports/readable/mathkg_readable.html")
NOTE = Path("reports/readable/note.txt")
EXTRA = [
    Path("reports/readable_v2/candidates.csv"),
]

def config() -> dict[str, str]:
    values = {}
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values

def main() -> int:
    cfg = config()
    user = cfg.get("GMAIL_USER") or cfg.get("SMTP_USER")
    password = cfg.get("GMAIL_APP_PASSWORD") or cfg.get("SMTP_PASS")
    dest = cfg.get("MAIL_DEST") or user
    if not (user and password and dest):
        sys.exit("missing mail credentials in ~/.config/enwise/mail.env")

    message = EmailMessage()
    message["From"] = user
    message["To"] = dest
    message["Subject"] = "[MathKG] Un graphe lisible — mesures, embeddings, et ce que ça ne résout pas"
    message.set_content(NOTE.read_text(encoding="utf-8"))
    message.add_attachment(
        HTML.read_bytes(), maintype="text", subtype="html", filename=HTML.name
    )
    for path in EXTRA:
        if path.exists():
            message.add_attachment(
                path.read_bytes(), maintype="text", subtype="csv", filename=path.name
            )

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=90) as server:
        server.login(user, password)
        server.send_message(message)
    print(f"sent to {dest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
