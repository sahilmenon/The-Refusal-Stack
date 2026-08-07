"""Stamp cache-busting content hashes onto the dashboard's asset URLs.

Cloudflare serves app.js / styles.css / data.js with `max-age=14400`, so a deploy
that reuses those filenames can pair fresh HTML with a stale 4-hour-cached asset
for up to four hours. Appending `?v=<content-hash>` changes the URL whenever an
asset's bytes change, forcing a fresh fetch while leaving unchanged assets cached.

Run before every deploy (after build_data.py): python dashboard/stamp_assets.py
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

D = Path("dashboard")
HTML = D / "index.html"
ASSETS = ["styles.css", "app.js", "data.js"]


def content_hash(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()[:8]


def main() -> None:
    html = HTML.read_text(encoding="utf-8")
    stamped = []
    for asset in ASSETS:
        ver = content_hash(D / asset)
        pattern = r'((?:href|src)=")' + re.escape(asset) + r'(?:\?v=[0-9a-f]+)?(")'
        html, n = re.subn(pattern, r"\g<1>" + f"{asset}?v={ver}" + r"\g<2>", html)
        if n:
            stamped.append(f"{asset}?v={ver}")
    HTML.write_text(html, encoding="utf-8")
    print("stamped:", ", ".join(stamped) if stamped else "nothing matched")


if __name__ == "__main__":
    main()
