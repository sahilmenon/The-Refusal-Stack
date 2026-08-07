# Forensic dashboard

A static, single-page dashboard for The Refusal Stack, hosted on Cloudflare Pages
at **[forensic.sahilmenon.com](https://forensic.sahilmenon.com)**. It presents the
committed results with plain-English explanations for non-experts, charts, and
links to the source papers.

## How it works

- `build_data.py` reads `../results/reported/*.json` and writes `data.js`
  (`window.FORENSIC`), so the page needs no backend and works from `file://` or
  a static host.
- `index.html` / `styles.css` / `app.js` render it. Charts use Chart.js (CDN).
- The "Plain English" toggle shows or hides the explainer boxes.

## Rebuild and deploy

```bash
python dashboard/build_data.py                              # refresh data.js from results/reported/
npx wrangler pages deploy dashboard --project-name=forensic-refusal-stack
```

Deploy needs a `CLOUDFLARE_API_TOKEN` with Pages edit permission in the
environment. Every number traces to a file in `results/reported/`.
