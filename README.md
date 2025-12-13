# CRT Rebound Scanner — Static (Option A)

Pure static GitHub Pages site.

- GitHub Action runs daily **after ~5PM New York time** and writes `docs/data/full.json`.
- The website renders **Top 10** (charts + details) and a table of **all** tickers.

## Build a new repo
1) Create a new GitHub repo (public)
2) Upload everything from this zip (same folder structure)
3) Repo → Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main` / Folder: `/docs`
4) Repo → Actions → **Daily CRT Scan** → Run workflow (first run)
5) Open your GitHub Pages URL

## “I can’t find Daily CRT Scan”
You will after you push the repo **with** `.github/workflows/daily_scan.yml` in it.
Then go to **Actions** in the repo navbar.

## If Actions time out while you’re testing
Edit `scripts/daily_scan.py` and set:
`MAX_TICKERS = 150`
Then set it back to `None` once you confirm everything works.
