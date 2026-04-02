"""
Vercel serverless cron — Short Strategy Scanner (10D/30D/60D).

Runs the daily_scan_short.py script, then pushes short/docs/data/ files to GitHub.
Schedule: 23:00 UTC Mon-Fri (6pm ET, after main scan)
"""

import os
import sys
import json
import base64
import traceback
from http.server import BaseHTTPRequestHandler

import requests as req


REPO = "viki-m13/crt"
BRANCH = "main"
GITHUB_API = "https://api.github.com"


def github_headers():
    token = os.environ.get("GH_TOKEN", "")
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def get_file_sha(path):
    """Get current SHA of a file in the repo (needed for updates)."""
    url = f"{GITHUB_API}/repos/{REPO}/contents/{path}"
    r = req.get(url, headers=github_headers(), params={"ref": BRANCH})
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def push_file(path, content_bytes, message):
    """Create or update a file in the repo via GitHub Contents API."""
    url = f"{GITHUB_API}/repos/{REPO}/contents/{path}"
    encoded = base64.b64encode(content_bytes).decode("utf-8")
    body = {
        "message": message,
        "content": encoded,
        "branch": BRANCH,
    }
    sha = get_file_sha(path)
    if sha:
        body["sha"] = sha
    r = req.put(url, headers=github_headers(), json=body)
    r.raise_for_status()
    return r.status_code


def push_directory(local_dir, repo_dir, commit_msg):
    """Walk a local directory and push all files to the corresponding repo path."""
    pushed = []
    for root, dirs, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, local_dir)
            repo_path = f"{repo_dir}/{rel}"
            with open(local_path, "rb") as f:
                content = f.read()
            push_file(repo_path, content, commit_msg)
            pushed.append(repo_path)
    return pushed


def run_scan():
    """Run the short strategy scan and push results to GitHub."""
    # Add short/scripts dir to path so daily_scan_short can be imported
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "short", "scripts")
    scripts_dir = os.path.abspath(scripts_dir)
    sys.path.insert(0, scripts_dir)

    # Force run (bypass time gate)
    os.environ["FORCE_RUN"] = "1"

    # Change to repo root so relative paths work
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    original_cwd = os.getcwd()
    os.chdir(repo_root)

    # Vercel's deployed filesystem is read-only except /tmp.
    # Redirect output to /tmp so file writes succeed.
    tmp_data_dir = "/tmp/short_docs_data"
    tmp_ticker_dir = os.path.join(tmp_data_dir, "tickers")

    try:
        import daily_scan_short
        daily_scan_short.OUT_DIR = tmp_data_dir
        daily_scan_short.TICKER_DIR = tmp_ticker_dir
        daily_scan_short.main()
    finally:
        os.chdir(original_cwd)

    # Push /tmp output to GitHub at short/docs/data/
    if not os.path.isdir(tmp_data_dir):
        raise RuntimeError(f"Scan produced no output: {tmp_data_dir} not found")

    pushed = push_directory(tmp_data_dir, "short/docs/data", "chore: short strategy daily scan update")
    return pushed


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            pushed = run_scan()
            body = json.dumps({
                "ok": True,
                "pushed": len(pushed),
                "files": pushed[:20],
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        except Exception as e:
            body = json.dumps({
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
