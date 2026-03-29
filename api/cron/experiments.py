"""
Vercel serverless cron — Experiments Daily Best Stock Picker.

Runs the experiments daily_scan.py, then pushes experiments/docs/data/ to GitHub.
Schedule: 22:30 UTC Mon-Fri (5:30pm ET)
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
    url = f"{GITHUB_API}/repos/{REPO}/contents/{path}"
    r = req.get(url, headers=github_headers(), params={"ref": BRANCH})
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def push_file(path, content_bytes, message):
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
    """Run the experiments daily scan and push results to GitHub."""
    # Navigate to experiments directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    experiments_dir = os.path.join(repo_root, "experiments")
    scripts_dir = os.path.join(experiments_dir, "scripts")

    sys.path.insert(0, experiments_dir)
    sys.path.insert(0, scripts_dir)

    original_cwd = os.getcwd()
    os.chdir(experiments_dir)

    try:
        # Import and run the experiments daily scan
        import importlib
        spec = importlib.util.spec_from_file_location(
            "experiments_scan", os.path.join(scripts_dir, "daily_scan.py")
        )
        scan_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scan_module)
        scan_module.main()
    finally:
        os.chdir(original_cwd)

    # Push experiments/docs/data/ to GitHub
    data_dir = os.path.join(experiments_dir, "docs", "data")
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Scan produced no output: {data_dir} not found")

    pushed = push_directory(
        data_dir, "experiments/docs/data", "chore: daily experiments scan"
    )
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
