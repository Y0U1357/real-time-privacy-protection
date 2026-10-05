"""Download the CC0 benchmark videos listed in benchmarks/videos.json.

Examples:
    python tools/download_benchmark_videos.py --list
    python tools/download_benchmark_videos.py
    python tools/download_benchmark_videos.py --only mumbai_street_cc0

Videos are intentionally not committed to Git. The manifest keeps the source,
author and license page so benchmark runs remain reproducible.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "videos.json"
OUTPUT_DIR = ROOT / "benchmarks" / "videos"
REDIRECT_BASE = "https://commons.wikimedia.org/wiki/Special:Redirect/file/"


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def download_url(item):
    return REDIRECT_BASE + urllib.parse.quote(item["commons_filename"], safe="()")


def human_bytes(value):
    units = ("B", "KiB", "MiB", "GiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def download(item, force=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / item["local_filename"]
    if target.exists() and not force:
        print(f"SKIP {item['id']}: {target} already exists")
        return target

    url = download_url(item)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "real-time-privacy-protection benchmark downloader/1.0"},
    )
    print(f"DOWNLOAD {item['id']}")
    print(f"  source : {item['source_page']}")
    print(f"  license: {item['license']}")
    print(f"  target : {target}")

    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0") or 0)
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            if total:
                pct = copied * 100 / total
                print(f"  {pct:5.1f}%  {human_bytes(copied)} / {human_bytes(total)}", end="\r")
        if total:
            print()
    if target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded empty file for {item['id']}")
    print(f"DONE {item['id']}: {human_bytes(target.stat().st_size)}")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list benchmark videos and exit")
    parser.add_argument("--only", action="append", default=[], help="download only this manifest id; repeatable")
    parser.add_argument("--force", action="store_true", help="overwrite existing local files")
    args = parser.parse_args()

    items = load_manifest()
    if args.list:
        for item in items:
            print(f"{item['id']}: {item['title']} | {item['license']} | "
                  f"{item['resolution']} | {item['duration_seconds']}s")
        return 0

    wanted = set(args.only)
    if wanted:
        known = {item["id"] for item in items}
        unknown = wanted - known
        if unknown:
            parser.error("unknown benchmark id(s): " + ", ".join(sorted(unknown)))
        items = [item for item in items if item["id"] in wanted]

    for item in items:
        download(item, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
