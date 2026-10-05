"""Static wiring checks for the browser demo in web/ + its headless self-test.

Run:  py tests/test_web.py      (or  py -m unittest tests.test_web)

Static part (always runs, no browser needed):
  * web text files decode as UTF-8 with no lost-character placeholders
  * every DOM id referenced from web/js exists in the matching HTML page
  * every relative ES-module import resolves to a real file
  * every local script/stylesheet/model asset referenced by the pages exists
  * the tuning ranges in index.html match LIMITS in web/js/config.js

Headless part (skipped when Chrome/Edge is missing):
  * web/selftest.html runs the real geometry/tracker/stable-ID/pixelation
    modules with synthetic detections and reports 0 failures.
"""
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_web import find_chrome, free_port, start_server  # noqa: E402

PLACEHOLDER = re.compile(r"\?{3,}")
TEXT_FILES = ("index.html", "selftest.html", "style.css")
GENERATED_PAGE_ASSETS = {
    "assets/shengli_night_market_cc0-processed-demo.gif":
        ROOT / "docs" / "assets" / "shengli_night_market_cc0-processed-demo.gif",
}

# Which page each script is loaded from (DOM ids are checked against it).
PAGE_OF_SCRIPT = {
    "app.js": "index.html",
    "selftest.js": "selftest.html",
}


def read(path):
    return Path(path).read_text(encoding="utf-8")


def js_files():
    return sorted((WEB / "js").glob("*.js"))


class TextIntegrityTests(unittest.TestCase):
    def test_web_text_files_decode_as_utf8_without_placeholders(self):
        names = list(TEXT_FILES) + [f"js/{path.name}" for path in js_files()]
        for name in names:
            with self.subTest(name=name):
                text = (WEB / name).read_bytes().decode("utf-8")
                self.assertNotIn("\ufffd", text)
                self.assertEqual(PLACEHOLDER.findall(text), [],
                                 "lost non-ASCII characters were written as '?' runs")


class WiringTests(unittest.TestCase):
    def test_dom_ids_used_by_scripts_exist_in_their_page(self):
        patterns = (re.compile(r"getElementById\(\s*['\"]([^'\"]+)['\"]\s*\)"),
                    re.compile(r"querySelector\(\s*['\"]#([A-Za-z0-9\-_]+)['\"]\s*\)"))
        checked = 0
        for script in js_files():
            page = PAGE_OF_SCRIPT.get(script.name)
            if page is None:
                continue
            html = read(WEB / page)
            ids = set(re.findall(r'id="([^"]+)"', html))
            for pattern in patterns:
                for element_id in pattern.findall(read(script)):
                    with self.subTest(script=script.name, id=element_id):
                        self.assertIn(element_id, ids,
                                      f"{script.name} references #{element_id} "
                                      f"which is missing from {page}")
                        checked += 1
        self.assertGreater(checked, 20, "expected the app to reference many ids")

    def test_relative_es_module_imports_resolve(self):
        import_pattern = re.compile(r"from\s+['\"]\./([^'\"]+)['\"]")
        for script in js_files():
            for target in import_pattern.findall(read(script)):
                with self.subTest(script=script.name, target=target):
                    self.assertTrue((script.parent / target).is_file(),
                                    f"{script.name} imports ./{target} which does not exist")

    def test_local_assets_referenced_by_pages_exist(self):
        asset_pattern = re.compile(r"(?:src|href)=\"\.\/([^\"#?]+)\"")
        for page in ("index.html", "selftest.html"):
            for asset in asset_pattern.findall(read(WEB / page)):
                with self.subTest(page=page, asset=asset):
                    local_asset = WEB / asset
                    if local_asset.is_file():
                        continue
                    generated_source = GENERATED_PAGE_ASSETS.get(asset)
                    self.assertIsNotNone(
                        generated_source,
                        f"{page} references ./{asset} which does not exist",
                    )
                    self.assertTrue(
                        generated_source.is_file(),
                        f"generated Pages asset source is missing: {generated_source}",
                    )

    def test_model_url_in_config_exists_or_is_documented_as_generated(self):
        config = read(WEB / "js" / "config.js")
        match = re.search(r"modelUrl:\s*['\"]\./([^'\"]+)['\"]", config)
        self.assertIsNotNone(match, "config.js must keep a local modelUrl")
        model = WEB / match.group(1)
        if model.is_file():
            self.assertGreater(model.stat().st_size, 0, "model file must not be empty")
            return
        # Public/portfolio copies intentionally exclude the generated ONNX blob.
        # Keep the wiring test meaningful by requiring both model metadata and
        # checked-in setup documentation when the binary is absent.
        self.assertTrue((WEB / "models" / "yolox_s_mot17.json").is_file(),
                        "model metadata must remain checked in")
        docs = ROOT / "docs" / "MODELS.md"
        self.assertTrue(docs.is_file(), "missing docs/MODELS.md for excluded model")
        self.assertIn("tools/export_yolox_onnx.py", read(docs))

    def test_tuning_ranges_in_html_match_config_limits(self):
        config = read(WEB / "js" / "config.js")
        limits = {name: (float(low), float(high), float(step))
                  for name, low, high, step in re.findall(
                      r"(\w+):\s*\{\s*min:\s*([\d.]+),\s*max:\s*([\d.]+),"
                      r"\s*step:\s*([\d.]+)\s*\}", config)}
        self.assertEqual(sorted(limits), ["holdFrames", "padding", "pixelScale"])
        html = read(WEB / "index.html")
        fields = {"padding": "padding-input", "pixelScale": "scale-input",
                  "holdFrames": "hold-input"}
        for name, element_id in fields.items():
            match = re.search(
                rf'<input id="{element_id}"[^>]*min="([\d.]+)"[^>]*'
                rf'max="([\d.]+)"[^>]*step="([\d.]+)"', html)
            self.assertIsNotNone(match, f"#{element_id} range input not found")
            found = tuple(float(value) for value in match.groups())
            with self.subTest(field=name):
                self.assertEqual(found, limits[name],
                                 f"{name} range drifted between index.html and LIMITS")


class HeadlessSelfTest(unittest.TestCase):
    """Runs web/selftest.html in headless Chrome/Edge and reads its verdict."""

    def test_selftest_page_reports_no_failures(self):
        import shutil
        import tempfile
        chrome = find_chrome()
        if chrome is None:
            self.skipTest("no Chrome/Edge found; set CHROME=<path> to run this")
        port = free_port()
        server = start_server(port)
        profile = tempfile.mkdtemp(prefix="privacy-selftest-")
        process = None
        try:
            args = [chrome, "--headless=new", "--no-first-run",
                    "--no-default-browser-check", f"--user-data-dir={profile}",
                    "--virtual-time-budget=15000", "--dump-dom"]
            # Chromium refuses to start its sandbox when the test itself runs as
            # root (common in containers). This flag is only added in that case.
            if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0:
                args.append("--no-sandbox")
            args.append(f"http://127.0.0.1:{port}/selftest.html")
            process = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            output, _ = process.communicate(timeout=120)
            dom = output.decode("utf-8", errors="replace")
            self.assertIn('data-status="passed"', dom,
                          "self-test page did not report success:\n" + dom[-2000:])
            self.assertNotIn('class="fail"', dom,
                             "self-test page contains failing checks:\n" + dom)
            summary = re.search(r'id="summary"[^>]*>([^<]+)<', dom)
            print(f"selftest.html: {summary.group(1) if summary else 'unknown'}")
        finally:
            if process is not None and process.poll() is None:
                process.kill()
            server.shutdown()
            server.server_close()
            shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
