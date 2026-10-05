"""Repository/document integrity checks for the browser-only portfolio."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_current_docs_decode_as_utf8(self):
        files = (
            "README.md",
            "docs/ARCHITECTURE.md",
            "docs/DEVELOPMENT_NOTES.md",
            "docs/MODELS.md",
            "docs/BENCHMARK.md",
            "docs/PORTFOLIO.md",
            "THIRD_PARTY.md",
        )
        for name in files:
            with self.subTest(name=name):
                text = (ROOT / name).read_bytes().decode("utf-8")
                self.assertNotIn("\ufffd", text)

    def test_legacy_runtime_is_not_documented_as_current(self):
        combined = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in ("README.md", "docs/ARCHITECTURE.md", "docs/DEVELOPMENT_NOTES.md")
        )
        for legacy in ("app/", "static/", "testcam.py", "testMOT.py", "testMOT2.py"):
            with self.subTest(legacy=legacy):
                self.assertNotIn(legacy, combined)

    def test_browser_product_files_are_present(self):
        required = (
            "web/index.html",
            "web/js/app.js",
            "web/js/detector.js",
            "web/js/tracker.js",
            "web/js/privacy.js",
            "tools/export_yolox_onnx.py",
            "tools/setup_ort_web.py",
            "tools/serve_web.py",
            "tools/download_benchmark_videos.py",
            "tools/run_portfolio_benchmark.py",
            "eval_wildtrack.py",
        )
        for name in required:
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
