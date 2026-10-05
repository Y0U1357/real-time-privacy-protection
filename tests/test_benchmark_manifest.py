"""Static checks for the public CC0 benchmark manifest."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "videos.json"


class BenchmarkManifestTests(unittest.TestCase):
    def test_manifest_is_reproducible_cc0_media(self):
        items = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(items), 2)
        ids = set()
        local_names = set()
        for item in items:
            with self.subTest(item=item.get("id")):
                self.assertNotIn(item["id"], ids)
                self.assertNotIn(item["local_filename"], local_names)
                ids.add(item["id"])
                local_names.add(item["local_filename"])
                self.assertTrue(item["license"].startswith("CC0"))
                self.assertTrue(item["source_page"].startswith(
                    "https://commons.wikimedia.org/wiki/File:"))
                self.assertTrue(item["commons_filename"].endswith(".webm"))
                self.assertTrue(item["local_filename"].endswith(".webm"))
                self.assertGreater(item["duration_seconds"], 0)
                self.assertRegex(item["resolution"], r"^\d+x\d+$")

    def test_downloader_and_docs_exist(self):
        self.assertTrue((ROOT / "tools" / "download_benchmark_videos.py").is_file())
        self.assertTrue((ROOT / "docs" / "BENCHMARK.md").is_file())


if __name__ == "__main__":
    unittest.main()
