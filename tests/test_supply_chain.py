"""Security regression checks for downloaded artifacts and GitHub Actions."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL_SHA_ACTION = re.compile(r"^\s*uses:\s*[^@\s]+@([0-9a-f]{40})(?:\s+#.*)?$", re.MULTILINE)


class SupplyChainTests(unittest.TestCase):
    def test_onnx_runtime_archive_has_pinned_integrity(self):
        text = (ROOT / "tools" / "setup_ort_web.py").read_text(encoding="utf-8")
        self.assertIn('VERSION = "1.23.2"', text)
        match = re.search(
            r'EXPECTED_ARCHIVE_INTEGRITY\s*=\s*\(\s*"([^"]+)"\s*"([^"]+)"\s*\)',
            text,
        )
        self.assertIsNotNone(match, "expected a checked-in npm SHA-512 integrity value")
        integrity = "".join(match.groups())
        self.assertTrue(integrity.startswith("sha512-"))
        self.assertGreater(len(integrity), 90)
        self.assertIn("verify_archive(archive)", text)

    def test_all_github_actions_are_pinned_to_full_commit_shas(self):
        workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
        self.assertTrue(workflows)
        for workflow in workflows:
            text = workflow.read_text(encoding="utf-8")
            uses_lines = [
                line for line in text.splitlines()
                if line.lstrip().startswith("uses:")
            ]
            for line in uses_lines:
                with self.subTest(workflow=workflow.name, line=line):
                    match = FULL_SHA_ACTION.match(line)
                    self.assertIsNotNone(
                        match,
                        f"{workflow.name} contains a mutable action ref: {line.strip()}",
                    )


if __name__ == "__main__":
    unittest.main()
