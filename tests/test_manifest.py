"""Manifest loading and validation."""

import json
import os
import tempfile
import unittest

from helpers import MANIFEST_JSON, REPO

from gfv.common import manifest

try:
    import tomllib
except ImportError:
    tomllib = None


class TestManifest(unittest.TestCase):
    def write(self, doc):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f)
        self.addCleanup(os.remove, path)
        return path

    def base(self):
        with open(MANIFEST_JSON) as f:
            return json.load(f)

    @unittest.skipUnless(tomllib, "reading TOML needs Python 3.11+")
    def test_json_copy_matches_shipped_toml(self):
        with open(os.path.join(REPO, "manifests", "hgx-h100-8x.toml"), "rb") as f:
            self.assertEqual(tomllib.load(f), self.base())

    @unittest.skipUnless(tomllib, "reading TOML needs Python 3.11+")
    def test_checkout_lists_shipped_classes(self):
        self.assertIn("hgx-h100-8x", manifest.bundled_classes())
        cls, spec, source, _ = manifest.load(cls="hgx-h100-8x")
        self.assertEqual(spec["gpus"], 8)
        self.assertEqual(source, os.path.join("manifests", "hgx-h100-8x.toml"))

    def test_single_class_is_implicit(self):
        cls, spec, _, _ = manifest.load(MANIFEST_JSON)
        self.assertEqual(cls, "hgx-h100-8x")

    def test_missing_and_wrong_types(self):
        doc = self.base()
        spec = doc["class"]["hgx-h100-8x"]
        del spec["driver"]
        spec["gpus"] = True
        spec["perf"]["rdma_write_bw_gbps"] = "fast"
        with self.assertRaises(manifest.ManifestError) as e:
            manifest.load(self.write(doc))
        msg = str(e.exception)
        for part in ("missing driver", "gpus must be int", "perf.rdma_write_bw_gbps must be a number"):
            self.assertIn(part, msg)

    def test_storage_needs_path(self):
        doc = self.base()
        doc["class"]["hgx-h100-8x"]["storage"] = {"size_gib": 4}
        with self.assertRaises(manifest.ManifestError):
            manifest.load(self.write(doc))

    def test_class_choice(self):
        doc = self.base()
        doc["class"]["other"] = dict(doc["class"]["hgx-h100-8x"])
        path = self.write(doc)
        with self.assertRaises(manifest.ManifestError):
            manifest.load(path)
        self.assertEqual(manifest.load(path, "other")[0], "other")
        with self.assertRaises(manifest.ManifestError):
            manifest.load(path, "missing")

    def test_unreadable(self):
        with self.assertRaises(manifest.ManifestError):
            manifest.load("/nonexistent/manifest.json")
        with self.assertRaises(manifest.ManifestError):
            manifest.load(self.write({"no": "classes"}))


if __name__ == "__main__":
    unittest.main()
