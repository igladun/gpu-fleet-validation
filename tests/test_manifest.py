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
    def test_shipped_toml_validates_with_power_thermal_and_burnin(self):
        with open(os.path.join(REPO, "manifests", "hgx-h100-8x.toml"), "rb") as f:
            spec = tomllib.load(f)["class"]["hgx-h100-8x"]
        manifest.validate("hgx-h100-8x", spec)
        self.assertEqual(spec["gpu_power_limit_w"], 700)
        self.assertEqual(spec["thermal"]["gpu_max_operating_c"], 87)
        self.assertLess(spec["burnin"]["gpu_temp_max_c"], spec["thermal"]["gpu_max_operating_c"])
        self.assertLess(spec["burnin"]["mem_temp_max_c"], spec["thermal"]["mem_max_operating_c"])

    def test_thermal_and_burnin_keys_are_checked(self):
        for table, key, value, msg in (("thermal", "gpu_max_operating", 87, "unknown thermal.gpu_max_operating"),
                                       ("thermal", "gpu_slowdown_c", "89", "thermal.gpu_slowdown_c must be a number"),
                                       ("burnin", "minutes", 0, "burnin.minutes has an invalid value"),
                                       ("burnin", "gpu_temp_max", 82, "unknown burnin.gpu_temp_max")):
            doc = self.base()
            doc["class"]["hgx-h100-8x"][table] = {key: value}
            with self.assertRaises(manifest.ManifestError) as e:
                manifest.load(self.write(doc))
            self.assertIn(msg, str(e.exception))
        doc = self.base()
        doc["class"]["hgx-h100-8x"]["gpu_power_limit_w"] = True
        with self.assertRaises(manifest.ManifestError):
            manifest.load(self.write(doc))

    @unittest.skipUnless(tomllib, "reading TOML needs Python 3.11+")
    def test_checkout_lists_shipped_classes(self):
        self.assertIn("hgx-h100-8x", manifest.bundled_classes())
        cls, spec, source, _ = manifest.load(cls="hgx-h100-8x")
        self.assertEqual(spec["gpus"], 8)
        self.assertEqual(source, os.path.join("manifests", "hgx-h100-8x.toml"))

    def test_single_class_is_implicit(self):
        cls, spec, _, _ = manifest.load(MANIFEST_JSON)
        self.assertEqual(cls, "hgx-h100-8x")

    def assert_rejected(self, doc, *parts):
        with self.assertRaises(manifest.ManifestError) as e:
            manifest.load(self.write(doc))
        for part in parts:
            self.assertIn(part, str(e.exception))

    def test_missing_and_wrong_types(self):
        doc = self.base()
        spec = doc["class"]["hgx-h100-8x"]
        del spec["driver"]
        spec["gpus"] = True
        self.assert_rejected(doc, "missing driver", "gpus must be int")

    def test_table_values(self):
        doc = self.base()
        spec = doc["class"]["hgx-h100-8x"]
        spec["perf"]["rdma_write_bw_gbps"] = "fast"
        spec["host"]["kernel_cmdline_contains"] = "iommu=pt"      # a string would be iterated per character
        spec["host"]["pcie_acs"] = "off"
        self.assert_rejected(doc, "perf.rdma_write_bw_gbps must be a number", "host.kernel_cmdline_contains",
                             "host.pcie_acs")

    def test_typos_do_not_silently_disable_checks(self):
        doc = self.base()
        spec = doc["class"]["hgx-h100-8x"]
        spec["perf"]["nccl_allreduce_busbw_gbs_1nod"] = spec["perf"].pop("nccl_allreduce_busbw_gbs_1node")
        spec["ncl"] = "2.28.9"
        spec["host"]["secureboot"] = True
        self.assert_rejected(doc, "unknown perf.nccl_allreduce_busbw_gbs_1nod", "unknown keys ['ncl']",
                             "unknown host.secureboot")

    def test_malformed_shapes(self):
        for doc in ({"class": {"x": []}}, [], {"class": []}):
            with self.assertRaises(manifest.ManifestError):
                manifest.load(self.write(doc))

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
