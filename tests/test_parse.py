"""Parsers against recorded output from a real 8x H100 HGX node (scrubbed), see fixtures/README.md."""

import json
import os
import unittest

from helpers import FIXTURES, read

from gfv.acceptance import parse


class TestParse(unittest.TestCase):
    def test_gpu_csv(self):
        gpus = parse.gpu_csv(read("gpus.csv"))
        self.assertEqual(len(gpus), 8)
        self.assertEqual(gpus[0]["bus"], "0000:0a:00.0")
        self.assertEqual({g["name"] for g in gpus}, {"NVIDIA H100 80GB HBM3"})
        self.assertEqual({g["pcie.link.width.current"] for g in gpus}, {"16"})

    def test_gpu_csv_column_mismatch(self):
        with self.assertRaises(ValueError):
            parse.gpu_csv("0, a, b\n")

    def test_norm_bus(self):
        self.assertEqual(parse.norm_bus("00000000:0A:00.0"), "0000:0a:00.0")
        self.assertEqual(parse.norm_bus("PCI:0000:3b:00"), "0000:3b:00.0")
        self.assertEqual(parse.norm_bus("3b:00.0"), "0000:3b:00.0")

    def test_remapped(self):
        rows = parse.remapped_csv(read("remapped.csv"))
        self.assertEqual(len(rows), 8)
        self.assertEqual({(r["remapped_rows.pending"], r["remapped_rows.failure"]) for r in rows}, {("No", "No")})

    def test_compute_apps(self):
        self.assertEqual(parse.compute_apps(""), [])
        apps = parse.compute_apps("4242, python3, 00000000:0A:00.0, 70000 MiB\n")
        self.assertEqual(apps[0]["process_name"], "python3")

    def test_smi_header(self):
        self.assertEqual(parse.smi_header(read("nvidia-smi.txt")), {"driver": "580.178.04", "cuda": "13.0"})

    def test_topo_rail_map(self):
        tp = parse.topo(read("topo.txt"))
        self.assertEqual(tp["nics"]["NIC0"], "mlx5_ib0")
        self.assertEqual(tp["nics"]["NIC18"], "mlx5_eth12")
        rails = {f"mlx5_ib{i}" for i in range(8)}
        rmap = parse.rail_map(tp, rails)
        self.assertEqual(rmap, {i: f"mlx5_ib{i}" for i in range(8)})

    def test_rail_map_ignores_non_rdma_nics(self):
        tp = parse.topo(read("topo.txt"))
        rmap = parse.rail_map(tp, {"mlx5_ib0"})
        self.assertEqual(rmap[0], "mlx5_ib0")
        self.assertEqual(rmap[1], "mlx5_ib0")       # PXB to NIC0 when its own NIC is not a rail
        self.assertIsNone(rmap[2])                  # NODE distance does not count

    def test_nvlink_status(self):
        st = parse.nvlink_status(read("nvlink-s.txt"))
        self.assertEqual(len(st), 8)
        self.assertEqual(sum(v["active"] for v in st.values()), 144)
        self.assertEqual(sum(v["inactive"] for v in st.values()), 0)

    def test_nvlink_errors(self):
        e = parse.nvlink_errors(read("nvlink-e.txt"))
        self.assertEqual(len(e), 8 * 18 * 3)
        self.assertEqual(set(e.values()), {0})
        self.assertIn((0, 0, "CRC Errors"), e)

    def test_fabric(self):
        fab = parse.fabric(read("smi-q.txt"))
        self.assertEqual(len(fab), 8)
        self.assertEqual({(v["state"], v["status"]) for v in fab.values()}, {("Completed", "Success")})

    def test_lspci_count(self):
        self.assertEqual(parse.lspci_count(read("lspci-nvidia.txt"), "22a3"), 4)
        self.assertEqual(parse.lspci_count(read("lspci-nvidia.txt"), "2330"), 8)

    def test_dcgmi(self):
        self.assertEqual(parse.dcgmi_nvswitches(read("dcgmi-discovery.txt")), 0)
        self.assertEqual(parse.dcgmi_version(read("dcgmi-version.txt")), "4.6.1")

    def test_dcgm_diag(self):
        rows, tests, meta = parse.dcgm_diag(read("dcgm-diag-r3.json"))
        self.assertEqual({t["summary"] for t in tests.values()}, {"Pass"})
        self.assertEqual(meta["version"], "4.6.1")
        tests = {r["test"] for r in rows}
        self.assertTrue({"memory", "diagnostic", "pcie", "nvbandwidth", "targeted_stress"} <= tests)
        self.assertEqual({parse.dcgm_verdict(r["status"]) for r in rows}, {"pass"})
        bidir = {r["entity"]: parse.dcgm_pcie_bidir(r["info"]) for r in rows if r["test"] == "pcie"}
        self.assertEqual(bidir[0], 76.08)
        self.assertEqual(bidir[1], 76.04)
        self.assertTrue(all(v > 100 for k, v in bidir.items() if k >= 2))

    def test_dcgm_unnamed_test(self):
        doc = {"DCGM Diagnostic": {"test_categories": [{"category": "Hardware", "tests": [
            {"results": [{"entity_group": "GPU", "entity_id": 0, "status": "Pass"}]}]}]}}
        rows, tests, _ = parse.dcgm_diag(json.dumps(doc))
        self.assertEqual(rows[0]["test"], "unnamed-Hardware-0")

    def test_dcgm_verdict(self):
        self.assertEqual(parse.dcgm_verdict("Fail"), "fail")
        self.assertEqual(parse.dcgm_verdict("Skip"), "skip")
        self.assertEqual(parse.dcgm_verdict("weird"), "error")

    def test_nccl_log(self):
        log = parse.nccl_log(read("all_reduce_perf.log"))
        self.assertEqual(log["nccl_version"], "2.28.9")
        self.assertEqual(log["ngpus"], 8)
        self.assertTrue(log["oob_ok"])
        top = max(log["rows"], key=lambda r: r["size"])
        self.assertEqual(top["size"], 8589934592)
        self.assertEqual(top["busbw"], 478.60)
        self.assertEqual({r["wrong"] for r in log["rows"]}, {"0"})

    def test_nccl_version(self):
        self.assertEqual(parse.nccl_version(22809), "2.28.9")
        self.assertEqual(parse.nccl_version(23203), "2.32.3")

    def test_perftest(self):
        text = (" #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]\n"
                " 65536      2094521          0.00               366.05 \t\t     0.698179\n")
        self.assertEqual(parse.perftest_bw(text), 366.05)
        self.assertIsNone(parse.perftest_bw("Couldn't connect"))

    def test_xids(self):
        self.assertEqual(parse.xids(read("dmesg-after.txt")), [])
        line = "[Tue Nov 10 10:00:00 2026] NVRM: Xid (PCI:0000:3b:00): 79, pid=1234, name=python3, GPU has fallen off the bus."
        self.assertEqual(parse.xids(line), [("0000:3b:00.0", 79, line)])

    def test_sxids(self):
        line = "[Tue Nov 10 10:00:00 2026] nvidia-nvswitch2: SXid (PCI:0000:11:00.0): 12028, Non-fatal, Link 32"
        self.assertEqual(parse.sxids(line), [("0000:11:00.0", 12028, line)])
        self.assertEqual(parse.sxids(read("dmesg-after.txt")), [])

    def test_mokutil(self):
        self.assertTrue(parse.mokutil("SecureBoot enabled\n"))
        self.assertFalse(parse.mokutil("SecureBoot disabled\n"))
        self.assertIsNone(parse.mokutil("EFI variables are not supported on this system\n"))

    def test_lspci_acs(self):
        flags, denied = parse.lspci_acs(read("lspci-vvv.txt"))
        self.assertEqual(denied, set())
        self.assertEqual(len(flags), 24)
        self.assertFalse(any(f["ReqRedir"] or f["CmpltRedir"] for f in flags.values()))
        _, denied = parse.lspci_acs("0000:08:00.0 PCI bridge: x\n\tCapabilities: <access denied>\n")
        self.assertEqual(denied, {"0000:08:00.0"})

    def test_fio(self):
        for rw in ("write", "read"):
            gbs, err = parse.fio_result(read(f"fio-{rw}.json"), rw)
            self.assertGreater(gbs, 0)
            self.assertEqual(err, 0)

    def test_ofed(self):
        self.assertEqual(parse.ofed_version("OFED-internal-25.10-2.4.1:\n"), "OFED-internal-25.10-2.4.1")

    def test_fixture_readme_lists_every_file(self):
        names = set(os.listdir(os.path.join(FIXTURES, "h100-8x-ib")))
        readme = read("README.md", FIXTURES)
        missing = sorted(n for n in names if f"`{n}`" not in readme)
        self.assertEqual(missing, [], "document the provenance of every fixture in tests/fixtures/README.md")

    def test_manifest_json_copy(self):
        doc = json.loads(read("manifest-hgx-h100-8x.json", FIXTURES))
        self.assertIn("hgx-h100-8x", doc["class"])


if __name__ == "__main__":
    unittest.main()
