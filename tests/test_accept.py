"""End to end: the real CLI against a fake node built from recorded H100 output, plus injected faults."""

import json
import os
import unittest

from helpers import MANIFEST_JSON, FakeNode

REFERENCE_FAILS = [("dcgm.pcie_bidir", "GPU0"), ("dcgm.pcie_bidir", "GPU1")]


def manifest_with(tmp, **perf):
    with open(MANIFEST_JSON) as f:
        doc = json.load(f)
    doc["class"]["hgx-h100-8x"]["perf"].update(perf)
    path = os.path.join(tmp, "manifest.json")
    with open(path, "w") as f:
        json.dump(doc, f)
    return path


class AcceptCase(unittest.TestCase):
    tools = None

    def setUp(self):
        self.node = FakeNode(tools=self.tools)
        self.addCleanup(self.node.cleanup)

    def run_accept(self, *args, **kw):
        self.node.build()
        return self.node.accept("--no-install", *args, **kw)


class TestReference(AcceptCase):
    def test_reference_node_fails_only_on_known_pcie_finding(self):
        r = self.run_accept()
        self.assertEqual(r.rc, 1, r.stderr)
        self.assertEqual(r.failing(), REFERENCE_FAILS, r.stderr)
        warn = {(x["check"], x["target"]) for x in r.results if x["verdict"] == "warn"}
        self.assertIn(("nvlink.dcgm_nvswitch_visible", None), warn)
        self.assertEqual(r.verdicts("nccl.allreduce_busbw"), ["pass"])
        self.assertEqual(r.by("nccl.allreduce_busbw")[0]["value"], 478.6)
        self.assertEqual(len(r.by("rdma.write_bw", "pass")), 8)
        self.assertEqual(r.verdicts("rdma.gdr_write_bw"), ["skip"])
        self.assertEqual(len(r.by("dcgm.pcie_bidir", "pass")), 6)
        for check in ("inventory.gpu_count", "inventory.driver", "inventory.cuda", "inventory.pcie_width",
                      "inventory.pcie_acs", "inventory.secure_boot", "nvlink.links_total", "nvlink.fabric_state",
                      "nvlink.nvswitch_count", "nvlink.fabricmanager_version", "rdma.nic_count", "rdma.port",
                      "rdma.rail_map", "health.xid", "health.xid_new", "health.ib_counter_delta",
                      "health.nvlink_error_delta", "nccl.correctness", "nccl.version"):
            self.assertEqual(r.verdicts(check), ["pass"], check)

    def test_outputs_and_schema(self):
        r = self.run_accept()
        out = self.node.out
        for name in ("results.jsonl", "junit.xml", "SUMMARY.md", "versions.json", "manifest.json",
                     "raw/commands.log", "raw/all_reduce_perf.out", "results.partial.jsonl"):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)
        stdout_lines = [json.loads(line) for line in r.stdout.splitlines()]
        self.assertEqual(len(stdout_lines), len(r.results))
        keys = {"ts", "run_id", "cluster", "node", "class", "rack", "leaf", "pod", "tier", "check", "target",
                "value", "expected", "verdict", "versions", "evidence", "detail"}
        for line in r.results:
            self.assertEqual(set(line), keys)
            self.assertEqual(line["tier"], "L2")
            self.assertEqual(line["node"], "gpu-node-01")
            self.assertIsNone(line["pod"])
        v = r.results[0]["versions"]
        self.assertEqual((v["driver"], v["nccl"], v["dcgm"], v["fabricmanager"]),
                         ("580.178.04", "2.28.9", "4.6.1", "580.178.04"))
        with open(os.path.join(out, "SUMMARY.md")) as f:
            self.assertIn("**Verdict: FAIL**", f.read())
        with open(os.path.join(out, "junit.xml")) as f:
            self.assertIn('failures="2"', f.read())

    def test_clean_pass(self):
        m = manifest_with(self.node.tmp, dcgm_pcie_bidir_gbs=70)
        r = self.run_accept(manifest=m)
        self.assertEqual(r.rc, 0, r.failing())
        self.assertEqual(r.failing(), [])

    def test_floor_omitted_records_info(self):
        with open(MANIFEST_JSON) as f:
            doc = json.load(f)
        del doc["class"]["hgx-h100-8x"]["perf"]["dcgm_pcie_bidir_gbs"]
        m = os.path.join(self.node.tmp, "m.json")
        with open(m, "w") as f:
            json.dump(doc, f)
        r = self.run_accept(manifest=m)
        self.assertEqual(r.rc, 0, r.failing())
        self.assertEqual(r.verdicts("dcgm.pcie_bidir"), ["info"])


class TestFaults(AcceptCase):
    def setUp(self):
        super().setUp()
        self.manifest = manifest_with(self.node.tmp, dcgm_pcie_bidir_gbs=70)

    def go(self, *args):
        return self.run_accept(*args, manifest=self.manifest)

    def test_seven_gpus(self):
        lines = self.node.text("gpus.csv").splitlines(keepends=True)
        self.node.write("gpus.csv", "".join(lines[:7]))
        r = self.go()
        self.assertEqual(r.rc, 1)
        self.assertIn(("inventory.gpu_count", None), r.failing())

    def test_gpu_at_x8(self):
        self.node.edit("gpus.csv", "00000000:44:00.0, 580.178.04, 96.00.99.00.01, 1650000000003, Enabled, Disabled, 16,",
                       "00000000:44:00.0, 580.178.04, 96.00.99.00.01, 1650000000003, Enabled, Disabled, 8,")
        r = self.go()
        self.assertEqual(r.rc, 1)
        self.assertIn(("inventory.pcie_width", "GPU3"), r.failing())

    def test_dcgm_failure(self):
        doc = json.loads(self.node.text("dcgm-diag-r3.json"))
        for cat in doc["DCGM Diagnostic"]["test_categories"]:
            for t in cat["tests"]:
                if t["name"] == "memory":
                    t["results"][5]["status"] = "Fail"
                    t["results"][5]["warnings"] = [{"message": "Error using CUDA API cudaMalloc"}]
        self.node.write("dcgm-diag-r3.json", json.dumps(doc))
        r = self.go()
        self.assertEqual(r.rc, 1)
        self.assertEqual(r.failing(), [("dcgm.diag", "memory/GPU5")])
        self.assertIn("cudaMalloc", r.by("dcgm.diag", "fail")[0]["detail"])

    def test_dcgm_garbage_output(self):
        self.node.write("dcgm-diag-r3.json", "Error: diag failed to start\n")
        r = self.go()
        self.assertEqual(r.rc, 1)
        self.assertIn(("dcgm.diag", None), r.failing())

    def test_nccl_wrong_values(self):
        self.node.edit("all_reduce_perf.log", "478.60       0", "478.60       3")
        r = self.go()
        self.assertEqual(r.rc, 1)
        self.assertIn(("nccl.correctness", None), r.failing())

    def test_nccl_below_floor(self):
        m = manifest_with(self.node.tmp, dcgm_pcie_bidir_gbs=70, nccl_allreduce_busbw_gbs_1node=500)
        r = self.run_accept(manifest=m)
        self.assertEqual(r.failing(), [("nccl.allreduce_busbw", "8589934592 B")])

    def test_ib_port_down(self):
        self.node.sysfs("/sys/class/infiniband/mlx5_ib3/ports/1/state", "1: DOWN")
        self.node.sysfs("/sys/class/infiniband/mlx5_ib3/ports/1/phys_state", "3: Disabled")
        r = self.go()
        self.assertEqual(r.rc, 1)
        self.assertIn(("rdma.port", "mlx5_ib3/1"), r.failing())
        self.assertEqual(len(r.by("rdma.write_bw")), 7)          # the down rail leaves the ring

    def test_rdma_slow_rail(self):
        bw = json.loads(self.node.text("rdma-bw.json"))
        bw["per_client"]["mlx5_ib5"] = 120.0
        self.node.write("rdma-bw.json", json.dumps(bw))
        r = self.go()
        self.assertEqual(r.failing(), [("rdma.write_bw", "mlx5_ib4<-mlx5_ib5")])

    def test_xid_79_after_load(self):
        line = "[Tue Nov 10 10:00:00 2026] NVRM: Xid (PCI:0000:3b:00): 79, pid=4242, GPU has fallen off the bus.\n"
        self.node.write("dmesg-after.txt", self.node.text("dmesg-after.txt") + line)
        r = self.go()
        self.assertEqual(r.rc, 1)
        self.assertEqual(r.failing(), [("health.xid_new", "0000:3b:00.0")])

    def test_application_xid_is_warning(self):
        line = "[Tue Nov 10 10:00:00 2026] NVRM: Xid (PCI:0000:3b:00): 13, pid=4242, Graphics Exception\n"
        self.node.write("dmesg-before.txt", self.node.text("dmesg-before.txt") + line)
        r = self.go()
        self.assertEqual(r.rc, 0, r.failing())
        self.assertEqual(r.verdicts("health.xid"), ["warn"])

    def test_ib_symbol_errors_under_load(self):
        self.node.write("after-load.json", json.dumps(
            {"/sys/class/infiniband/mlx5_ib2/ports/1/counters/symbol_error": "5\n"}))
        r = self.go()
        self.assertEqual(r.failing(), [("health.ib_counter_delta", "mlx5_ib2/1")])

    def test_nvlink_crc_under_load(self):
        self.node.edit("nvlink-e-after.txt", "Link 3: CRC Errors: 0", "Link 3: CRC Errors: 12")
        r = self.go()
        self.assertEqual(r.failing(), [("health.nvlink_error_delta", "GPU0/link3")])

    def test_fabric_not_trained(self):
        self.node.edit("smi-q.txt", "State                                          : Completed",
                       "State                                          : In Progress")
        r = self.go()
        self.assertEqual(r.failing(), [("nvlink.fabric_state", "GPU0")])

    def test_acs_enabled(self):
        self.node.edit("lspci-vvv.txt", "ACSCtl:\tSrcValid- TransBlk- ReqRedir- CmpltRedir-",
                       "ACSCtl:\tSrcValid+ TransBlk- ReqRedir+ CmpltRedir+")
        r = self.go()
        self.assertEqual(r.failing(), [("inventory.pcie_acs", "GPU0 via 0000:08:00.0")])

    def test_secure_boot_off(self):
        self.node.write("mokutil.txt", "SecureBoot disabled\n")
        r = self.go()
        self.assertEqual(r.failing(), [("inventory.secure_boot", None)])

    def test_rail_map_broken(self):
        # GPU5's own NIC moves to another PCIe switch: no PIX/PXB NIC left for it
        t = self.node.text("topo.txt").splitlines(keepends=True)
        t = [line.replace("PXB\tPIX\tNODE\tNODE\tSYS", "SYS\tSYS\tNODE\tNODE\tSYS") if line.startswith("GPU5") else line
             for line in t]
        self.node.write("topo.txt", "".join(t))
        r = self.go()
        self.assertIn(("rdma.rail_map", "GPU5"), r.failing())

    def test_gpus_busy_cannot_run(self):
        self.node.write("compute-apps.csv", "4242, vllm, 00000000:0A:00.0, 70000 MiB\n")
        r = self.go()
        self.assertEqual(r.rc, 2)
        self.assertEqual(r.failing(), [("prereq.gpus_idle", None)])
        self.assertFalse(any(c[0] == "dcgmi" for c in self.node.calls()))

    def test_gpus_busy_force(self):
        self.node.write("compute-apps.csv", "4242, vllm, 00000000:0A:00.0, 70000 MiB\n")
        r = self.go("--force")
        self.assertEqual(r.rc, 0, r.failing())
        self.assertEqual(r.verdicts("prereq.gpus_idle"), ["warn"])

    def test_hostengine_started_and_stopped(self):
        self.node.write("no-hostengine", "")
        r = self.go()
        self.assertEqual(r.rc, 0, r.failing())
        calls = self.node.calls()
        self.assertIn(["nv-hostengine"], calls)
        self.assertIn(["nv-hostengine", "--term"], calls)

    def test_fail_fast_skips_heavy_stages(self):
        self.node.write("mokutil.txt", "SecureBoot disabled\n")
        r = self.go("--fail-fast")
        self.assertEqual(r.rc, 1)
        self.assertEqual({x["check"] for x in r.results if x["verdict"] == "skip" and x["check"].endswith(".load")},
                         {"rdma.load", "dcgm.load", "nccl.load", "fio.load"})
        self.assertFalse(any(c[0] in ("dcgmi", "all_reduce_perf") and c[1:2] != ["--version"]
                             and c[1:2] != ["discovery"] for c in self.node.calls()))

    def test_only_stages(self):
        r = self.go("--only", "inventory,nvlink")
        self.assertEqual(r.rc, 0, r.failing())
        stages = {x["check"].split(".")[0] for x in r.results}
        self.assertEqual(stages, {"prereq", "inventory", "nvlink"})


class TestMissingTools(AcceptCase):
    tools = ["nvidia-smi", "dcgmi", "nv-hostengine", "ib_write_bw", "lspci", "mokutil", "ofed_info", "systemctl",
             "dpkg-query", "dmesg"]           # no all_reduce_perf, no fio

    def test_missing_all_reduce_perf_is_error(self):
        m = manifest_with(self.node.tmp, dcgm_pcie_bidir_gbs=70)
        r = self.run_accept(manifest=m)
        self.assertEqual(r.rc, 1)
        self.assertEqual(r.failing(), [("nccl.all_reduce", None)])


class TestCannotRun(AcceptCase):
    def test_no_driver(self):
        self.node.build()
        env = self.node.env(PATH=os.path.join(self.node.tmp, "empty"))
        r = self.node.accept("--no-install", env=env)
        self.assertEqual(r.rc, 2)
        self.assertEqual(r.failing(), [("prereq.nvidia_smi", None)])

    def test_not_root(self):
        self.node.build()
        env = self.node.env()
        del env["GFV_ALLOW_NONROOT"]
        if os.geteuid() == 0:
            self.skipTest("running as root")
        r = self.node.accept("--no-install", env=env)
        self.assertEqual(r.rc, 2)
        self.assertIn("run as root", r.stderr)

    def test_bad_manifest(self):
        bad = os.path.join(self.node.tmp, "bad.json")
        with open(bad, "w") as f:
            json.dump({"class": {"x": {"gpus": "eight"}}}, f)
        r = self.node.accept(manifest=bad)
        self.assertEqual(r.rc, 2)
        self.assertIn("gpus must be int", r.stderr)

    def test_unknown_stage(self):
        r = self.node.accept("--only", "inventroy")
        self.assertEqual(r.rc, 2)
        self.assertIn("unknown stage", r.stderr)


class TestStorage(AcceptCase):
    def test_fio_with_storage_table(self):
        with open(MANIFEST_JSON) as f:
            doc = json.load(f)
        scratch = os.path.join(self.node.tmp, "scratch")
        os.makedirs(scratch)
        spec = doc["class"]["hgx-h100-8x"]
        spec["perf"]["dcgm_pcie_bidir_gbs"] = 70
        spec["storage"] = {"path": scratch, "size_gib": 1, "seq_write_gbs": 1.0, "seq_read_gbs": 9.0}
        m = os.path.join(self.node.tmp, "m.json")
        with open(m, "w") as f:
            json.dump(doc, f)
        r = self.run_accept(manifest=m)
        self.assertEqual(r.verdicts("fio.seq_write"), ["pass"])
        self.assertEqual(r.failing(), [("fio.seq_read", scratch)])      # 6.39 GB/s < 9.0


if __name__ == "__main__":
    unittest.main()
