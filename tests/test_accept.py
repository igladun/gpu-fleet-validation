"""End to end: the real CLI against a fake node built from recorded H100 output, plus injected faults."""

import json
import os
import signal
import subprocess
import sys
import time
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
        self.assertEqual(r.verdicts("burnin"), ["skip"])              # the reference manifest has no [burnin]
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
        self.assertEqual(r.failing(), [("inventory.pcie_acs", "GPU0 via 0000:08:00.0"),
                                       ("inventory.pcie_acs", "mlx5_ib0 via 0000:08:00.0")])

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
                         {"rdma.load", "dcgm.load", "burnin.load", "nccl.load", "fio.load"})
        self.assertFalse(any(c[0] in ("dcgmi", "all_reduce_perf") and c[1:2] != ["--version"]
                             and c[1:2] != ["discovery"] for c in self.node.calls()))

    def test_only_stages_with_allow_partial(self):
        r = self.go("--only", "inventory,nvlink", "--allow-partial")
        self.assertEqual(r.rc, 0, r.failing())
        ran = {x["check"].split(".")[0] for x in r.results if not x["check"].endswith(".not_selected")}
        self.assertEqual(ran, {"run", "prereq", "inventory", "nvlink"})
        self.assertEqual(r.verdicts("run.partial"), ["warn"])
        self.assertEqual({x["check"] for x in r.by("health.not_selected")}, {"health.not_selected"})

    def test_partial_run_fails_by_default(self):
        r = self.go("--only", "fio")
        self.assertEqual(r.rc, 1)
        self.assertEqual(r.failing(), [("run.partial", None)])
        with open(os.path.join(self.node.out, "SUMMARY.md")) as f:
            self.assertIn("**Verdict: FAIL**", f.read())

    def test_no_stages_left(self):
        r = self.go("--only", "install")          # --no-install removes it: nothing would run
        self.assertEqual(r.rc, 2)

    def test_evidence_dir_must_be_empty(self):
        os.makedirs(self.node.out)
        open(os.path.join(self.node.out, "old.txt"), "w").close()
        r = self.go()
        self.assertEqual(r.rc, 2)
        self.assertIn("not empty", r.stderr)

    def test_test_overrides_are_recorded(self):
        r = self.go()
        self.assertEqual(r.verdicts("run.test_overrides"), ["warn"])

    # --- review regressions: missing or odd data must never pass -------------------------
    def dcgm_doc(self):
        return json.loads(self.node.text("dcgm-diag-r3.json"))

    def dcgm_tests(self, doc):
        return {t["name"]: t for c in doc["DCGM Diagnostic"]["test_categories"] for t in c["tests"]}

    def test_dcgm_nonzero_exit_with_all_pass_json(self):
        self.node.write("dcgm-rc-fail", "")
        r = self.go()
        self.assertEqual(r.failing(), [("dcgm.diag", None)])

    def test_dcgm_required_test_skipped(self):
        doc = self.dcgm_doc()
        for res in self.dcgm_tests(doc)["memory"]["results"]:
            res["status"] = "Skip"
        self.node.write("dcgm-diag-r3.json", json.dumps(doc))
        r = self.go()
        self.assertEqual(r.rc, 1)
        self.assertEqual({t for c, t in r.failing()}, {f"memory/GPU{i}" for i in range(8)})

    def test_dcgm_gpu_missing_from_results(self):
        doc = self.dcgm_doc()
        for t in self.dcgm_tests(doc).values():
            t["results"] = [x for x in t["results"] if x.get("entity_id") != 7]
        self.node.write("dcgm-diag-r3.json", json.dumps(doc))
        r = self.go()
        self.assertEqual(r.rc, 1)
        self.assertIn(("dcgm.diag", "memory"), r.failing())

    def test_dcgm_test_summary_fail_without_results(self):
        doc = self.dcgm_doc()
        t = self.dcgm_tests(doc)["diagnostic"]
        t["results"] = []
        t["test_summary"] = {"status": "Fail", "errors": [{"message": "plugin crashed"}]}
        self.node.write("dcgm-diag-r3.json", json.dumps(doc))
        r = self.go()
        self.assertEqual(r.failing(), [("dcgm.diag", "diagnostic")])
        self.assertIn("plugin crashed", r.by("dcgm.diag", "fail")[0]["detail"])

    def test_dcgm_required_test_missing(self):
        doc = self.dcgm_doc()
        for c in doc["DCGM Diagnostic"]["test_categories"]:
            c["tests"] = [t for t in c["tests"] if t["name"] != "targeted_power"]
        self.node.write("dcgm-diag-r3.json", json.dumps(doc))
        r = self.go()
        self.assertEqual(r.failing(), [("dcgm.diag", "targeted_power")])

    def test_acs_unresolvable_gpu_path_is_error(self):
        del self.node.spec["symlinks"]["/sys/bus/pci/devices/0000:0a:00.0"]
        self.node.edit("lspci-vvv.txt", "0000:09:00.0 PCI bridge", "0000:09:00.0 PCI bridge")
        r = self.go()
        self.assertIn(("inventory.pcie_acs", None), r.failing())

    def test_saturated_soft_counter_is_warning(self):
        self.node.sysfs("/sys/class/infiniband/mlx5_ib2/ports/1/counters/port_xmit_discards", "65535")
        r = self.go()
        self.assertEqual(r.rc, 0, r.failing())
        self.assertIn("warn", r.verdicts("health.ib_counter"))

    def test_compute_apps_query_failure_cannot_run(self):
        os.remove(self.node.path("compute-apps.csv"))
        r = self.go()
        self.assertEqual(r.rc, 2)
        self.assertEqual(r.failing(), [("prereq.gpus_idle", None)])

    def test_ecc_disabled(self):
        lines = [line.replace(", 0, 0, 9.0, Enabled", ", [N/A], [N/A], 9.0, Disabled")
                 for line in self.node.text("gpus.csv").splitlines(keepends=True)]
        self.node.write("gpus.csv", "".join(lines))
        r = self.go()
        self.assertEqual(r.rc, 1)
        failing = r.failing()
        self.assertIn(("inventory.ecc_mode", "GPU0"), failing)
        self.assertIn(("health.ecc_uncorrected", "GPU0"), failing)
        self.assertIn(("health.ecc_delta", "GPU0"), failing)

    def test_gpu_query_broken_after_load(self):
        self.node.write("gpus-after.csv", "Unable to determine the device handle for GPU0000:3B:00.0: Unknown Error\n")
        r = self.go()
        self.assertEqual(r.rc, 1)
        self.assertIn(("health.ecc_delta", None), r.failing())

    def test_acs_nothing_checkable(self):
        for link in [k for k in self.node.spec["symlinks"] if "/bus/pci/devices/" in k or k.endswith("/device")]:
            del self.node.spec["symlinks"][link]
        r = self.go()
        self.assertIn(("inventory.pcie_acs", None), r.failing())
        self.assertEqual(r.verdicts("inventory.pcie_acs"), ["error"])

    def test_nvlink_counters_unreadable(self):
        self.node.write("nvlink-e.txt", "Failed to get NVLink error counters: Unknown Error\n")
        self.node.write("nvlink-e-after.txt", "Failed to get NVLink error counters: Unknown Error\n")
        r = self.go()
        self.assertIn(("health.nvlink_counters_readable", None), r.failing())
        self.assertIn(("health.nvlink_error_delta", None), r.failing())

    def test_ib_counters_missing(self):
        for k in [k for k in self.node.spec["files"] if k.startswith("/sys/class/infiniband/mlx5_ib4/ports/1/counters/")]:
            del self.node.spec["files"][k]
        r = self.go()
        self.assertIn(("health.ib_counters_readable", "mlx5_ib4"), r.failing())

    def test_ib_counter_saturated(self):
        self.node.sysfs("/sys/class/infiniband/mlx5_ib2/ports/1/counters/symbol_error", "65535")
        r = self.go()
        self.assertEqual(r.failing(), [("health.ib_counter", "mlx5_ib2/1")])

    def test_topology_unparseable(self):
        self.node.write("topo.txt", self.node.text("topo.txt").replace("\t", "  "))
        r = self.go()
        self.assertIn(("rdma.rail_map", None), r.failing())

    def test_unclassified_xid_fails(self):
        line = "[Tue Nov 10 10:00:00 2026] NVRM: Xid (PCI:0000:3b:00): 154, GPU recovery action changed (Node Reboot Required)\n"
        self.node.write("dmesg-before.txt", self.node.text("dmesg-before.txt") + line)
        r = self.go()
        self.assertEqual(r.failing(), [("health.xid", "0000:3b:00.0")])

    def test_sxid_during_load_fails(self):
        line = "[Tue Nov 10 10:00:00 2026] nvidia-nvswitch1: SXid (PCI:0000:10:00.0): 20034, Fatal, Link 12 LTSSM Fault Up\n"
        self.node.write("dmesg-after.txt", self.node.text("dmesg-after.txt") + line)
        r = self.go()
        self.assertEqual(r.failing(), [("health.sxid_new", "0000:10:00.0")])

    def test_nccl_truncated_log(self):
        lines = [x for x in self.node.text("all_reduce_perf.log").splitlines(keepends=True)
                 if not x.lstrip().startswith(("4294967296", "8589934592"))]
        self.node.write("all_reduce_perf.log", "".join(lines))
        r = self.go()
        self.assertEqual(r.failing(), [("nccl.allreduce_busbw", "2147483648 B")])


class TestSignals(AcceptCase):
    """SIGTERM unwinds and records an interrupted run (exit 2); an inherited SIGHUP ignore (nohup) is kept."""

    def start(self, ignore_hup):
        self.node.build()
        m = os.path.join(self.node.tmp, "m.json")
        with open(MANIFEST_JSON) as f:
            doc = json.load(f)
        doc["class"]["hgx-h100-8x"]["perf"]["dcgm_pcie_bidir_gbs"] = 70
        with open(m, "w") as f:
            json.dump(doc, f)
        cmd = [sys.executable, "-m", "gfv", "accept", "--manifest", m, "--node", "gpu-node-01", "--out", self.node.out,
               "--tools-dir", self.node.tools_dir, "--no-install"]
        pre = (lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN)) if ignore_hup else None
        p = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             env=self.node.env(GFV_SETTLE_SECONDS="0.5"), stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, text=True, preexec_fn=pre)
        deadline = time.time() + 30
        while time.time() < deadline and not os.path.exists(os.path.join(self.node.out, "raw", "commands.log")):
            time.sleep(0.05)
        time.sleep(1.0)                       # inside the RDMA ring (8 pairs x 0.5 s settle)
        return p

    def results(self):
        with open(os.path.join(self.node.out, "results.jsonl")) as f:
            return [json.loads(line) for line in f]

    def test_sigterm_records_interrupted_run(self):
        p = self.start(ignore_hup=False)
        p.send_signal(signal.SIGTERM)
        p.communicate(timeout=60)
        self.assertEqual(p.returncode, 2)
        self.assertIn(("run", "error"), {(r["check"], r["verdict"]) for r in self.results()})

    def test_nohup_keeps_running(self):
        p = self.start(ignore_hup=True)
        p.send_signal(signal.SIGHUP)
        p.communicate(timeout=60)
        self.assertEqual(p.returncode, 0)


class TestMissingTools(AcceptCase):
    tools = ["nvidia-smi", "dcgmi", "nv-hostengine", "ib_write_bw", "lspci", "mokutil", "ofed_info", "systemctl",
             "dpkg-query", "dmesg"]           # no all_reduce_perf, no fio

    def test_missing_all_reduce_perf_is_error(self):
        m = manifest_with(self.node.tmp, dcgm_pcie_bidir_gbs=70)
        r = self.run_accept(manifest=m)
        self.assertEqual(r.rc, 1)
        self.assertEqual(r.failing(), [("nccl.all_reduce", None)])


class TestCannotRun(AcceptCase):
    def test_nvidia_smi_not_executable(self):
        self.node.build()
        b = os.path.join(self.node.tmp, "noexec")
        os.makedirs(b)
        with open(os.path.join(b, "nvidia-smi"), "w") as f:
            f.write("#!/bin/sh\n")
        env = self.node.env(PATH=b)
        r = self.node.accept("--no-install", env=env)
        self.assertEqual(r.rc, 2)
        self.assertEqual(r.failing(), [("prereq.nvidia_smi", None)])
        with open(os.path.join(self.node.out, "SUMMARY.md")) as f:
            self.assertIn("**Verdict: FAIL**", f.read())

    def test_malformed_manifest_shape(self):
        bad = os.path.join(self.node.tmp, "shape.json")
        with open(bad, "w") as f:
            json.dump({"class": {"x": []}}, f)
        r = self.node.accept(manifest=bad)
        self.assertEqual(r.rc, 2)

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
    def test_fio_job_error(self):
        doc = json.loads(self.node.text("fio-write.json"))
        doc["jobs"][0]["error"] = 5
        doc["jobs"][0]["write"]["bw_bytes"] = 0
        self.node.write("fio-write.json", json.dumps(doc))
        with open(MANIFEST_JSON) as f:
            m = json.load(f)
        scratch = os.path.join(self.node.tmp, "scratch")
        os.makedirs(scratch)
        spec = m["class"]["hgx-h100-8x"]
        spec["perf"]["dcgm_pcie_bidir_gbs"] = 70
        spec["storage"] = {"path": scratch, "size_gib": 1}
        path = os.path.join(self.node.tmp, "m.json")
        with open(path, "w") as f:
            json.dump(m, f)
        r = self.run_accept(manifest=path)
        self.assertEqual(r.failing(), [("fio.seq_write", scratch)])

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


def manifest_doc(tmp, perf=None, **spec):
    """The reference manifest with extra top-level keys / tables (e.g. thermal=..., burnin=...)."""
    with open(MANIFEST_JSON) as f:
        doc = json.load(f)
    cls = doc["class"]["hgx-h100-8x"]
    cls["perf"].update({"dcgm_pcie_bidir_gbs": 70, **(perf or {})})
    cls.update(spec)
    path = os.path.join(tmp, "manifest-extra.json")
    with open(path, "w") as f:
        json.dump(doc, f)
    return path


GDR_HELP = ("      --use_cuda=<cuda device id> Use CUDA specific device for GPUDirect RDMA testing\n"
            "      --use_cuda_dmabuf Use CUDA DMA-BUF for GPUDirect RDMA testing\n")


class TestNewNode(AcceptCase):
    """Behaviour found on the first hardware run: driver 615 banner, held packages, power cap, PCIe split,
    GPUDirect over DMA-BUF, NUMA-pinned RDMA."""

    def test_driver_and_cuda_from_r615_banner(self):
        self.node.write("nvidia-smi.txt", self.node.text("nvidia-smi-r615.txt"))
        r = self.run_accept(manifest=manifest_doc(self.node.tmp, driver="615.71.10", cuda="13.4"))
        self.assertEqual(r.verdicts("inventory.driver"), ["pass"])
        self.assertEqual(r.by("inventory.cuda")[0]["value"], "13.4")

    def test_unreadable_banner_falls_back_to_csv_driver_and_errors_on_cuda(self):
        self.node.write("nvidia-smi.txt", "NVIDIA-SMI something new\n")
        r = self.run_accept(manifest=manifest_doc(self.node.tmp))
        self.assertEqual(r.verdicts("inventory.driver"), ["pass"])        # 580.178.04 from --query-gpu
        self.assertEqual(r.verdicts("inventory.cuda"), ["error"])
        self.assertIn("could not read", r.by("inventory.cuda")[0]["detail"])

    def test_held_fabric_manager_counts_as_installed(self):
        self.node.write("fabricmanager-dpkg.txt", "hi  nvidia-fabricmanager 580.178.04-1ubuntu1\n"
                                                  "un  nvidia-fabricmanager-580 \n")
        r = self.run_accept(manifest=manifest_doc(self.node.tmp))
        self.assertEqual(r.verdicts("nvlink.fabricmanager_version"), ["pass"])

    def test_power_limit(self):
        r = self.run_accept(manifest=manifest_doc(self.node.tmp, gpu_power_limit_w=700))
        self.assertEqual(r.rc, 0, r.failing())
        self.assertEqual(r.verdicts("inventory.power_limit"), ["pass"])

    def test_capped_gpu_fails_power_limit(self):
        self.node.write("power-limit.csv", self.node.text("power-limit.csv").replace("2, 700.00", "2, 600.00"))
        r = self.run_accept(manifest=manifest_doc(self.node.tmp, gpu_power_limit_w=700))
        self.assertEqual(r.failing(), [("inventory.power_limit", "GPU2")])
        self.assertIn("capped", r.by("inventory.power_limit", "fail")[0]["detail"])

    def test_pcie_oneway_gate_turns_low_bidir_into_warning(self):
        r = self.run_accept(manifest=manifest_doc(self.node.tmp, perf={"dcgm_pcie_bidir_gbs": 95,
                                                                       "dcgm_pcie_oneway_gbs": 50}))
        self.assertEqual(r.rc, 0, r.failing())
        self.assertEqual(len(r.by("dcgm.pcie_oneway", "pass")), 8)
        self.assertEqual(sorted(x["target"] for x in r.by("dcgm.pcie_bidir", "warn")), ["GPU0", "GPU1"])

    def test_pcie_oneway_below_floor_keeps_bidir_failing(self):
        r = self.run_accept(manifest=manifest_doc(self.node.tmp, perf={"dcgm_pcie_bidir_gbs": 95,
                                                                       "dcgm_pcie_oneway_gbs": 56}))
        self.assertEqual(len(r.by("dcgm.pcie_oneway", "fail")), 8)
        self.assertEqual(sorted(x["target"] for x in r.by("dcgm.pcie_bidir", "fail")), ["GPU0", "GPU1"])

    def test_gpudirect_over_dmabuf(self):
        self.node.write("ib-write-bw-help.txt", GDR_HELP)
        r = self.run_accept(manifest=manifest_doc(self.node.tmp))
        self.assertEqual(r.rc, 0, r.failing())
        gdr = r.by("rdma.gdr_write_bw", "pass")
        self.assertEqual(len(gdr), 8)
        self.assertIn("via dmabuf", gdr[0]["detail"])
        self.assertTrue(any("--use_cuda_dmabuf" in c for c in self.node.calls() if c[0] == "ib_write_bw"))

    def test_gpudirect_falls_back_to_peermem(self):
        self.node.write("ib-write-bw-help.txt", GDR_HELP)
        self.node.write("gdr-dmabuf-fail", "")
        r = self.run_accept(manifest=manifest_doc(self.node.tmp))
        self.assertEqual(r.rc, 0, r.failing())
        gdr = r.by("rdma.gdr_write_bw", "pass")
        self.assertEqual(len(gdr), 8)
        self.assertIn("via peermem (dmabuf:", gdr[0]["detail"])
        self.assertTrue(all("via peermem" in x["detail"] for x in gdr))
        dmabuf_tries = [c for c in self.node.calls() if c[0] == "ib_write_bw" and "--use_cuda_dmabuf" in c
                        and c[-1] == "127.0.0.1"]
        self.assertEqual(len(dmabuf_tries), 1, "the working path is reused after the first pair")

    def test_gpudirect_broken_is_error(self):
        self.node.write("ib-write-bw-help.txt", GDR_HELP)
        self.node.write("gdr-fail", "")
        r = self.run_accept(manifest=manifest_doc(self.node.tmp))
        self.assertEqual(r.verdicts("rdma.gdr_write_bw"), ["error"])
        self.assertIn("Couldn't create IB resources", r.by("rdma.gdr_write_bw")[0]["detail"])
        self.assertEqual(r.verdicts("inventory.nvidia_peermem"), ["pass"])     # still required by this manifest

    def test_gpudirect_required_but_untestable_is_error(self):
        with open(MANIFEST_JSON) as f:
            host = json.load(f)["class"]["hgx-h100-8x"]["host"]
        host.pop("nvidia_peermem")
        host["gpudirect"] = True
        r = self.run_accept(manifest=manifest_doc(self.node.tmp, host=host))     # fixture perftest has no CUDA
        self.assertEqual(r.failing(), [("rdma.gdr_write_bw", None)])
        self.assertIn("without CUDA support", r.by("rdma.gdr_write_bw")[0]["detail"])

    def test_reinstall_required_fabric_manager_is_not_installed(self):
        self.node.write("fabricmanager-dpkg.txt", "iiR nvidia-fabricmanager 580.178.04-1ubuntu1\n")
        r = self.run_accept(manifest=manifest_doc(self.node.tmp))
        self.assertEqual(r.failing(), [("nvlink.fabricmanager_version", None)])

    def test_peermem_is_info_when_not_required(self):
        with open(MANIFEST_JSON) as f:
            host = json.load(f)["class"]["hgx-h100-8x"]["host"]
        host.pop("nvidia_peermem")
        r = self.run_accept(manifest=manifest_doc(self.node.tmp, host=host))
        self.assertEqual(r.verdicts("inventory.nvidia_peermem"), ["info"])

    def test_rdma_is_numa_pinned(self):
        for dev, target in self.node.spec["symlinks"].items():
            if dev.startswith("/sys/class/infiniband/mlx5_ib") and dev.endswith("/device"):
                self.node.sysfs(target + "/local_cpulist", "0-27,112-139")
        r = self.run_accept(manifest=manifest_doc(self.node.tmp))
        self.assertEqual(r.rc, 0, r.failing())
        bw = [c for c in self.node.calls() if "ib_write_bw" in c and "--help" not in c]
        self.assertTrue(bw and all(c[:3] == ["taskset", "-c", "0-27,112-139"] for c in bw), bw[:2])
        self.assertIn("NUMA-pinned", r.by("rdma.write_bw")[0]["detail"])


def dmon_text(peak, n=240, base=35.0, tau_min=3.0, power=690.0, ramp=6, rate=None, mem_offset=8.0, mem_na=False,
              cooldown=0):
    """dcgmi dmon -e 150,140,155,100 output, one sample per 5 s. peak: {gpu: steady temperature}.
    rate: degrees C per minute for a linear rise instead of the exponential approach."""
    import math
    lines = ["#Entity  TMPTR  MMTMP  POWER  SMCLK", "ID       C      C      W      MHZ"]
    for i in range(n + cooldown):
        minutes = min(i, n - 1) * 5 / 60
        if i == n // 2:
            lines += ["#Entity  TMPTR  MMTMP  POWER  SMCLK", "ID       C      C      W      MHZ"]
        for g in range(8):
            if rate is not None:
                t = base + rate * minutes
            else:
                t = base + (peak[g] - base) * (1 - math.exp(-minutes / tau_min))
            w = 75.0 if i < ramp or i >= n else power
            if i >= n:                        # after the load: cooling down
                t -= 8 * (i - n + 1)
            mem = "N/A" if mem_na else f"{t + mem_offset:.0f}"
            lines.append(f"GPU {g}    {t:.0f}     {mem}     {w:.1f}  {1980 if i >= ramp else 345}")
    return "\n".join(lines) + "\n"


class TestBurnin(AcceptCase):
    THERMAL = {"gpu_max_operating_c": 87, "gpu_slowdown_c": 89, "gpu_shutdown_c": 95, "mem_max_operating_c": 95}
    BURNIN = {"minutes": 20, "power_min_w": 665, "gpu_temp_warn_c": 77, "gpu_temp_max_c": 82, "mem_temp_max_c": 85,
              "gpu_temp_spread_c": 15}

    def go(self, *args, thermal=None, burnin=None, peak=70, **dmon):
        peaks = peak if isinstance(peak, dict) else {g: peak for g in range(8)}
        self.node.write("dcgmi-dmon-burnin.txt", dmon_text(peaks, **dmon))
        m = manifest_doc(self.node.tmp, thermal=thermal or self.THERMAL, burnin={**self.BURNIN, **(burnin or {})})
        return self.run_accept(*args, manifest=m)

    def test_cool_node_passes_every_burnin_check(self):
        r = self.go()
        self.assertEqual(r.rc, 0, r.failing())
        self.assertEqual(len(r.by("burnin.limits", "pass")), 4)
        for check in ("burnin.dcgm", "burnin.steady_state", "burnin.temp_spread", "burnin.throttle"):
            self.assertEqual(r.verdicts(check), ["pass"], check)
        for check in ("burnin.gpu_temp", "burnin.mem_temp", "burnin.power"):
            self.assertEqual(len(r.by(check, "pass")), 8, check)
        gpu0 = r.by("burnin.gpu_temp")[0]
        self.assertEqual(gpu0["value"], 70)
        self.assertIn("17 °C under the 87 °C max operating temperature", gpu0["detail"])
        self.assertEqual(r.by("burnin.power")[0]["value"], 690.0)
        diag = [c for c in self.node.calls() if c[:3] == ["dcgmi", "diag", "-r"] and c[3] == "targeted_power"]
        self.assertEqual(diag[0][4:7], ["-p", "targeted_power.test_duration=1200", "-j"])
        self.assertTrue(os.path.exists(os.path.join(self.node.out, "burnin-temperatures.csv")))

    def test_minutes_override(self):
        self.go("--burnin-minutes", "2")
        diag = [c for c in self.node.calls() if c[:4] == ["dcgmi", "diag", "-r", "targeted_power"]]
        self.assertEqual(diag[0][5], "targeted_power.test_duration=120")

    def test_hot_gpu_fails_and_warm_gpu_warns(self):
        peak = {g: 70 for g in range(8)}
        peak[3], peak[5] = 84, 79
        r = self.go(peak=peak, mem_offset=0.0)
        self.assertEqual(r.rc, 1)
        self.assertEqual(r.failing(), [("burnin.gpu_temp", "GPU3")])
        self.assertEqual([x["target"] for x in r.by("burnin.gpu_temp", "warn")], ["GPU5"])

    def test_hot_hbm_fails(self):
        r = self.go(mem_offset=20.0)
        self.assertEqual({t for c, t in r.failing()}, {f"GPU{g}" for g in range(8)})
        self.assertEqual({c for c, t in r.failing()}, {"burnin.mem_temp"})

    def test_one_gpu_much_hotter_than_the_rest_warns(self):
        peak = {g: 60 for g in range(8)}
        peak[7] = 76
        r = self.go(peak=peak)
        self.assertEqual(r.rc, 0, r.failing())
        self.assertEqual(r.verdicts("burnin.temp_spread"), ["warn"])
        self.assertIn("hottest GPU7", r.by("burnin.temp_spread")[0]["detail"])

    def test_still_rising_temperature_warns(self):
        r = self.go("--burnin-minutes", "15", rate=1.0, base=45.0, n=180)     # +1 C/min for 15 min
        self.assertEqual(r.rc, 0, r.failing())
        self.assertEqual(r.verdicts("burnin.steady_state"), ["warn"])

    def test_short_run_does_not_judge_steady_state(self):
        r = self.go("--burnin-minutes", "5", n=66)
        self.assertEqual(r.rc, 0, r.failing())
        self.assertEqual(r.verdicts("burnin.steady_state"), ["info"])

    def test_cooldown_tail_does_not_hide_a_rising_temperature(self):
        r = self.go("--burnin-minutes", "15", rate=0.9, base=45.0, n=180, cooldown=3)
        self.assertEqual(r.verdicts("burnin.steady_state"), ["warn"])

    def test_telemetry_shorter_than_the_load_is_error(self):
        r = self.go(n=24)                                       # 2 min of samples for a 20 min burn-in
        self.assertEqual(r.rc, 1)
        self.assertEqual({c for c, t in r.failing()}, {"burnin.telemetry"})
        self.assertEqual(len(r.by("burnin.telemetry", "error")), 8)
        self.assertEqual(r.verdicts("burnin.steady_state"), ["skip"])

    def test_missing_hbm_temperature_with_threshold_is_error(self):
        r = self.go(mem_na=True)
        self.assertEqual({c for c, t in r.failing()}, {"burnin.mem_temp"})
        self.assertEqual(r.verdicts("burnin.mem_temp"), ["error"])

    def test_missing_hbm_temperature_without_threshold_is_skip(self):
        burnin = {k: v for k, v in self.BURNIN.items() if k != "mem_temp_max_c"}
        self.node.write("dcgmi-dmon-burnin.txt", dmon_text({g: 70 for g in range(8)}, mem_na=True))
        r = self.run_accept(manifest=manifest_doc(self.node.tmp, thermal=self.THERMAL, burnin=burnin))
        self.assertEqual(r.rc, 0, r.failing())
        self.assertEqual(r.verdicts("burnin.mem_temp"), ["skip"])

    def test_throttle_counters_not_reported_is_error(self):
        self.node.write("throttle.csv", "".join(f"{g}, [N/A], [N/A], [N/A]\n" for g in range(8)))
        r = self.go()
        self.assertEqual({c for c, t in r.failing()}, {"burnin.throttle"})

    def test_throttle_counters_unparseable_is_error(self):
        self.node.write("throttle.csv", "Field is not supported\n")
        r = self.go()
        self.assertEqual(r.failing(), [("burnin.throttle", None)])

    def test_thermal_limit_not_reported_is_error(self):
        self.node.write("dcgmi-dmon-limits.txt", self.node.text("dcgmi-dmon-limits.txt").replace(
            "GPU 2           87", "GPU 2           N/A"))
        r = self.go()
        self.assertEqual(r.failing(), [("burnin.limits", "gpu_max_operating_c")])
        self.assertIn("GPUs [2]", r.by("burnin.limits", "error")[0]["detail"])

    def test_minutes_without_burnin_table_is_error(self):
        r = self.run_accept("--burnin-minutes", "240", manifest=manifest_doc(self.node.tmp))
        self.assertEqual(r.failing(), [("burnin", None)])

    def test_low_power_fails(self):
        r = self.go(power=600.0)
        self.assertEqual({c for c, t in r.failing()}, {"burnin.power"})

    def test_thermal_throttle_during_burnin_fails(self):
        self.node.write("throttle-after.csv", self.node.text("throttle.csv").replace("5, 0, 0,", "5, 0, 2500000,"))
        r = self.go()
        self.assertEqual(r.failing(), [("burnin.throttle", "GPU5")])
        self.assertEqual(r.by("burnin.throttle", "fail")[0]["value"], {"HW thermal slowdown": 2.5})

    def test_thermal_limits_differ_from_manifest(self):
        r = self.go(thermal={**self.THERMAL, "gpu_max_operating_c": 90})
        self.assertEqual({(c, t) for c, t in r.failing()}, {("burnin.limits", f"gpu_max_operating_c GPU{g}")
                                                          for g in range(8)})

    def test_no_telemetry_is_error(self):
        m = manifest_doc(self.node.tmp, thermal=self.THERMAL, burnin=self.BURNIN)
        r = self.run_accept(manifest=m)
        self.assertEqual(r.failing(), [("burnin.telemetry", None)])

    def test_targeted_power_failure(self):
        doc = json.loads(self.node.text("dcgm-diag-r3.json"))
        for cat in doc["DCGM Diagnostic"]["test_categories"]:
            for t in cat["tests"]:
                if t["name"] == "targeted_power":
                    t["results"][4]["status"] = "Fail"
                    t["results"][4]["warnings"] = [{"message": "Max power of 520 W did not reach 700 W"}]
        self.node.write("dcgm-targeted-power.json", json.dumps(doc))
        r = self.go()
        self.assertEqual(r.failing(), [("burnin.dcgm", "GPU4")])

    def test_skip_burnin_with_allow_partial(self):
        r = self.go("--skip", "burnin", "--allow-partial")
        self.assertEqual(r.rc, 0, r.failing())
        self.assertFalse(any(c[:4] == ["dcgmi", "diag", "-r", "targeted_power"] for c in self.node.calls()))


if __name__ == "__main__":
    unittest.main()
