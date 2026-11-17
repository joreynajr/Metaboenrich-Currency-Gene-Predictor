import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "benchmarks")]
import null_calibration as N  # noqa: E402


class RegistryUnitsTest(unittest.TestCase):
    def test_unknown_collection_is_not_run(self):
        # A collection the runner has no branch for is reported, not read as Workbench.
        row = {c: "" for c in pd.read_csv(N.REGISTRY, sep="\t", nrows=0).columns}
        row.update(collection="Z_new", study="S1", sub_unit="AN1", status="active")
        with tempfile.TemporaryDirectory() as d:
            reg = Path(d) / "registry.tsv"
            pd.DataFrame([row]).to_csv(reg, sep="\t", index=False)
            units, not_run = N.registry_units(None, None, None, "decreased", registry=reg)
        self.assertEqual(units, [])
        self.assertEqual([r["reason"] for r in not_run], ["unknown collection Z_new"])


if __name__ == "__main__":
    unittest.main()
