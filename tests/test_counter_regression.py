"""Счётчик на сохранённых записях должен давать то же число повторений, что в tests/baseline_v1.json.
Пропускается, если нет data/keypoints или data/penn."""
import glob, json, os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import baseline


class Regression(unittest.TestCase):
    def test_counts_same_as_baseline(self):
        base = json.load(open(os.path.join(ROOT, "tests", "baseline_v1.json"), encoding="utf-8"))
        files = glob.glob(os.path.join(ROOT, "data", "keypoints", "*.npz")) + glob.glob(os.path.join(ROOT, "data", "penn", "*.npz"))
        if not files:
            self.skipTest("нет данных")
        for p in files:
            name = os.path.basename(p)[:-4]
            n, evs = baseline.run(p)
            self.assertEqual(n, base[name]["count"], name)
            self.assertEqual([e["valid"] for e in evs], [e["valid"] for e in base[name]["events"]], name)


if __name__ == "__main__":
    unittest.main()
