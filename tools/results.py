"""Table of all PASTIS-R runs in the shared runs folder (all users). Smoke runs hidden unless --smoke."""
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path("/lustre/shared/agriscience-benchmark/runs/pastis")
rows = []
for m in ROOT.glob("*/*/metrics.json"):
    run = m.parent
    try:
        r = json.loads(m.read_text())
        r = max(r, key=lambda x: x["macro_iou"]) if isinstance(r, list) else r
        cfg = yaml.safe_load((run / "resolved.yaml").read_text())
    except Exception:
        continue
    if r.get("smoke") and "--smoke" not in sys.argv:
        continue
    tables = run / "feature-tables.json"
    rep = json.loads(tables.read_text())[0].get("representation", "-") if tables.exists() else "-"
    rows.append((r.get("split", "validation"), -r["macro_iou"], run.parent.name, run.name.rsplit("-", 1)[-1],
                 (cfg.get("model") or {}).get("name", "?"), rep, r["accuracy"], r["macro_iou"]))
print(f"{'user':<10} {'job':>9} {'model':<20} {'representation':<46} {'split':<11} {'acc':>7} {'mIoU':>7}")
for split, _, user, job, model, rep, acc, iou in sorted(rows):
    print(f"{user:<10} {job:>9} {model:<20} {rep:<46} {split:<11} {acc:>7.4f} {iou:>7.4f}")
ledger = ROOT / "test-ledger.jsonl"
if ledger.exists():
    print("\nFold-5 test results (ledger):")
    for line in ledger.read_text().splitlines():
        e = json.loads(line)
        print(f"  {e['time']}  {e['user']:<10} {e['encoder']}--{e['input']} / {e['model']}  "
              f"acc {e['accuracy']:.4f}  mIoU {e['macro_iou']:.4f}")
