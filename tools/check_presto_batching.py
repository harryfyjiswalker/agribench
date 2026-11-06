"""Check that batching parcels by observed-month count does not change Presto embeddings."""
import pathlib
import sys
import tempfile
import time

import numpy as np

from tools.adapters.presto import extract

ROOT = pathlib.Path("/lustre/shared/agriscience-benchmark")
src = np.load(ROOT / "inputs/pastis-r/pastis-s2-and-location.npz", allow_pickle=False)
arrays = {k: src[k] for k in src.files}
sel = np.sort(np.random.default_rng(0).choice(len(arrays["ids"]), 64, replace=False))
with tempfile.TemporaryDirectory() as d:
    f = pathlib.Path(d) / "sample.npz"
    np.savez(f, ids=arrays["ids"][sel], s2=arrays["s2"][sel], observed=arrays["observed"][sel],
             coords=arrays["coords"][sel], months=arrays["months"], provenance=arrays["provenance"])
    kw = dict(inputs=f, source_dir=ROOT / "upstream/presto", device="cpu")
    t0 = time.time()
    _, a = extract(batch_size=64, **kw)
    _, b = extract(batch_size=1, **kw)
diff = float(np.abs(a - b).max())
print(f"width {a.shape[1]} | distinct observed-month counts {len(set(arrays['observed'][sel].sum(1)))} "
      f"| batched vs one-by-one max abs diff {diff:.2e} | {time.time() - t0:.0f}s")
sys.exit(0 if diff < 1e-4 else 1)
