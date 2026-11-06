"""Score one predictions CSV with every mIoU definition in use, to see if they agree."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score

UPSTREAM = Path("/lustre/shared/agriscience-benchmark/upstream/ltae")
NUM_CLASSES = 18

df = pd.read_csv(sys.argv[1])
if {"label_index", "prediction_index"} <= set(df.columns):    # dev-track files, 0-based
    truth, pred = df.label_index.to_numpy(), df.prediction_index.to_numpy()
elif {"label", "prediction"} <= set(df.columns):              # test-track files, 1-based
    truth, pred = df.label.to_numpy() - 1, df.prediction.to_numpy() - 1
else:
    raise SystemExit(f"Unrecognised columns: {list(df.columns)}")
classes = np.arange(NUM_CLASSES)

cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
np.add.at(cm, (truth, pred), 1)
tp = np.diag(cm)
union = cm.sum(0) + cm.sum(1) - tp
with np.errstate(invalid="ignore", divide="ignore"):
    iou = tp / union                      # NaN where a class is in neither truth nor prediction
present = cm.sum(1) > 0
print(f"parcels={len(truth)}  classes in truth={present.sum()}/{NUM_CLASSES}  "
      f"classes never predicted={(cm.sum(0) == 0).sum()}")
print(f"accuracy                          {np.mean(truth == pred):.4f}")
print(f"macro F1 (sklearn, all 18)        {f1_score(truth, pred, labels=classes, average='macro', zero_division=0):.4f}")
print(f"mIoU sklearn jaccard, all 18      {jaccard_score(truth, pred, labels=classes, average='macro', zero_division=0):.4f}")
print(f"mIoU own, NaN->0, all 18          {np.nan_to_num(iou).mean():.4f}")
print(f"mIoU own, nanmean (union > 0)     {np.nanmean(iou):.4f}")
print(f"mIoU own, classes in truth only   {np.nanmean(iou[present]):.4f}")
try:
    sys.path.insert(0, str(UPSTREAM))
    from learning.metrics import mIou
    print(f"mIoU upstream mIou                {float(mIou(truth, pred, NUM_CLASSES)):.4f}")
except Exception as error:                # report rather than hide an import problem
    print(f"upstream mIou unavailable: {error!r}")
