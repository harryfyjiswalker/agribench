"""Train and validate the adapted PASTIS-R parcel baseline."""
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch import nn


def batches(dataset, sensor, size, rng=None):
    """Bucket by real acquisitions. BatchNorm requires two training parcels."""
    buckets = defaultdict(list)
    lengths = {}
    for index, parcel in enumerate(dataset.id_parcels):
        patch = dataset.id_patches[parcel]
        if patch not in lengths:
            lengths[patch] = len(dataset.get_dates(patch, sensor))
        buckets[lengths[patch]].append(index)
    result = []
    for indices in buckets.values():
        if rng is not None:
            rng.shuffle(indices)
        chunks = [indices[i:i + size] for i in range(0, len(indices), size)]
        if rng is not None and len(chunks[-1]) == 1:
            if len(chunks) == 1:
                continue
            chunks[-2].extend(chunks.pop())
        result.extend(chunks)
    if rng is not None:
        rng.shuffle(result)
    return result


def read_batch(dataset, indices, sensor, device):
    samples = [dataset[i] for i in indices]
    pixels = torch.stack([s[sensor][0] for s in samples]).float()
    masks = torch.stack([s[sensor][1] for s in samples])
    masks = masks[:, None, :].expand(-1, pixels.shape[1], -1).contiguous()
    dates = torch.stack([s[f"dates-{sensor}"] for s in samples]).long()
    labels = torch.stack([s["label"] for s in samples]).long()
    if pixels.shape[:2] != dates.shape:
        raise ValueError("Acquisition arrays and dates disagree")
    return tuple(t.to(device) for t in (pixels, masks, dates, labels))


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    upstream = Path(cfg.upstream_root).expanduser().resolve()
    repos = {name: upstream / name for name in ("pastis-benchmark", "ltae", "utae-paps")}
    out = Path(HydraConfig.get().runtime.output_dir)
    for name, repo in repos.items():
        (out / f"{name}.patch").write_text(git(repo, "diff", "HEAD", "--binary"))
    commits = {name: git(repo, "rev-parse", "HEAD").strip() for name, repo in repos.items()}
    (out / "upstream-commits.json").write_text(json.dumps(commits, indent=2))
    (out / "requirements-snapshot.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    (out / "resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    sys.path[:0] = [str(repos["pastis-benchmark"] / "code"), str(repos["ltae"]),
                    str(repos["utae-paps"])]

    from dataloader_pixelset import PASTIS_Dataset_PixelSet
    from learning.metrics import mIou
    from models.pse import PixelSetEncoder
    from src.backbones.ltae import LTAE2d

    data, exp = cfg.dataset, cfg.experiment
    split = cfg.validation
    folds = [list(split[key]) for key in ("train_folds", "val_folds", "test_folds")]
    if any(not group for group in folds) or sorted(sum(folds, [])) != [1, 2, 3, 4, 5]:
        raise ValueError("Train, validation and reserved test must partition folds 1–5")
    if data.num_classes != 18 or cfg.evaluation.selection_metric != "macro_iou":
        raise ValueError("Use the verified 18-class mapping and validation macro IoU")
    def score_predictions(truth, pred):
        return {"accuracy": float(np.mean(np.asarray(truth) == pred)),
                "macro_iou": float(mIou(truth, pred, data.num_classes))}

    if "estimator" in cfg.model:
        from tabular import run
        run(cfg, folds, out, PASTIS_Dataset_PixelSet, score_predictions)
        return
    if (cfg.model.name != "adapted_pse_ltae2d" or data.sensor != "S2"
            or list(cfg.inputs.sensors) != ["S2"] or "embedding_path" in cfg.inputs
            or exp.train_limit is not None
            or split.get("train_limit") is not None):
        raise ValueError("The neural pilot uses S2 and its existing batch-budget settings")

    if exp.batch_size < 2 or exp.epochs < 1 or (exp.max_batches is not None and exp.max_batches < 1):
        raise ValueError("Invalid epoch or batch settings")
    np.random.seed(exp.seed)
    torch.manual_seed(exp.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Submit this runner on a GPU node")

    def load(fold_ids, norm):
        return PASTIS_Dataset_PixelSet(
            folder=data.root, sats=[data.sensor], folds=fold_ids, norm=norm,
            n_pixel=data.n_pixel, label_offset=1, cache=False, geom_features=False,
        )

    train = load(folds[0], True)
    val = load(folds[1], False)
    val.norm[data.sensor] = train.norm[data.sensor]
    val_batches = batches(val, data.sensor, exp.batch_size)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.pse = PixelSetEncoder(10, mlp1=[10, 32, 64],
                                      pooling="mean_std", mlp2=[128, 128], with_extra=False)
            self.temporal = LTAE2d(in_channels=128)
            self.head = nn.Linear(128, data.num_classes)

        def forward(self, pixels, masks, dates):
            features = self.pse((pixels, masks))
            return self.head(self.temporal(
                features[..., None, None], batch_positions=dates).flatten(1))

    model = Classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    best = -float("inf")
    history = []
    for epoch in range(1, exp.epochs + 1):
        model.train()
        train_batches = batches(train, data.sensor, exp.batch_size,
                                np.random.default_rng(exp.seed + epoch))
        seen = 0
        loss_sum = 0.0
        for step, indices in enumerate(train_batches[:exp.max_batches], 1):
            pixels, masks, dates, labels = read_batch(train, indices, data.sensor, device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(pixels, masks, dates), labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}, batch {step}")
            loss.backward()
            optimizer.step()
            seen += len(indices)
            loss_sum += loss.item() * len(indices)
            if step % 200 == 0:
                print(f"Epoch {epoch}: {step} batches, {seen} parcels", flush=True)
        if not seen:
            raise RuntimeError("No training batches")

        # Fix validation pixel sampling across epochs without changing training RNG.
        rng_state = np.random.get_state()
        np.random.seed(exp.seed)
        model.eval()
        rows = []
        with torch.inference_mode():
            for indices in val_batches[:exp.max_batches]:
                pixels, masks, dates, labels = read_batch(val, indices, data.sensor, device)
                logits = model(pixels, masks, dates)
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError("Non-finite validation logits")
                rows.extend(zip((int(val.id_parcels[i]) for i in indices),
                                labels.cpu().tolist(), logits.argmax(1).cpu().tolist()))
        np.random.set_state(rng_state)
        if not rows or (exp.max_batches is None and len(rows) != len(val)):
            raise RuntimeError("Incomplete validation")
        _, truth, pred = zip(*rows)
        scores = score_predictions(truth, pred)
        score = scores["macro_iou"]
        record = {"epoch": epoch, "train_parcels": seen, "validation_parcels": len(rows),
                  "train_loss": loss_sum / seen,
                  **scores}
        history.append(record)
        (out / "metrics.json").write_text(json.dumps(history, indent=2))
        if score > best:
            best = score
            torch.save({"epoch": epoch, "model": model.state_dict()}, out / "best.pt")
            with (out / "validation-predictions.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["parcel_id", "label_index", "prediction_index"])
                writer.writerows(rows)
        print(record, flush=True)


if __name__ == "__main__":
    main()
