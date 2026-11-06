"""sklearn-compatible Lightning heads, so neural heads run under the same protocol as RF/XGBoost/TabPFN."""
import copy
import logging
import warnings

import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*does not have many workers.*")


class _MLP(LightningModule):
    def __init__(self, n_in, n_out, hidden, dropout, lr, weight_decay):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)
        self.lr, self.weight_decay = lr, weight_decay

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.cross_entropy(self(x), y))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class _Best(Callback):
    """Remember the weights with the lowest validation loss."""

    def __init__(self):
        self.best, self.state = float("inf"), None

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("val_loss")
        if value is not None and float(value) < self.best:
            self.best = float(value)
            self.state = copy.deepcopy(module.state_dict())


class LightningMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(256,), dropout=0.1, lr=1e-3, weight_decay=1e-4, max_epochs=50,
                 patience=5, batch_size=256, val_fraction=0.1, seed=0, accelerator="auto"):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.seed = seed
        self.accelerator = accelerator

    def fit(self, X, y):
        seed_everything(self.seed, workers=True)
        X = np.asarray(X, dtype=np.float32)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        self.mean_ = X.mean(0)
        self.std_ = X.std(0) + 1e-6
        Xs = (X - self.mean_) / self.std_
        stratify = y_idx if np.bincount(y_idx).min() >= 2 else None
        tr, va = train_test_split(np.arange(len(Xs)), test_size=self.val_fraction,
                                  random_state=self.seed, stratify=stratify)

        def loader(idx, shuffle):
            ds = TensorDataset(torch.from_numpy(Xs[idx]), torch.from_numpy(y_idx[idx]).long())
            return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

        module = _MLP(Xs.shape[1], len(self.classes_), list(self.hidden),
                      self.dropout, self.lr, self.weight_decay)
        best = _Best()
        trainer = Trainer(max_epochs=self.max_epochs, accelerator=self.accelerator, devices=1,
                          logger=False, enable_checkpointing=False, enable_progress_bar=False,
                          enable_model_summary=False, num_sanity_val_steps=0,
                          callbacks=[EarlyStopping("val_loss", patience=self.patience), best])
        trainer.fit(module, loader(tr, True), loader(va, False))
        if best.state is not None:
            module.load_state_dict(best.state)
        self.module_ = module.eval()
        return self

    def predict_proba(self, X):
        x = torch.from_numpy((np.asarray(X, dtype=np.float32) - self.mean_) / self.std_)
        device = next(self.module_.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                out.append(torch.softmax(self.module_(x[i:i + 4096].to(device)), 1).cpu())
        return torch.cat(out).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
