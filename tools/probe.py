"""Linear probe for frozen representations: closed-form ridge, the regression probe of torchgeo/geo-eval-harness
(CoordBench `linear_probe_score`), adapted to AgriBench's nested tuning (protocols/yieldsat.md, 2.6).

As in the harness: features are z-scored with the training fold's statistics, the intercept is not penalised,
the ridge solution is computed in closed form in float64, and the penalty is chosen by cross-validation over a
log-spaced grid, with a warning when the best value sits on the grid edge. Different from the harness: the
inner folds are supplied by the runner (they mirror the outer scheme) and selection uses pixel RMSE.

The penalty is per training sample (loss = mean squared error + lam * ||w||^2), so the same grid means the same
amount of shrinkage whatever the number of training pixels. One eigendecomposition of X'X per fold gives the
solution for every lam at once, so the whole grid costs about one ridge fit.
"""
import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

DEFAULT_LAMBDAS = tuple(float(v) for v in np.logspace(-6, 3, 19))
CHUNK = 32768


def _stats(x, rows):
    mean = np.zeros(x.shape[1])
    sq = np.zeros(x.shape[1])
    for a in range(0, len(rows), CHUNK):
        c = x[rows[a:a + CHUNK]].astype(np.float64)
        mean += c.sum(0)
        sq += (c * c).sum(0)
    mean /= len(rows)
    std = np.sqrt(np.maximum(sq / len(rows) - mean ** 2, 0))
    std[std < 1e-12] = 1.0
    return mean, std


def _gram(x, y, rows, mean, std):
    d = x.shape[1]
    g, b = np.zeros((d, d)), np.zeros(d)
    ym = float(y[rows].mean())
    for a in range(0, len(rows), CHUNK):
        r = rows[a:a + CHUNK]
        c = (x[r].astype(np.float64) - mean) / std
        g += c.T @ c
        b += c.T @ (y[r] - ym)
    return g, b, ym


def _solve(g, b, n, lambdas):
    """Ridge weights for every lam: (d, len(lambdas))."""
    evals, evecs = np.linalg.eigh(g)
    evals = np.maximum(evals, 0)
    proj = evecs.T @ b
    return evecs @ (proj[:, None] / (evals[:, None] + n * np.asarray(lambdas)[None, :]))


def _predict(x, rows, mean, std, w, intercept):
    out = []
    for a in range(0, len(rows), CHUNK):
        c = (x[rows[a:a + CHUNK]].astype(np.float64) - mean) / std
        out.append(c @ w + intercept)
    return np.concatenate(out)


class RidgeProbe(BaseEstimator, RegressorMixin):
    """Closed-form ridge with the penalty chosen on supplied CV folds: fit(X, y, cv=[(train_idx, val_idx), ...])."""

    def __init__(self, lambdas=DEFAULT_LAMBDAS):
        self.lambdas = lambdas

    def fit(self, X, y, cv=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError("RidgeProbe needs imputed (finite) features")
        lambdas = np.asarray(self.lambdas, dtype=np.float64)
        self.cv_results_ = []
        if cv is not None and len(lambdas) > 1:
            sse = np.zeros(len(lambdas))
            per_fold = []
            for tr, va in cv:
                tr, va = np.asarray(tr), np.asarray(va)
                mean, std = _stats(X, tr)
                g, b, ym = _gram(X, y, tr, mean, std)
                w = _solve(g, b, len(tr), lambdas)
                p = _predict(X, va, mean, std, w, ym)
                err = ((p - y[va, None]) ** 2).mean(0)
                per_fold.append(np.sqrt(err))
            rmse = np.mean(per_fold, axis=0)  # mean over folds of fold RMSE, as GridSearchCV would report
            best = int(np.argmin(rmse))
            self.cv_results_ = [{"lambda": float(l), "inner_rmse": float(r)} for l, r in zip(lambdas, rmse)]
            if best in (0, len(lambdas) - 1):
                warnings.warn(f"RidgeProbe: best lambda {lambdas[best]:g} is on the grid edge; widen the grid")
            self.lambda_ = float(lambdas[best])
            self.best_score_ = float(rmse[best])
        else:
            self.lambda_ = float(lambdas[0])
            self.best_score_ = None
        rows = np.arange(len(X))
        self.mean_, self.std_ = _stats(X, rows)
        g, b, ym = _gram(X, y, rows, self.mean_, self.std_)
        self.coef_ = _solve(g, b, len(X), [self.lambda_])[:, 0]
        self.intercept_ = ym
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return _predict(X, np.arange(len(X)), self.mean_, self.std_, self.coef_[:, None], self.intercept_)[:, 0]
