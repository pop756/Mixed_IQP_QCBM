"""Bernoulli-Bernoulli RBM trained with PCD (persistent contrastive divergence).

Training settings follow Hinton, "A Practical Guide to Training Restricted
Boltzmann Machines" (UTML TR 2010-003):
  - W ~ N(0, 0.01), visible bias = log[p/(1-p)] of data marginals, hidden bias 0
  - momentum 0.5 for the first 5% of updates, then 0.9
  - weight decay 1e-4 on W only (not biases)
  - PCD with #persistent chains = batch size
Learning-rate / hidden-size / PCD-k grids are set in run_baselines.py and are
grounded in the recio2025 repo's own baseline search
(paper/hyperparam_opt/hyperparameter_settings.py + training/best_hyperparameters.yaml).

Exact marginal p(v) for n_v <= 16 via the closed-form free energy
  F(v) = -b.v - sum_j softplus(c_j + (vW)_j),   p(v) = e^{-F(v)} / Z.
"""
from __future__ import annotations

import numpy as np
import torch


class RBM:
    def __init__(self, n_visible: int, n_hidden: int, seed: int = 0):
        self.nv = int(n_visible)
        self.nh = int(n_hidden)
        g = torch.Generator().manual_seed(int(seed))
        self.W = 0.01 * torch.randn(self.nv, self.nh, generator=g, dtype=torch.float32)
        self.b = torch.zeros(self.nv, dtype=torch.float32)   # visible bias
        self.c = torch.zeros(self.nh, dtype=torch.float32)   # hidden bias
        self._g = g

    @property
    def n_params(self) -> int:
        return self.nv * self.nh + self.nv + self.nh

    # ---- conditionals ----
    def _ph(self, v):                      # p(h=1|v)
        return torch.sigmoid(v @ self.W + self.c)

    def _pv(self, h):                      # p(v=1|h)
        return torch.sigmoid(h @ self.W.T + self.b)

    def _bern(self, p):
        return torch.bernoulli(p, generator=self._g)

    # ---- training (PCD) ----
    def fit(self, data: np.ndarray, *, n_updates: int, lr: float, k: int = 1,
            batch_size: int = 128, weight_decay: float = 1e-4,
            momentum_early: float = 0.5, momentum_late: float = 0.9,
            val_data: np.ndarray | None = None, log_every: int = 1000):
        """Train with PCD-k. Returns a history list of dicts (recon MSE +
        free-energy gap train-vs-val, the overfitting monitor from Hinton sec.6)."""
        X = torch.as_tensor(np.asarray(data, dtype=np.float32))
        Xval = None if val_data is None else torch.as_tensor(np.asarray(val_data, dtype=np.float32))
        n = X.shape[0]
        # Hinton: visible bias init to log odds of the data marginals
        m = X.mean(0).clamp(1e-4, 1 - 1e-4)
        self.b = torch.log(m / (1 - m))
        # persistent chains = batch size (Hinton sec.4.1)
        chains = self._bern(torch.full((batch_size, self.nv), 0.5))
        dW = torch.zeros_like(self.W)
        db = torch.zeros_like(self.b)
        dc = torch.zeros_like(self.c)
        warm = max(1, int(0.05 * n_updates))
        hist = []
        for step in range(1, n_updates + 1):
            mom = momentum_early if step <= warm else momentum_late
            idx = torch.randint(0, n, (batch_size,), generator=self._g)
            v0 = X[idx]
            ph0 = self._ph(v0)
            # negative phase from the persistent chains
            for _ in range(max(1, k)):
                h = self._bern(self._ph(chains))
                chains = self._bern(self._pv(h))
            phk = self._ph(chains)
            gW = (v0.T @ ph0 - chains.T @ phk) / batch_size - weight_decay * self.W
            gb = (v0 - chains).mean(0)
            gc = (ph0 - phk).mean(0)
            dW.mul_(mom).add_(lr * gW)
            db.mul_(mom).add_(lr * gb)
            dc.mul_(mom).add_(lr * gc)
            self.W += dW
            self.b += db
            self.c += dc
            if step == 1 or step % log_every == 0 or step == n_updates:
                with torch.no_grad():
                    rec = ((v0 - self._pv(self._bern(ph0))) ** 2).mean().item()
                    row = {"step": step, "recon_mse": rec}
                    if Xval is not None:
                        # free-energy gap (train minus val); growing |gap| = overfitting
                        ii = torch.randint(0, n, (min(n, 1024),), generator=self._g)
                        ft = self.free_energy(X[ii]).mean().item()
                        fv = self.free_energy(Xval[: 1024]).mean().item()
                        row["fe_gap"] = ft - fv
                    hist.append(row)
        return hist

    # ---- free energy / exact distribution ----
    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        return -(v @ self.b) - torch.nn.functional.softplus(v @ self.W + self.c).sum(-1)

    def exact_logp(self) -> torch.Tensor:
        """Exact log p over all 2^nv states (nv <= 16 only); state index bit i = (idx>>i)&1
        (matches evaluate_sweep.emp_z bit order)."""
        assert self.nv <= 16, "exact enumeration only for nv <= 16"
        ns = 1 << self.nv
        a = torch.arange(ns, dtype=torch.int64)
        V = torch.stack([(a >> i) & 1 for i in range(self.nv)], dim=1).to(torch.float32)
        with torch.no_grad():
            logp = -self.free_energy(V).to(torch.float64)
        return logp - torch.logsumexp(logp, 0)

    # ---- sampling (block Gibbs, independent parallel chains) ----
    @torch.no_grad()
    def sample(self, n_samples: int, burn_in: int, seed: int,
               init: np.ndarray | None = None) -> np.ndarray:
        """One sample per independent chain after burn_in block-Gibbs sweeps
        (random uniform init unless `init` rows are given)."""
        g = torch.Generator().manual_seed(int(seed))
        if init is None:
            v = torch.bernoulli(torch.full((n_samples, self.nv), 0.5), generator=g)
        else:
            v = torch.as_tensor(np.asarray(init, dtype=np.float32)).clone()
        for _ in range(int(burn_in)):
            h = torch.bernoulli(self._ph(v), generator=g)
            v = torch.bernoulli(self._pv(h), generator=g)
        return v.to(torch.int8).numpy()

    # ---- (de)serialization ----
    def state_dict(self):
        return {"nv": self.nv, "nh": self.nh, "W": self.W, "b": self.b, "c": self.c}

    @classmethod
    def from_state(cls, sd):
        rbm = cls(sd["nv"], sd["nh"])
        rbm.W = sd["W"].clone().to(torch.float32)
        rbm.b = sd["b"].clone().to(torch.float32)
        rbm.c = sd["c"].clone().to(torch.float32)
        return rbm
