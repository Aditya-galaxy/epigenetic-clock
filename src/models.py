"""Epigenetic Aging Clock Models.

Implements the classical Horvath non-linear transformation, ElasticNet penalized linear clock,
and DeepEpigeneticClock (sparse regularized MLP in PyTorch).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import ElasticNetCV
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def horvath_transform(y: Union[float, np.ndarray], adult_age: float = 20.0) -> np.ndarray:
    """Horvath non-linear transformation of chronological age.

    Models logarithmic rate of epigenetic changes during childhood/adolescence (<20 yrs)
    and linear progression throughout adult life.

    F(y) = log(y + 1) - log(adult_age + 1) if y <= adult_age
         = (y - adult_age) / (adult_age + 1) if y > adult_age
    """
    y_arr = np.asarray(y, dtype=float)
    c = np.log(adult_age + 1.0)
    denom = adult_age + 1.0
    transformed = np.where(y_arr <= adult_age, np.log(y_arr + 1.0) - c, (y_arr - adult_age) / denom)
    return transformed


def inverse_horvath_transform(z: Union[float, np.ndarray], adult_age: float = 20.0) -> np.ndarray:
    """Inverse of the Horvath age transformation, mapping model logits to years.

    F^-1(z) = (adult_age + 1) * exp(z) - 1 if z < 0
            = (adult_age + 1) * z + adult_age if z >= 0
    """
    z_arr = np.asarray(z, dtype=float)
    mult = adult_age + 1.0
    raw_age = np.where(z_arr < 0.0, mult * np.exp(z_arr) - 1.0, mult * z_arr + adult_age)
    return np.maximum(raw_age, 0.0)


class ElasticNetClock:
    """Canonical Horvath/Hannum style Elastic Net regularized epigenetic clock."""

    def __init__(
        self,
        l1_ratios: Optional[List[float]] = None,
        cv: int = 5,
        random_state: int = 42,
        use_horvath_transform: bool = True,
    ):
        if l1_ratios is None:
            l1_ratios = [0.2, 0.5, 0.7, 0.85, 0.95, 0.99]
        self.l1_ratios = l1_ratios
        self.cv = cv
        self.random_state = random_state
        self.use_horvath_transform = use_horvath_transform
        self.model = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            cv=self.cv,
            random_state=self.random_state,
            max_iter=3000,
            n_jobs=-1,
        )
        self.cpg_ids: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, cpg_ids: Optional[List[str]] = None) -> "ElasticNetClock":
        """Train regularized Elastic Net clock on methylation beta matrix X and ages y."""
        if cpg_ids is not None:
            self.cpg_ids = list(cpg_ids)
        else:
            self.cpg_ids = [f"cpg_{i}" for i in range(X.shape[1])]

        target_y = horvath_transform(y) if self.use_horvath_transform else y
        self.model.fit(X, target_y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict biological age in years for test samples."""
        raw_pred = self.model.predict(X)
        if self.use_horvath_transform:
            return inverse_horvath_transform(raw_pred)
        return np.maximum(raw_pred, 0.0)

    def get_selected_cpgs(self) -> Dict[str, float]:
        """Return non-zero CpG coefficients sorted by absolute weight."""
        coefs = self.model.coef_
        selected = {}
        for cpg, weight in zip(self.cpg_ids, coefs):
            if abs(weight) > 1e-5:
                selected[cpg] = float(weight)
        return dict(sorted(selected.items(), key=lambda item: abs(item[1]), reverse=True))


class _MLPClock(nn.Module):
    """Feedforward neural network for non-linear epigenetic age modeling."""

    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DeepEpigeneticClock:
    """Deep learning epigenetic clock with L1 weight sparsity and non-linear interactions."""

    def __init__(
        self,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        l1_penalty: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_penalty = l1_penalty
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model: Optional[_MLPClock] = None
        self.cpg_ids: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, cpg_ids: Optional[List[str]] = None) -> "DeepEpigeneticClock":
        """Train deep neural clock with AdamW and sparsity penalties."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if cpg_ids is not None:
            self.cpg_ids = list(cpg_ids)
        else:
            self.cpg_ids = [f"cpg_{i}" for i in range(X.shape[1])]

        n_samples, n_features = X.shape
        self.model = _MLPClock(in_features=n_features, hidden_dim=self.hidden_dim)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        y_trans = horvath_transform(y).astype(np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_trans, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, n_samples), shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                mse_loss = criterion(pred, batch_y)

                # L1 penalty on input weights for biological feature selection
                l1_reg = torch.sum(torch.abs(self.model.net[0].weight))
                loss = mse_loss + self.l1_penalty * l1_reg

                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict age in years."""
        assert self.model is not None, "Model must be fitted before predict."
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            raw_pred = self.model(X_tensor).cpu().numpy()
        return inverse_horvath_transform(raw_pred)

    def get_feature_importances(self) -> Dict[str, float]:
        """Compute relative importance based on L1-norm of input layer weights."""
        assert self.model is not None
        first_layer_weights = self.model.net[0].weight.detach().cpu().numpy()  # (hidden_dim, in_features)
        importances = np.mean(np.abs(first_layer_weights), axis=0)
        importances /= np.sum(importances) + 1e-12

        feat_dict = {cpg: float(imp) for cpg, imp in zip(self.cpg_ids, importances)}
        return dict(sorted(feat_dict.items(), key=lambda x: x[1], reverse=True))
