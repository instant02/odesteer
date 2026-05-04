from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import torch
from torch import Tensor, nn

from ._rff import RFF
from ._poly_cnt_sketch import PolyCntSketch, NormedPolyCntSketch


class KernelClassifier(nn.Module):
    def __init__(self, lin_clf_type: Literal['lr', 'svm'] = 'lr'):
        super().__init__()
        self.lin_clf_type = lin_clf_type
        self.kernel: nn.Module = None
        self.fitted: bool = False
        
    def fit(self, pos_X: Tensor, neg_X_or_labels: Tensor) -> 'KernelClassifier':
        if neg_X_or_labels.ndim == 1:
            return self._fit_with_labels(pos_X, neg_X_or_labels)
        else:
            assert neg_X_or_labels.shape[1] == pos_X.shape[1], \
                f'neg_X_or_labels.shape[1] = {neg_X_or_labels.shape[1]} != pos_X.shape[1] = {pos_X.shape[1]}'
            return self._fit_with_two_sets(pos_X, neg_X_or_labels)
    
    def forward(self, X: Tensor) -> Tensor:
        assert self.fitted, 'KernelClassifier is not fitted'
        return self.predict_proba(X)
        
    def predict(self, X: Tensor) -> Tensor:
        assert self.fitted, 'KernelClassifier is not fitted'
        self.kernel.to(X.device)
        return self.predict_proba(X) > 0.5
    
    def predict_proba(self, X: Tensor) -> Tensor:
        assert self.fitted, 'KernelClassifier is not fitted'
        self.kernel.to(X.device)
        Z = self.kernel.transform(X)
        return (Z @ self.coef + self.intercept).sigmoid()
    
    def score(self, X: Tensor, y: Tensor) -> float:
        assert self.fitted, 'KernelClassifier is not fitted'
        return (self.predict(X) == y).float().mean().item()
    
    def density_ratio(self, X: Tensor) -> Tensor:
        assert self.fitted, 'RFFClassifier is not fitted'
        self.kernel.to(X.device)
        Z = self.rff.transform(X)
        return (Z @ self.coef + self.intercept).exp() * self.dre_coeff
    
    def log_dre(self, X: Tensor) -> Tensor:
        assert self.fitted, 'RFFClassifier is not fitted'
        Z = self.kernel.transform(X)
        return Z @ self.coef + self.intercept + np.log(self.dre_coeff)
    
    def grad(self, X: Tensor) -> Tensor:
        assert self.fitted, 'KernelClassifier is not fitted'
        self.kernel.to(X.device)
        return self.kernel.vjp(X, self.coef) * self.dre_coeff
    
    def _fit_with_two_sets(self, pos_X: Tensor, neg_X: Tensor) -> 'KernelClassifier':
        pi = len(pos_X) / (len(pos_X) + len(neg_X))
        self.dre_coeff = (1 - pi) / pi
        X = torch.cat([pos_X, neg_X], dim = 0)
        y = torch.cat([torch.ones(pos_X.shape[0]), torch.zeros(neg_X.shape[0])])
        return self._fit_with_labels(X, y)
    
    def _fit_with_labels(self, X: Tensor, y: Tensor) -> 'KernelClassifier':
        pi = len(X[y == 1]) / len(X)
        self.dre_coeff = (1 - pi) / pi
        Z = self.kernel.fit_transform(X)
        coef, intercept = self._fit_linear_clf(Z, y)
        self.register_buffer(
            'coef', 
            torch.as_tensor(coef, dtype = X.dtype, device = X.device),
        )
        self.register_buffer(
            'intercept', 
            torch.as_tensor(intercept, dtype = X.dtype, device = X.device),
        )
        return self
    
    def _fit_linear_clf(self, X: Tensor, y: Tensor):
        if self.lin_clf_type == 'lr':
            clf = LogisticRegression(max_iter = 1000)
        elif self.lin_clf_type == 'svm':
            clf = LinearSVC(max_iter = 1000)
        else:
            raise ValueError(f'Invalid linear classifier type: {self.lin_clf_type}')
        clf.fit(X, y)
        coef = torch.as_tensor(clf.coef_.ravel(), dtype = X.dtype, device = X.device)
        intercept = torch.as_tensor(clf.intercept_.ravel(), dtype = X.dtype, device = X.device)
        self.fitted = True
        return coef, intercept    


class LinearClassifier(nn.Module):
    """
    Kernel 없이 raw activation 위에서 직접 LR/SVM 학습.
    grad(X) = coef (상수 — X에 무관).
    """
    def __init__(self, lin_clf_type: Literal['lr', 'svm'] = 'lr', **kwargs):
        super().__init__()
        self.lin_clf_type = lin_clf_type
        self.fitted = False

    def fit(self, pos_X: Tensor, neg_X: Tensor) -> 'LinearClassifier':
        pi = len(pos_X) / (len(pos_X) + len(neg_X))
        self.dre_coeff = (1 - pi) / pi
        X = torch.cat([pos_X, neg_X], dim=0)
        y = torch.cat([torch.ones(len(pos_X)), torch.zeros(len(neg_X))])

        if self.lin_clf_type == 'lr':
            clf = LogisticRegression(max_iter=1000)
        elif self.lin_clf_type == 'svm':
            clf = LinearSVC(max_iter=1000)
        else:
            raise ValueError(f'Invalid lin_clf_type: {self.lin_clf_type}')

        clf.fit(X.cpu().numpy(), y.cpu().numpy())
        self.register_buffer('coef', torch.as_tensor(clf.coef_.ravel(), dtype=X.dtype))
        self.register_buffer('intercept', torch.as_tensor(clf.intercept_.ravel(), dtype=X.dtype))
        self.fitted = True
        return self

    def predict_proba(self, X: Tensor) -> Tensor:
        return (X @ self.coef + self.intercept).sigmoid()

    def grad(self, X: Tensor) -> Tensor:
        """선형 분류기의 gradient는 coef 상수 — X 전체에 동일하게 broadcast."""
        return self.coef.unsqueeze(0).expand_as(X) * self.dre_coeff


class RFFClassifier(KernelClassifier):
    def __init__(
        self,
        n_components: int,
        sigma: float | Literal['median', 'scale'] = 'median',
        lin_clf_type: Literal['lr', 'svm'] = 'lr',
    ):
        super().__init__(lin_clf_type)
        self.kernel = RFF(n_components, sigma)


class PolyClassifier(KernelClassifier):
    def __init__(
        self,
        degree: int = 2,
        n_components: int = 100,
        gamma: float = 1.0,
        coef0: float = 0.1,
        lin_clf_type: str = 'lr',
    ):
        super().__init__(lin_clf_type)
        self.kernel = PolyCntSketch(degree, n_components, gamma, coef0)    
        

class NormedPolyClassifier(KernelClassifier):
    def __init__(
        self,
        degree: int = 2,
        n_components: int = 100,
        gamma: float = 1.0,
        coef0: float = 0.1,
        lin_clf_type: str = 'lr',
    ):
        super().__init__(lin_clf_type)
        self.kernel = NormedPolyCntSketch(degree, n_components, gamma, coef0)


class MLPClassifier(nn.Module):
    def __init__(self, hidden_dims=[512, 256], lr=1e-3, epochs=50, batch_size=256, **kwargs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.fitted = False
        self.net = None  # fit() 할 때 input_dim 보고 생성

    def _build_net(self, input_dim):
        dims = [input_dim] + self.hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def fit(self, pos_X, neg_X):
        pi = len(pos_X) / (len(pos_X) + len(neg_X))
        self.dre_coeff = (1 - pi) / pi

        X = torch.cat([pos_X, neg_X], dim=0).float()
        y = torch.cat([torch.ones(len(pos_X)), torch.zeros(len(neg_X))]).float()

        self._build_net(X.shape[1])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self.net(xb).squeeze(-1), yb
                )
                loss.backward()
                optimizer.step()

        self.net.eval()
        self.fitted = True
        return self

    def grad(self, X):
        with torch.enable_grad():
            X_req = X.detach().float().requires_grad_(True)
            logit = self.net(X_req).squeeze(-1).sum()
            grad = torch.autograd.grad(logit, X_req)[0]
        return grad * self.dre_coeff