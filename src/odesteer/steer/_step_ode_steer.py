'''
Step-based ODE steering methods
'''

from abc import abstractmethod

from torch import Tensor

from ._base_steer import Steer
from ..utils.kernels import KernelClassifier, LinearClassifier, MLPClassifier, RFFClassifier, MLPClassifier
from ..utils.kernels import NormedPolyClassifier


class BaseStepODESteer(Steer):
    def __init__(self, **kwargs):
        super().__init__()    
        self.clf = self._init_clf(**kwargs)
        
    def fit(self, pos_X: Tensor, neg_X_or_labels: Tensor) -> 'BaseStepODESteer':
        self.clf.fit(pos_X, neg_X_or_labels)
        return self
    
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        vf = self.vector_field(X)
        return X + T * vf
    
    def vector_field(self, X: Tensor) -> Tensor:
        self.clf.to(X.device)
        raw_grad = self.clf.grad(X)
        return raw_grad / (raw_grad.norm(dim = -1, keepdim = True) + 1e-10)
    
    @abstractmethod
    def _init_clf(self, **kwargs) -> KernelClassifier:
        raise NotImplementedError

        
class StepODESteer(BaseStepODESteer):
    '''
    One-step ODESteer used in the paper with NormedPolyCntSketch classifier
    '''
    def _init_clf(self, **kwargs) -> NormedPolyClassifier:
        return NormedPolyClassifier(**kwargs)
    
    
class RFFStepODESteer(BaseStepODESteer):
    '''
    Ablation study for One-step ODESteer with RFF classifier
    '''
    def _init_clf(self, **kwargs) -> RFFClassifier:
        return RFFClassifier(**kwargs)


class LinearStepODESteer(BaseStepODESteer):
    '''
    커널 없이 raw activation 위에서 직접 LR/SVM을 학습하는 one-step 스티어.
    grad(X) = coef (상수)이므로 vector field가 X 전체에서 동일한 방향을 가리킴.
    ablation: kernel approximation이 실제로 얼마나 기여하는지 확인용.
    '''
    def _init_clf(self, **kwargs) -> LinearClassifier:
        return LinearClassifier(**kwargs)
    
class MLPStepODESteer(BaseStepODESteer):
    def _init_clf(self, **kwargs):
        return MLPClassifier(**kwargs)

