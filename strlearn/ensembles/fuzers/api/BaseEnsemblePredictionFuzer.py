from abc import ABCMeta, abstractmethod


class BaseEnsemblePredictionFuzer(metaclass=ABCMeta):

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x):
        raise NotImplementedError
