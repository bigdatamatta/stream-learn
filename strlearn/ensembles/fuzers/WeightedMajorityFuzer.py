from typing import List

import numpy as np
from attr import attrs, attrib
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from strlearn.ensembles.fuzers.SupportsExtractor import SupportsExtractor
from strlearn.ensembles.fuzers.api.BaseEnsemblePredictionFuzer import BaseEnsemblePredictionFuzer


@attrs(auto_attribs=True)
class WeightedMajorityFuzer(BaseEnsemblePredictionFuzer):
    _ensemble: List[BaseEstimator] = attrib()
    _weights: List[float] = attrib()
    _classes: List = attrib()

    def get_supports(self, x):
        return SupportsExtractor(self._ensemble, self._weights, self._classes).extract(x)

    def predict(self, x):
        x = check_array(x)

        supports_sum_by_sample = self.get_supports(x)

        predictions = [self._classes[idx] for idx in np.argmax(supports_sum_by_sample, axis=1)]

        return np.array(predictions)

    def predict_proba(self, x):
        x = check_array(x)

        return self.get_supports(x)
