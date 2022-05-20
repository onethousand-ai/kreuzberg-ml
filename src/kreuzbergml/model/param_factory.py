from abc import ABC, abstractmethod
from typing import Dict, Iterable

import numpy as np
import sklearn


class AbstractGridSearchParamsFactory(ABC):
    @abstractmethod
    def get_model_class(self):
        pass

    @abstractmethod
    def get_param_dict(self) -> Dict[str, Iterable]:
        pass


class L2RegularizedLRParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.linear_model.LogisticRegression

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
            "C": list(range(11)),
        }
        return param_dict


class ENRegularizedLRParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.linear_model.LogisticRegression

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "penalty": ["elasticnet"],
            "solver": ["saga"],
            "C": list(range(11)),
            "l1_ratio": list(np.linspace(0, 1, num=5)),
        }
        return param_dict

class RFCParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.ensemble.RandomForestClassifier

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "max_depth": list(range(2,11)),
            "n_estimator": [8, 16, 32, 64, 100, 200, 500, 1000],
            "min_samples_split": [2, 3, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
        }
        return param_dict
